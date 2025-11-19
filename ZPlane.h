#ifndef ZPLANE_HPP
#define ZPLANE_HPP

#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <algorithm>
#include <map>
#include <set>
#include <cmath>
#include <stdexcept>
#include <iomanip>

namespace zp {

    // =======================================================================
    // 基础定义
    // =======================================================================

    enum class OpType {
        LoadConst, LoadCoeff, LoadInput,
        ReadDelay, // 核心：单位延迟 z^-1
        Add, Sub, Mul, Div,
        BlackBox
    };

    template<typename T> using CoeffFunc = std::function<T()>;
    template<typename T> using BlackBoxFunc = std::function<T(T)>;

    template<typename T>
    struct Node {
        OpType type;
        T value = T(0);
        CoeffFunc<T> coeffFunc = nullptr;
        BlackBoxFunc<T> bbFunc = nullptr;
        std::shared_ptr<Node<T>> left = nullptr;
        std::shared_ptr<Node<T>> right = nullptr;

        // 编译期状态
        int regIndex = -1;
        int visitState = 0;

        Node(OpType t) : type(t) {}

        std::shared_ptr<Node<T>> clone() const {
            auto n = std::make_shared<Node<T>>(type);
            n->value = value; n->coeffFunc = coeffFunc; n->bbFunc = bbFunc;
            if (left) n->left = left->clone();
            if (right) n->right = right->clone();
            return n;
        }
    };

    // =======================================================================
    // 运行时实例
    // =======================================================================

    struct MacOp { uint16_t targetReg; uint16_t sourceReg; uint16_t coeffIdx; };
    struct LinearBlock { std::vector<MacOp> ops; std::vector<uint16_t> clearTargets; };
    enum class GeneralOpType { BlackBox, Div, MulSignal, Assign, Load };
    struct GeneralBlock { GeneralOpType type; uint16_t targetReg; uint16_t src1Reg; uint16_t src2Reg; int funcIdx; };
    struct ExecutionStep { bool isLinear; LinearBlock linear; GeneralBlock general; };
    struct DelayUpdate { uint16_t delayIdx; uint16_t srcReg; };

    template<typename T> class ZCompiler;

    template<typename T>
    class ZInstance {
        std::vector<T> registers;
        std::vector<T> delayState;
        std::vector<CoeffFunc<T>> coeffPool;
        std::vector<BlackBoxFunc<T>> bbPool;
        std::vector<ExecutionStep> pipeline;
        std::vector<DelayUpdate> delayUpdates;
        std::vector<std::pair<int, int>> delayReadMap;
        int outputRegIndex = 0;

    public:
        T Tick(T input) {
            if (registers.empty()) return T(0);
            registers[0] = input;

            // z^-1: 读上一帧状态
            for (const auto& mapping : delayReadMap) {
                registers[mapping.second] = delayState[mapping.first];
            }

            for (const auto& step : pipeline) {
                if (step.isLinear) {
                    for (auto t : step.linear.clearTargets) registers[t] = T(0);
                    for (const auto& op : step.linear.ops) {
                        registers[op.targetReg] += registers[op.sourceReg] * coeffPool[op.coeffIdx]();
                    }
                }
                else {
                    const auto& g = step.general;
                    switch (g.type) {
                    case GeneralOpType::BlackBox:
                        registers[g.targetReg] = bbPool[g.funcIdx](registers[g.src1Reg]);
                        break;
                    case GeneralOpType::Div: {
                        T den = registers[g.src2Reg];
                        registers[g.targetReg] = (std::abs(den) > 1e-12) ? registers[g.src1Reg] / den : T(0);
                    } break;
                    case GeneralOpType::MulSignal:
                        registers[g.targetReg] = registers[g.src1Reg] * registers[g.src2Reg];
                        break;
                    case GeneralOpType::Assign:
                        registers[g.targetReg] = registers[g.src1Reg];
                        break;
                    case GeneralOpType::Load:
                        registers[g.targetReg] = coeffPool[g.funcIdx]();
                        break;
                    }
                }
            }

            // 更新延迟状态
            for (const auto& du : delayUpdates) {
                delayState[du.delayIdx] = registers[du.srcReg];
            }

            return registers[outputRegIndex];
        }
        friend class ZCompiler<T>;
    };

    // =======================================================================
    // 编译器
    // =======================================================================

    template<typename T>
    class ZCompiler {
        std::shared_ptr<Node<T>> root;
        ZInstance<T> instance;
        int regCounter = 0;
        int delayCounter = 0;
        std::vector<Node<T>*> topoOrder;
        std::map<Node<T>*, int> delayIdMap;
        std::vector<Node<T>*> delayNodes;

        int getCoeff(CoeffFunc<T> func) { instance.coeffPool.push_back(func); return (int)instance.coeffPool.size() - 1; }
        int getConstCoeff(T val) { return getCoeff([val]() { return val; }); }

        // 支持任意深度的 z(k) 扫描
        void preScanDelays(Node<T>* n, std::set<Node<T>*>& visited) {
            if (!n || visited.count(n)) return;
            visited.insert(n);
            if (n->type == OpType::ReadDelay) {
                n->regIndex = regCounter++;
                delayIdMap[n] = delayCounter++;
                delayNodes.push_back(n);
                instance.delayReadMap.push_back({ delayIdMap[n], n->regIndex });
            }
            if (n->left) preScanDelays(n->left.get(), visited);
            if (n->right) preScanDelays(n->right.get(), visited);
        }

        void topoSort(Node<T>* n) {
            if (!n) return;
            if (n->visitState == 2) return;
            if (n->visitState == 1) {
                throw std::runtime_error("Algebraic Loop detected! (Feedback without delay)");
            }
            n->visitState = 1;

            if (n->type != OpType::ReadDelay) {
                if (n->left) topoSort(n->left.get());
                if (n->right) topoSort(n->right.get());
            }

            if (n->type == OpType::LoadInput) n->regIndex = 0;
            else if (n->type != OpType::ReadDelay) n->regIndex = regCounter++;

            topoOrder.push_back(n);
            n->visitState = 2;
        }

        bool isSignal(Node<T>* n) { return !(n->type == OpType::LoadConst || n->type == OpType::LoadCoeff); }
        bool isLinearOp(Node<T>* n) {
            if (n->type == OpType::Add || n->type == OpType::Sub) return true;
            if (n->type == OpType::Mul) return (isSignal(n->left.get()) ^ isSignal(n->right.get()));
            return false;
        }

        void generatePipeline() {
            LinearBlock currentLinear;
            auto flushLinear = [&]() {
                if (currentLinear.ops.empty() && currentLinear.clearTargets.empty()) return;
                ExecutionStep step; step.isLinear = true; step.linear = currentLinear;
                instance.pipeline.push_back(step);
                currentLinear = LinearBlock();
                };

            for (Node<T>* n : topoOrder) {
                if (n->type == OpType::ReadDelay) {
                    if (n->left) {
                        DelayUpdate du; du.delayIdx = delayIdMap[n]; du.srcReg = n->left->regIndex;
                        instance.delayUpdates.push_back(du);
                    }
                    continue;
                }
                if (n->type == OpType::LoadInput) continue;

                if (isLinearOp(n)) {
                    currentLinear.clearTargets.push_back(n->regIndex);
                    auto addMac = [&](Node<T>* src, T constVal, int coeffIdx = -1) {
                        int cIdx = coeffIdx; if (cIdx == -1) cIdx = getConstCoeff(constVal);
                        currentLinear.ops.push_back({ (uint16_t)n->regIndex, (uint16_t)src->regIndex, (uint16_t)cIdx });
                        };
                    if (n->type == OpType::Add) { addMac(n->left.get(), T(1)); addMac(n->right.get(), T(1)); }
                    else if (n->type == OpType::Sub) { addMac(n->left.get(), T(1)); addMac(n->right.get(), T(-1)); }
                    else if (n->type == OpType::Mul) {
                        Node<T>* sig = isSignal(n->left.get()) ? n->left.get() : n->right.get();
                        Node<T>* cf = isSignal(n->left.get()) ? n->right.get() : n->left.get();
                        int cIdx = -1;
                        if (cf->type == OpType::LoadConst) cIdx = getConstCoeff(cf->value);
                        else if (cf->type == OpType::LoadCoeff) cIdx = getCoeff(cf->coeffFunc);
                        addMac(sig, T(0), cIdx);
                    }
                }
                else {
                    flushLinear();
                    ExecutionStep step; step.isLinear = false; step.general.targetReg = n->regIndex;
                    bool push = true;
                    if (n->type == OpType::BlackBox) {
                        step.general.type = GeneralOpType::BlackBox;
                        step.general.src1Reg = n->left->regIndex;
                        instance.bbPool.push_back(n->bbFunc);
                        step.general.funcIdx = (int)instance.bbPool.size() - 1;
                    }
                    else if (n->type == OpType::LoadConst) { step.general.type = GeneralOpType::Load; step.general.funcIdx = getConstCoeff(n->value); }
                    else if (n->type == OpType::LoadCoeff) { step.general.type = GeneralOpType::Load; step.general.funcIdx = getCoeff(n->coeffFunc); }
                    else if (n->type == OpType::Div) { step.general.type = GeneralOpType::Div; step.general.src1Reg = n->left->regIndex; step.general.src2Reg = n->right->regIndex; }
                    else if (n->type == OpType::Mul) { step.general.type = GeneralOpType::MulSignal; step.general.src1Reg = n->left->regIndex; step.general.src2Reg = n->right->regIndex; }
                    else push = false;
                    if (push) instance.pipeline.push_back(step);
                }
            }
            flushLinear();
        }

        void resetVisitState(Node<T>* n) {
            if (!n || n->visitState == 0) return;
            n->visitState = 0;
            if (n->left) resetVisitState(n->left.get());
            if (n->right) resetVisitState(n->right.get());
        }

    public:
        ZCompiler(std::shared_ptr<Node<T>> r) : root(r) {}

        ZInstance<T> Compile() {
            regCounter = 1; delayCounter = 0;
            std::set<Node<T>*> visited;
            instance.delayReadMap.clear();

            preScanDelays(root.get(), visited);

            resetVisitState(root.get());
            for (auto* dn : delayNodes) if (dn->left) resetVisitState(dn->left.get());

            try {
                topoSort(root.get());
                for (auto* dn : delayNodes) if (dn->left) topoSort(dn->left.get());
            }
            catch (...) { throw; }

            instance.registers.resize(regCounter, T(0));
            instance.delayState.resize(delayCounter, T(0));
            generatePipeline();
            if (root) instance.outputRegIndex = root->regIndex;
            return instance;
        }
    };

    // =======================================================================
    // 前端构建类 (ZPlane)
    // =======================================================================

    template<typename T>
    class ZPlane {
    public:
        std::shared_ptr<Node<T>> root;

        static std::shared_ptr<Node<T>> makeNode(OpType t, std::shared_ptr<Node<T>> l, std::shared_ptr<Node<T>> r) {
            auto n = std::make_shared<Node<T>>(t); n->left = l; n->right = r; return n;
        }

        static void replaceInput(std::shared_ptr<Node<T>> curr, std::shared_ptr<Node<T>> repl) {
            if (!curr) return;
            if (curr->left) {
                if (curr->left->type == OpType::LoadInput) curr->left = repl;
                else replaceInput(curr->left, repl);
            }
            if (curr->right) {
                if (curr->right->type == OpType::LoadInput) curr->right = repl;
                else replaceInput(curr->right, repl);
            }
        }

        static void injectFeedback(std::shared_ptr<Node<T>> curr, std::shared_ptr<Node<T>> feedbackSource, std::set<Node<T>*>& visited) {
            if (!curr || visited.count(curr.get())) return;
            visited.insert(curr.get());
            if (curr->left) {
                if (curr->left->type == OpType::LoadInput) curr->left = feedbackSource;
                else injectFeedback(curr->left, feedbackSource, visited);
            }
            if (curr->right) {
                if (curr->right->type == OpType::LoadInput) curr->right = feedbackSource;
                else injectFeedback(curr->right, feedbackSource, visited);
            }
        }

        bool isOpenBox() const { return (root->type == OpType::BlackBox); }

    public:
        ZPlane() : ZPlane(T(1)) {}
        ZPlane(std::shared_ptr<Node<T>> n) : root(n) {}
        ZPlane(T val) : root(std::make_shared<Node<T>>(OpType::LoadConst)) { root->value = val; }
        ZPlane(CoeffFunc<T> func) : root(std::make_shared<Node<T>>(OpType::LoadCoeff)) { root->coeffFunc = func; }

        ZPlane(BlackBoxFunc<T> func, bool implicitDelay) : root(std::make_shared<Node<T>>(OpType::BlackBox)) {
            root->bbFunc = func;
            if (implicitDelay) {
                auto delayNode = std::make_shared<Node<T>>(OpType::ReadDelay);
                delayNode->left = std::make_shared<Node<T>>(OpType::LoadInput);
                root->left = delayNode;
            }
            else {
                root->left = std::make_shared<Node<T>>(OpType::LoadInput);
            }
        }

        static ZPlane Box(BlackBoxFunc<T> func) { return ZPlane(func, false); }
        static ZPlane SequentialBox(BlackBoxFunc<T> func) { return ZPlane(func, true); }
        static ZPlane Ref(T& var) { return ZPlane([&var]() { return var; }); }
        static ZPlane Input() { return ZPlane(std::make_shared<Node<T>>(OpType::LoadInput)); }

        friend ZPlane operator+(const ZPlane& l, const ZPlane& r) { return ZPlane(makeNode(OpType::Add, l.root, r.root)); }
        friend ZPlane operator-(const ZPlane& l, const ZPlane& r) { return ZPlane(makeNode(OpType::Sub, l.root, r.root)); }
        /*
        friend ZPlane operator*(const ZPlane& l, const ZPlane& r) {
            if (l.isOpenBox()) return l(r);
            return ZPlane(makeNode(OpType::Mul, l.root, r.root));
        }*/
        // 1. 增强检测逻辑
        static bool isPipe(std::shared_ptr<Node<T>> n) {
            if (!n) return false;
            if (n->type == OpType::LoadInput) return true;
            if (n->type == OpType::BlackBox || n->type == OpType::ReadDelay) {
                return isPipe(n->left);
            }
            return false;
        }

        // 2. 全能乘法重载
        friend ZPlane operator*(const ZPlane& l, const ZPlane& r) {
            bool l_is_pipe = isPipe(l.root);
            bool r_is_pipe = isPipe(r.root);

            if (l_is_pipe) {
                // Pipe * Signal 或 Pipe * Pipe -> 把 r 塞进 l
                // 这解决了 z(12) * ExternalDelay (Pipe * Pipe)
                // 和 ExternalDelay * H (Pipe * Signal)
                return l(r);
            }

            if (r_is_pipe) {
                // Signal * Pipe -> 把 l 塞进 r
                // 这解决了 ExternalDelay * H * z(12)
                // 此时左边 (Ext*H) 是 Signal，右边 z(12) 是 Pipe
                // 结果变成 z(12)( Ext(H) ) -> 也就是 H 被两层延迟包裹，正确！
                return r(l);
            }

            // 都是信号，执行普通乘法
            return ZPlane(makeNode(OpType::Mul, l.root, r.root));
        }

        // 除法：含 Lattice 智能检测
        friend ZPlane operator/(const ZPlane& Num, const ZPlane& Den) {
            std::shared_ptr<Node<T>> k_node = nullptr, link_node = nullptr;
            bool denMatched = false;
            auto checkMul = [&](std::shared_ptr<Node<T>> n) -> bool {
                if (n->type == OpType::Mul) {
                    if (n->left->type == OpType::LoadCoeff || n->left->type == OpType::LoadConst) {
                        k_node = n->left; link_node = n->right; return true;
                    }
                    if (n->right->type == OpType::LoadCoeff || n->right->type == OpType::LoadConst) {
                        k_node = n->right; link_node = n->left; return true;
                    }
                }
                return false;
                };

            if (Den.root->type == OpType::Add) {
                if (Den.root->left->type == OpType::LoadConst && std::abs(Den.root->left->value - 1.0) < 1e-9) {
                    if (checkMul(Den.root->right)) denMatched = true;
                }
                else if (Den.root->right->type == OpType::LoadConst && std::abs(Den.root->right->value - 1.0) < 1e-9) {
                    if (checkMul(Den.root->left)) denMatched = true;
                }
            }

            if (denMatched && Num.root->type == OpType::Add) {
                std::shared_ptr<Node<T>> n_k = nullptr, n_link = nullptr;
                bool numMatched = false;
                auto isSame = [](std::shared_ptr<Node<T>> a, std::shared_ptr<Node<T>> b) { return a == b; };
                if (isSame(Num.root->left, k_node)) { n_k = Num.root->left; n_link = Num.root->right; numMatched = true; }
                else if (isSame(Num.root->right, k_node)) { n_k = Num.root->right; n_link = Num.root->left; numMatched = true; }

                if (numMatched && isSame(n_link, link_node)) {
                    ZPlane X = ZPlane::Input();
                    ZPlane K(k_node);
                    ZPlane V(link_node);
                    ZPlane W = X - K * V;
                    std::set<Node<T>*> visited;
                    injectFeedback(V.root, W.root, visited);
                    ZPlane Y = V + K * W;
                    return Y;
                }
            }
            return ZPlane(makeNode(OpType::Div, Num.root, Den.root));
        }

        friend ZPlane operator+(const ZPlane& l, T r) { return l + ZPlane(r); }
        friend ZPlane operator+(T l, const ZPlane& r) { return ZPlane(l) + r; }
        friend ZPlane operator-(const ZPlane& l, T r) { return l - ZPlane(r); }
        friend ZPlane operator-(T l, const ZPlane& r) { return ZPlane(l) - r; }
        friend ZPlane operator*(const ZPlane& l, T r) { return l * ZPlane(r); }
        friend ZPlane operator*(T l, const ZPlane& r) { return ZPlane(l) * r; }
        friend ZPlane operator/(const ZPlane& l, T r) { return l / ZPlane(r); }
        friend ZPlane operator/(T l, const ZPlane& r) { return ZPlane(l) / r; }

        ZPlane operator()(const ZPlane& replacement) const {
            auto newRoot = root->clone();
            replaceInput(newRoot, replacement.root);
            return ZPlane(newRoot);
        }

        ZInstance<T> MakeInstance() const {
            ZCompiler<T> compiler(root);
            return compiler.Compile();
        }

        // ===================================================================
        // [Restored] Z 工厂类：支持 z(1), z(k) 语法
        // ===================================================================
        struct ZFactory {
            // z(k) 返回一个包含 k 个延迟节点的 ZPlane
            ZPlane operator()(int k = 1) const {
                if (k < 1) return ZPlane(T(1)); // z^0 = 1

                // 创建链：ReadDelay -> ReadDelay ... -> LoadInput
                // 例如 z(2): ReadDelay -> ReadDelay -> LoadInput
                auto head = std::make_shared<Node<T>>(OpType::ReadDelay);
                auto curr = head;
                for (int i = 1; i < k; ++i) {
                    curr->left = std::make_shared<Node<T>>(OpType::ReadDelay);
                    curr = curr->left;
                }
                curr->left = std::make_shared<Node<T>>(OpType::LoadInput);

                return ZPlane(head);
            }
        } static constexpr z{};
    };

    template<typename T> constexpr typename ZPlane<T>::ZFactory ZPlane<T>::z;

} // namespace ZLab

#endif