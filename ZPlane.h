#pragma once

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
        ReadDelay, // 读延迟寄存器（环路切断点）
        Add, Sub, Mul, Div,
        BlackBox   // 外部函数/黑盒
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
        int visitState = 0; // 0: Unvisited, 1: Visiting (Gray), 2: Visited (Black)

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
    // 运行时实例 (Runtime Instance)
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
        int outputRegIndex = 0;

    public:
        T ProcessSample(T input) {
            if (registers.empty()) return T(0);
            registers[0] = input; // Reg 0 is always Input
        }

        std::vector<std::pair<int, int>> delayReadMap; // delayIndex -> registerIndex

        T Tick(T input) {
            if (registers.empty()) return T(0);
            registers[0] = input;

            // 1. 读取延迟：将上一帧的 DelayState 写入对应的 ReadDelay 寄存器
            for (const auto& mapping : delayReadMap) {
                registers[mapping.second] = delayState[mapping.first];
            }

            // 2. 执行流水线
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

            // 3. 更新延迟状态 (为下一帧做准备)
            for (const auto& du : delayUpdates) {
                delayState[du.delayIdx] = registers[du.srcReg];
            }

            return registers[outputRegIndex];
        }

        friend class ZCompiler<T>;
    };

    // =======================================================================
    // 编译器 (Compiler)
    // =======================================================================

    template<typename T>
    class ZCompiler {
        std::shared_ptr<Node<T>> root;
        ZInstance<T> instance;
        int regCounter = 0;
        int delayCounter = 0;
        std::vector<Node<T>*> topoOrder;
        std::map<Node<T>*, int> delayIdMap; // Node* -> DelayIndex in State
        std::vector<Node<T>*> delayNodes;

        int getCoeff(CoeffFunc<T> func) { instance.coeffPool.push_back(func); return (int)instance.coeffPool.size() - 1; }
        int getConstCoeff(T val) { return getCoeff([val]() { return val; }); }

        // 预扫描：分配 Delay ID 和寄存器
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

        // 拓扑排序 + 代数环检测 (三色标记法)
        void topoSort(Node<T>* n) {
            if (!n) return;
            if (n->visitState == 2) return; // Black: 已处理
            if (n->visitState == 1) {
                // Gray: 正在访问 -> 发现环！
                // 如果环中包含 ReadDelay，通常不会进入这里，因为 ReadDelay 是叶子节点（在这个DFS视角下）。
                // ReadDelay 不会递归调用 left。
                // 所以如果在这里遇到 Gray，说明这是一个无延迟环（代数环）。
                throw std::runtime_error("Algebraic Loop detected! (Recursion without Delay)");
            }

            n->visitState = 1; // Gray

            // ReadDelay 节点视为源点，不向下递归
            if (n->type != OpType::ReadDelay) {
                if (n->left) topoSort(n->left.get());
                if (n->right) topoSort(n->right.get());
            }

            // 分配寄存器 (Input 固定为 0, ReadDelay 已经在 preScan 分配)
            if (n->type == OpType::LoadInput) n->regIndex = 0;
            else if (n->type != OpType::ReadDelay) n->regIndex = regCounter++;

            topoOrder.push_back(n);
            n->visitState = 2; // Black
        }

        // 辅助判断
        bool isSignal(Node<T>* n) { return !(n->type == OpType::LoadConst || n->type == OpType::LoadCoeff); }

        bool isLinearOp(Node<T>* n) {
            if (n->type == OpType::Add || n->type == OpType::Sub) return true;
            if (n->type == OpType::Mul) {
                // 仅当一边是常数/系数时，乘法才是线性的
                bool l = isSignal(n->left.get());
                bool r = isSignal(n->right.get());
                return (l ^ r); // 异或：一个信号，一个系数
            }
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
                // ReadDelay 和 Input 已经是源，不需要计算指令
                if (n->type == OpType::ReadDelay) {
                    // 但需要记录 Delay 写入的操作 (从 left 写入 state)
                    if (n->left) {
                        DelayUpdate du;
                        du.delayIdx = delayIdMap[n];
                        du.srcReg = n->left->regIndex;
                        instance.delayUpdates.push_back(du);
                    }
                    continue;
                }
                if (n->type == OpType::LoadInput) continue;

                if (isLinearOp(n)) {
                    // 线性优化块
                    currentLinear.clearTargets.push_back(n->regIndex);

                    auto addMac = [&](Node<T>* src, T constVal, int coeffIdx = -1) {
                        int cIdx = coeffIdx; if (cIdx == -1) cIdx = getConstCoeff(constVal);
                        currentLinear.ops.push_back({ (uint16_t)n->regIndex, (uint16_t)src->regIndex, (uint16_t)cIdx });
                        };

                    if (n->type == OpType::Add) {
                        addMac(n->left.get(), T(1));
                        addMac(n->right.get(), T(1));
                    }
                    else if (n->type == OpType::Sub) {
                        addMac(n->left.get(), T(1));
                        addMac(n->right.get(), T(-1));
                    }
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
                    // 非线性/通用操作
                    flushLinear();
                    ExecutionStep step; step.isLinear = false; step.general.targetReg = n->regIndex;
                    bool push = true;

                    if (n->type == OpType::LoadConst) {
                        step.general.type = GeneralOpType::Load;
                        step.general.funcIdx = getConstCoeff(n->value);
                    }
                    else if (n->type == OpType::LoadCoeff) {
                        step.general.type = GeneralOpType::Load;
                        step.general.funcIdx = getCoeff(n->coeffFunc);
                    }
                    else if (n->type == OpType::BlackBox) {
                        step.general.type = GeneralOpType::BlackBox;
                        step.general.src1Reg = n->left->regIndex;
                        instance.bbPool.push_back(n->bbFunc);
                        step.general.funcIdx = (int)instance.bbPool.size() - 1;
                    }
                    else if (n->type == OpType::Div) {
                        step.general.type = GeneralOpType::Div;
                        step.general.src1Reg = n->left->regIndex;
                        step.general.src2Reg = n->right->regIndex;
                    }
                    else if (n->type == OpType::Mul) {
                        step.general.type = GeneralOpType::MulSignal;
                        step.general.src1Reg = n->left->regIndex;
                        step.general.src2Reg = n->right->regIndex;
                    }
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
            regCounter = 1; // 0 is Input
            delayCounter = 0;

            // 1. 扫描延迟节点
            std::set<Node<T>*> visited;
            delayNodes.clear();
            instance.delayReadMap.clear();
            preScanDelays(root.get(), visited);

            // 2. 拓扑排序 (包含代数环检测)
            // 先重置状态，防止多次Compile复用Node出错
            resetVisitState(root.get());
            for (auto* dn : delayNodes) resetVisitState(dn->left.get()); // Delay的输入源也要重置

            topoOrder.clear();
            try {
                topoSort(root.get());
                // 确保延迟节点的输入源也被计算
                for (auto* dn : delayNodes) {
                    if (dn->left) topoSort(dn->left.get());
                }
            }
            catch (const std::exception& e) {
                std::cerr << "[Compiler Error] " << e.what() << std::endl;
                throw;
            }

            // 3. 生成实例
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

        static ZPlane makeBin(OpType t, const ZPlane& l, const ZPlane& r) {
            return ZPlane(makeNode(t, l.root, r.root));
        }

        static void replaceZ(std::shared_ptr<Node<T>> curr, std::shared_ptr<Node<T>> repl) {
            if (!curr) return;
            // 替换逻辑：ReadDelay 的输入端 或 BlackBox(Input) 的输入端
            if (curr->type == OpType::ReadDelay) { curr->left = repl; return; }
            if (curr->type == OpType::BlackBox && curr->left && curr->left->type == OpType::LoadInput) {
                curr->left = repl; return;
            }
            // 递归
            if (curr->left) replaceZ(curr->left, repl);
            if (curr->right) replaceZ(curr->right, repl);
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

        // 检查是否为“待输入”的 BlackBox
        bool isOpenBox() const {
            return (root->type == OpType::BlackBox && root->left && root->left->type == OpType::LoadInput);
        }

    public:
        ZPlane() : ZPlane(T(1)) {}
        ZPlane(std::shared_ptr<Node<T>> n) : root(n) {}
        ZPlane(T val) : root(std::make_shared<Node<T>>(OpType::LoadConst)) { root->value = val; }
        ZPlane(CoeffFunc<T> func) : root(std::make_shared<Node<T>>(OpType::LoadCoeff)) { root->coeffFunc = func; }
        ZPlane(BlackBoxFunc<T> func) : root(std::make_shared<Node<T>>(OpType::BlackBox)) {
            root->bbFunc = func;
            root->left = std::make_shared<Node<T>>(OpType::LoadInput);
        }

        static ZPlane Box(BlackBoxFunc<T> func) { return ZPlane(func); }
        static ZPlane Ref(T& var) { return ZPlane([&var]() { return var; }); }
        static ZPlane Input() { return ZPlane(std::make_shared<Node<T>>(OpType::LoadInput)); }

        // 运算符重载
        friend ZPlane operator+(const ZPlane& l, const ZPlane& r) { return makeBin(OpType::Add, l, r); }
        friend ZPlane operator-(const ZPlane& l, const ZPlane& r) { return makeBin(OpType::Sub, l, r); }

        // 关键修改：智能乘法
        friend ZPlane operator*(const ZPlane& l, const ZPlane& r) {
            // 如果左边是待输入的 BlackBox，解释为 Cascade (Function Application)
            if (l.isOpenBox()) {
                return l(r);
            }
            return makeBin(OpType::Mul, l, r);
        }

        // 智能除法 (Lattice 识别)
        friend ZPlane operator/(const ZPlane& Num, const ZPlane& Den) {
            // 匹配 H = (k + Link) / (1 + k * Link)
            std::shared_ptr<Node<T>> k_node = nullptr;
            std::shared_ptr<Node<T>> link_node = nullptr;
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
                std::shared_ptr<Node<T>> n_k = nullptr;
                std::shared_ptr<Node<T>> n_link = nullptr;

                bool numPattern = false;
                // 检查 (k + Link)
                // 注意：这里需要判断指针是否和分母中的 k, link 对应
                // 简单起见，假设构造时复用了对象，直接比对指针
                auto matchPtr = [](std::shared_ptr<Node<T>> a, std::shared_ptr<Node<T>> b) { return a == b; };

                if (matchPtr(Num.root->left, k_node)) { n_k = Num.root->left; n_link = Num.root->right; numPattern = true; }
                else if (matchPtr(Num.root->right, k_node)) { n_k = Num.root->right; n_link = Num.root->left; numPattern = true; }

                if (numPattern && matchPtr(n_link, link_node)) {
                    // === 构建 Lattice ===
                    // 结构: Y = (K + V) / (1 + K*V)
                    // 变换为: W = X - K*V; Y = V + K*W; 且 V 的输入设为 W
                    ZPlane X = ZPlane::Input();
                    ZPlane K(k_node);
                    ZPlane V(link_node); // V 是 Link

                    // 1. 构造反馈信号 W
                    ZPlane W = X - K * V;

                    // 2. 将 W 注入到 V 的内部输入端 (形成闭环)
                    std::set<Node<T>*> visited;
                    injectFeedback(V.root, W.root, visited);

                    // 3. 构造输出 Y
                    ZPlane Y = V + K * W;
                    return Y;
                }
            }
            return makeBin(OpType::Div, Num, Den);
        }

        friend ZPlane operator+(const ZPlane& l, T r) { return l + ZPlane(r); }
        friend ZPlane operator+(T l, const ZPlane& r) { return ZPlane(l) + r; }
        friend ZPlane operator-(const ZPlane& l, T r) { return l - ZPlane(r); }
        friend ZPlane operator-(T l, const ZPlane& r) { return ZPlane(l) - r; }
        friend ZPlane operator*(const ZPlane& l, T r) { return l * ZPlane(r); }
        friend ZPlane operator*(T l, const ZPlane& r) { return ZPlane(l) * r; }
        friend ZPlane operator/(const ZPlane& l, T r) { return l / ZPlane(r); }
        friend ZPlane operator/(T l, const ZPlane& r) { return ZPlane(l) / r; }

        // 替换/连接操作 (Functor 语法)
        ZPlane operator()(const ZPlane& replacement) const {
            auto newRoot = root->clone();
            replaceZ(newRoot, replacement.root);
            return ZPlane(newRoot);
        }

        ZInstance<T> MakeInstance() const {
            ZCompiler<T> compiler(root);
            return compiler.Compile();
        }

        struct ZFactory {
            // z(k) 返回一个延迟 k 次的 Delay 算子
            // 目前简单实现 z(1)
            ZPlane operator()(int k = 1) const {
                auto n = std::make_shared<Node<T>>(OpType::ReadDelay);
                n->left = std::make_shared<Node<T>>(OpType::LoadInput);
                return ZPlane(n);
            }
        } static constexpr z{};
    };

    template<typename T> constexpr typename ZPlane<T>::ZFactory ZPlane<T>::z;

} // namespace ZP
