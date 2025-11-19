#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <algorithm>
#include <map>
#include <set>
#include <complex>
#include <cmath>
#include <iomanip>

// =======================================================================
// ZPlane 核心库 (智能代数优化版)
// =======================================================================

enum class OpType {
	LoadConst, LoadCoeff, LoadInput,
	ReadDelay, StoreDelay,
	Add, Sub, Mul, Div,
	BlackBox,
	// [新增] 专门的 Lattice 全通节点类型（虽然会被分解，但标记一下逻辑）
	LatticeAllpass
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
	int regIndex = -1;
	bool visited = false;

	Node(OpType t) : type(t) {}

	std::shared_ptr<Node<T>> clone() const {
		auto n = std::make_shared<Node<T>>(type);
		n->value = value; n->coeffFunc = coeffFunc; n->bbFunc = bbFunc;
		if (left) n->left = left->clone();
		if (right) n->right = right->clone();
		return n;
	}
};

// --- 流水线结构 ---
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
		if (!registers.empty()) registers[0] = input;

		// 更新 Delay
		for (size_t i = 0; i < delayState.size(); ++i) {
			if (i + 1 < registers.size()) registers[i + 1] = delayState[i];
		}

		// 执行流水线
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

		for (const auto& du : delayUpdates) {
			delayState[du.delayIdx] = registers[du.srcReg];
		}

		return registers.empty() ? T(0) : registers[outputRegIndex];
	}
	friend class ZCompiler<T>;
};

template<typename T>
class ZCompiler {
	std::shared_ptr<Node<T>> root;
	ZInstance<T> instance;
	int regCounter = 0; int delayCounter = 0;
	std::vector<Node<T>*> topoOrder;
	std::map<Node<T>*, int> delayIdMap;
	std::vector<Node<T>*> delayNodes;

	int getCoeff(CoeffFunc<T> func) { instance.coeffPool.push_back(func); return (int)instance.coeffPool.size() - 1; }
	int getConstCoeff(T val) { return getCoeff([val]() { return val; }); }

	void preScanDelays(Node<T>* n, std::set<Node<T>*>& visited) {
		if (!n || visited.count(n)) return;
		visited.insert(n);
		if (n->type == OpType::ReadDelay) {
			n->regIndex = regCounter++;
			delayIdMap[n] = delayCounter++;
			delayNodes.push_back(n);
		}
		if (n->left) preScanDelays(n->left.get(), visited);
		if (n->right) preScanDelays(n->right.get(), visited);
	}

	void topoSort(Node<T>* n) {
		if (!n || n->visited) return;
		if (n->left) { if (n->type != OpType::ReadDelay) topoSort(n->left.get()); }
		if (n->right) topoSort(n->right.get());
		if (n->type != OpType::ReadDelay && n->type != OpType::LoadInput) n->regIndex = regCounter++;
		else if (n->type == OpType::LoadInput) n->regIndex = 0;
		topoOrder.push_back(n); n->visited = true;
	}

	bool isSignal(Node<T>* n) { return !(n->type == OpType::LoadConst || n->type == OpType::LoadCoeff); }

	bool isLinearOp(Node<T>* n) {
		if (n->type == OpType::Add || n->type == OpType::Sub) return true;
		if (n->type == OpType::Mul) {
			bool l = isSignal(n->left.get()), r = isSignal(n->right.get());
			return (l ^ r);
		}
		return false;
	}

	void generatePipeline() {
		LinearBlock currentLinear;
		auto flushLinear = [&]() {
			if (currentLinear.ops.empty()) return;
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

public:
	ZCompiler(std::shared_ptr<Node<T>> r) : root(r) {}
	ZInstance<T> Compile() {
		regCounter = 1; delayCounter = 0; std::set<Node<T>*> visited; delayNodes.clear();
		preScanDelays(root.get(), visited);
		topoSort(root.get());
		for (auto* dn : delayNodes) if (dn->left) topoSort(dn->left.get());

		instance.registers.resize(regCounter, T(0));
		instance.delayState.resize(delayCounter, T(0));
		generatePipeline();
		if (root) instance.outputRegIndex = root->regIndex;
		return instance;
	}
};

template<typename T>
class ZPlane {
public:
	std::shared_ptr<Node<T>> root;

	// 基础节点构造
	static std::shared_ptr<Node<T>> makeNode(OpType t, std::shared_ptr<Node<T>> l, std::shared_ptr<Node<T>> r) {
		auto n = std::make_shared<Node<T>>(t); n->left = l; n->right = r; return n;
	}

	static ZPlane makeBin(OpType t, const ZPlane& l, const ZPlane& r) { return ZPlane(makeNode(t, l.root, r.root)); }

	static void replaceZ(std::shared_ptr<Node<T>> curr, std::shared_ptr<Node<T>> repl) {
		if (!curr) return;
		// 递归替换：如果遇到 Delay 或者 BlackBox(Input)，替换为新的结构
		if (curr->type == OpType::ReadDelay) { curr->left = repl; return; }
		if (curr->type == OpType::BlackBox && curr->left && curr->left->type == OpType::LoadInput) { curr->left = repl; return; }
		replaceZ(curr->left, repl); replaceZ(curr->right, repl);
	}

	// 辅助函数：深度递归注入 W (用于 Lattice 反馈)
	static void injectFeedback(std::shared_ptr<Node<T>> curr, std::shared_ptr<Node<T>> feedbackSource, std::set<Node<T>*>& visited) {
		if (!curr || visited.count(curr.get())) return;
		visited.insert(curr.get());

		// 关键逻辑：找到 "Input" 节点并替换为 Feedback
		// 这里的 Input 指的是被 Link 包裹的子系统的输入端
		if (curr->left) {
			if (curr->left->type == OpType::LoadInput) curr->left = feedbackSource;
			else injectFeedback(curr->left, feedbackSource, visited);
		}
		if (curr->right) {
			if (curr->right->type == OpType::LoadInput) curr->right = feedbackSource;
			else injectFeedback(curr->right, feedbackSource, visited);
		}
	}

	ZPlane() : ZPlane(T(1)) {}
	ZPlane(std::shared_ptr<Node<T>> n) : root(n) {}
	ZPlane(T val) : root(std::make_shared<Node<T>>(OpType::LoadConst)) { root->value = val; }
	ZPlane(CoeffFunc<T> func) : root(std::make_shared<Node<T>>(OpType::LoadCoeff)) { root->coeffFunc = func; }
	ZPlane(BlackBoxFunc<T> func) : root(std::make_shared<Node<T>>(OpType::BlackBox)) { root->bbFunc = func; root->left = std::make_shared<Node<T>>(OpType::LoadInput); }

	static ZPlane Box(BlackBoxFunc<T> func) { return ZPlane(func); }
	static ZPlane Ref(T& var) { return ZPlane([&var]() { return var; }); }
	static ZPlane Input() { return ZPlane(std::make_shared<Node<T>>(OpType::LoadInput)); }

	friend ZPlane operator+(const ZPlane& l, const ZPlane& r) { return makeBin(OpType::Add, l, r); }
	friend ZPlane operator-(const ZPlane& l, const ZPlane& r) { return makeBin(OpType::Sub, l, r); }
	friend ZPlane operator*(const ZPlane& l, const ZPlane& r) { return makeBin(OpType::Mul, l, r); }

	// =======================================================================
	// [核心黑科技] 智能除法运算符：自动识别全通公式并生成 Lattice 结构
	// =======================================================================
	friend ZPlane operator/(const ZPlane& Num, const ZPlane& Den) {
		// 模式匹配目标：H = (k + Link) / (1 + k * Link)

		// 1. 分析分母 Den 是否为 Add(Const(1), Mul(k, Link)) 形式
		//    或者 Add(Mul(k, Link), Const(1))
		std::shared_ptr<Node<T>> k_node = nullptr;
		std::shared_ptr<Node<T>> link_node = nullptr;
		bool denMatched = false;

		auto checkMul = [&](std::shared_ptr<Node<T>> n) -> bool {
			if (n->type == OpType::Mul) {
				// 假设左边是系数 k，右边是 Link (或反之，乘法交换律)
				// 这里简单起见，假设 k 是 LoadConst 或 LoadCoeff
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
				// 1 + ...
				if (checkMul(Den.root->right)) denMatched = true;
			}
			else if (Den.root->right->type == OpType::LoadConst && std::abs(Den.root->right->value - 1.0) < 1e-9) {
				// ... + 1
				if (checkMul(Den.root->left)) denMatched = true;
			}
		}

		// 2. 如果分母匹配，检查分子 Num 是否为 (k + Link) 形式
		if (denMatched && Num.root->type == OpType::Add) {
			std::shared_ptr<Node<T>> n_k = nullptr;
			std::shared_ptr<Node<T>> n_link = nullptr;

			if (Num.root->left == k_node || (Num.root->left->type == k_node->type && Num.root->left->coeffFunc.target_type() == k_node->coeffFunc.target_type())) {
				// 这里指针比较可能不够，但为了演示，假设对象重用
				// 在实际工程中应比较值的来源 ID
				n_k = Num.root->left; n_link = Num.root->right;
			}
			else {
				n_k = Num.root->right; n_link = Num.root->left;
			}

			// 3. 核心校验：分子分母中的 Link 必须是同一个对象（或者拓扑结构相同）
			// 简便起见，直接比较 shared_ptr (用户代码中复用了变量 Link，所以指针应该相同)
			bool linkMatched = (n_link == link_node);

			if (linkMatched) {
				// === 匹配成功！检测到 Lattice 全通公式 ===
				// 构建 Lattice Wave Digital Filter 结构
				// 拓扑: 
				// W = Input - k * Link(Output_Prev) -- wait, standard lattice
				// Correct One-Multiplier Lattice Allpass:
				// W = X - k * V (V is output of delay block)
				// Inject W into input of delay block
				// Y = V + k * W

				// 这里的 link_node 就是 V (延迟后的值)
				// 但我们需要修改 V 的输入源！

				ZPlane X = ZPlane::Input();
				ZPlane K(k_node);
				ZPlane V(link_node); // V 是上一级输出经过延迟后的信号

				// 构造反馈节点 W
				// 注意：这里利用了编译器会将 Add/Mul 优化为线性块的能力
				ZPlane W = X - K * V;

				// 【关键步骤】将 W 注入到 V (Link) 内部的输入端
				// 这打破了原来的前馈连接，建立了反馈回路
				std::set<Node<T>*> visited;
				injectFeedback(V.root, W.root, visited);

				// 构造输出 Y
				ZPlane Y = V + K * W;

				return Y;
			}
		}

		// 如果不匹配，执行普通除法
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

	ZPlane operator()(const ZPlane& replacement) const { auto newRoot = root->clone(); replaceZ(newRoot, replacement.root); return ZPlane(newRoot); }
	ZInstance<T> MakeInstance() const { ZCompiler<T> compiler(root); return compiler.Compile(); }
	struct ZFactory { ZPlane operator()(int k = 1) const { auto n = std::make_shared<Node<T>>(OpType::ReadDelay); n->left = std::make_shared<Node<T>>(OpType::LoadInput); return ZPlane(n); } } static constexpr z{};
};
template<typename T> constexpr typename ZPlane<T>::ZFactory ZPlane<T>::z;

// =======================================================================
// 测试代码
// =======================================================================

struct ExtraDelay {
	std::vector<double> buf;
	int head = 0;
	ExtraDelay(int len) : buf(len, 0.0) {}
	double process(double x) {
		double y = buf[head];
		buf[head] = x;
		head = (head + 1) % buf.size();
		return y;
	}
};

int main() {
	using Type = double;
	auto z = ZPlane<Type>::z;

	std::cout << "=== Smart Division Lattice Allpass Test ===\n";
	std::cout << "Note: operator/ automatically detects H=(k+z)/(1+kz) and builds a lattice.\n\n";

	const int STAGES = 8;
	Type k[STAGES] = { 0.8, -0.9, 0.9, -0.99, 0.99, -0.999, 0.9999, -0.9 };

	std::vector<std::unique_ptr<ExtraDelay>> delays;
	for (int i = 0; i < STAGES; ++i) {
		delays.push_back(std::make_unique<ExtraDelay>(i + 5));
	}

	ZPlane<Type> H = 1;

	for (int i = STAGES - 1; i >= 0; --i) {
		// 1. 准备 Link (上一级 H 经过延迟)
		// 此时 Link 看起来是前馈的：Link <--- DelayedH
		ZPlane<Type> DelayedH = z(1)(H);
		ZPlane<Type> Link = ZPlane<Type>::Box([ptr = delays[i].get()](Type x) {
			return ptr->process(x);
			});
		Link.root->left = DelayedH.root;

		// 2. 准备系数 K
		auto K = ZPlane<Type>::Ref(k[i]);

		// 3. 【魔法时刻】直接写传递函数除法！
		// operator/ 会检测到 (K+Link)/(1+K*Link) 模式
		// 它会自动断开 Link 的前馈输入，改接到内部反馈节点 W 上
		// 从而生成能量守恒的 Lattice 结构。
		H = (K + Link) / (Type(1.0) + K * Link);
	}

	std::cout << "Compiling..." << std::endl;
	auto proc = H.MakeInstance();
	std::cout << "Ready.\n" << std::endl;

	double totalEnergy = 0.0;
	std::cout << "Time | Input | Output" << std::endl;
	std::cout << "-----|-------|--------" << std::endl;

	for (int t = 0; t < 3000000; ++t) {
		Type input = (t == 0) ? 1.0 : 0.0;
		Type output = proc.ProcessSample(input);

		totalEnergy += output * output;

		if (t < 15) {
			std::cout << std::setw(4) << t << " | "
				<< std::setw(5) << input << " | "
				<< std::setw(8) << output << std::endl;
		}
	}

	std::cout << "...\n";
	std::cout << "Total Energy: " << std::setprecision(6) << totalEnergy << std::endl;

	if (std::abs(totalEnergy - 1.0) < 1e-3) {
		std::cout << "[PASS] Energy Conserved (1.0)" << std::endl;
	}
	else {
		std::cout << "[FAIL] Energy Mismatch" << std::endl;
	}

	return 0;
}