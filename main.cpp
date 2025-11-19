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

// ... (ZPlane 库代码与上一版完全一致，确保包含 outputRegIndex 修复) ...
// ... (ExtraDelay 类保持不变) ...

// 为了方便，重新粘贴一下必要的库代码部分，确保这是一个独立可运行的文件
// =======================================================================
enum class OpType {
	LoadConst, LoadCoeff, LoadInput,
	ReadDelay, StoreDelay,
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
		for (size_t i = 0; i < delayState.size(); ++i) if (i + 1 < registers.size()) registers[i + 1] = delayState[i];
		for (const auto& step : pipeline) {
			if (step.isLinear) {
				for (auto t : step.linear.clearTargets) registers[t] = T(0);
				for (const auto& op : step.linear.ops) registers[op.targetReg] += registers[op.sourceReg] * coeffPool[op.coeffIdx]();
			}
			else {
				const auto& g = step.general;
				switch (g.type) {
				case GeneralOpType::BlackBox: registers[g.targetReg] = bbPool[g.funcIdx](registers[g.src1Reg]); break;
				case GeneralOpType::Div: { T den = registers[g.src2Reg]; registers[g.targetReg] = (std::abs(den) > 1e-12) ? registers[g.src1Reg] / den : T(0); } break;
				case GeneralOpType::MulSignal: registers[g.targetReg] = registers[g.src1Reg] * registers[g.src2Reg]; break;
				case GeneralOpType::Assign: registers[g.targetReg] = registers[g.src1Reg]; break;
				case GeneralOpType::Load: registers[g.targetReg] = coeffPool[g.funcIdx](); break;
				}
			}
		}
		for (const auto& du : delayUpdates) delayState[du.delayIdx] = registers[du.srcReg];
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
		if (n->type == OpType::ReadDelay) { n->regIndex = regCounter++; delayIdMap[n] = delayCounter++; delayNodes.push_back(n); }
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
		if (n->type == OpType::Mul) { bool l = isSignal(n->left.get()), r = isSignal(n->right.get()); return (l ^ r); }
		return false;
	}
	void generatePipeline() {
		LinearBlock currentLinear;
		auto flushLinear = [&]() {
			if (currentLinear.ops.empty()) return;
			ExecutionStep step; step.isLinear = true; step.linear = currentLinear; instance.pipeline.push_back(step); currentLinear = LinearBlock();
			};
		for (Node<T>* n : topoOrder) {
			if (n->type == OpType::ReadDelay) {
				if (n->left) { DelayUpdate du; du.delayIdx = delayIdMap[n]; du.srcReg = n->left->regIndex; instance.delayUpdates.push_back(du); } continue;
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
				if (n->type == OpType::LoadConst) { step.general.type = GeneralOpType::Load; step.general.funcIdx = getConstCoeff(n->value); }
				else if (n->type == OpType::LoadCoeff) { step.general.type = GeneralOpType::Load; step.general.funcIdx = getCoeff(n->coeffFunc); }
				else if (n->type == OpType::BlackBox) { step.general.type = GeneralOpType::BlackBox; step.general.src1Reg = n->left->regIndex; instance.bbPool.push_back(n->bbFunc); step.general.funcIdx = (int)instance.bbPool.size() - 1; }
				else if (n->type == OpType::Div) { step.general.type = GeneralOpType::Div; step.general.src1Reg = n->left->regIndex; step.general.src2Reg = n->right->regIndex; }
				else if (n->type == OpType::Mul) { step.general.type = GeneralOpType::MulSignal; step.general.src1Reg = n->left->regIndex; step.general.src2Reg = n->right->regIndex; }
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
		generatePipeline();
		instance.registers.resize(regCounter, T(0)); instance.delayState.resize(delayCounter, T(0));
		if (root) instance.outputRegIndex = root->regIndex;
		return instance;
	}
};
template<typename T>
class ZPlane {
public:
	std::shared_ptr<Node<T>> root;
	static ZPlane makeBin(OpType t, const ZPlane& l, const ZPlane& r) { auto n = std::make_shared<Node<T>>(t); n->left = l.root; n->right = r.root; return ZPlane(n); }
	static void replaceZ(std::shared_ptr<Node<T>> curr, std::shared_ptr<Node<T>> repl) {
		if (!curr) return;
		if (curr->type == OpType::ReadDelay) { curr->left = repl; return; }
		if (curr->type == OpType::BlackBox && curr->left && curr->left->type == OpType::LoadInput) { curr->left = repl; return; }
		replaceZ(curr->left, repl); replaceZ(curr->right, repl);
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
	friend ZPlane operator/(const ZPlane& l, const ZPlane& r) { return makeBin(OpType::Div, l, r); }
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

// 辅助函数：手动构建 Lattice Allpass 拓扑
template<typename T>
ZPlane<T> makeLatticeAllpass(ZPlane<T> Link_H_prev, T k_val) {
	auto K = ZPlane<T>::Ref(k_val);
	T k2_val = 1.0 - k_val * k_val;

	ZPlane<T> X = ZPlane<T>::Input();
	std::shared_ptr<Node<T>> v_node = Link_H_prev.root;

	// W = (1-k^2)X - k*V
	auto w_node = (ZPlane<T>(k2_val) * X - ZPlane<T>(k_val) * ZPlane<T>(v_node)).root;

	std::set<Node<T>*> visited;
	std::function<void(std::shared_ptr<Node<T>>)> injectW;
	injectW = [&](std::shared_ptr<Node<T>> curr) {
		if (!curr || visited.count(curr.get())) return;
		visited.insert(curr.get());

		if (curr->left) {
			if (curr->left->type == OpType::LoadInput) curr->left = w_node;
			else injectW(curr->left);
		}
		if (curr->right) {
			if (curr->right->type == OpType::LoadInput) curr->right = w_node;
			else injectW(curr->right);
		}
		};

	injectW(v_node);

	ZPlane<T> Y = ZPlane<T>(k_val) * X + ZPlane<T>(v_node);
	return Y;
}

int main() {
	using Type = double;
	auto z = ZPlane<Type>::z;

	std::cout << "=== 8-Stage Nested Lattice Allpass Test (Fixed) ===\n";

	const int STAGES = 16;
	Type k[STAGES] = { 0.5, -0.4, 0.3, -0.2, 0.6, -0.5, 0.4, -0.3,0.1,0.2,0.4,0.7,0.01,0.001,0.9,0.95 };

	std::vector<std::unique_ptr<ExtraDelay>> delays;
	for (int i = 0; i < STAGES; ++i) {
		delays.push_back(std::make_unique<ExtraDelay>(i + 1));
	}

	ZPlane<Type> H = z(1);

	for (int i = STAGES - 1; i >= 0; --i) {
		// 【关键修改】：先通过 z(1) (VM 原生 Delay) 包裹 H，打断代数环！
		// 物理意义：Lattice Filter 的级联标准就是需要一个 Delay。
		ZPlane<Type> DelayedH = z(1)(H);

		// 然后再串联黑箱延时
		ZPlane<Type> Link = ZPlane<Type>::Box([ptr = delays[i].get()](Type x) {
			return ptr->process(x);
			});

		// 挂载：Link = BlackBox( DelayedH )
		Link.root->left = DelayedH.root;

		// 构建 Lattice
		H = makeLatticeAllpass(Link, k[i]);
	}

	std::cout << "Compiling..." << std::endl;
	auto proc = H.MakeInstance();
	std::cout << "Ready.\n" << std::endl;

	std::vector<double> impulseResponse;
	double totalEnergy = 0.0;

	std::cout << "Time | Input | Output" << std::endl;
	std::cout << "-----|-------|--------" << std::endl;

	for (int t = 0; t < 300000; ++t) {
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