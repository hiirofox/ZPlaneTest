#include <iostream>
#include <vector>
#include <memory>
#include "ZPlane.h"

using namespace ZLab;

// 模拟外部音频处理模块（如 Delay Line）
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

	const int STAGES = 8;
	Type k[STAGES] = { 0.8, -0.9, 0.9, -0.99, 0.99, -0.999, 0.9999, -0.9 };

	std::vector<std::unique_ptr<ExtraDelay>> delays;
	for (int i = 0; i < STAGES; ++i) {
		delays.push_back(std::make_unique<ExtraDelay>(i + 1));
	}

	ZPlane<Type> H = 1; // 初始全通传递函数为 1

	try {
		for (int i = STAGES - 1; i >= 0; --i) {
			// 1. 创建外部延迟 Box
			auto ExternalDelay = ZPlane<Type>::Box([ptr = delays[i].get()](Type x) {
				return ptr->process(x);
				});

			// 2. 准备级联信号
			// 用户需求：希望写成 Link = Link * DelayedH;
			// 这里利用重载的 operator* 实现级联：ExternalDelay 是 Box，DelayedH 是信号
			// 语义：Link 的输入来源于 z(1)(H)

			// 正常延迟写法：
			ZPlane<Type> DelayedH = z(1)(H);

			// 如果用户不小心写成 ZPlane<Type> DelayedH = H; (没有 z(1))
			// 那么 Link * DelayedH 会形成直接代数环，Compiler 将会报错。

			// 使用 operator* 进行级联连接 (Syntactic Sugar)
			ZPlane<Type> Link = ExternalDelay * DelayedH;

			auto K = ZPlane<Type>::Ref(k[i]);

			// 3. 全通滤波器公式 (Operator / 自动转换为 Lattice)
			// H_new = (K + Link) / (1 + K * Link)
			// Link 内部包含了对旧 H 的引用
			H = (K + Link) / (Type(1.0) + K * Link);
		}

		std::cout << "Compiling Pipeline..." << std::endl;
		auto proc = H.MakeInstance();
		std::cout << "Compilation Successful.\n" << std::endl;

		// === 运行测试 ===
		double totalEnergy = 0.0;
		std::cout << "Time | Input | Output" << std::endl;
		std::cout << "-----|-------|--------" << std::endl;

		for (int t = 0; t < 5000000; ++t) {
			Type input = (t == 0) ? 1.0 : 0.0; // 脉冲输入
			Type output = proc.Tick(input);

			totalEnergy += output * output;

			if (t < 10) {
				std::cout << std::setw(4) << t << " | "
					<< std::setw(5) << input << " | "
					<< std::setw(8) << std::fixed << std::setprecision(5) << output << std::endl;
			}
		}
		std::cout << "...\nTotal Energy (Impulse Response): " << totalEnergy << std::endl;

		if (std::abs(totalEnergy - 1.0) < 1e-2)
			std::cout << "[PASS] Energy Conserved (Allpass Property Verified)" << std::endl;
		else
			std::cout << "[FAIL] Energy Mismatch" << std::endl;

	}
	catch (const std::exception& e) {
		std::cerr << "\n[FATAL ERROR] " << e.what() << std::endl;
		std::cerr << "Tip: Did you forget z(1) when creating the loop?" << std::endl;
	}

	return 0;
}