#include <iostream>
#include <vector>
#include <memory>
#include <complex>
#include "ZPlane.h"

using namespace zp;

// 模拟外部音频处理模块（如 Delay Line）
template<typename T>
struct ExtraDelay {
	std::vector<T> buf;
	int head = 0;
	ExtraDelay(int len) : buf(len, 0.0) {}
	T process(T x) {
		T y = buf[head];
		buf[head] = x;
		head = (head + 1) % buf.size();
		return y;
	}
};

int main() {
	using Type = std::complex<double>;
	auto z = ZPlane<Type>::z;

	std::cout << "=== Smart Division Lattice Allpass Test ===\n";

	const int STAGES = 16;
	Type k[STAGES] = { (Type)0.8, (Type)-0.9,(Type)0.9, (Type)-0.99,(Type)0.99,(Type)-0.999,(Type)0.9999, (Type)-0.9 ,(Type)0.8, (Type)-0.9, (Type)0.9, (Type)-0.99,(Type)0.99,(Type)-0.999, (Type)0.9999, (Type)-0.9 };

	std::vector<std::unique_ptr<ExtraDelay<Type>>> delays;
	for (int i = 0; i < STAGES; ++i) {
		delays.push_back(std::make_unique<ExtraDelay<Type>>(i + 1));
	}

	ZPlane<Type> H = (ZPlane<Type>)1; // 初始全通传递函数为 1

	try {
		for (int i = STAGES - 1; i >= 0; --i) {
			auto ExternalDelay = ZPlane<Type>::SequentialBox([ptr = delays[i].get()](Type x) {
				return ptr->process(x);
				});

			ZPlane<Type> Link = ExternalDelay * H;//测试部分

			auto K = ZPlane<Type>::Ref(k[i]);//引用

			H = (K + Link) / (Type(1.0) + K * Link);
		}

		std::cout << "Compiling Pipeline..." << std::endl;
		auto proc = H.MakeInstance();
		std::cout << "Compilation Successful.\n" << std::endl;

		// === 运行测试 ===
		Type totalEnergy = (Type)0.0;
		std::cout << "Time | Input | Output" << std::endl;
		std::cout << "-----|-------|--------" << std::endl;

		for (int t = 0; t < 5000000; ++t) {
			Type input;
			if (t == 0) input = Type{ cos(1.14514),sin(1.14514) }; // 单位冲激
			else input = (Type)0.0;
			Type output = proc.Tick(input);

			totalEnergy += output * output;

			if (t < 30) {
				std::cout << std::setw(4) << t << " | "
					<< std::setw(5) << input << " | "
					<< std::setw(8) << std::fixed << std::setprecision(5) << output << std::endl;
			}
		}
		std::cout << "...\nTotal Energy (Impulse Response): " << std::abs(totalEnergy) << std::endl;

		if (std::abs(std::abs(totalEnergy) - 1.0) < 1e-2)
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