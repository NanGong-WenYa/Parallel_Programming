#include <iostream>
#include <CL/sycl.hpp>
#include <fstream>
#include <chrono>
#include <ratio>
#include <functional>
#include <iomanip>
#include <random>
#include <thread>


using namespace cl::sycl;
long long f, first, last;

void LUMethod(buffer<float, 2>& M, queue& A) {
	host_accessor m{ M ,read_write };
	int n = m.get_range()[0];
	for (int k = 0; k < n; k++) {
		for (int j = k + 1; j < n; j++) {
			m[k][j] = m[k][j] / m[k][k];
		}
		m[k][k] = 1;
		for (int i = k + 1; i < n; i++) {
			for (int j = k + 1; j < n; j++) {
				m[i][j] = m[i][j] - m[i][k] * m[k][j];
			}
			m[i][k] = 0;
		}
	}
}

void print(buffer<float, 2>& M) {
	host_accessor m{ M ,read_only };
	auto range = m.get_range();
	for (int i = 0; i < range[0]; i++) {
		for (int j = 0; j < range[1]; j++) {
			std::cout << std::setw(16) << m[i][j];
		}
		std::cout << std::endl;
	}
}

void copy(buffer<float, 2>& to, buffer<float, 2>& from) {
	host_accessor src{ from ,read_only };
	host_accessor des{ to ,write_only };
	assert(src.get_range() == des.get_range());
	auto range = src.get_range();
	for (int i = 0; i < range[0]; i++) {
		for (int j = 0; j < range[1]; j++) {
			des[i][j] = src[i][j];
		}
	}
}

void matrix_init(buffer<float, 2>& M) {
	host_accessor A{ M ,read_write };

	static std::default_random_engine generator(1337);
	static std::uniform_real_distribution<float> distribution(-1.0, 1.0);

	int N = A.get_range()[0];
	for (int i = 0; i < N; i++) {
        A[i][i] = 1.0;
        for (int j = i + 1; j < N; j++) {
            A[i][j] = rand() % 5000;
        }

    }
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] += A[k][j];
                A[i][j] = (int)A[i][j] % 5000;
            }
        }
    }

}



void OneAPI(buffer<float, 2>& M, queue& A) {


	int n = M.get_range()[0];
	for (int k = 0; k < n; k++) {

		A.submit([&](handler& h) {
			accessor m{ M, h, read_write };
			h.parallel_for(range(n - k), [=](auto idx) {
				int j = k + idx;
				m[k][j] = m[k][j] / m[k][k];
				});
			});

		A.submit([&](handler& h) {
			accessor m{ M, h, read_write };
			h.parallel_for(range(n - (k + 1), n - (k + 1)), [=](auto idx) {
				int i = k + 1 + idx.get_id(0);
				int j = k + 1 + idx.get_id(1);
				m[i][j] = m[i][j] - m[i][k] * m[k][j];
				});
			});

		A.submit([&](handler& h) {
			accessor m{ M, h, read_write };
			h.parallel_for(range(n - (k + 1)), [=](auto idx) {
				int i = k + 1 + idx;
				m[i][k] = 0;
				});
			});
	}
	A.wait();
}


void test(int n, queue& A) {
	buffer<float, 2> M1(range(n, n));
    buffer<float, 2> M2(range(n, n));

	matrix_init(M1);
    copy(M1,M1);
    auto start = std::chrono::high_resolution_clock::now();
    LUMethod(M1,A);
    auto end = std::chrono::high_resolution_clock::now();
   
    double time_ordinary = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout<<"ordinary time:"<<time_ordinary<<std::endl;
    start = std::chrono::high_resolution_clock::now();
    OneAPI(M2,A);
    end = std::chrono::high_resolution_clock::now();
    double time_oneapi = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout<<"oneapi time:"<<time_oneapi<<std::endl;
	return ;
}


int main() {
	queue A(gpu_selector{});   
	device my_device = A.get_device();
	std::cout << "Device: " << my_device.get_info<info::device::name>() << std::endl;
    std::cin>>n;
    test(n,A);
}
