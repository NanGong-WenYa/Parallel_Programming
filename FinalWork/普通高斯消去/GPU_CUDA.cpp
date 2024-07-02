#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include <iomanip>
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include"device_functions.h"
#include<windows.h>

using namespace std;
const int N = 1024;
const int BLOCK_SIZE = 1024;
float **M;
float elm[N][N] = { 0 };

float* result = new float[N * N];
float* temp = new float[N * N];

long long head, tail, freq;

void M_init() {     
    M = new float* [N];
    for (int i = 0; i < N; i++) {
        M[i] = new float[N];
    }
    for (int i = 0; i < N; i++) {
        M[i][i] = 1.0;
        for (int j = i + 1; j < N; j++) {
            M[i][j] = rand() % 5000;
        }

    }
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            for (int j = 0; j < N; j++) {
                M[i][j] += M[k][j];
                M[i][j] = (int)M[i][j] % 5000;
            }
        }
    }
}

void copy() {
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			temp[i * N + j] = M[i][j];
		}
	}
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            elm[i][j] = M[i][j];
        }
    }
}

void trans() {
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			elm[i][j] = result[i * N + j];
		}
	}
}

void print_final() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << elm[i][j] << " ";
        }
        cout << endl;
    }
}

void LUMethod() {    
    for (int k = 0; k < N; k++) {
        for (int j = k + 1; j < N; j++) {
            elm[k][j] = elm[k][j] / elm[k][k];
        }
        elm[k][k] = 1.0;

        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++) {
                elm[i][j] = elm[i][j] - elm[i][k] * elm[k][j];
            }
            elm[i][k] = 0;
        }
    }
}


__global__ void division_Method(float* data, int k, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int element = data[k * N + k];
    int temp = data[k * N + tid];
    data[k * N + tid] = (float)temp / element;
    return;
}

__global__ void eliminate_kernel(float* data, int k, int N) {
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tx == 0)
        data[k * N + k] = 1.0;
    int row = k + 1 + blockIdx.x;
    while (row < N) {
        int tid = threadIdx.x;
        while (k + 1 + tid < N) {
            int col = k + 1 + tid;
            float temp_1 = data[(row * N) + col];
            float temp_2 = data[(row * N) + k];
            float temp_3 = data[k * N + col];
            data[(row * N) + col] = temp_1 - temp_2 * temp_3;
            tid = tid + blockDim.x;
        }
        __syncthreads();//块内同步
        if (threadIdx.x == 0) {
            data[row * N + k] = 0;
        }
        row += gridDim.x;
    }
    return;
}

int main() {
    M_init();
    copy();
    QueryPerformanceFrequency((LMRGE_INTEGER*)&freq);

    QueryPerformanceCounter((LMRGE_INTEGER*)&head);
    LUMethod();
    QueryPerformanceCounter((LMRGE_INTEGER*)&tail);
    cout << "CPU_LUMethod:" << (tail - head) * 1000 / freq << "ms" << endl;
    print_final();       

	cudaError_t ret;
	float* gpudata;
	int size = N * N * sizeof(float);
	ret = cudaMalloc(&gpudata, size);
	if (ret != cudaSuccess) {
		printf("cudaMalloc gpudata failed!\n");

	}
	ret = cudaMemcpy(gpudata, temp, size, cudaMemcpyHostToDevice);
	if (ret != cudaSuccess) {
		printf("cudaMemcpyHostToDevice failed!\n");
	}

	Dimension dimGrid(BLOCK_SIZE, 1);
	Dimension Grid(1, 1);

	cudaEvent_t start, stop;
	float elapsedTime = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	for (int k = 0; k < N; k++) {
		division_Method << <Grid, dimGrid >> > (gpudata, k, N);
		cudaDeviceSynchronize();
		ret = cudaGetLastError();
		if (ret != cudaSuccess) {
			printf("division_Method failed, %s\n", cudaGetErrorString(ret));
		}
		eliminate_kernel << <Grid, dimGrid >> > (gpudata, k,N);
		cudaDeviceSynchronize();
		ret = cudaGetLastError();
		if (ret != cudaSuccess) {
			printf("eliminate_kernel failed, %s\n", cudaGetErrorString(ret));
		}
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf(elapsedTime);
	cudaError_t cudaStatus2 = cudaGetLastError();
	if (cudaStatus2 != cudaSuccess) {
		fprintf( cudaGetErrorString(cudaStatus2));
	}
	ret = cudaMemcpy(result, gpudata, size, cudaMemcpyDeviceToHost);
	if (ret != cudaSuccess) {
		printf("failed!\n");
	}
	trans();
    print_final();
	cudaFree(gpudata);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
