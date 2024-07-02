#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <cstring>
#include <ctime>
#include <chrono>
#include <vector>

using namespace std;

string Xiaoyuanzi = "/home/s2213893/MPI-para/5x.txt";
string Beixiaoyuanzi = "/home/s2213893/MPI-para/5b.txt";

const int maxsize = 3000;
const int maxrow = 60000;
const int numBasis = 100000;

map<int, int*> ans;

fstream RowFile(Beixiaoyuanzi, ios::in | ios::out);
fstream BasisFile(Xiaoyuanzi, ios::in | ios::out);

int gRows[maxrow][maxsize];
int gBasis[numBasis][maxsize];

int ifBasis[numBasis] = {0};

void reset() {
    memset(gRows, 0, sizeof(gRows));
    memset(gBasis, 0, sizeof(gBasis));
    memset(ifBasis, 0, sizeof(ifBasis));
    RowFile.close();
    BasisFile.close();
    RowFile.open(Beixiaoyuanzi, ios::in | ios::out);
    BasisFile.open(Xiaoyuanzi, ios::in | ios::out);
    ans.clear();
}

int readBasis() {
    for (int i = 0; i < numBasis; i++) {
        if (BasisFile.eof()) {
            cout << "读取消元子" << i - 1 << "行" << endl;
            return i - 1;
        }
        string tmp;
        bool flag = false;
        int row = 0;
        getline(BasisFile, tmp);
        stringstream s(tmp);
        int pos;
        while (s >> pos) {
            if (!flag) {
                row = pos;
                flag = true;
                ifBasis[row] = 1;
            }
            int index = pos / 32;
            int offset = pos % 32;
            gBasis[row][index] = gBasis[row][index] | (1 << offset);
        }
    }
    return numBasis;
}

int readRowsFrom(int pos) {
    if (RowFile.is_open())
        RowFile.close();
    RowFile.open(Beixiaoyuanzi, ios::in | ios::out);
    memset(gRows, 0, sizeof(gRows));
    string line;
    for (int i = 0; i < pos; i++) {
        getline(RowFile, line);
    }
    for (int i = pos; i < pos + maxrow; i++) {
        int tmp;
        getline(RowFile, line);
        if (line.empty()) {
            cout << "读取被消元行 " << i << " 行" << endl;
            return i;
        }
        bool flag = false;
        stringstream s(line);
        while (s >> tmp) {
            int index = tmp / 32;
            int offset = tmp % 32;
            gRows[i - pos][index] = gRows[i][index] | (1 << offset);
            flag = true;
        }
    }
    return -1;
}

int findfirst(int row) {
    int first;
    for (int i = maxsize - 1; i >= 0; i--) {
        if (gRows[row][i] == 0)
            continue;
        else {
            int pos = i * 32;
            int offset = 0;
            for (int k = 31; k >= 0; k--) {
                if (gRows[row][i] & (1 << k)) {
                    offset = k;
                    break;
                }
            }
            first = pos + offset;
            return first;
        }
    }
    return -1;
}

void writeResult(ofstream& out) {
    for (auto it = ans.rbegin(); it != ans.rend(); it++) {
        int* result = it->second;
        int max = it->first / 32 + 1;
        for (int i = max; i >= 0; i--) {
            if (result[i] == 0)
                continue;
            int pos = i * 32;
            for (int k = 31; k >= 0; k--) {
                if (result[i] & (1 << k)) {
                    out << k + pos << " ";
                }
            }
        }
        out << endl;
    }
}

__global__ void xorKernel(int* rows, int* basis, int maxsize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < maxsize) {
        rows[idx] ^= basis[idx];
    }
}

__global__ void findFirstKernel(int* gRows, int* first, int maxsize, int maxrow) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < maxrow) {
        for (int i = maxsize - 1; i >= 0; i--) {
            if (gRows[row * maxsize + i] != 0) {
                int pos = i * 32;
                int offset = 0;
                for (int k = 31; k >= 0; k--) {
                    if (gRows[row * maxsize + i] & (1 << k)) {
                        offset = k;
                        break;
                    }
                }
                first[row] = pos + offset;
                return;
            }
        }
        first[row] = -1;
    }
}

__global__ void updateBasisKernel(int* gBasis, int* gRows, int* ifBasis, int row, int first, int maxsize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < maxsize) {
        gBasis[first * maxsize + idx] = gRows[row * maxsize + idx];
        ifBasis[first] = 1;
    }
}

bool try_eliminate(int* basis, int* gBasis, int* ifBasis, int maxsize) {
    int first = findfirst(*basis);
    if (ifBasis[first] == 1) {
        int blockSize = 256;
        int numBlocks = (maxsize + blockSize - 1) / blockSize;
        xorKernel<<<numBlocks, blockSize>>>(basis, &gBasis[first * maxsize], maxsize);
        cudaDeviceSynchronize();
        for (int j = 0; j < maxsize; ++j) {
            if (basis[j] != 0) {
                return true;
            }
        }
        return false;
    }
    return false;
}

void GE_cuda(int rank, int size, int num_threads) {
    int begin = 0;
    int flag;

    if (rank == 0) {
        flag = readRowsFrom(begin);
    }
    MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int num_rows = (flag == -1) ? maxrow : flag;
    auto start_time = chrono::high_resolution_clock::now();

    vector<MPI_Request> send_requests;

    int* d_gRows;
    int* d_gBasis;
    int* d_ifBasis;
    int* d_first;

    cudaMalloc(&d_gRows, maxrow * maxsize * sizeof(int));
    cudaMalloc(&d_gBasis, numBasis * maxsize * sizeof(int));
    cudaMalloc(&d_ifBasis, numBasis * sizeof(int));
    cudaMalloc(&d_first, maxrow * sizeof(int));

    cudaMemcpy(d_gRows, gRows, maxrow * maxsize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gBasis, gBasis, numBasis * maxsize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ifBasis, ifBasis, numBasis * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (maxrow + blockSize - 1) / blockSize;

    findFirstKernel<<<numBlocks, blockSize>>>(d_gRows, d_first, maxsize, maxrow);
    cudaDeviceSynchronize();

    for (int i = rank; i < num_rows; i += size) {
        while (true) {
           int first;
            cudaMemcpy(&first, &d_first[i], sizeof(int), cudaMemcpyDeviceToHost);
        if (first == -1)
        break;
        if (ifBasis[first] == 1) {
            updateBasisKernel<<<numBlocks, blockSize>>>(d_gBasis, d_gRows, d_ifBasis, i, first, maxsize);
            cudaDeviceSynchronize();

            ans.insert({first, new int[maxsize]});
            cudaMemcpy(ans[first], &gBasis[first], maxsize * sizeof(int), cudaMemcpyHostToHost);

            for (int dest = 0; dest < size; dest++) {
                if (dest != rank) {
                    MPI_Request request;
                    MPI_Isend(&gBasis[first], maxsize, MPI_INT, dest, first, MPI_COMM_WORLD, &request);
                    send_requests.push_back(request);
                }
            }
            break;
        } else {
            int blockSize = 256;
            int numBlocks = (maxsize + blockSize - 1) / blockSize;
            xorKernel<<<numBlocks, blockSize>>>(d_gRows + i * maxsize, d_gBasis + first * maxsize, maxsize);
            cudaDeviceSynchronize();
        }
    }
    }

    bool all_done = false;
    while (!all_done) {
    all_done = true;

    for (int source = 0; source < size; source++) {
        if (source != rank) {
            MPI_Status status;
            int flag;
            MPI_Iprobe(source, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
            if (flag) {
                int first = status.MPI_TAG;
                int* received_basis = new int[maxsize];
                MPI_Recv(received_basis, maxsize, MPI_INT, source, first, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                ifBasis[first] = 1;
                cudaMemcpy(&gBasis[first], received_basis, maxsize * sizeof(int), cudaMemcpyHostToHost);

                while (try_eliminate(received_basis, gBasis, ifBasis, maxsize)) {
                    for (int dest = 0; dest < size; dest++) {
                        if (dest != rank) {
                            MPI_Request request;
                            MPI_Isend(received_basis, maxsize, MPI_INT, dest, first, MPI_COMM_WORLD, &request);
                            send_requests.push_back(request);
                        }
                    }
                }
                delete[] received_basis;
            }
        }
    }

    for (auto& request : send_requests) {
        int flag;
        MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
        if (!flag) {
            all_done = false;
        }
    }
    }

    cudaFree(d_gRows);
    cudaFree(d_gBasis);
    cudaFree(d_ifBasis);
    cudaFree(d_first);

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end_time - start_time;
    if (rank == 0) {
        cout << "GE_cuda (" << num_threads << " threads): " << duration.count() << "ms" << endl;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0) {
        readBasis();
    }

    if (rank == 0) {
        reset();
    }
    GE_cuda(rank, size, 8);

    MPI_Finalize();
    return 0;
}
