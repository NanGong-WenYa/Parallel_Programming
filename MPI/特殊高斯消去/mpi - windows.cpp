#include <mpi.h>
#include <omp.h>
#include <immintrin.h>
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

int ifBasis[numBasis] = { 0 };

void reset() {
    memset(gRows, 0, sizeof(gRows));
    memset(gBasis, 0, sizeof(gBasis));
    memset(ifBasis, 0, sizeof(ifBasis));
    RowFile.close();
    BasisFile.close();
    RowFile.open(Beixiaoyuanzi, ios::in | ios::out);
    BasisFile.open(Xiaoyuanzi, ios::in | ios_out);
    ans.clear();
}

int readBasis() {
    for (int i = 0; i < numBasis; i++) {
        if (BasisFile.eof()) {
            cout << "Read basis line " << i - 1 << endl;
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
            cout << "Read eliminated row " << i << endl;
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
    for (int i = maxsize - 1; i >= 0; i--) {
        if (gRows[row][i] == 0)
            continue;
        int pos = i * 32;
        int offset = 0;
        for (int k = 31; k >= 0; k--) {
            if (gRows[row][i] & (1 << k)) {
                offset = k;
                break;
            }
        }
        return pos + offset;
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

// Attempt to eliminate function with AVX
bool try_eliminate_avx(int* basis) {
    int first = findfirst(0); // Find the first entry position
    if (ifBasis[first] == 1) {
        for (int j = 0; j < maxsize; j += 8) {
            __m256i rows = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&gBasis[first][j]));
            __m256i basis_chunk = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&basis[j]));
            __m256i result = _mm256_xor_si256(rows, basis_chunk);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&basis[j]), result);
        }

        for (int j = 0; j < maxsize; ++j) {
            if (basis[j] != 0) {
                return true;
            }
        }
        return false; // Entire row eliminated
    }
    return false; // No elimination
}

void GE() {
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < maxrow; i++) {
        while (findfirst(i) != -1) {
            int first = findfirst(i);
            if (ifBasis[first] == 1) {
                for (int j = 0; j < maxsize; j++) {
                    gRows[i][j] = gRows[i][j] ^ gBasis[first][j];
                }
            }
            else {
                for (int j = 0; j < maxsize; j++) {
                    gBasis[first][j] = gRows[i][j];
                }
                ifBasis[first] = 1;
                ans.insert(pair<int, int*>(first, gBasis[first]));
                break;
            }
        }
    }
    auto finish = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = finish - start;
    cout << "GE: " << duration.count() << "ms" << endl;
}
void GE_mpi(int rank, int size) {
    int begin = 0;
    int flag;
    if (rank == 0) {
        flag = readRowsFrom(begin);
    }

    MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int num = (flag == -1) ? maxrow : flag;
    auto start = chrono::high_resolution_clock::now();

    vector<MPI_Request> send_requests;
    vector<int> send_indices;

    for (int i = rank; i < num; i += size) {
        bool basis_updated = false;

        while (findfirst(i) != -1) {
            int first = findfirst(i);
            if (ifBasis[first] == 1) {
                for (int j = 0; j < maxsize; j++) {
                    gRows[i][j] = gRows[i][j] ^ gBasis[first][j];
                }
            }
            else {
                for (int j = 0; j < maxsize; j++) {
                    gBasis[first][j] = gRows[i][j];
                }
                ifBasis[first] = 1;
                ans.insert(pair<int, int*>(first, gBasis[first]));

                basis_updated = true;

                // 将新消元子发送给其他进程
                for (int dest = 0; dest < size; dest++) {
                    if (dest != rank) {
                        MPI_Send(gBasis[first], maxsize, MPI_INT, dest, first, MPI_COMM_WORLD);
                    }
                }
                break;
            }
        }
    }

    // Wait until all send operations are complete
    bool all_done = false;
    while (!all_done) {
        all_done = true;

        // Receive and eliminate with new reduced rows from other processes
        for (int source = 0; source < size; source++) {
            if (source != rank) {
                MPI_Status status;
                int flag;
                MPI_Iprobe(source, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
                if (flag) {
                    int first = status.MPI_TAG;
                    int* received_basis = new int[maxsize];
                    MPI_Recv(received_basis, maxsize, MPI_INT, source, first, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

#pragma omp critical
                    {
                        ifBasis[first] = 1;
                        for (int j = 0; j < maxsize; j++) {
                            gBasis[first][j] = received_basis[j];
                        }

                        while (try_eliminate(received_basis)) {
                            for (int dest = 0; dest < size; dest++) {
                                if (dest != rank) {
                                    MPI_Request request;
                                    MPI_Isend(received_basis, maxsize, MPI_INT, dest, first, MPI_COMM_WORLD, &request);
                                    send_requests.push_back(request);
                                }
                            }
                        }
                    }
                    delete[] received_basis;
                }
            }
        }

        // Check if all send operations are complete
        for (auto& request : send_requests) {
            int flag;
            MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
            if (!flag) {
                all_done = false;
            }
        }
    }

    auto finish = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = finish - start;
    if (rank == 0) {
        cout << "GE_mpi: " << duration.count() << "ms" << endl;
    }
}
void GE_mpi_avx_openmp(int rank, int size, int num_threads) {
    int begin = 0;
    int flag;

    // Rank 0 initializes flag and broadcasts it to all MPI ranks
    if (rank == 0) {
        flag = readRowsFrom(begin);
    }
    MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int num_rows = (flag == -1) ? maxrow : flag;
    auto start_time = chrono::high_resolution_clock::now();

    vector<MPI_Request> send_requests;

    // Parallel region to perform elimination with AVX acceleration
#pragma omp parallel for num_threads(num_threads)
    for (int i = rank; i < num_rows; i += size) {
        while (findfirst(i) != -1) {
            int first = findfirst(i);
            if (ifBasis[first] == 1) {
                // AVX acceleration for XOR operation
                for (int j = 0; j < maxsize; j += 8) { // AVX registers are 256-bit (32 bytes), process 8 integers at a time
                    __m256i vec1 = _mm256_load_si256((__m256i*) & gRows[i][j]);
                    __m256i vec2 = _mm256_load_si256((__m256i*) & gBasis[first][j]);
                    __m256i result = _mm256_xor_si256(vec1, vec2);
                    _mm256_store_si256((__m256i*) & gRows[i][j], result);
                }
            }
            else {
                // AVX acceleration for assignment operation
                for (int j = 0; j < maxsize; j += 8) {
                    __m256i vec = _mm256_load_si256((__m256i*) & gRows[i][j]);
                    _mm256_store_si256((__m256i*) & gBasis[first][j], vec);
                }
                ifBasis[first] = 1;
                ans.insert({ first, gBasis[first] });

                // Send the new reduced row to other processes
                for (int dest = 0; dest < size; dest++) {
                    if (dest != rank) {
                        MPI_Request request;
                        MPI_Isend(&gBasis[first], maxsize, MPI_INT, dest, first, MPI_COMM_WORLD, &request);
                        send_requests.push_back(request);
                    }
                }
                break;
            }
        }
    }

    // Wait until all send operations are complete
    bool all_done = false;
    while (!all_done) {
        all_done = true;

        // Receive and eliminate with new reduced rows from other processes
        for (int source = 0; source < size; source++) {
            if (source != rank) {
                MPI_Status status;
                int flag;
                MPI_Iprobe(source, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
                if (flag) {
                    int first = status.MPI_TAG;
                    int* received_basis = new int[maxsize];
                    MPI_Recv(received_basis, maxsize, MPI_INT, source, first, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

#pragma omp critical
                    {
                        ifBasis[first] = 1;
                        for (int j = 0; j < maxsize; j += 8) {
                            __m256i vec = _mm256_load_si256((__m256i*) & received_basis[j]);
                            _mm256_store_si256((__m256i*) & gBasis[first][j], vec);
                        }

                        while (try_eliminate(received_basis)) {
                            for (int dest = 0; dest < size; dest++) {
                                if (dest != rank) {
                                    MPI_Request request;
                                    MPI_Isend(received_basis, maxsize, MPI_INT, dest, first, MPI_COMM_WORLD, &request);
                                    send_requests.push_back(request);
                                }
                            }
                        }
                    }
                    delete[] received_basis;
                }
            }
        }

        // Check if all send operations are complete
        for (auto& request : send_requests) {
            int flag;
            MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
            if (!flag) {
                all_done = false;
            }
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end_time - start_time;
    if (rank == 0) {
        cout << "GE_mpi_avx_openmp (" << num_threads << " threads): " << duration.count() << "ms" << endl;
    }
}
// 主函数
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        readBasis();
    }

    // 基础算法测试
    if (rank == 0) {
        reset();
        GE();
    }

    // MPI算法测试
    reset();
    GE_mpi(rank, size);

    reset();
    GE_mpi_avx_openmp(rank, size, 8);

    MPI_Finalize();
    return 0;
}
