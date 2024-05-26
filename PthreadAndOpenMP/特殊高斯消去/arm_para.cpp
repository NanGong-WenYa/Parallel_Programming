#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <pthread.h>
#include <chrono>
#include <arm_neon.h>
#include <cstring>
#include <omp.h>

using namespace std;

fstream Brows("/home/s2213893/pthread/open/7b.txt", ios::in | ios::out);
fstream Xrows("/home/s2213893/pthread/open/7x.txt", ios::in | ios::out);

const int maxsize = 3000;
const int maxrow = 60000;
const int numBasis = 100000;

pthread_mutex_t lock;

map<int, int*> iToBasis;
map<int, int*> ans;

int Brow[maxrow][maxsize];
int Xrow[numBasis][maxsize];

void reset() {
    memset(Brow, 0, sizeof(Brow));
    memset(Xrow, 0, sizeof(Xrow));
    Brows.close();
    Xrows.close();
    Brows.open("/home/s2213893/pthread/open/7b.txt", ios::in | ios::out);
    Xrows.open("/home/s2213893/pthread/open/7x.txt", ios::in | ios::out);
    if (!Brows.is_open()) {
        cerr << "Failed to open RowFile" << endl;
    }
    if (!Xrows.is_open()) {
        cerr << "Failed to open BasisFile" << endl;
    }

    iToBasis.clear();
    ans.clear();
}

int readBasis() {
    int i;
    for (i = 0; i < numBasis; i++) {
        string tmp;
        if (!getline(Xrows, tmp)) {
            cout << "读取消元子" << i << "行" << endl;
            return i;
        }
        if (tmp.empty()) continue;

        bool flag = false;
        int row = 0;
        stringstream s(tmp);
        int pos;
        while (s >> pos) {
            if (!flag) {
                row = pos;
                flag = true;
                iToBasis.insert(pair<int, int*>(row, Xrow[row]));
            }
            int index = pos / 32;
            int offset = pos % 32;
            Xrow[row][index] |= (1 << offset);
        }
    }
    return i;
}

int readRowsFrom(int pos) {
    Brows.clear();
    Brows.seekg(0, ios::beg);
    for (int i = 0; i < pos; i++) {
        string dummy;
        if (!getline(Brows, dummy)) {
            cout << "读取被消元行 " << i << " 行" << endl;
            return i;
        }
    }

    int i;
    for (i = 0; i < maxrow; i++) {
        string line;
        if (!getline(Brows, line)) {
            cout << "读取被消元行 " << pos + i << " 行" << endl;
            return pos + i;
        }
        if (line.empty()) continue;

        stringstream s(line);
        int tmp;
        while (s >> tmp) {
            int index = tmp / 32;
            int offset = tmp % 32;
            Brow[i][index] |= (1 << offset);
        }
    }
    cout << "read max rows" << endl;
    return pos + i;
}

int findfirst(int row) {
    int first;
    for (int i = maxsize - 1; i >= 0; i--) {
        if (Brow[row][i] == 0)
            continue;
        else {
            int pos = i * 32;
            int offset = 0;
            for (int k = 31; k >= 0; k--) {
                if (Brow[row][i] & (1 << k)) {
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

#define NUM_THREADS 7

struct threadParam_t {
    int t_id;
    int num;
};

void GE() {
    int begin = 0;
    int flag;
    flag = readRowsFrom(begin);

    int num = (flag == -1) ? maxrow : flag;
    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < num; i++) {
        while (findfirst(i) != -1) {
            int first = findfirst(i);
            if (iToBasis.find(first) != iToBasis.end()) {
                int* basis = iToBasis.find(first)->second;
                for (int j = 0; j < maxsize; j++) {
                    Brow[i][j] = Brow[i][j] ^ basis[j];
                }
            }
            else {
                for (int j = 0; j < maxsize; j++) {
                    Xrow[first][j] = Brow[i][j];
                }
                iToBasis.insert(pair<int, int*>(first, Xrow[first]));
                ans.insert(pair<int, int*>(first, Xrow[first]));
                break;
            }
        }
    }

    auto finish = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = finish - start;
    cout << "basic: " << duration.count() << "ms" << endl;
}

void AVX_GE() {
    int begin = 0;
    int flag;
    flag = readRowsFrom(begin);

    int num = (flag == -1) ? maxrow : flag;
    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < num; i++) {
        while (findfirst(i) != -1) {
            int first = findfirst(i);
            if (iToBasis.find(first) != iToBasis.end()) {
                int* basis = iToBasis.find(first)->second;
                int j = 0;
                for (; j + 8 < maxsize; j += 8) {
                    uint32x4_t vij1 = vld1q_u32((uint32_t*)&Brow[i][j]);
                    uint32x4_t vj1 = vld1q_u32((uint32_t*)&basis[j]);
                    uint32x4_t vx1 = veorq_u32(vij1, vj1);
                    vst1q_u32((uint32_t*)&Brow[i][j], vx1);

                    uint32x4_t vij2 = vld1q_u32((uint32_t*)&Brow[i][j + 4]);
                    uint32x4_t vj2 = vld1q_u32((uint32_t*)&basis[j + 4]);
                    uint32x4_t vx2 = veorq_u32(vij2, vj2);
                    vst1q_u32((uint32_t*)&Brow[i][j + 4], vx2);
                }
                for (; j < maxsize; j++) {
                    Brow[i][j] = Brow[i][j] ^ basis[j];
                }
            }
            else {
                int j = 0;
                for (; j + 8 < maxsize; j += 8) {
                    uint32x4_t vij1 = vld1q_u32((uint32_t*)&Brow[i][j]);
                    vst1q_u32((uint32_t*)&Xrow[first][j], vij1);

                    uint32x4_t vij2 = vld1q_u32((uint32_t*)&Brow[i][j + 4]);
                    vst1q_u32((uint32_t*)&Xrow[first][j + 4], vij2);
                }
                for (; j < maxsize; j++) {
                    Xrow[first][j] = Brow[i][j];
                }
                iToBasis.insert(pair<int, int*>(first, Xrow[first]));
                ans.insert(pair<int, int*>(first, Xrow[first]));
                break;
            }
        }
    }

    auto finish = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = finish - start;
    cout << "NEON: " << duration.count() << "ms" << endl;
}

void* GE_lock_thread(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    int num = p->num;

    for (int i = t_id; i + NUM_THREADS < num; i += NUM_THREADS) {
        while (findfirst(i) != -1) {
            int first = findfirst(i);
            pthread_mutex_lock(&lock);
            if (iToBasis.find(first) != iToBasis.end()) {
                int* basis = iToBasis.find(first)->second;
                pthread_mutex_unlock(&lock);
                for (int j = 0; j < maxsize; j++) {
                    Brow[i][j] ^= basis[j];
                }
            }
            else {
                for (int j = 0; j < maxsize; j++) {
                    Xrow[first][j] = Brow[i][j];
                }
                iToBasis.insert(pair<int, int*>(first, Xrow[first]));
                ans.insert(pair<int, int*>(first, Xrow[first]));
                pthread_mutex_unlock(&lock);
                break;
            }
        }
    }
    pthread_exit(NULL);
}

void GE_pthread() {
    int begin = 0;
    int flag = readRowsFrom(begin);
    int num = (flag == -1) ? maxrow : flag;
    auto start = chrono::high_resolution_clock::now();

    pthread_t handles[NUM_THREADS];
    threadParam_t param[NUM_THREADS];

    pthread_mutex_init(&lock, NULL);

    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].num = num;
        pthread_create(&handles[t_id], NULL, GE_lock_thread, (void*)&param[t_id]);
    }

    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handles[t_id], NULL);
    }

    pthread_mutex_destroy(&lock);

    auto finish = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = finish - start;
    cout << "pthread: " << duration.count() << "ms" << endl;
}

void* AVX_GE_thread(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    int num = p->num;

    for (int i = t_id; i + NUM_THREADS < num; i += NUM_THREADS) {
        while (findfirst(i) != -1) {
            int first = findfirst(i);
            pthread_mutex_lock(&lock);
            if (iToBasis.find(first) != iToBasis.end()) {
                int* basis = iToBasis.find(first)->second;
                pthread_mutex_unlock(&lock);
                int j = 0;
                for (; j + 8 < maxsize; j += 8) {
                    uint32x4_t vij1 = vld1q_u32((uint32_t*)&Brow[i][j]);
                    uint32x4_t vj1 = vld1q_u32((uint32_t*)&basis[j]);
                    uint32x4_t vx1 = veorq_u32(vij1, vj1);
                    vst1q_u32((uint32_t*)&Brow[i][j], vx1);

                    uint32x4_t vij2 = vld1q_u32((uint32_t*)&Brow[i][j + 4]);
                    uint32x4_t vj2 = vld1q_u32((uint32_t*)&basis[j + 4]);
                    uint32x4_t vx2 = veorq_u32(vij2, vj2);
                    vst1q_u32((uint32_t*)&Brow[i][j + 4], vx2);
                }
                for (; j < maxsize; j++) {
                    Brow[i][j] ^= basis[j];
                }
            }
            else {
                int j = 0;
                for (; j + 8 < maxsize; j += 8) {
                    uint32x4_t vij1 = vld1q_u32((uint32_t*)&Brow[i][j]);
                    vst1q_u32((uint32_t*)&Xrow[first][j], vij1);

                    uint32x4_t vij2 = vld1q_u32((uint32_t*)&Brow[i][j + 4]);
                    vst1q_u32((uint32_t*)&Xrow[first][j + 4], vij2);
                }
                for (; j < maxsize; j++) {
                    Xrow[first][j] = Brow[i][j];
                }
                iToBasis.insert(pair<int, int*>(first, Xrow[first]));
                ans.insert(pair<int, int*>(first, Xrow[first]));
                pthread_mutex_unlock(&lock);
                break;
            }
        }
    }
    pthread_exit(NULL);
}

void AVX_pthread() {
    int begin = 0;
    int flag = readRowsFrom(begin);
    int num = (flag == -1) ? maxrow : flag;
    auto start = chrono::high_resolution_clock::now();

    pthread_t handles[NUM_THREADS];
    threadParam_t param[NUM_THREADS];

    pthread_mutex_init(&lock, NULL);

    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].num = num;
        pthread_create(&handles[t_id], NULL, AVX_GE_thread, (void*)&param[t_id]);
    }

    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handles[t_id], NULL);
    }

    pthread_mutex_destroy(&lock);

    auto finish = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = finish - start;
    cout << "NEON + pthread: " << duration.count() << "ms" << endl;
}

void GE_openmp() {
    int begin = 0;
    int flag = readRowsFrom(begin);     //读取被消元行
    int num = (flag == -1) ? maxrow : flag;
    auto start = chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < num; i++) {
        bool flag_break = false; // 用于控制循环结束的标志
        while (findfirst(i) != -1 && !flag_break) {
            int first = findfirst(i);
#pragma omp critical
            {
                if (iToBasis.find(first) != iToBasis.end()) {
                    int* basis = iToBasis.find(first)->second;
                    for (int j = 0; j < maxsize; j++) {
                        Brow[i][j] ^= basis[j];
                    }
                }
                else {
                    for (int j = 0; j < maxsize; j++) {
                        Xrow[first][j] = Brow[i][j];
                    }
                    iToBasis.insert(pair<int, int*>(first, Xrow[first]));
                    ans.insert(pair<int, int*>(first, Xrow[first]));
                    flag_break = true; // 设置标志位，表示循环结束
                }
            }
        }
    }

    auto finish = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = finish - start;
    cout << "OpenMP: " << duration.count() << "ms" << endl;
}


int main() {
    cout << "result of sample 6" << endl;
    for (int i = 0; i < 1; i++) {
        ofstream out1("100.txt");
        ofstream out2("200.txt");
        ofstream out3("300.txt");
        ofstream out4("400.txt");
        ofstream out5("500.txt");

        reset();
        cout << "读取消元子" << readBasis() << "行" << endl;
        GE();
        writeResult(out1);

        reset();
        cout << "读取消元子" << readBasis() << "行" << endl;
        AVX_GE();
        writeResult(out2);

        reset();
        cout << "读取消元子" << readBasis() << "行" << endl;
        GE_pthread();
        writeResult(out3);

        reset();
        cout << "读取消元子" << readBasis() << "行" << endl;
        AVX_pthread();
        writeResult(out4);

        reset();
        cout << "读取消元子" << readBasis() << "行" << endl;
        GE_openmp();
        writeResult(out5);
    }
    return 0;
}
