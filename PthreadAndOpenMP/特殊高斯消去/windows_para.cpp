#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<map>
#include<windows.h>
#include<time.h>
#include<tmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h>
#include<pthread.h>
#include<omp.h>
using namespace std;

#define NUM_THREADS 10


struct threadParam_t {    //�������ݽṹ
	int t_id;
	int num;
};


const int maxsize = 3000;
const int maxrow = 60000; //3000*32>90000 ,����������90000�ı���Ԫ�о���60000��
const int numx = 100000;   //���洢90000*100000����Ԫ��

pthread_mutex_t lock;  //д����Ԫ��ʱ��Ҫ����

//long long read = 0;
long long head, tail, freq;

map<int, int*>xiaoyuanzi;    //����Ϊi����Ԫ�ӵ�ӳ��
map<int, int*>beixiaoyuanzi;			//��

fstream Barrays("D://Data//Groebner//sample6//6b.txt", ios::in | ios::out);
fstream Xrows("D://Data//Groebner//sample6//6x.txt", ios::in | ios::out);


int Barrays[maxrow][maxsize];   //����Ԫ�����60000�У�3000��
int Xarrays[numx][maxsize];  //��Ԫ�����40000�У�3000��


void clearRow() {
	//	read = 0;
	memset(Barrays, 0, sizeof(Barrays));
	memset(Xarrays, 0, sizeof(Xarrays));
	Barrays.close();
	Xrows.close();
	Barrays.open("D://Data//Groebner//sample6//6b.txt", ios::in | ios::out);
	Xrows.open("D://Data//Groebner//sample6//6x.txt", ios::in | ios::out);
	xiaoyuanzi.clear();

	beixiaoyuanzi.clear();
}

int readX() {          //��ȡ��Ԫ��
	for (int i = 0; i < numx; i++) {
		if (Xrows.eof()) {
			cout << "��ȡ��Ԫ��" << i - 1 << "��" << endl;
			return i - 1;
		}
		string tmp;
		bool flag = false;
		int row = 0;
		getline(Xrows, tmp);
		stringstream s(tmp);
		int pos;
		while (s >> pos) {
			//cout << pos << " ";
			if (!flag) {
				row = pos;
				flag = true;
				xiaoyuanzi.insert(pair<int, int*>(row, Xarrays[row]));
			}
			int index = pos / 32;
			int offset = pos % 32;
			Xarrays[row][index] = Xarrays[row][index] | (1 << offset);
		}
		flag = false;
		row = 0;
	}
}

int readRow(int pos) {       //��ȡ����Ԫ��
	if (Barrays.is_open())
		Barrays.close();
	Barrays.open("D://Data//Groebner//sample6//6b.txt", ios::in | ios::out);
	memset(Barrays, 0, sizeof(Barrays));   //����Ϊ0
	string line;
	for (int i = 0; i < pos; i++) {       //��ȡposǰ���޹���
		getline(Barrays, line);
	}
	for (int i = pos; i < pos + maxrow; i++) {
		int tmp;
		getline(Barrays, line);
		if (line.empty()) {
			cout << "��ȡ����Ԫ�� " << i << " ��" << endl;
			return i;   //���ض�ȡ������
		}
		bool flag = false;
		stringstream s(line);
		while (s >> tmp) {
			int index = tmp / 32;
			int offset = tmp % 32;
			Barrays[i - pos][index] = Barrays[i - pos][index] | (1 << offset);
			flag = true;
		}
	}
	cout << "read max rows" << endl;
	return -1;  //�ɹ���ȡmaxrow��

}

int Rowfirst(int row) {  //Ѱ�ҵ�row�б���Ԫ�е�����
	int first;
	for (int i = maxsize - 1; i >= 0; i--) {
		if (Barrays[row][i] == 0)
			continue;
		else {
			int pos = i * 32;
			int offset = 0;
			for (int k = 31; k >= 0; k--) {
				if (Barrays[row][i] & (1 << k))
				{
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



void outputRes(ofstream& out) {
	for (auto it = beixiaoyuanzi.rbegin(); it != beixiaoyuanzi.rend(); it++) {
		int* result = it->second;
		int max = it->first / 32 + 1;
		for (int i = max; i >= 0; i--) {
			if (result[i] == 0)
				continue;
			int pos = i * 32;
			//int offset = 0;
			for (int k = 31; k >= 0; k--) {
				if (result[i] & (1 << k)) {
					out << k + pos << " ";
				}
			}
		}
		out << endl;
	}
}

void GE() {
	int begin = 0;
	int flag;
	flag = readRow(begin);     //��ȡ����Ԫ��

	int num = (flag == -1) ? maxrow : flag;
	double duration=0;
	clock_t start,finish;
	start=clock();

	for (int i = 0; i < num; i++) {
		while (Rowfirst(i)!= -1) {     //��������
			int first =Rowfirst(i);      //first������
			if (xiaoyuanzi.find(first) != xiaoyuanzi.end()) {  //��������Ϊfirst��Ԫ��
				int* basis = xiaoyuanzi.find(first)->second;  //�ҵ�����Ԫ�ӵ�����
				for (int j = 0; j < maxsize; j++) {
					Barrays[i][j] = Barrays[i][j] ^ basis[j];     //���������Ԫ

				}
			}
			else {   //����Ϊ��Ԫ��
				for (int j = 0; j < maxsize; j++) {
					Xarrays[first][j] = Barrays[i][j];
				}
				xiaoyuanzi.insert(pair<int, int*>(first, Xarrays[first]));
				beixiaoyuanzi.insert(pair<int, int*>(first, Xarrays[first]));
				break;
			}
		}
	}

	finish=clock();
	cout << "basic:" << (finish - start)*1000 / CLOCKS_PER_SEC << "ms" << endl;
}

void AVX_GE() {
	int begin = 0;
	int flag;
	flag = readRow(begin);     //��ȡ����Ԫ��
	int num = (flag == -1) ? maxrow : flag;
	double duration=0;
	clock_t start,finish;
	start=clock();
	for (int i = 0; i < num; i++) {
		while (Rowfirst(i) != -1) {
			int first = Rowfirst(i);
			if (xiaoyuanzi.find(first) != xiaoyuanzi.end()) {  //���ڸ���Ԫ��
				int* basis = xiaoyuanzi.find(first)->second;
				int j = 0;
				for (; j + 8 < maxsize; j += 8) {
					__m256i vij = _mm256_loadu_si256((__m256i*) & Barrays[i][j]);
					__m256i vj = _mm256_loadu_si256((__m256i*) & basis[j]);
					__m256i vx = _mm256_xor_si256(vij, vj);
					_mm256_storeu_si256((__m256i*) & Barrays[i][j], vx);
				}
				for (; j < maxsize; j++) {
					Barrays[i][j] = Barrays[i][j] ^ basis[j];
				}
			}
			else {
				int j = 0;
				for (; j + 8 < maxsize; j += 8) {
					__m256i vij = _mm256_loadu_si256((__m256i*) & Barrays[i][j]);
					_mm256_storeu_si256((__m256i*) & Xarrays[first][j], vij);
				}
				for (; j < maxsize; j++) {
					Xarrays[first][j] = Barrays[i][j];
				}
				xiaoyuanzi.insert(pair<int, int*>(first, Xarrays[first]));
				beixiaoyuanzi.insert(pair<int, int*>(first, Xarrays[first]));
				break;
			}
		}
	}
	finish=clock();
	cout << "AVX:" << (finish - start)*1000 / CLOCKS_PER_SEC << "ms" << endl;
}

void* GE_lock_thread(void* param) {

	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;
	int num = p->num;

	for (int i = t_id; i + NUM_THREADS < num; i += NUM_THREADS) {
		while (Rowfirst(i) != -1) {
			int first = Rowfirst(i);      //first������
			if (xiaoyuanzi.find(first) != xiaoyuanzi.end()) {  //��������Ϊfirst��Ԫ��
				int* basis = xiaoyuanzi.find(first)->second;  //�ҵ�����Ԫ�ӵ�����
				for (int j = 0; j < maxsize; j++) {
					Barrays[i][j] = Barrays[i][j] ^ basis[j];     //���������Ԫ

				}
			}
			else {   //����Ϊ��Ԫ��
				pthread_mutex_lock(&lock); //�����first����Ԫ��û�б�ռ�ã������
				if (xiaoyuanzi.find(first) != xiaoyuanzi.end())
				{
					pthread_mutex_unlock(&lock);
					continue;
				}
				for (int j = 0; j < maxsize; j++) {
					Xarrays[first][j] = Barrays[i][j];     //��Ԫ�ӵ�д��
				}
				xiaoyuanzi.insert(pair<int, int*>(first, Xarrays[first]));
				beixiaoyuanzi.insert(pair<int, int*>(first, Xarrays[first]));
				pthread_mutex_unlock(&lock);          //����
				break;
			}

		}
	}
	cout << t_id << "�߳����" << endl;
	pthread_exit(NULL);
	return NULL;
}

void GE_pthread() {
	int begin = 0;
	int flag;
	flag = readRow(begin);     //��ȡ����Ԫ��

	int num = (flag == -1) ? maxrow : flag;

	pthread_mutex_init(&lock, NULL);  //��ʼ����

	pthread_t* handle = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
	threadParam_t* param = (threadParam_t*)malloc(NUM_THREADS * sizeof(threadParam_t));

    double duration=0;
	clock_t start,finish;
	start=clock();
	for (int t_id = 0; t_id < NUM_THREADS; t_id++) {//��������
		param[t_id].t_id = t_id;
		param[t_id].num = num;
		pthread_create(&handle[t_id], NULL, GE_lock_thread, &param[t_id]);
	}

	for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
		pthread_join(handle[t_id], NULL);
	}

    finish=clock();
	cout << "pthread:" << (finish - start)*1000 / CLOCKS_PER_SEC << "ms" << endl;
	free(handle);
	free(param);
	pthread_mutex_destroy(&lock);
}

void* AVX_lock_thread(void* param) {

	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;
	int num = p->num;

	for (int i = t_id; i + NUM_THREADS < num; i += NUM_THREADS) {
		while (Rowfirst(i) != -1) {
			int first = Rowfirst(i);
			if (xiaoyuanzi.find(first) != xiaoyuanzi.end()) {  //���ڸ���Ԫ��
				int* basis = xiaoyuanzi.find(first)->second;
				int j = 0;
				for (; j + 8 < maxsize; j += 8) {
					__m256i vij = _mm256_loadu_si256((__m256i*) & Barrays[i][j]);
					__m256i vj = _mm256_loadu_si256((__m256i*) & basis[j]);
					__m256i vx = _mm256_xor_si256(vij, vj);
					_mm256_storeu_si256((__m256i*) & Barrays[i][j], vx);
				}
				for (; j < maxsize; j++) {
					Barrays[i][j] = Barrays[i][j] ^ basis[j];
				}
			}
			else {
				pthread_mutex_lock(&lock); //�����first����Ԫ��û�б�ռ�ã������
				if (xiaoyuanzi.find(first) != xiaoyuanzi.end())
				{
					pthread_mutex_unlock(&lock);
					continue;
				}
				int j = 0;
				for (; j + 8 < maxsize; j += 8) {
					__m256i vij = _mm256_loadu_si256((__m256i*) & Barrays[i][j]);
					_mm256_storeu_si256((__m256i*) & Xarrays[first][j], vij);
				}
				for (; j < maxsize; j++) {
					Xarrays[first][j] = Barrays[i][j];
				}
				xiaoyuanzi.insert(pair<int, int*>(first, Xarrays[first]));
				beixiaoyuanzi.insert(pair<int, int*>(first, Xarrays[first]));
				pthread_mutex_unlock(&lock);
				break;
			}
		}
	}
	cout << t_id << "�߳����" << endl;
	pthread_exit(NULL);
	return NULL;
}

void AVX_pthread() {
	int begin = 0;
	int flag;
	flag = readRow(begin);     //��ȡ����Ԫ��

	int num = (flag == -1) ? maxrow : flag;

	pthread_mutex_init(&lock, NULL);  //��ʼ����

	pthread_t* handle = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
	threadParam_t* param = (threadParam_t*)malloc(NUM_THREADS * sizeof(threadParam_t));

    double duration=0;
	clock_t start,finish;
	start=clock();
	for (int t_id = 0; t_id < NUM_THREADS; t_id++) {//��������
		param[t_id].t_id = t_id;
		param[t_id].num = num;
		pthread_create(&handle[t_id], NULL, AVX_lock_thread, &param[t_id]);
	}

	for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
		pthread_join(handle[t_id], NULL);
	}

    finish=clock();
	cout << "AVX+pthread:" << (finish - start)*1000 / CLOCKS_PER_SEC << "ms" << endl;
	free(handle);
	free(param);
	pthread_mutex_destroy(&lock);
}

void GE_OpenMP() {
    int begin = 0;
    int flag;
    flag = readRow(begin);
    int num = (flag == -1) ? maxrow : flag;
    for (int i = 0; i < num; i++) {
        while (Rowfirst(i) != -1) {
            int first = Rowfirst(i);
            if (xiaoyuanzi.find(first) != xiaoyuanzi.end()) {
                int* basis = xiaoyuanzi.find(first)->second;
                for (int j = 0; j < maxsize; j++) {
                    Barrays[i][j] = Barrays[i][j] ^ basis[j];
                    elimination
                }
            } else {
                #pragma omp critical
                {
                    for (int j = 0; j < maxsize; j++) {
                        Xarrays[first][j] = Barrays[i][j];
                    }
                    xiaoyuanzi.insert(pair<int, int*>(first, Xarrays[first]));
                    beixiaoyuanzi.insert(pair<int, int*>(first, Xarrays[first]));
                }
                break;
            }
        }
    }
}

void AVX_GE_OpenMP() {
    int begin = 0;
    int flag;
    flag = readRow(begin);     // Read elimination rows
    int num = (flag == -1) ? maxrow : flag;
    for (int i = 0; i < num; i++) {
        while (Rowfirst(i) != -1) {
            int first = Rowfirst(i);
            if (xiaoyuanzi.find(first) != xiaoyuanzi.end()) {  // If a reducer exists
                int* basis = xiaoyuanzi.find(first)->second;
                int j = 0;
                for (; j + 8 < maxsize; j += 8) {
                    __m256i vij = _mm256_loadu_si256((__m256i*) & Barrays[i][j]);
                    __m256i vj = _mm256_loadu_si256((__m256i*) & basis[j]);
                    __m256i vx = _mm256_xor_si256(vij, vj);
                    _mm256_storeu_si256((__m256i*) & Barrays[i][j], vx);
                }
                for (; j < maxsize; j++) {
                    Barrays[i][j] = Barrays[i][j] ^ basis[j];
                }
            } else {
                #pragma omp critical
                {
                    int j = 0;
                    for (; j + 8 < maxsize; j += 8) {
                        __m256i vij = _mm256_loadu_si256((__m256i*) & Barrays[i][j]);
                        _mm256_storeu_si256((__m256i*) & Xarrays[first][j], vij);
                    }
                    for (; j < maxsize; j++) {
                        Xarrays[first][j] = Barrays[i][j];
                    }
                    xiaoyuanzi.insert(pair<int, int*>(first, Xarrays[first]));
                    beixiaoyuanzi.insert(pair<int, int*>(first, Xarrays[first]));
                }
                break;
            }
        }
    }
}

int main() {
	double time1 = 0;
	double time2 = 0;


	for (int i = 0; i < 1; i++) {
		ofstream out3("D://Data//Groebner//sample6//��Ԫ���(AVX_lock).txt");


		readX();
		AVX_pthread();
		outputRes(out3);

		clearRow();
		out3.close();
	}
	cout << "time1:" << time1 / 5 << endl << "time2:" << time2 / 5;
}
