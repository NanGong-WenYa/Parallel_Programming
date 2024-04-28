#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<map>
#include<windows.h>
#include<tmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h>
using namespace std;
//sample:column:div:be_div
//sample1:130:22:8
//sample2:254:106:53
//sample3:562:170:53
//sample4:1011:539:263
//sample5:2362:1226:453
//sample6:3799:2759:1953
//sample7:8399:6375:4535
//sample8:23045:18748:14325
//sample9:37960:29304:14921
//sample10:43577:39477:54274
//sample11:85401:5724:756
const int column_count = 3000;
const int row_count = 60000;
const int basic_count = 40000;

//long long read = 0;
long long start, endTime, frequency;

map<int, int*>mir1;
map<int, int>mir2;
map<int, int*>result;

fstream Bxyz("D://Data//Groebner//sample5//被消元行.txt", ios::in | ios::out);
fstream Xyz("D://Data//Groebner//sample5//消元子.txt", ios::in | ios::out);


int Xiaoyuanzi[row_count][column_count];
int Beixiaoyuanzi[basic_count][column_count];

void zero_init() {
	//	read = 0;
	memset(Xiaoyuanzi, 0, sizeof(Xiaoyuanzi));
	memset(Beixiaoyuanzi, 0, sizeof(Beixiaoyuanzi));
	Bxyz.close();
	Xyz.close();
	Bxyz.open("D://Data//Groebner//sample5//被消元行.txt", ios::in | ios::out);
	Xyz.open("D://Data//Groebner//sample5//消元子.txt", ios::in | ios::out);
	mir1.clear();
	mir2.clear();
	result.clear();

}

void Rxiaoyuanzi() {
	for (int i = 0; i < basic_count; i++) {
		if (Xyz.eof()) {
			cout << "读取消元子" << i-1 << "行" << endl;
			return;
		}
		string tmp;
		bool flag = false;
		int row = 0;
		getline(Xyz, tmp);
		stringstream s(tmp);
		int pos;
		while (s >> pos) {
			if (!flag) {
				row = pos;
				flag = true;
				mir1.insert(pair<int, int*>(row, Beixiaoyuanzi[row]));
			}
			int index = pos / 32;
			int offset = pos % 32;
			Beixiaoyuanzi[row][index] = Beixiaoyuanzi[row][index] | (1 << offset);
		}
		flag = false;
		row = 0;
	}
}

int Rbeixiaoyuanzi(int pos) {
	mir2.clear();
	if (Bxyz.is_open())
		Bxyz.close();
	Bxyz.open("D://Data//Groebner//sample5//被消元行.txt", ios::in | ios::out);
	memset(Xiaoyuanzi, 0, sizeof(Xiaoyuanzi));
	string line;
	for (int i = 0; i < pos; i++) {
		getline(Bxyz, line);
	}
	for (int i = pos; i < pos + row_count; i++) {
		int tmp;
		getline(Bxyz, line);
		if (line.empty()) {
			cout << "读取被消元行 "<<i<<" 行" << endl;
			return i;
		}
		bool flag = false;
		stringstream next_num(line);
		while (next_num >> tmp) {
			if (!flag) {
				mir2.insert(pair<int, int>(i - pos, tmp));
			}
			int index = tmp / 32;
			int offset = tmp % 32;
			Xiaoyuanzi[i - pos][index] = Xiaoyuanzi[i - pos][index] | (1 << offset);
			flag = true;
		}
	}
	cout << "read max rows" << endl;
	return -1;

}

void add_xiaoyuanzi(int row) {
	bool flag = 0;
	for (int i = column_count - 1; i >= 0; i--) {
		if (Xiaoyuanzi[row][i] == 0)
			continue;
		else {
			if (!flag)
				flag = true;
			int pos = i * 32;
			int offset = 0;
			for (int k = 31; k >= 0; k--) {
				if (Xiaoyuanzi[row][i] & (1 << k))
				{
					offset = k;
					break;
				}
			}
			int newfirst = pos + offset;
			mir2.erase(row);
			mir2.insert(pair<int, int>(row, newfirst));
			break;
		}
	}
	if (!flag) {
		mir2.erase(row);
	}
	return;
}

void out_result(ofstream& out) {
	for (auto it = result.rbegin(); it != result.rend(); it++) {
		int* result = it->second;
		int column_count = it->first / 32 + 1;
		for (int i = column_count; i >= 0; i--) {
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

void basic_func() {
	long long readBegin, readEnd;
	int begin = 0;
	int flag;
	QueryPerformanceCounter((LARGE_INTEGER*)&readBegin);
	flag = Rbeixiaoyuanzi(begin);
	QueryPerformanceCounter((LARGE_INTEGER*)&readEnd);
	start += (readEnd - readBegin);

	int num = (flag == -1) ? row_count : flag;
	for (int i = 0; i < num; i++) {
		while (mir2.find(i) != mir2.end()) {
			int first = mir2.find(i)->second;
			if (mir1.find(first) != mir1.end()) {
				int* basis = mir1.find(first)->second;
				for (int j = 0; j < column_count; j++) {
					Xiaoyuanzi[i][j] = Xiaoyuanzi[i][j] ^ basis[j];
				}
				add_xiaoyuanzi(i);
			}
			else {
				for (int j = 0; j < column_count; j++) {
					Beixiaoyuanzi[first][j] = Xiaoyuanzi[i][j];
				}
				mir1.insert(pair<int, int*>(first, Beixiaoyuanzi[first]));
				result.insert(pair<int, int*>(first, Beixiaoyuanzi[first]));
				mir2.erase(i);
			}
		}
	}


}

void AVX_func() {
	long long readBegin, readEnd;
	int begin = 0;
	int flag;

	QueryPerformanceCounter((LARGE_INTEGER*)&readBegin);
	flag = Rbeixiaoyuanzi(begin);
	QueryPerformanceCounter((LARGE_INTEGER*)&readEnd);
	start += (readEnd - readBegin);
	int num = (flag == -1) ? row_count : flag;
	for (int i = 0; i < num; i++) {
		while (mir2.find(i) != mir2.end()) {
			int first = mir2.find(i)->second;
			if (mir1.find(first) != mir1.end()) {
				int* basis = mir1.find(first)->second;
				int j = 0;
				for (; j + 8 < column_count; j += 8) {
					__m256i vij = _mm256_loadu_si256((__m256i*) & Xiaoyuanzi[i][j]);
					__m256i vj = _mm256_loadu_si256((__m256i*) & basis[j]);
					__m256i vx = _mm256_xor_si256(vij, vj);
					_mm256_storeu_si256((__m256i*) & Xiaoyuanzi[i][j], vx);
				}
				for (; j < column_count; j++) {
					Xiaoyuanzi[i][j] = Xiaoyuanzi[i][j] ^ basis[j];
				}
				add_xiaoyuanzi(i);
			}
			else {
				int j = 0;
				for (; j + 8 < column_count; j += 8) {
					__m256i vij = _mm256_loadu_si256((__m256i*) & Xiaoyuanzi[i][j]);
					_mm256_storeu_si256((__m256i*) & Beixiaoyuanzi[first][j], vij);
				}
				for (; j < column_count; j++) {
					Beixiaoyuanzi[first][j] = Xiaoyuanzi[i][j];
				}
				mir1.insert(pair<int, int*>(first, Beixiaoyuanzi[first]));
				result.insert(pair<int, int*>(first, Beixiaoyuanzi[first]));
				mir2.erase(i);
			}
		}
	}


}


int main() {
	double time1 = 0;
	double time2 = 0;


	for (int i = 0; i < 3; i++) {
		ofstream out("D://Data//Groebner//sample5//消元结果_test.txt");
		ofstream out1("D://Data//Groebner//sample5//消元结果(AVX)_test.txt");
		out << "__________" << endl;
		out1 << "__________" << endl;
		QueryPerformanceFrequency((LARGE_INTEGER*)&frequency);
		Rxiaoyuanzi();
		//writeResult();
		QueryPerformanceCounter((LARGE_INTEGER*)&start);
		basic_func();
		QueryPerformanceCounter((LARGE_INTEGER*)&endTime);
		cout << "Ordinary time:" << (endTime - start) * 1000 / frequency << "ms" << endl;
		time1 += (endTime - start) * 1000 / frequency;
		out_result(out);

		zero_init();

		Rxiaoyuanzi();
		QueryPerformanceCounter((LARGE_INTEGER*)&start);
		AVX_func();
		QueryPerformanceCounter((LARGE_INTEGER*)&endTime);
		cout << "AVX time:" << (endTime - start) * 1000 / frequency << "ms" << endl;
		time2 += (endTime - start) * 1000 / frequency;
		out_result(out1);

		zero_init();
		out.close();
		out1.close();
	}
	cout << "time1:" << time1 / 3 << endl << "time2:" << time2 / 3;
}


