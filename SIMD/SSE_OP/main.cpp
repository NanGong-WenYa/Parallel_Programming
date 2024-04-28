#include<iostream>
#include<windows.h>
#include <stdio.h>
#include<typeinfo>
#include <stdlib.h>
#include<tmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h>
using namespace std;
#define N 2000
float** matrix = NULL;
void matrix_init() {
    matrix = new float*[N];
    for (int i = 0; i < N; i++) {
        matrix[i] = new float[N];
    }
    for (int i = 0; i < N; i++) {
        matrix[i][i] = 1.0;
        for (int j = i + 1; j < N; j++) {
            matrix[i][j] = rand() % 1000;
        }

    }
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix[i][j] += matrix[k][j];
                matrix[i][j] = (int)matrix[i][j] % 1000;
            }
        }
    }
}
\


void print(float **a) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << a[i][j] << " ";
        }
        cout << endl;
    }
}

void basic_func(float **m) {
    for (int k = 0; k < N; k++) {
        for (int j = k + 1; j < N; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

void sse_func(float **matrix) {
    for (int k = 0; k < N; k++) {
        __m128 t1 = _mm_set1_ps(matrix[k][k]);
        int j = 0;
        for (j = k + 1; j + 4 <= N; j += 4) {
            __m128 t2 = _mm_loadu_ps(&matrix[k][j]);
            t2 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(&matrix[k][j], t2);
        }
        for (; j < N; j++) {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            __m128 vik = _mm_set1_ps(matrix[i][k]);
            for (j = k + 1; j + 4 <= N; j += 4) {
                __m128 vkj = _mm_loadu_ps(&matrix[k][j]);
                __m128 vij = _mm_loadu_ps(&matrix[i][j]);
                __m128 vx = _mm_mul_ps(vik, vkj);
                vij = _mm_sub_ps(vij, vx);
                _mm_storeu_ps(&matrix[i][j], vij);
            }
            for (; j < N; j++) {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

void sse_align(float **matrix) {
    for (int k = 0; k < N; k++) {
        __m128 t1 = _mm_set1_ps(matrix[k][k]);
        int j = k+1;
        while ((int)(&matrix[k][j])%16)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
            j++;
        }
        for ( ; j + 4 <= N; j += 4) {
            __m128 t2 = _mm_load_ps(&matrix[k][j]);
            t2 = _mm_div_ps(t2, t1);
            _mm_store_ps(&matrix[k][j], t2);
        }
        for (; j < N; j++) {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            __m128 vik = _mm_set1_ps(matrix[i][k]);
            j = k + 1;
            while ((int)(&matrix[k][j])%16)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
                j++;
            }
            for ( ; j + 4 <= N; j += 4) {
                __m128 vkj = _mm_load_ps(&matrix[k][j]);
                __m128 vij = _mm_loadu_ps(&matrix[i][j]);
                __m128 vx = _mm_mul_ps(vik, vkj);
                vij = _mm_sub_ps(vij, vx);
                _mm_storeu_ps(&matrix[i][j], vij);
            }
            for (; j < N; j++) {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

int main() {
    long long start, endTime, frequency;
    QueryPerformanceFrequency((LARGE_INTEGER*)&frequency);
    matrix_init();
    QueryPerformanceCounter((LARGE_INTEGER*)&start);
    basic_func(matrix);
    QueryPerformanceCounter((LARGE_INTEGER*)&endTime);
    cout << "Æ½·²Ëã·¨:" << (endTime - start) * 1000 / frequency << "ms" << endl;
    cout << "------------------" << endl;

    for (int i = 0; i < N; i++) {
        delete[] matrix[i];
    }
    delete matrix;

    matrix_init();
    QueryPerformanceCounter((LARGE_INTEGER*)&start);
    sse_func(matrix);
    QueryPerformanceCounter((LARGE_INTEGER*)&endTime);
    //print(unalign);
    cout<<"SSE:"<<(endTime-start)*1000/frequency<<"ms"<<endl;



}

