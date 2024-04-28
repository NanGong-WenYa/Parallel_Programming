#include<iostream>
#include<time.h>
#include<arm_neon.h>
using namespace std;
#define N 4000
float matrix[N][N];
float **resu;
void matrix_init(){
    for(int i = 0,j=0 ; i < N ; i++){
        matrix[i][i] = 1.0;
        for(int j = i+1; j < N; j++){
            matrix[i][j] = rand()%1000;
        }

    }
    for(int k = 0; k < N; k++){
        for(int i = k+1; i < N; i++){
            for(int j = 0; j < N; j++){
                matrix[i][j] += matrix[k][j];
                matrix[i][j] = (int)matrix[i][j]%1000;
            }
        }
    }
}

void copyMatrix(float** resu, float matrix[N][N]){
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++)
			resu[i][j]=matrix[i][j];
	}
}

void print(){
   for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
        cout<<matrix[i][j]<<" ";
    }
    cout<<endl;
   }
}

void basic_func(float** matrix){      //平凡算法
    for(int k = 0; k<N; k++){
        for(int j = k+1; j<N; j++ ){
            matrix[k][j] = matrix[k][j]/matrix[k][k];
        }
        matrix[k][k] = 1.0;
        for(int i = k+1; i<N; i++){
            for(int j = k+1; j<N; j++){
                matrix[i][j] = matrix[i][j]-matrix[i][k]*matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

void neon_func(float** matrix){
    for(int k = 0; k < N; k++){
        float32x4_t vt = vdupq_n_f32(matrix[k][k]);
        int j = 0;
        for(j = k+1; j+4 <= N; j+=4){
                float32x4_t va = vld1q_f32(&matrix[k][j]);
                va = vdivq_f32(va, vt);
                vst1q_f32(&matrix[k][j], va);
        }
        for( ;j < N; j++){
                matrix[k][j] = matrix[k][j]/matrix[k][k];
        }
        matrix[k][k] = 1.0;
        for(int i = k+1; i < N; i++){
                float32x4_t vaik = vdupq_n_f32(matrix[i][k]);
                for(j = k+1; j+4 <= N; j+=4){
                        float32x4_t vakj = vld1q_f32(&matrix[k][j]);
                        float32x4_t vaij = vld1q_f32(&matrix[i][j]);
                        float32x4_t vx = vmulq_f32(vakj, vaik);
                        vaij = vsubq_f32(vaij , vx);
                        vst1q_f32(&matrix[i][j] , vaij);
        }
        for( ; j < N; j++){
                matrix[i][j] = matrix[i][j]-matrix[i][k]*matrix[k][j];
        }
        matrix[i][k] = 0;

    }
}
}


int main(){
    matrix_init();
    resu = new float*[N];
    for(int i=0;i<N;i++)
         resu[i] = new float[N];
    copyMatrix(resu,matrix);
    struct timespec sts,ets;
   // print();
    timespec_get(&sts, TIME_UTC);
    basic_func(resu);
    timespec_get(&ets, TIME_UTC);
    time_t dsec = ets.tv_sec-sts.tv_sec;
    long dnsec = ets.tv_nsec - sts.tv_nsec;
    if(dnsec<0){
        dsec--;
        dnsec+=1000000000ll;
    }


    printf("平凡算法：",dsec,dnsec);
    copyMatrix(resu,matrix);
    timespec_get(&sts,TIME_UTC);
    neon_func(resu);
    timespec_get(&ets,TIME_UTC);
    dsec = ets.tv_sec - sts.tv_sec;
    dnsec = ets.tv_nsec - sts.tv_nsec;
    if(dnsec < 0){
        dsec--;
        dnsec += 1000000000ll;
    }
    printf("NEON：",dsec,dnsec);


}
