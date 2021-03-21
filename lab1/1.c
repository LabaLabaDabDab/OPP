#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>

#define N 4096
#define EPS 0.000000001
#define TAU 0.01

struct timeval tv1,tv2,dtv;
void time_start() { gettimeofday(&tv1, NULL); }

long time_stop(){
    gettimeofday(&tv2, NULL);
    dtv.tv_sec= tv2.tv_sec -tv1.tv_sec;
    dtv.tv_usec=tv2.tv_usec-tv1.tv_usec;
    if(dtv.tv_usec<0) { dtv.tv_sec--; dtv.tv_usec+=1000000; }
    return dtv.tv_sec*1000+dtv.tv_usec/1000;
}


void fulling(double* A, double* b, double* x, size_t size){
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            A[i*size + j] = (i == j) ? 2.0 : 1.0;

    for (int i = 0; i < size; ++i){
        b[i] = size + 1;
        x[i] = 0;
    }
}

void VECTxSCAL(double* res, double* vect, double scal, size_t size){
    for (int i = 0; i < size; ++i)
        res[i] = vect[i] * scal;
}

void MATxVECT(double* res, double* mat, double* vect, size_t size){
    for (int i = 0; i < size; ++i){
        double tmp = 0;
        for (int j = 0; j < size; ++j)
            tmp += mat[i * size + j] * vect[j];
        res[i] = tmp;
    }
}

void VECTsubVECT(double* res, double* vect1, double* vect2, size_t size){
    for (int i = 0; i < size; ++i)
        res[i] = vect1[i] - vect2[i];
}

void approx(double* res, double* xn, double* b, double* A, size_t size){
    double tau = TAU;
    MATxVECT(res, A, xn, size);
    VECTsubVECT(res, res, b, size);
    VECTxSCAL(res, res, tau, size);
    VECTsubVECT(res, xn, res, size);
}

double norm(double* vect, size_t size){
    double res = 0;
    for (int i = 0; i < size; ++i)
        res += vect[i] * vect[i];
    return sqrt(res);
}

double condition(double* xn, double* b, double* A, size_t size){
    double* tmp = (double*)malloc(sizeof(double) * N);

    MATxVECT(tmp, A, xn, size);
    VECTsubVECT(tmp, tmp, b, size);

    double n = norm(tmp, size) / norm(b, size);
    free(tmp);
    return n;
}

int main(int argc, char **argv){
    time_start();
    size_t size = N;

    double* A = (double*)malloc(sizeof(double) * N * N);
    double* b = (double*)malloc(sizeof(double) * N);
    double* x = (double*)malloc(sizeof(double) * N);
    double* next_x = (double*)malloc(sizeof(double) * N);
    fulling(A, b, x, size);
    double E = condition(x, b, A, size);
    while (E >= EPS){
        approx(next_x, x, b, A, size);
        double* tmp = next_x;
        next_x = x;
        x = tmp;
        E = condition(x, b, A, size);
    }
    long dt = time_stop();
    printf("time diff %ld ms\n", dt);
    free(next_x);
    free(x);
    free(b);
    free(A);
    return 0;
}
