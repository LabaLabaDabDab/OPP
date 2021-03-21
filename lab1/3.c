#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <mpi.h>
#include <time.h>
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

void fulling(double* A, double* x, double* b, size_t size, int tid){
    for(int i = 0; i < size; i++)
        for (int j = 0; j < N; j++)
            A[i*N + j] = (tid*size + i) == j ? 2.0:1.0;
    for (int i = 0; i < size; ++i){
        b[i] = N + 1;
        x[i] = 0;
    }
}

void VECTxSCAL( double* res, double* vect, double scal, size_t size){
    for (int i = 0; i < size; ++i)
        res[i] = vect[i] * scal;
}

void VECTsubVECT( double* res, size_t sRes, double* vect1, double* vect2){
    for(int i = 0; i < sRes; ++i)
        res[i] = vect1[i] - vect2[i];
}

void MATxVECT(  double* res, size_t sRes,
                double* vect, size_t sVect,
                double* mat)
{
    for(int i = 0; i < sRes; ++i){
        double tmp = 0;
        for (int j = 0; j < sVect; ++j)
            tmp += mat[i * sRes + j] * vect[j];
        res[i] = tmp;
    }
}

void approx(  double* res, size_t sRes,
              double* xn, double* b,
              double* A, size_t sMatG)
{
    double* X = (double*)malloc(sizeof(double) * N);
    MPI_Allgather(xn, sRes, MPI_DOUBLE, X, sRes, MPI_DOUBLE, MPI_COMM_WORLD);
    MATxVECT(res, sRes, X, sMatG, A); // нужен целый векор xn и только часть res
    VECTsubVECT(res, sRes, res, b); // нужна только часть b
    VECTxSCAL(res, res, TAU, sRes); // только часть res
    VECTsubVECT(res, sRes, xn, res); // только часть res, xn
    free(X);
}

double qNorm(double* vect, size_t size){
    double res = 0;
    for (int i = 0; i < size; ++i)
        res += vect[i] * vect[i];
    return res;
}


double condition( double* A, size_t sMatV,
                  double* xn, double* b)
{
    double* tmp = (double*)malloc(sizeof(double) * sMatV);
    double* X = (double*)malloc(sizeof(double) * N);
    MPI_Allgather(xn, sMatV, MPI_DOUBLE, X, sMatV, MPI_DOUBLE, MPI_COMM_WORLD);

    MATxVECT(tmp, sMatV, X, N, A); // нужен весь xn, часть tmp
    VECTsubVECT(tmp, sMatV, tmp, b); // часть tmp, b

    double n1 = qNorm(tmp, sMatV); // часть tmp
    double n2 = qNorm(b, sMatV); // часть b

    double sum1 = 0, sum2 = 0;

    MPI_Allreduce(&n1,&sum1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&n2,&sum2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    free(X);
    free(tmp);
    return sqrt(sum1 / sum2);
}

int main(int argc, char **argv){
    int size_proc, tid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &tid);

    if (N % size_proc!=0) {
        if (tid == 0)
            printf("Invalid parameters\n");
        MPI_Finalize();
        return 0;
    }

    time_start();

    size_t size = N / size_proc;

    double* A = (double*)malloc(sizeof(double) * N * size);
    double* b = (double*)malloc(sizeof(double) * size);
    double* x = (double*)malloc(sizeof(double) * size);
    double* next_x = (double*)malloc(sizeof(double) * size);

    fulling(A, x, b, size, tid);

    double E = condition(A, size, x, b);
    while (E >= EPS ){
        approx(next_x, size, x, b, A, N);
        double* tmp = next_x;
        next_x = x;
        x = tmp;
        E = condition( A, size, x, b);
    }
    long dt = time_stop();
    if(tid == 0) printf("time diff %ld ms \n",dt);

    free(next_x);
    free(x);
    free(b);
    free(A);
    MPI_Finalize();
    return 0;
}
