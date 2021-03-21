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
    for (int i = 0; i < N; ++i){
        b[i] = N + 1;
        x[i] = 0;
    }
}

void VECTxSCAL( double* res, double* vect, double scal, size_t size)
{
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
              double* A, size_t sMatG,
              int tid)
{
    int shift = tid * sRes;
    MATxVECT(res, sRes, xn, sMatG, A);
    VECTsubVECT(res, sRes, res, b + shift);
    VECTxSCAL(res, res, TAU, sRes);
    VECTsubVECT(res, sRes, xn + shift, res);
}

double qNorm(double* vect, size_t size){
    double res = 0;
    for (int i = 0; i < size; ++i)
        res += vect[i] * vect[i];
    return res;
}


double condition( double* A, size_t sMatV,
                  double* xn, double* b,
                  size_t sVects, int tid)
{
    int shift = tid * sMatV;

    double* tmp = (double*)malloc(sizeof(double) * sMatV);

    MATxVECT(tmp, sMatV, xn, sVects, A);
    VECTsubVECT(tmp, sMatV, tmp, b + shift);

    double n1 = qNorm(tmp, sMatV);
    double n2 = qNorm(b + shift, sMatV);

    double sum1 = 0, sum2 = 0;

    MPI_Allreduce(&n1,&sum1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&n2,&sum2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

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
    double* b = (double*)malloc(sizeof(double) * N);
    double* x = (double*)malloc(sizeof(double) * N);
    double* next_x = (double*)malloc(sizeof(double) * N);

    fulling(A, x, b, size, tid);

    double E = condition(A, size, x, b, N, tid);
    while (E >= EPS ){
        approx(next_x, size, x, b, A, N, tid);
        MPI_Allgather(next_x, size, MPI_DOUBLE, x, size, MPI_DOUBLE, MPI_COMM_WORLD);
        E = condition( A, size, x, b, N, tid);
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
