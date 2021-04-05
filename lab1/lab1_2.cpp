#include <iostream>
#include <cmath>
#include <ctime>
#include <mpi.h>
#include <cstdio>

#define N 4096
#define t 10e-6
#define eps 10e-9

void init(int **perProcess, int **processDispls, double **matrix, double **b, double **x, int size, int rank) {
    *perProcess = new int[size]();
    *processDispls = new int[size]();
    int offset = 0;
    for(int i = 0, tmp = size - (N % size); i < size; ++i) {
        (*processDispls)[i] = offset;
        (*perProcess)[i] = i < tmp ? (N / size) : (N / size + 1);
        offset += ((*perProcess)[i]);
    }

    *matrix = new double[(*perProcess)[rank] * N];
    for(int i = 0; i < (*perProcess)[rank]; ++i) {
        for(int j = 0; j < N; ++j) {
            (*matrix)[i * N + j] = ((*processDispls)[rank] + i) == j ? 2 : 1;
        }
    }

    *b = new double[(*perProcess)[rank]];
    for(int i = 0; i < (*perProcess)[rank]; ++i) {
        (*b)[i] = N + 1;
    }

    *x = new double[N / size + 1]();
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int *perProcess = 0;
    int *processDispls = 0;
    double *matrix = 0, *b = 0, *x = 0;
    init(&perProcess, &processDispls, &matrix, &b, &x, size, rank);

    double startTime = 0, normB = 0, tmpNormB = 0 ;

    for(int i = 0; i < perProcess[rank]; ++i) {
        tmpNormB += b[i] * b[i];
    }
    MPI_Allreduce(&tmpNormB, &normB, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


    double *tmpSum = new double[perProcess[rank]];
    double *tmpX = new double[N / size + 1]();
    int keepCalc = 1;
    while(keepCalc) {
        for(int i = 0; i < perProcess[rank]; ++i) {
            tmpSum[i] = 0;
        }


        for(int i = 0, currentCrds = processDispls[rank]; i < size; ++i) {

            for(int j = 0; j < perProcess[rank]; ++j) {
                for(int k = currentCrds, c = currentCrds + perProcess[i]; k < c; ++k) {
                    tmpSum[j] += matrix[j * N + k] * x[k - currentCrds];
                }
            }

            MPI_Status status;
            MPI_Sendrecv(x, N / size + 1, MPI_DOUBLE, (rank - 1 + size) % size, 0,
                         tmpX, N / size + 1, MPI_DOUBLE, (rank + 1) % size, 0, MPI_COMM_WORLD, &status);

            std::swap(x, tmpX);
            currentCrds = (currentCrds + perProcess[i]) % N;
        }

        double processAnswer = 0;
        for(int i = 0; i < perProcess[rank]; ++i) {
            tmpSum[i] -= b[i];
            x[i] = x[i] - tmpSum[i] * t;
            processAnswer += tmpSum[i] * tmpSum[i];
        }

        double sum = 0;
        MPI_Allreduce(&processAnswer, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (rank == 0) {
            keepCalc = sqrt(sum) / normB >= eps;
        }

        MPI_Bcast(&keepCalc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    double *fullX;
    if (rank == 0) {
        fullX = new double[N];
    }

    MPI_Gatherv(x, perProcess[rank], MPI_DOUBLE, fullX, perProcess, processDispls, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double endTime = MPI_Wtime();
        std::cout << "Size: " << size << ", time: " << (endTime - startTime) << std::endl;

        bool correctAnswer = true;
        for(int i = 0; i < N; ++i) {
            if(fabs(fabs(fullX[i]) - 1) >= eps) {
                correctAnswer = false;
                break;
            }
        }

        if(correctAnswer)
            std::cout << "Accepted." << std::endl;
        else
            std::cout << "WA." << std::endl;

        delete[] fullX;
    }

    delete[] tmpX;
    delete[] x;
    delete[] b;
    delete[] matrix;
    delete[] perProcess;
    MPI_Finalize();
    return 0;
}
