#include <iostream>
#include <cmath>
#include <ctime>
#include <mpi.h>

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

    *x = new double[N]();
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

    double normB = 0, startTime = 0;
    if(rank == 0) {
        startTime = MPI_Wtime();

        for(int i = 0; i < N; ++i) {
            normB += b[i] * b[i];
        }
        normB = sqrt(normB);
    }

    double *processX = new double[perProcess[rank]];
    int keepCalc = 1;
    while(keepCalc) {
        double processAnswer = 0;
        for(int i = 0; i < perProcess[rank]; ++i) {
            double sum = 0;
            for(int j = 0; j < N; ++j) {
                sum += matrix[i * N + j] * x[j];
            }
            sum -= b[i];
            processX[i] = x[i + processDispls[rank]] - t * sum;
            processAnswer += sum * sum;
        }

        double sum = 0;
        MPI_Allreduce(&processAnswer, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (rank == 0) {
            keepCalc = sqrt(sum) / normB >= eps;
        }
        MPI_Bcast(&keepCalc, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Allgatherv(processX, perProcess[rank], MPI_DOUBLE, x, perProcess, processDispls, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    if(rank == 0) {
        double endTime = MPI_Wtime();
        std::cout << "Size: " << size << ", time: " << (endTime - startTime) << std::endl;

        bool correctAnswer = true;
        for(int i = 0; i < N; ++i) {
            if(fabs(fabs(x[i]) - 1) >= eps) {
                correctAnswer = false;
                break;
            }
        }

        if(correctAnswer)
            std::cout << "Accepted." << std::endl;
        else
            std::cout << "WA." << std::endl;
    }

    delete[] processX;
    delete[] x;
    delete[] b;
    delete[] matrix;
    delete[] perProcess;
    MPI_Finalize();
    return 0;
}
