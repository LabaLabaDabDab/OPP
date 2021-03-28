#include <iostream>
#include <cmath>
#include <ctime>
#include <mpi.h>
#include <cstdio>

#define N 4096
#define t 10e-6
#define eps 10e-9

void init(int **perProcess, int *startLine, double **matrix, double **b, double **x, int size, int rank) {
    *perProcess = new int[size]();
    for(int i = 0, tmp = size - (N % size); i < size; ++i) {
        (*perProcess)[i] = i < tmp ? (N / size) : (N / size + 1);
        if (i < rank) {
            *startLine += (*perProcess)[i];
        }
    }

    *matrix = new double[(*perProcess)[rank] * N];
    for(int i = 0; i < (*perProcess)[rank]; ++i) {
        for(int j = 0; j < N; ++j) {
            (*matrix)[i * N + j] = ((*startLine) + i) == j ? 2 : 1;
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

    int startLine = 0;
    int *perProcess = 0;
    double *matrix = 0, *b = 0, *x = 0;
    init(&perProcess, &startLine, &matrix, &b, &x, size, rank);

    double startTime = 0, normB = 0;
    if(rank != 0) {
        for(int i = 0; i < perProcess[rank]; ++i) {
            normB += b[i] * b[i];
        }
        MPI_Send(&normB, 1, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
    } else {
        startTime = MPI_Wtime();

        for(int i = 0; i < perProcess[rank]; ++i) {
            normB += b[i] * b[i];
        }

        for(int i = 1; i < size; ++i) {
            double tmp;
            MPI_Status status;
            MPI_Recv(&tmp, 1, MPI_DOUBLE, i, 4, MPI_COMM_WORLD, &status);
            normB += tmp;
        }

        normB = sqrt(normB);
    }

    double *tmpSum = new double[perProcess[rank]];
    double *tmpX = new double[N / size + 1]();
    int keepCalc = 1;
    while(keepCalc) {
        for(int i = 0; i < perProcess[rank]; ++i) {
            tmpSum[i] = 0;
        }

        for(int i = 0, currentCrds = startLine; i < size; ++i) {

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

        if(rank != 0) {
            MPI_Send(&processAnswer, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        } else {
            double sum = processAnswer;
            for(int i  = 1; i < size; ++i) {
                MPI_Status status;
                double tmp;
                MPI_Recv(&tmp, 1, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, &status);
                sum += tmp;
            }
            sum = sqrt(sum);

            keepCalc = sum / normB >= eps;
        }

        MPI_Bcast(&keepCalc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if(rank != 0) {
        MPI_Send(x, perProcess[rank], MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    } else {
        double *fullX = new double[N];
        for(int i = 0; i < perProcess[rank]; ++i) {
            fullX[i] = x[i];
        }

        for(int i = 1, currentLine = perProcess[rank]; i < size; ++i) {
            MPI_Status status;
            MPI_Recv(&fullX[currentLine], perProcess[i], MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status);
            currentLine += perProcess[i];
        }

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
