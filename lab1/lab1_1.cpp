#include <iostream>
#include <cmath>
#include <ctime>
#include <mpi.h>
 
#define N 4096
#define t 10e-6
#define eps 10e-9
 
void init(int **perProcess, int *startLine, double **matrix, double **b, double **x, int size, int rank) {
    *perProcess = new int[size]();
    for(int i = 0, tmp = size - (N % size); i < size; ++i) {
        (*perProcess)[i] = i < tmp ? (N / size) : (N / size + 1);
        if(i < rank) {
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
 
    *x = new double[N]();
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
            sum -= b[i]; //Ax - b
            processX[i] = x[i + startLine] - t * sum;
            processAnswer += sum * sum;
        }
 
        if(rank != 0) {
            MPI_Send(processX, perProcess[rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&processAnswer, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        } else {
            for(int i = startLine, c = startLine + perProcess[rank]; i < c; ++i) {
                x[i] = processX[i - startLine];
            }
 
            double sum = processAnswer;
            for(int i  = 1, currentLine = perProcess[rank]; i < size; ++i) {
                MPI_Status status;
                MPI_Recv(&x[currentLine], perProcess[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
                currentLine += perProcess[i];
 
                double tmp;
                MPI_Recv(&tmp, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status);
                sum += tmp;
            }
            sum = sqrt(sum);
 
 
            keepCalc = sum / normB >= eps;
        }
        MPI_Bcast(&keepCalc, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
