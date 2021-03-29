#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

using namespace std;

#define N 15000
#define eps 1e-6

void matrMul(double* A, double* x, double* xnew) {
#pragma omp for
    for (int i = 0; i < N; i++) {
        xnew[i] = 0;
        for (int j = 0; j < N; j++) {
            xnew[i] += A[j + i * N] * x[j];
        }
    }

}

int main() {
    double startTime = omp_get_wtime( );

    double *x = nullptr, *b = nullptr, *xnew = nullptr, tau = 0.0001, norm = 0, norm_b = 0, sum = 0;

    x = (double *)calloc(N, sizeof(double));
    b = (double *)malloc((N + 1) * sizeof(double));
    xnew = (double *)calloc(N, sizeof(double));

#pragma omp for
    for (int i = 0; i < N; ++i) {
        b[i] = N + 1;
    }

    double *A = nullptr;
    A = (double *)malloc(N * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i == j) ? 2.0 : 1.0;
        }
    }

    for (int i = 0; i < N; i++)
        sum += b[i] * b[i];
    norm_b = sqrt(sum);

    matrMul(A, x, xnew);
    for (int i = 0; i < N; i++)
        xnew[i] = xnew[i] - b[i];
    for (int i = 0; i < N; i++)
        sum += xnew[i] * xnew[i];
    double norm_0 = sqrt(sum);
    for (int i = 0; i < N; i++)
        xnew[i] = -tau * xnew[i];
    for (int i = 0; i < N; i++)
        xnew[i] = x[i] + xnew[i];
    for (int i = 0; i < N; i++)
        x[i] = xnew[i];

    sum = 0;
    matrMul(A, x, xnew);
    for (int i = 0; i < N; i++)
        xnew[i] = xnew[i] - b[i];
    for (int i = 0; i < N; i++)
        sum += xnew[i] * xnew[i];
    norm = sqrt(sum);

    if (norm_0 <= norm)
        tau = -tau;

    norm /= norm_b;

#pragma omp parallel
    while (norm >= eps) {
        sum = 0;
        matrMul(A, x, xnew);

#pragma omp for
        for (int i = 0; i < N; i++)
            xnew[i] = xnew[i] - b[i];

#pragma omp shared(sum) for reduction(+:sum)
        for (int i = 0; i < N; i++)
            sum += xnew[i] * xnew[i];
        norm = sqrt(sum) / norm_b;

#pragma omp for
        for (int i = 0; i < N; i++)
            xnew[i] = -tau * xnew[i];

#pragma omp for
        for (int i = 0; i < N; i++)
            xnew[i] = x[i] + xnew[i];

#pragma omp for
        for (int i = 0; i < N; i++)
            x[i] = xnew[i];
    }

    double endTime = omp_get_wtime( );

    cout << "Res: " << "\n";
    for (int i = 0; i < N; i++) {
        cout << "x[" << i << "] = " << xnew[i] << "\n";
    }
    cout << "Time: "<< (double)(endTime - startTime) << endl;

    return 0;
}
