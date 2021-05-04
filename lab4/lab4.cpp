#include <cstring>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <algorithm>

double phi(double x, double y, double z) {
    return pow(x, x) + pow(y, y) + pow(z, z);
}

double ro(double x, double y, double z, const double a) {
    return 6 - a * phi(x,y,z);
}

double updLayer(
        int base_z, int height, double *omega_part, double *tmp_omega_part,
        double hx, double hy, double hz,
        const int N, const double x0, const double y0, const double z0, const double a) {

    int abs_z = base_z + height;

    if (abs_z == 0 || abs_z == N - 1) {
        memcpy(tmp_omega_part + height * N * N, omega_part + height * N * N, N * N * sizeof(double));
        return 0;
    }


    double max_delta = 0;
    double z = z0 + abs_z * hz;

    for (int i = 0; i < N; i++) {
        double x = x0 + i * hx;

        for (int j = 0; j < N; j++) {
            double y = y0 + j * hy;

            int cell = height * N * N + i * N + j;

            if (i == 0 || i == N - 1 || j == 0 || j == N - 1) {
                tmp_omega_part[cell] = omega_part[cell];
                continue;
            }

            tmp_omega_part[cell] = ((omega_part[height * N * N + (i + 1) * N + j]
                                     + omega_part[height * N * N + (i - 1) * N + j]) / (hx * hx)
                                    + (omega_part[height * N * N + i * N + (j + 1)]
                                       + omega_part[height * N * N + i * N + (j - 1)]) / (hy * hy)
                                    + (omega_part[(height + 1) * N * N + i * N + j]
                                       + omega_part[(height - 1) * N * N + i * N + j]) / (hz * hz)
                                    - ro(x, y, z, a)) /
                                   ((2 / (hx * hx)) + (2 / (hy * hy)) + (2 / (hz * hz)) + a);

            max_delta = std::max(max_delta, std::abs(tmp_omega_part[cell] - omega_part[cell]));
        }
    }

    return max_delta;
}

double JacobiMethod(const double epsilon, const double a, const int N,
                    const double x0, const double y0, const double z0,
                    const double x1, const double y1, const double z1) {

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (N % size) {
        if(rank == 0) {
            std::cout << "Invalid number of processes" << std::endl;
            return -1;
        }
    }

    double start_time = MPI_Wtime();

    double hx = (x1 - x0) / (N - 1);
    double hy = (y1 - y0) / (N - 1);
    double hz = (z1 - z0) / (N - 1);

    int part_height = N / size;

    int part_base_z = rank * part_height - 1;


    double *omega = new double[(part_height + 2) * N * N];
    double *tmp_omega = new double[(part_height + 2) * N * N];

    int iterationsCounter = 0;


    for (int i = 0; i < part_height + 2; i++) {
        int omega_z = i + part_base_z;
        double real_z = z0 + hz * omega_z;

        for (int j = 0; j < N; j++) {

            double x = x0 + hx * j;

            for (int k = 0; k < N; k++) {

                double y = y0 + hy * k;

                if (omega_z == 0 || omega_z == N - 1 || j == 0 || j == N - 1 || k == 0 || k == N - 1) {

                    omega[i * N * N + j * N + k] = phi(x, y, real_z);
                } else {

                    omega[i * N * N + j * N + k] = 0;
                }
            }
        }
    }



    double max_delta_shared;
    do {
        double max_delta = 0;
        double tmp_delta = updLayer(part_base_z, 1, omega, tmp_omega, hx, hy, hz, N, x0, y0, z0, a);
        max_delta = std::max(max_delta, tmp_delta);

        tmp_delta = updLayer(part_base_z, part_height, omega, tmp_omega, hx, hy, hz, N, x0, y0, z0, a);
        max_delta = std::max(max_delta, tmp_delta);

        MPI_Request rq[4];

        if (rank != 0) {
            MPI_Isend(tmp_omega + N * N, N * N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &rq[0]);
            MPI_Irecv(tmp_omega, N * N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &rq[2]);
        }

        if (rank != size - 1) {
            MPI_Isend(tmp_omega + part_height * N * N, N * N, MPI_DOUBLE,
                      rank + 1, 0, MPI_COMM_WORLD, &rq[1]);
            MPI_Irecv(tmp_omega + (part_height + 1) * N * N, N * N, MPI_DOUBLE,
                      rank + 1, 0, MPI_COMM_WORLD, &rq[3]);
        }

        for (int i = 2; i < part_height; i++) {
            double tmpdelta = updLayer(part_base_z, i, omega, tmp_omega, hx, hy, hz, N, x0, y0, z0, a);
            max_delta = std::max(max_delta, tmpdelta);
        }

        if (rank != 0) {
            MPI_Wait(&rq[0], MPI_STATUS_IGNORE);
            MPI_Wait(&rq[2], MPI_STATUS_IGNORE);
        }

        if (rank != size - 1) {
            MPI_Wait(&rq[1], MPI_STATUS_IGNORE);
            MPI_Wait(&rq[3], MPI_STATUS_IGNORE);
        }

        memcpy(omega, tmp_omega, (part_height + 2) * N * N  * sizeof(double));

        MPI_Reduce(&max_delta, &max_delta_shared, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Bcast(&max_delta_shared, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        iterationsCounter++;
    } while (max_delta_shared >= epsilon);

    delete[] tmp_omega;

    double *fullResult = nullptr;
    if (rank == 0) {
        fullResult = new double[N * N * N];
    }

    MPI_Gather(omega + N * N, part_height * N * N, MPI_DOUBLE, fullResult,
               part_height * N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (rank == 0) {

        double max_delta = 0;

        for (int layer = 0; layer < N; layer++){
            double z = z0 + layer * hz;

            for (int j = 0; j < N; j++) {
                double x = x0 + j * hx;

                for (int k = 0; k < N; k++) {
                    double y = y0 + k * hy;

                    max_delta = std::max(max_delta, std::abs(fullResult[layer * N * N + j * N + k] - phi(x, y, z)));
                }
            }
        }

        std::cout << "Answer: delta = " << max_delta << std::endl;
        printf("Time: %lf\n", end_time - start_time);
        std::cout << iterationsCounter << " cycle iterations" << std::endl;

        delete[] fullResult;
    }

    delete[] omega;

    return end_time - start_time;
}


void JacobiMethodTest(const int repeats, const double epsilon, const double a, const int N,
    const double x0, const double y0, const double z0,
    const double x1, const double y1, const double z1) {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double current_time, best_time = std::numeric_limits<double>::max();

    for (int i = 1; i <= repeats; i++) {
        if (rank == 0) {
        std::cout << "Try " << i << "/" << repeats << std::endl;
        }
        current_time = JacobiMethod(epsilon, a, N, x0, y0, z0, x1, y1, z1);
        if (rank == 0) {
            best_time = (current_time < best_time) ? current_time : best_time;
        }
    }

    if (rank == 0) {
        printf("Best time: %lf sec\n", best_time);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    const int repeats = 1;
    const double epsilon = 1e-8;
    const double a = 1e5;
    const int N = 240;
    const double x0 = -1;
    const double y0 = -1;
    const double z0 = -1;
    const double x1 = 1;
    const double y1 = 1;
    const double z1 = 1;
    JacobiMethodTest(repeats, epsilon, a, N, x0, y0, z0, x1, y1, z1);
    MPI_Finalize();
    return 0;
}
