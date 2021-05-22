#include <mpi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <cstddef>

#define REQUEST_TAG 10
#define ANSWER_TAG 20
#define SUCCESS 500
#define FAIL 404
#define NEED_TASKS 220
#define TURN_OFF 312
#define TASKS_IN_LIST 200
#define L 10000

typedef struct Task {
    int repeatNum;
} Task;

typedef Task TaskList[TASKS_IN_LIST];

int rank, size;
int iterCounter = 0, iterMax = 5;
int curTask = 0, listSize = 0;
double globalRes = 0;
TaskList taskList;
pthread_mutex_t list_mutex;
MPI_Datatype MPI_TASK;

void getTaskList(int iter) {
    listSize = TASKS_IN_LIST;
    for (int i = rank * TASKS_IN_LIST; i < (rank + 1) * TASKS_IN_LIST; i++) {
        taskList[i % TASKS_IN_LIST].repeatNum = abs(TASKS_IN_LIST / 2 - i % TASKS_IN_LIST) * abs(rank - (iterCounter % size)) * L;
    }
}

int getTaskFrom(int from) {
    int flag = NEED_TASKS;
    MPI_Send(&flag, 1, MPI_INT, from, REQUEST_TAG, MPI_COMM_WORLD);
    MPI_Recv(&flag, 1, MPI_INT, from, ANSWER_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (flag == FAIL) return FAIL;

    Task recvTask;
    MPI_Recv(&recvTask, 1, MPI_TASK, from, ANSWER_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    pthread_mutex_lock(&list_mutex);
    taskList[listSize] = recvTask;
    listSize++;
    pthread_mutex_unlock(&list_mutex);
    return SUCCESS;
}

void doTask(Task task) {
    for (int i = 0; i < task.repeatNum; i++) {
        globalRes += sin(i);
    }
}

void *taskSenderThread(void *args) {
    int flag;
    while (iterCounter < iterMax) {
        MPI_Status status;
        MPI_Recv(&flag, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_TAG, MPI_COMM_WORLD, &status);
        if (flag == TURN_OFF) break;

        pthread_mutex_lock(&list_mutex);
        if (curTask >= listSize - 1) {
            pthread_mutex_unlock(&list_mutex);

            flag = FAIL;
            MPI_Send(&flag, 1, MPI_INT, status.MPI_SOURCE, ANSWER_TAG, MPI_COMM_WORLD);
            continue;
        }

        listSize--;
        Task sendTask = taskList[listSize];
        pthread_mutex_unlock(&list_mutex);

        flag = SUCCESS;
        MPI_Send(&flag, 1, MPI_INT, status.MPI_SOURCE, ANSWER_TAG, MPI_COMM_WORLD);
        MPI_Send(&sendTask, 1, MPI_TASK, status.MPI_SOURCE, ANSWER_TAG, MPI_COMM_WORLD);
    }
    return NULL;
}

void *taskEvaluatorThread(void *args) {
    MPI_Barrier(MPI_COMM_WORLD);

    struct timespec start, end;
    iterCounter = 0;
    while (iterCounter < iterMax) {
        int tasksDone = 0;
        int hasTasks = 1;

        pthread_mutex_lock(&list_mutex);
        curTask = 0;
        getTaskList(iterCounter);
        pthread_mutex_unlock(&list_mutex);

        clock_gettime(CLOCK_MONOTONIC, &start);
        while (hasTasks) {
            pthread_mutex_lock(&list_mutex);
            if (curTask < listSize) {
                Task task = taskList[curTask];
                pthread_mutex_unlock(&list_mutex);
                doTask(task);
                tasksDone++;
                pthread_mutex_lock(&list_mutex);
                curTask++;
                pthread_mutex_unlock(&list_mutex);
                continue;
            }

            curTask = 0;
            listSize = 0;
            pthread_mutex_unlock(&list_mutex);
            hasTasks = 0;
            for (int i = 0; i < size; i++) {
                if (i == rank) continue;
                if (getTaskFrom(i) == SUCCESS) {
                    hasTasks = 1;
                }
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &end);

        double timeTaken = end.tv_sec - start.tv_sec + 0.000000001 * (end.tv_nsec - start.tv_nsec);
        printf("Iteration %d, Process %d - tasks done = %d\n", iterCounter, rank, tasksDone);
        printf("Iteration %d, Process %d - globalRes = %lf\n", iterCounter, rank, globalRes);
        printf("Iteration %d, Process %d - time taken = %lf sec.\n", iterCounter, rank, timeTaken);
        fflush(stdout);

        double minTime, maxTime;
        MPI_Reduce(&timeTaken, &minTime, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&timeTaken, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Iteration %d, Imbalance time = %lf sec., %lf%\n", iterCounter, maxTime - minTime, (maxTime - minTime) / maxTime * 100);
            fflush(stdout);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        iterCounter++;
    }

    int flag = TURN_OFF;
    MPI_Send(&flag, 1, MPI_INT, rank, REQUEST_TAG, MPI_COMM_WORLD);

    return NULL;
}

void createTaskType() {
    const int num = 1;
    int lenOfBlocks[num] = { 1 };
    MPI_Datatype types[num] = { MPI_INT };
    MPI_Aint offsets[num];

    offsets[0] = offsetof(Task, repeatNum);

    MPI_Type_create_struct(num, lenOfBlocks, offsets, types, &MPI_TASK);
    MPI_Type_commit(&MPI_TASK);
}

int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "Couldn't init MPI with MPI_THREAD_MULTIPLE level support\n");
        MPI_Finalize();
        return 0;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    createTaskType();

    pthread_t threads[2];
    pthread_mutex_init(&list_mutex, NULL);
    pthread_attr_t attrs;
    if (pthread_attr_init(&attrs) != 0) {
        perror("Cannot initialize attributes");
        abort();
    }
    if (pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE) != 0) {
        perror("Error in setting attributes");
        abort();
    }

    if (pthread_create(&threads[0], &attrs, taskSenderThread, NULL) != 0 ||
        pthread_create(&threads[1], &attrs, taskEvaluatorThread, NULL) != 0) {
        perror("Cannot create a thread");
        abort();
    };
    pthread_attr_destroy(&attrs);
    for (int i = 0; i < 2; i++) {
        if (pthread_join(threads[i], NULL) != 0) {
            perror("Cannot join a thread");
            abort();
        }
    }

    pthread_mutex_destroy(&list_mutex);
    MPI_Type_free(&MPI_TASK);
    MPI_Finalize();
    return 0;
}
