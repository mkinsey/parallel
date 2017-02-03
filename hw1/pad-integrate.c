#include <stdio.h>
#include <omp.h>
#define REQ_NUM 4
#define PAD 8 // assume 64 byte L1 cache line size

static long num_steps = 10000000;
double step;

int main() {
    double pi, sum = 0.0;

    // request 4
    omp_set_num_threads(REQ_NUM);
    int actual_num_threads = 0;

    double sums[REQ_NUM][PAD];
    step = 1.0/(double) num_steps;

    double start_time = omp_get_wtime();
#pragma omp parallel
    {
        double x;
        int i;
        int thread = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        // update num threads in shared memory
        if(thread == 0) actual_num_threads = num_threads; 

        for (i=thread; i<num_steps; i+=num_threads) {
            x = (i + 0.5) * step;
            sums[thread][0] += 4.0/(1.0 + x*x);
        }
    }

    // combine data
    for(int i=0; i<actual_num_threads; i++) {
        pi += sums[i][0] * step;
    }

    printf("Threads: %d\nResult: %f\nSteps: %d\n", 
            actual_num_threads, pi, (int)num_steps);

    double end_time = omp_get_wtime();
    printf("Seconds elapsed: %.5f\n", end_time - start_time);
}
