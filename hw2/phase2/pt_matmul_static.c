// The following iteration can be used to solve linear systems
//   t_{i+1} = A t_i + b
// If the itteration converges to t, then t == t_{i+1} == t_i
// So t = A t + b
//   or  (I-a) t = b
//   where, I is the n*n idenity matrix
// There are several important applied problems where convergence 
// will take place. One such case is when for
// each row of A ( rows 0 <= i < n)
//             sum(j=0 ... n-1) abs(a[i][j])  < 1.0    
// Then the iteration will converge, assuming no roundoff or overflow.
// Example
// % ./matmul_static 4 10 5
//
//  a=
//  0.189331   0.147829  -0.009582   0.012830
// -0.020409   0.222627   0.073037   0.042701
//  0.069882   0.228326  -0.001161   0.024936
//  0.116375  -0.100117   0.229832   0.022235
//
//  b=
//  2.411774   9.837874   6.251698   6.576916
//
//  itt  error
//    0   2.878398e+00
//    1   8.266521e-01
//    2   2.688652e-01
//    3   8.817662e-02
//    4   2.832084e-02
//    5   9.015857e-03
//

#define _XOPEN_SOURCE 600

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

void srand48(long int seedval);
double drand48(void);

#define N 10000
#define ITT_MAX 10

int	n=4;	// problem size
int nthreads=4; // number of threads to use
double 	a[N][N];// transformation matrix
double 	b[N];   // transformation vector 
double  ts[ITT_MAX+2][N];	// solution vector. +1 for extra iteration +1 for filling 0 with b

int	nit=5;// number of iterations to preform
double 	errori[ITT_MAX+1][N]; // | t1[i] - t[i] |
FILE *fp; // file pointer
pthread_barrier_t barrier;


/*
 * This function is called by each thread after it is initialized.
 */
void *dotproduct(void* arg){
    int itt;
    int j, i;
    long id = (long)arg; // thread id
    double	sum = 0;	// computes the inner products for A * t

    // dot product of a and t
    for(itt=0; itt<=nit; itt++) {

        printf("Hello thread %li iteration %d\n", id, itt);
        // column i in a
        for(i=id; i< n; i+=nthreads) {

            sum = 0.0;
//             row j in a, col j in t
            for(j=0; j< n; j++) {
                sum += a[i][j] * ts[itt][j];
            }

            ts[itt+1][i] = sum + b[i];

            errori[itt][i] = fabs(ts[itt+1][i]-ts[itt][i]);
        }

        // wait for all threads to finish this iteration
        pthread_barrier_wait(&barrier);

        if(id == 0){
            // TODO find max error
            double current_max = errori[itt][0];
            for(i=1; i<nit; i++){
                if(errori[itt][i] > current_max)
                    current_max = errori[itt][i];
            }
            fprintf(fp, "%5d %14.6e\n", itt, current_max);
        }

    }
    return NULL;
}

int main(int argc, char *argv[]) {
    int	seed=10;// seed for srand48() / drand48()

//    double  *t=ts;  // pointer to solution vector
//    double  *t1=ts1;// pointer to next itteration of solution vector
//    double	*ttemp;	// used to swap t1 and t at each itteration

    int	i, j;
    long threadnum =0;

    char	ch;	// for error checking on command line args.
    char   *filename; // specify output file
    filename = "static-out-pthreads.txt"; // default file name

    // parse args
    if( argc == 6 ) {
        if( (sscanf(argv[1],"%d %[^ /t]", &n, &ch) != 1) ||
            (sscanf(argv[2],"%d %[^ /t]", &seed, &ch) != 1) ||
            (sscanf(argv[3],"%d %[^ /t]", &nit, &ch) != 1) ||
            (sscanf(argv[4],"%d %[^ /t]", &nthreads, &ch) != 1))
        {
            fprintf(stderr," ERROR : useage: %s [ <n> <seed> <itt_max> <n-threads> <file>]\n", argv[0]);
            return(1);
        }
        filename = argv[5];

    }
    else if(argc != 1 ) {
        fprintf(stderr," ERROR : useage: %s [ <n> <seed> <itt_max>]\n", argv[0]);
        return(1);
    }
    if( (n<1) || (N<n) ) {
        fprintf(stderr," ERROR :  n must be positive and <= %d.\n", N);
        return(1);
    }
    if( nit<1 || nit > ITT_MAX){
        fprintf(stderr," ERROR :  itt_max must be positive and <= %d.\n", ITT_MAX);
        return(1);
    }
    // open and test file
    fp = fopen(filename, "w");
    if(fp == 0){
        fprintf(stderr, "ERROR : could not open file: %s", argv[5]);
        return(1);
    }

    pthread_t threads[nthreads]; // threads

    // Generate matrix a with | eigenvalues | < 1
    srand48((long int)seed);
    fprintf(fp, "\n  a=\n");
    for(i=0; i< n; i++) {
        for(j=0; j< n; j++) {
            a[i][j] = 1.999 * (drand48() - 0.5) / n;
            fprintf( fp, "%10.6f ", a[i][j]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n  b=\n");

    // Generate vector b
    for(i=0; i< n; i++) {
        b[i] = 10.0 * drand48();
        fprintf(fp, "%10.6f ", b[i]);
    }
    fprintf(fp ,"\n");

    // Initialize t to elements in b
    for(i=0; i< n; i++) {
        ts[0][i] = b[i];
    }
    fprintf(fp, "\n  itt  error\n");

    // init barrier
    pthread_barrier_init(&barrier, NULL, nthreads);

    // set up pthreads
    for(threadnum=0; threadnum<nthreads; threadnum++){
        pthread_create(&threads[threadnum], NULL, dotproduct, (void*)threadnum);
    }


    // join threads
    for (threadnum=0; threadnum<nthreads; threadnum++){
        pthread_join(threads[threadnum], NULL);
    }

    // cleanup
    fclose(fp);

    return(0);
}
