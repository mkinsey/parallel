#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

#define N 10000
#define ITT_MAX 10000

void srand48(long int seedval);
double drand48(void);

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
