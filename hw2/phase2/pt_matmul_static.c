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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

void srand48(long int seedval);
double drand48(void);

#define N 10000

double 	a[N][N];// transformation matrix
double 	b[N];   // transformation vector 
double  ts[N];	// solution vector
double  ts1[N];	// solution vector

void *dotproduct(void* i){
    int id = (int)i;

    printf("Hello %d\n", id);
    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    int	n=4;	// problem size
    int	seed=10;// seed for srand48() / drand48()

    double  *t=ts;  // pointer to solution vector
    double  *t1=ts1;// pointer to next itteration of solution vector
    double	*ttemp;	// used to swap t1 and t at each itteration

    int	itt_max=5;// number of itterations to preform
    int	itt;	// current itteration
    int	i, j;   // indices into arrays
    int nthreads=4; // number of threads to use

    double	sum;	// computes the inner products for A * t
    double 	error;  // max | t1[i] - t[i] |
    double 	errori; // | t1[i] - t[i] |
    char	ch;	// for error checking on command line args.
    char   *filename; // specify output file
    FILE *fp; //
    filename = "static-out-original.txt"; // default file name
    fp = fopen(filename, "w"); // open file


    // parse args
    if( argc == 6 ) {
        if( (sscanf(argv[1],"%d %[^ /t]", &n, &ch) != 1) ||
            (sscanf(argv[2],"%d %[^ /t]", &seed, &ch) != 1) ||
            (sscanf(argv[3],"%d %[^ /t]", &itt_max, &ch) != 1) ||
            (sscanf(argv[4],"%d %[^ /t]", &nthreads, &ch) != 1))
        {
            fprintf(stderr," ERROR : useage: %s [ <n> <seed> <itt_max> <n-threads> <file>]\n", argv[0]);
            return(1);
        }

        // open and test file
        fp = fopen(argv[5], "w");
        if(fp == 0){
            fprintf(stderr, "ERROR : could not open file: %s", argv[5]);
            return(1);
        }
    }
    else if(argc != 1 ) {
        fprintf(stderr," ERROR : useage: %s [ <n> <seed> <itt_max>]\n", argv[0]);
        return(1);
    }
    if( (n<1) || (N<n) ) {
        fprintf(stderr," ERROR :  n must be positive and <= %d.\n", N);
        return(1);
    }


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
        t[i] = b[i];
    }

    // set up pthreads
    pthread_t threads[nthreads];
    for(i=0; i<nthreads; i++){
        pthread_create(&threads[i], NULL, dotproduct, (void*)i);
    }

    // join threads
    for (i=0; i<nthreads; i++){
        pthread_join(threads[i], NULL);
    }

    // dot product of a and t
    fprintf(fp, "\n  itt  error\n");
    for(itt=0; itt<=itt_max; itt++) {
        error=0.0;
        // column i in a
        for(i=0; i< n; i++) {

            sum = 0.0;
            // row j in a, col j in t
            for(j=0; j< n; j++) {
                sum += a[i][j] * t[j];
            }

            {
                t1[i] = sum + b[i];
            }

            errori = fabs(t1[i]-t[i]);
            if(errori > error) {
                error=errori;
            }
        }

        // swap t and t1
        ttemp = t1;
        t1 = t;
        t = ttemp;
        fprintf(fp, "%5d %14.6e\n", itt, error);
    }

    return(0);
}
