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
// Then the itteration will converge, assuming no roundoff or overflow.
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

void srand48(long int seedval);
double drand48(void);

#define N 100

double 	a[N][N];// transformation matrix
double 	b[N];   // transformation vector 
double  ts[N];	// solution vector
double  ts1[N];	// solution vector

int main(int argc, char *argv[]) {
    int	n=4;	// problem size
    int	seed=10;// seed for srand48() / drand48()
    double  *t=ts;  // pointer to solution vector
    double  *t1=ts1;// pointer to next itteration of solution vector
    double	*ttemp;	// used to swap t1 and t at each itteration

    int	itt_max=5;// number of itterations to preform
    int	itt;	// current itteration
    int	i, j;   // indices into arrays

    double	sum;	// computes the inner products for A * t
    double 	error;  // max | t1[i] - t[i] |
    double 	errori; // | t1[i] - t[i] |
    char	ch;	// for error checking on command line args.

    FILE *fp;
    fp = fopen("../static-out.txt", "w");

    if( argc == 4 ) {
        if( (sscanf(argv[1],"%d %[^ /t]", &n, &ch) != 1) ||
            (sscanf(argv[2],"%d %[^ /t]", &seed, &ch) != 1) ||
            (sscanf(argv[3],"%d %[^ /t]", &itt_max, &ch) != 1) ) {
            fprintf(stderr," ERROR : useage: %s [ <n> <seed> <itt_max>]\n", argv[0]);
            return(1);
        }
    } else if(argc != 1 ) {
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
// Initialize t
    for(i=0; i< n; i++) {
        t[i] = b[i];
    }

    fprintf(fp, "\n  itt  error\n");

    // multiplication
    for(itt=0; itt<=itt_max; itt++) {
        error=0.0;
        for(i=0; i< n; i++) {
            sum = 0.0;
            for(j=0; j< n; j++) {
                sum += a[i][j] * t[j];
            }
            t1[i] = sum + b[i];
            errori = fabs(t1[i]-t[i]);
            if(errori > error) {
                error=errori;
            }
        }
        ttemp = t1;
        t1 = t;
        t = ttemp;
        fprintf(fp, "%5d %14.6e\n", itt, error);
    }

    return(0);
}
