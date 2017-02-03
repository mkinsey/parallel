#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <memory.h>
#include <time.h>
#define SIZE 500

void countsort(int a[], int n) {
    int i, j, count;
    int* temp = malloc(n * sizeof(int));
    for (i = 0; i < n; i++) {
        count = 0;
        for (j = 0; j < n; j++) {
            if (a[j] < a[i])
                count++;
            else if (a[j] == a[i] && j < i)
                count++;
        }
        temp[count] = a[i];
    }
    memcpy(a, temp, n*sizeof(int));
    free(temp);
} // Count sort

// Print first n numbers in given array
void printarray(int a[], int n) {
    int i;
    for (i = 0; i < n-1; i++){ printf("%d, ", a[i]);
    }
    printf("%d \n", a[n-1]);
}

void build_array(int a[], int n) {
    int i; 
    for (i = 0; i < n-1; i++){
        a[i] = rand() % 100;
    }
}

int cmpfunc (const void * a, const void * b) {
    return ( *(int*)a - *(int*)b );
}

int main() {
    int arr[SIZE];

    srand(time(NULL));
    build_array(arr, SIZE);

    double start = omp_get_wtime();
    qsort(arr, SIZE, sizeof(int), cmpfunc);
    double end = omp_get_wtime();
    //printarray(arr, SIZE);
    printf("Qsort %d in %.4f\n", SIZE, end-start);
    return 0;
}

