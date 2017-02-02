//
// Created by mdk on 2/1/17.
//
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <omp.h>
#define SIZE 5

// parallelized version of the count sort program
void count_sort(int a[], int n) {
    int i;
    int* temp = malloc(n * sizeof(int));
#pragma omp parallel for
    for (i = 0; i < n; i++) {
        int j, count = 0;
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
}

// Print first n numbers in given array
void print_array(int a[], int n) {
    int i;
    for (i = 0; i < n-1; i++){
        printf("%d, ", a[i]);
    }
    printf("%d \n", a[n-1]);
}

int main() {
    int arr[SIZE] = {1, 3, 2, 6, 4};

    double start = omp_get_wtime();
    count_sort(arr, SIZE);
    double end = omp_get_wtime();

    print_array(arr, SIZE);
    printf("Completed in %.2f seconds\n", end-start);
    return 0;
}

