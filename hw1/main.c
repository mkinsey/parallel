#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#define SIZE 5

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
    for (i = 0; i < n-1; i++){
        printf("%d, ", a[i]);
    }
    printf("%d \n", a[n-1]);
}

int notmain() {
    int arr[SIZE] = {5, 3, 2, 6, 9};
    countsort(arr, SIZE);
    printarray(arr, SIZE);
    return 0;
}

