# Homework 1 Exercises 

Questions from the text Parallel Programming and OpenMP

## 25.2.1

The output will vary by machine for both programs. For my machine, the serial 
program will output:

```
procs 8
threads 1
num 0
```

The parallelized version will likely have a different output each time it is 
executed because calls to various print statements happen concurrently. The 
output may look something like this:

```
procs_8
threads_8
procs_8
threads_8
num_5
procs_8
threads_8
num_4
(...)
```


## 25.2.2

### Variant 1

The loop can be parallelized with a simple workflow construct 
(`#pragma omp parallel for`) before the for loop.

### Variant 2

### Variant 3

### Variant 4


