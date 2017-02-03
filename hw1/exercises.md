# Homework 1 Exercises 

Questions from the text Parallel Programming and OpenMP

## 25.2.1

The output will vary by machine for both programs. For my machine, the serial 
program will output:

```
procs_8
threads_1
num_0
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

Variant 2 cannot be parallelized because of an inter-loop dependency. Each loop 
updates value `x[i]` and accesses value `x[i+1]`, giving race conditions
between threads.

### Variant 3

Variant 3 cannot be parallelized due to the same reason listed for variant 2,
replacing `x[i-1]` for `x[i+1]`.

### Variant 4

Variant 4 cannot be parallelized for the same reasons listed above. Additionaly,
`a[i]` determines the value of `a[i+1]`, another inter-loop dependency.

