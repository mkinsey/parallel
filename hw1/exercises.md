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

Variant 2 requires some extra work to be parallelized while avoiding race 
conditions. First, create a temporary array, say `t` to hold the previous 
values of `x`. Next, split the for loop into two parallel loops. In the first,
simply add the line updating `x[i]`. In the second loop, update the value of
`a[i]` but use `t[i+1]` in place of `x[i+1]`.

### Variant 3

Variant 3 can be parallelized by breaking the for loop into two parallel 
worksharing loops. Because calculating `a[i]` requires first updating and then
accessing `x[i-1]`, we must be sure that `x[i-1]` is computed first. The 
implicit barrier at the end of the first worksharing section accomplishes this.

### Variant 4

Variant 4 uses the same solution as Variant 3. By using two separate loops,
the index updated in `a` is irrelevant, as long as the accessed value in `x` has
been computed. 
