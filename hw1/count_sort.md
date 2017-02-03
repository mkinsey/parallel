# Homework 1 

Michael Kinsey

2-7-17

## Question A

If we attempt to parallelize the outer for loop in the
provided count_sort method, we will need to share the 
variables `i` and `temp`. The variables `j` and `count` 
should be private to each thread.

## Question B 

There are no loop-carried dependencies. Due to the nature
of the algorithm, each index of the `temp` array is only updated
once. Using the work-sharing construct, each iteration of the for
loop is computed once. This iteration then works through the collection
independently of other iterations.

## Question C

The `memcpy` call can be replaced with a second parallel loop, replacing 
indexes of `a` with `temp` however I suspect it would be faster to copy 
the whole array at once. 
 
## Question E

The following table compares averaged execution time of sorting n numbers using 
the given sorting algorithm. Both parallelized versions ran on 4 processors
with 4 threads.

 n    | Serial Time | Parallel Time | Worsharing | Library Qsort
---   |    ---      |     ---       |    ---     |   ---
500   | 0.0047 s    | 0.0091 s      | 0.0083 s   | 0.0001 s
1000  | 0.0131 s    | 0.0061 s      | 0.0092 s   | 0.0003 s
5000  | 0.2713 s    | 0.0993 s      | 0.0985 s   | 0.0013 s
10000 | 1.0748 s    | 0.3909 s      | 0.3734 s   | 0.0024 s
