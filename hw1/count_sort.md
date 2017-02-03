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


