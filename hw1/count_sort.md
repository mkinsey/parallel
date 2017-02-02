# Homework 1 

Michael Kinsey

2-7-17

## Question A

If we attempt to parallelize the outer for loop in the
provided count_sort method, we will need to share the 
variables `i` and `temp`. The variables `j` and `count` 
should be private to each thread.

## Question B 

There should be no loop-carried dependencies.

## Question C

Because the `temp` array is updated one index at a time,
and each index should only be updated once, we could call
`memcpy` for each index directly inside the outer loop. This
would parallelize the `memcpy` call, however I suspect it would
be faster to copy the whole array at once.

## Question E

...

