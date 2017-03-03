# Makefile 
# cs454 Fall 2010
# kapenga
CC = gcc
CFLAGS = -Wall -pedantic -std=gnu99
SRC = matmul_stack.c matmul_static.c matmul_heap.c
EXE = matmul_stack matmul_static matmul_heap
OTHER = Makefile 

all:  matmul_static matmul_omp matmul_pt
matmul_static: matmul_static.c
	${CC} ${CFLAGS} -o matmul_static matmul_static.c -lm
matmul_omp: matmul_static_openmp.c
	${CC} ${CFLAGS} -o matmul_omp matmul_static_openmp.c -lm -fopenmp
matmul_pt: pt_matmul_static.c
	${CC} ${CFLAGS} -o pt_matmul pt_matmul_static.c -lpthread
test: all
	./pt_matmul
	./matmul_static
	diff static-out-pthreads.txt static-out-original.txt
tar:
	tar -cvf matmul.tar ${SRC} ${OTHER}

clean:
	rm -f core a.out *.txt ${EXE} matmul.tar
