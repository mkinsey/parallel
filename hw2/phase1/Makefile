# Settings for Solaris 9.0
# CC= cc
# CFLAGS=  -mt
# LIBS= -lpthread

# Settings for  Linux
CC= gcc
CFLAGS=  -Wall -pedantic -std=gnu99
LIBS= -lpthread

all: pthreads-pro-con pthreads-read-write pthreads-dp
pthreads-pro-con: pthreads-pro-con.c
	${CC} ${CFLAGS} -o pthreads-pro-con pthreads-pro-con.c ${LIBS}
pthreads-read-write: pthreads-read-write.c
	${CC} ${CFLAGS} -o pthreads-read-write pthreads-read-write.c ${LIBS}
pthreads-dp: pthreads-dp.c
	${CC} ${CFLAGS} -o pthreads-dp pthreads-dp.c ${LIBS}
#man:
#	# pthreads.txt
#tutorial:
#	# pthreads-tutorial
#clean:
#	rm -rf core a.out *.o pthreads-pro-con pthreads-read-write pthreads-dp
