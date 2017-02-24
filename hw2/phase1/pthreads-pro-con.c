/*
* The following tutorial on PTHREADS was taken from:
* http://www.cs.nmsu.edu/~jcook/Tools/pthreads/pthreads.html
* Minor corrections/changes were made. I believe some of the text
* here may have come directly from the linux/posix documentation.
* John Kapenga 2002
*/
/*
 *      File    : pc.c
 *      Title   : Demo Producer/Consumer.
 *      Short   : A solution to the producer consumer problem using
 *        pthreads.       
 *      Author  : Andrae Muys
 *      Date    : 18 September 1997
 */

/*
 * Documentation added by Michael Kinsey, 2-23-17
 *
 * This file demonstrates the producer-consumer problem using pthreads
 */

#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#define QUEUESIZE 10
#define LOOP 20

// declare functions
void *producer (void *args);
void *consumer (void *args);

// define the structure of the shared queue
typedef struct {
    int buf[QUEUESIZE];
    long head, tail;
    int full, empty;
    pthread_mutex_t *mut;
    pthread_cond_t *notFull, *notEmpty;
} queue;

queue *queueInit (void);
void queueDelete (queue *q);
void queueAdd (queue *q, int in);
void queueDel (queue *q, int *out);

int main ()
{
    queue *fifo;
    pthread_t pro, con;

    fifo = queueInit ();
    if (fifo ==  NULL) {
        fprintf (stderr, "main: Queue Init failed.\n");
        exit (1);
    }

    // initialize a thread for both the producer and consumer
    // starts respective functions w/ arg fifo
    pthread_create (&pro, NULL, producer, fifo);
    pthread_create (&con, NULL, consumer, fifo);

    // wait for pro and con terminate
    pthread_join (pro, NULL);
    pthread_join (con, NULL);
    queueDelete (fifo);

    return 0;
}

/**
 * This is the start routine for the producer thread. It utilizes mutexes and conditional variables to produce in the
 * same queue that the consumer consumes in.
 * @param q  shared queue
 * @return  null
 */
void *producer (void *q)
{
    queue *fifo;
    int i;

    // set fifo to queue passed from void pointer
    fifo = (queue *)q;

    for (i = 0; i < LOOP; i++) {
        // lock queue for production
        pthread_mutex_lock (fifo->mut);

        // when full, unlock and wait until not full
        while (fifo->full) {
            printf ("producer: queue FULL.\n");
            pthread_cond_wait (fifo->notFull, fifo->mut);
        }

        // produce by adding variable i to queue. Unlock and notify consumer
        queueAdd (fifo, i);
        pthread_mutex_unlock (fifo->mut);
        pthread_cond_signal (fifo->notEmpty);
        usleep (100000);
    }

    for (i = 0; i < LOOP; i++) {
        // lock queue for production
        pthread_mutex_lock (fifo->mut);

        // when full, unlock and wiat until not full
        while (fifo->full) {
            printf ("producer: queue FULL.\n");
            pthread_cond_wait (fifo->notFull, fifo->mut);
        }

        // produce by adding variable i to queue. Unlock and notify consumer
        queueAdd (fifo, i);
        pthread_mutex_unlock (fifo->mut);
        pthread_cond_signal (fifo->notEmpty);

        // Note: different wait time than above loop
        usleep (200000);
    }
    return (NULL);
}

/**
 * This is the start routine for the consumer thread. It utilizes mutexes and conditional variables to consume
 * from the same queue that the producer uses to produce.
 * @param q shared queue
 * @return  null
 */
void *consumer (void *q)
{
    queue *fifo;
    int i, d;

    // set fifo to queue passed from void pointer
    fifo = (queue *)q;

    for (i = 0; i < LOOP; i++) {
        // lock queue for consumption
        pthread_mutex_lock (fifo->mut);

        // if empty, unlock and wait until not empty
        while (fifo->empty) {
            printf ("consumer: queue EMPTY.\n");
            pthread_cond_wait (fifo->notEmpty, fifo->mut);
        }

        // 'consume' d by removing it from queue. Unlock and then notify producer
        queueDel (fifo, &d);
        pthread_mutex_unlock (fifo->mut);
        pthread_cond_signal (fifo->notFull);
        printf ("consumer: recieved %d.\n", d);
        usleep(200000);
    }
    for (i = 0; i < LOOP; i++) {
        // lock queue for consumption
        pthread_mutex_lock (fifo->mut);

        // if empty, unlock and wait until no empty
        while (fifo->empty) {
            printf ("consumer: queue EMPTY.\n");
            pthread_cond_wait (fifo->notEmpty, fifo->mut);
        }

        // 'consume' d by removing it from queue. Unlock then notify producer
        queueDel (fifo, &d);
        pthread_mutex_unlock (fifo->mut);
        pthread_cond_signal (fifo->notFull);
        printf ("consumer: recieved %d.\n", d);

        // Note: different wait time than above loop
        usleep (50000);
    }
    return (NULL);
}

/**
 * Initialize a queue. Init 'tracker' variables and conditional variables
 * @return the queue
 */
queue *queueInit (void)
{
    queue *q;

    q = (queue *)malloc (sizeof (queue));
    if (q == NULL) return (NULL);

    q->empty = 1;
    q->full = 0;
    q->head = 0;
    q->tail = 0;
    q->mut = (pthread_mutex_t *) malloc (sizeof (pthread_mutex_t));
    pthread_mutex_init (q->mut, NULL);
    q->notFull = (pthread_cond_t *) malloc (sizeof (pthread_cond_t));
    pthread_cond_init (q->notFull, NULL);
    q->notEmpty = (pthread_cond_t *) malloc (sizeof (pthread_cond_t));
    pthread_cond_init (q->notEmpty, NULL);

    return (q);
}

/**
 * Deallocate memory used to track queue. Destroy conditional variables
 * @param q queue to be deleted
 */
void queueDelete (queue *q)
{
    pthread_mutex_destroy (q->mut);
    free (q->mut);
    pthread_cond_destroy (q->notFull);
    free (q->notFull);
    pthread_cond_destroy (q->notEmpty);
    free (q->notEmpty);
    free (q);
}

/**
 * Add item to queue. Update 'tracker' variables
 * @param q pointer to queue
 * @param in  int to be added to queue
 */
void queueAdd (queue *q, int in)
{
    q->buf[q->tail] = in;
    q->tail++;
    if (q->tail == QUEUESIZE)
        q->tail = 0;
    if (q->tail == q->head)
        q->full = 1;
    q->empty = 0;

    return;
}

/**
 * Remove item from queue. Update 'tracker variables
 * @param q pointer to queue
 * @param out pointer to variable to be removed
 */
void queueDel (queue *q, int *out)
{
    *out = q->buf[q->head];

    q->head++;
    if (q->head == QUEUESIZE)
        q->head = 0;
    if (q->head == q->tail)
        q->empty = 1;
    q->full = 0;

    return;
}
