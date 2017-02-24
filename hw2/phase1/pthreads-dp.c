/*
* The following tutorial on PTHREADS was taken from:
* http://www.cs.nmsu.edu/~jcook/Tools/pthreads/pthreads.html
* Minor corrections/changes were made. I believe some of the text
* here may have come directly from the linux/posix documentation.
* John Kapenga 2002
*/
/*
 *      File    : dining.c
 *      Title   : Dining Philosophers.
 *      Short   : Solution to the dining philosophers problem.
 *      Long    : Sets up a fifo fifo of threads, and activates threads
 *          as forks become available.
 *      Author  : Andrae Muys
 *      Date    : 18 September 1997
 *
 */

/*
 *      Documentation added by Michael Kinsey 02/21/2017
 *
 *      Dining Philosophers - This file presents a solution to the dining
 *      philosophers problem by implementing a central authority via a mutex.
 *      Philosophers must first request a fork, but can release a fork at any
 *      time.
 */
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include <sys/errno.h>

#define TABLE 5
#define DELAY ((rand()%5+1)*1000)
#define DISHES 4

// define node for linked list queue
typedef struct node {
    struct node *next;
    pthread_cond_t sleep;
} node;

// define fifo structure for queue
typedef struct {
    node *head, *tail;
    pthread_mutex_t *mut;
    int runmax;
    int size;
} fifo;

// define functions
fifo *fifoInit (int runmax);
void fifoDelete (fifo *q);
void fifoLock (fifo *q, pthread_mutex_t *mut, int id);
void fifoUnlock (fifo *q, int id);

void *philosopher (void *id);

// define global scope variables
pthread_mutex_t forks[TABLE]; // create an array of mutex variables, these represent the forks
pthread_cond_t newfork; // declare conditional variable
fifo *queue; // declare queue

int main (int argc, char **argv)
{
    pthread_t p[TABLE];
    int i;

    if (argc<2) return 1;  // require program input
    srand(atoi(argv[1]));  // use input as seed for srand()

    queue = fifoInit (TABLE / 2);
    printf ("Begin %d-diners\n", TABLE );
    pthread_cond_init (&newfork, NULL);

    // set up mutex for each fork
    for (i = 0; i < TABLE; i++)
        pthread_mutex_init (&(forks[i]), NULL);

    // initialize each thread and call philosopher function
    for (i = 0; i < TABLE; i++)
        pthread_create (&p[i], NULL, philosopher, &i);

    // wait for each thread to finish in sequence
    for (i = 0; i < TABLE; i++)
        pthread_join (p[i], NULL);

    // end
    printf ("End %d-diners\n",TABLE/2);
    fifoDelete (queue);
    return 0;
}

/**
 * function for each philosopher thread.
 * @param num
 * @return
 */
void *philosopher (void *num)
{
    int id;
    int i;
    pthread_mutex_t dummy = PTHREAD_MUTEX_INITIALIZER;

    id = *(int*)num;
    pthread_mutex_lock (&dummy);
    printf ("P-%d sit\n", id);

    // TODO
    for (i = 0; i < 2; i++) {
        fifoLock (queue, &dummy, id);
        usleep (DELAY);
        fifoUnlock (queue, id);
    }

    //
    for (i = 0; i < DISHES; i++) {
        printf ("P-%d Start\n", id);
        while (1) {

            fifoLock (queue, &dummy, id);
            if (pthread_mutex_trylock (&(forks[(id+1)%TABLE])) == EBUSY) {
                fifoUnlock (queue, id);
                continue;
            }
            printf ("P-%d gotfork%d\n", id,
                    (id+1)%TABLE);
            usleep(DELAY);
            if (pthread_mutex_trylock (&(forks[id])) == EBUSY) {
                //printf ("P-%d reqfork%d\n"
                //    , id, id);
                pthread_mutex_unlock (&(forks[(id+1)%TABLE]));
                printf ("P-%d dropfork%d\n"
                        , id, (id+1)%TABLE);
                fifoUnlock (queue, id);
                continue;
            }
            printf ("P-%d gotfork%d\n", id, id);
            printf ("P-%d eating\n", id);
            usleep (DELAY * 3);
            pthread_mutex_unlock (&(forks[id]));
            pthread_mutex_unlock (&(forks[(id+1)%TABLE]));
            printf ("P-%d retforks%d-%d\n", id, id, (id+1)%TABLE);
            fifoUnlock (queue, id);
            break;
        }
        printf ("P-%d finished\n", id);
    }
    printf ("P-%d finishedMEAL\n", id);
    pthread_mutex_unlock (&dummy);
    pthread_mutex_destroy (&dummy);

    return (0); // Modified NULL
}

/**
 *  Initialize queue
 * @param runmax
 * @return new queue
 */
fifo *fifoInit (int runmax)
{
    fifo *q;

    // allocate memory and test that malloc was successful
    q = (fifo *)malloc (sizeof (fifo));
    if (q == NULL) return (NULL);

    // allocate memory for queue mutex and test that malloc was successful
    q->mut = (pthread_mutex_t *)malloc (sizeof (pthread_mutex_t));
    if (q->mut == NULL) { free (q); return (NULL); }

    // initialize mutex with default settings
    pthread_mutex_init (q->mut, NULL);

    // init queue variables
    q->runmax = runmax;
    q->head = NULL;
    q->tail = NULL;
    q->size = 0;

    return (q);
}

void fifoDelete (fifo *q)
{
    if (q->head != NULL) {
        //printf ("fifoDelete: Things that make you say mmmmm.\n");
        exit (1);
    }
    pthread_mutex_destroy (q->mut);
    free (q->mut);
    free (q);

    return;
}

/**
 * Insert new node into queue. Use mutexes to prevent race conditions
 * @param q queue
 * @param mut mutex to lock
 * @param id - thread id, used for debugging
 */
void fifoLock (fifo *q, pthread_mutex_t *mut, int id)
{
    node *new;

    pthread_mutex_lock (q->mut);
    q->size++;
    //fprintf (stderr, "Lock %d : size = %d\n", id, q->size);
    if (q->size > q->runmax) {
        new = (node *)malloc (sizeof (node));
        if (new == NULL) {
            //printf ("fifoLock: malloc failed.\n");
            exit (1);
        }

        // init conditional and vars for new node
        pthread_cond_init (&(new->sleep), NULL);
        new->next = NULL;
        if (q->tail == NULL) {
            q->head = q->tail = new;
        } else {
            q->tail->next = new;
            q->tail = new;
        }

        pthread_mutex_unlock (q->mut);
        pthread_cond_wait (&(new->sleep), mut);
    } else {
        pthread_mutex_unlock (q->mut);
    }

    return;
}

/**
 * Remove first element from queue. Use a mutex lock to prevent race conditions
 * @param q  - queue to unlock
 * @param id  - thread id, used for debugging
 */
void fifoUnlock (fifo *q, int id)
{
    node *old;

    // lock the queue
    pthread_mutex_lock (q->mut);
    q->size--;
    //fprintf (stderr, "Unlock : %d size = %d\n", id, q->size);
    if (q->head != NULL) {

        // set head pointer to next in linked list
        old = q->head;
        q->head = old->next;

        if (q->head == NULL) {  // list is now empty
            q->tail = NULL;
        }
        old->next = NULL;
        //fprintf (stderr, "%d Waking head.\n", id);
        pthread_cond_signal (&(old->sleep));
    }

    // unlock queue to make it available again
    pthread_mutex_unlock (q->mut);

    return;
}
