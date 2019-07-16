/**
* @file ptask_handler.c
* @author Gianluca D'Amico
* @brief File containing ptasks handling function
*
* HANDLING PTAKS: It manages all the periodic tasks in the application.
*
* It can creats a task, assigning them a period, a deadline and a priority. 
* It can check if each task respect its deadline and manages their activation.
*
*/

#include <pthread.h>
#include <sched.h>
#include <assert.h>
#include <time.h>

#include "ptask_handler.h"

#define MAX_TASK (10)       /**< Max number of active task. */

/**
* LOCAL STRUCTS
*/

struct task_par {
    int id;                 /**< Task id. */
    long wcet;              /**< Task worst case execution time. */
    int period;             /**< Task period in ms. */
    int deadline;           /**< Task relative deadline in ms. */
    int priority;           /**< Task priority [0,99]. */
    int dmiss;              /**< Task numbero of deadline miss. */
    struct timespec at;     /**< Task next activation time. */
    struct timespec dl;     /**< Task absolute deadline. */
};

struct task_par tp[MAX_TASK];   /**< Task array parameters. */
pthread_t tid[MAX_TASK];        /**< Thread ID. */
size_t task_counter = 0;        /**< Counter of active tasks. */

/**
* GLOBAL FUNCTIONS
*/

/**< Copy time structure. */
void time_copy(struct timespec *td, struct timespec ts) {
    td->tv_sec = ts.tv_sec;
    td->tv_nsec = ts.tv_nsec;
}

/**< Add a fixed ms to a time struct. */
void time_add_ms(struct timespec *t, int ms) {
    t->tv_sec += ms/1000;
    t->tv_nsec += (ms%1000)*1000000;

    if (t->tv_nsec > 1000000000) {
        t->tv_nsec -= 1000000000;
        t->tv_sec += 1;
    }
}

/**< Compare 2 time structs. */
int time_cmp(struct timespec t1, struct timespec t2) {
    if (t1.tv_sec > t2.tv_sec) return 1;
    if (t1.tv_sec < t2.tv_sec) return -1;
    if (t1.tv_nsec > t2.tv_nsec) return 1;
    if (t1.tv_nsec < t2.tv_nsec) return -1;
    return 0;
}

/**< Create a periodic task with the given parameters. */
int task_create(void * (*task_handler)(void *), int period, 
                                                    int drel, int prio) {
    pthread_attr_t task_attributes;
    struct sched_param task_sched_params;
    int ret;

    int id = task_counter; 
    task_counter += 1;  
    assert(id < MAX_TASK);

    tp[id].id = id;         
    tp[id].period = period;
    tp[id].deadline = drel;
    tp[id].priority = prio;
    tp[id].dmiss = 0;       

    pthread_attr_init(&task_attributes);
    pthread_attr_setinheritsched(&task_attributes, PTHREAD_EXPLICIT_SCHED);
    pthread_attr_setschedpolicy(&task_attributes, SCHED_RR);
    task_sched_params.sched_priority = tp[id].priority;
    pthread_attr_setschedparam(&task_attributes, &task_sched_params);
    
    ret = pthread_create(&tid[id], &task_attributes, task_handler, 
                                                            (void *)(&tp[id]));

    return ret;
}

/**< Return the index of a task. */
int get_task_index(void * arg) {
    struct task_par * tp;
    tp = (struct task_par *)arg;
    return tp->id;
}

/**< Compute the next activation time of the task. */
void set_activation(const int id) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    time_copy(&(tp[id].at), t);
    time_copy(&(tp[id].dl), t);
    time_add_ms(&(tp[id].at), tp[id].period);
    time_add_ms(&(tp[id].dl), tp[id].deadline);
}

/**< Check if the task miss its deadline. */
int deadline_miss(const int i) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    if (time_cmp(t, tp[i].dl) > 0) {
        tp[i].dmiss++;
        return 1;
    }
    return 0;
}

/**< Suspend the task untill the next activation instant. */
void wait_for_activation(int i) {
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &(tp[i].at), NULL);
    time_add_ms(&(tp[i].at), tp[i].period);
    time_add_ms(&(tp[i].dl), tp[i].period);
}

/**< Waits untill all task finish. */
void wait_tasks() {
    for (int i = 0; i < task_counter; ++i) {
        pthread_join(tid[i], NULL);
    }
}
