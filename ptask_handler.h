#ifndef PTASK_HANDLER_H
#define PTASK_HANDLER_H

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

/**
* GLOBAL FUNCTIONS
*/

/**< Copy time structure. */
void time_copy(struct timespec *td, struct timespec ts);

/**< Add a fixed ms to a time struct. */
void time_add_ms(struct timespec *t, int ms);

/**< Compare 2 time structs. */
int time_cmp(struct timespec t1, struct timespec t2);

/**< Create a periodic task with the given parameters. */
int task_create(void * (*task_handler)(void *), int period, int drel, int prio);

/**< Return the index of a task. */
int get_task_index(void * arg);

/**< Compute the next activation time of the task. */
void set_activation(int id);

/**< Check if the task miss its deadline. */
int deadline_miss(int id);

/**< Suspend the task untill the next activation instant. */
void wait_for_activation(int id);

/**< Waits untill all task finish. */
void wait_tasks();

#endif
