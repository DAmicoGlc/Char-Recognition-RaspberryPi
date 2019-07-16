#ifndef NN_HANDLER_H
#define NN_HANDLER_H

/**
* @file nn_handler.h
* @author Gianluca D'Amico
* @brief File containing neural network handling functions
*
* HANDLING MODEL FUNCTIONS: It manages all the functions needed to load the
* model weights of the neural network and to utilize the network.
*
* At the start of the application, 3 different predefined models will be 
* loaded, one for the recognition of digits, one for the recognition of 
* letters, one for the recognition of both. Each model corresponds to different
* sinapsi weights trained offline and saved in a txt file. This file include 
* also the function needed to utilize the neural network in the application, 
* taking the preprocessed images captured by the camera it will feed the active
* network and compute the result of the recognition.
*
* @note Only one model is active at time.
*
*/

#include "common.h"

/**
* RETURN CONSTANT
*/

#define NN_SUCCESS              0
#define NN_ERROR_NO_FILE        1
#define NN_ERROR_READING_FILE   2

/**
* GLOBAL DATA
*/

/**< Struct that identify actual input and output.*/
typedef struct {
    char rec_char;          /**< Recognized character.*/
    float prob;             /**< Recognition accurancy.*/
} data_network_t;

/**< Requested active model. */
extern network_target requested_model;
/**< Result of the computation. */
extern data_network_t nn_result;

/**
* GLOBAL MUTEX
*/
extern pthread_mutex_t actual_model_mutex;          /**< Active model.*/

/**
* GLOBAL FUNCTION PROTOTYPES
*/

/**< Initialize all the 3 differet model and load the corresponding weights. */
int init_networks();

/**< Compute the output of the active neural network.*/
void recognize_character(BITMAP* input_image);

#endif