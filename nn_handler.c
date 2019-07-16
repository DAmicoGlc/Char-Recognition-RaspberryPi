#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <allegro.h>
#include <math.h>
#include <pthread.h>

#include "nn_handler.h"

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

/**
* LOCAL DATA
*/

/**< Neural network model size. */
#define INPUT_SIZE INPUT_DIM * INPUT_DIM     /**< Input layer size. */

#define DIGIT_HID_SIZE_1    64      /**< First hidden size of digits. */
#define DIGIT_HID_SIZE_2    32      /**< Second hidden size of digits. */
#define DIGIT_OUTPUT_SIZE   10      /**< Output layer size of digits. */
#define LET_HID_SIZE        128     /**< Hidden layer size of letters. */
#define LETTER_OUTPUT_SIZE  26      /**< Output layer size of letters. */
#define MIX_HID_SIZE        512     /**< Hidden layer size of mixed. */
#define MIXED_OUTPUT_SIZE   47      /**< Output layer size of mixed. */

#define HID_DIGITS   2              /**< Number of hidden layers of digits. */
#define HID_LET_MIX  3              /**< Number of hidden layers of letters and 
                                                                /**< mixed. */

#define MAX_HID_SIZE 512            /**< Max hidden layer size of all models. */
#define MAX_OUT_SIZE 47             /**< Max output layer size of all models. */
#define MAX_HID_NUM  3              /**< Max number of hidden layers. */

#define ERROR -1        /**< Error returning value. */
#define SUCCESS 1       /**< Success returning value. */

/**< Filename of digits model weights. */
static const char digits_filename[30]   = "digits_2_64_32.txt";

/**< Filename of letters model weights. */
static const char letters_filename[30]  = "letters_3_128_128_128.txt";

/**< Filename of mixed model weights. */
static const char mixed_filename[30]    = "mixed_3_512_512_512.txt";

/**
* LOCAL STRUTCS
*/

/**
* NEURAL NETWORK SINAPSI 
*/

/**<Input sinapsi, the size of the Sinapsi is taken as the max of all sizes. */
typedef  struct {
    float weights[MAX_HID_SIZE][INPUT_SIZE];  /**< Weights of the connection.*/
    float bias[MAX_HID_SIZE];                 /**< Bias of the connection. */

    int card_in;   /**< Number of incoming neurons connetcted to the sinapsi.*/
    int card_out;  /**< Number of outgoing neurons connetcted to the sinapsi.*/
} sinapsi_in_t;

// Hidden sinapsi, the size of the Sinapsi is taken as the maximum of all sizes
typedef struct {
    float weights[MAX_HID_SIZE][MAX_HID_SIZE];/**< Weights of the connection.*/
    float bias[MAX_HID_SIZE];                 /**< Bias of the connection. */

    int card_in;    /**< Number of incoming neurons connetcted to the sinapsi.*/
    int card_out;   /**< Number of outgoing neurons connetcted to the sinapsi.*/
} sinapsi_hid_t;

/**< Output sinapsi, the size of the Sinapsi is taken as the max of all sizes.*/
typedef struct {
    float weights[MAX_OUT_SIZE][MAX_HID_SIZE];/**< Weights of the connection.*/
    float bias[MAX_OUT_SIZE];                 /**< Bias of the connection. */

    int card_in;   /**< Number of incoming neurons connetcted to the sinapsi.*/
    int card_out;  /**< Number of outgoing neurons connetcted to the sinapsi.*/
} sinapsi_out_t;

/**
* NEURAL NETWORK LAYER 
*/

/**< Input layer, the size of the Layer is taken as the maximum of all sizes.*/
typedef struct {
    float z_value[INPUT_SIZE];          /**< Weighted input of each neuron.*/
    float act_value[INPUT_SIZE];        /**< Activation value of each neruon.*/

    int num_neuron;     /**< Number of neurons of the layer.*/
} layer_in_t;

/**< Hidden layer, the size of the Layer is taken as the maximum of all sizes.*/
typedef struct {
    float z_value[MAX_HID_SIZE];        /**< Weighted input of each neuron.*/
    float act_value[MAX_HID_SIZE];      /**< Activation value of each neruon.*/

    int num_neuron;     /**< Number of neurons of the layer.*/
} layer_hid_t;

/**< Output layer, the size of the Layer is taken as the maximum of all sizes.*/
typedef struct {
    float z_value[MAX_OUT_SIZE];        /**< Weighted input of each neuron.*/
    float act_value[MAX_OUT_SIZE];      /**< Activation value of each neruon.*/

    int num_neuron;     /**< Number of neurons of the layer.*/
} layer_out_t;

/**
* NEURAL NETWORK STRUCT
*/

/**< Struct of each neural network model.*/
typedef struct {
    /**< Sinapsi from first layer to first hidden one.*/
    sinapsi_in_t in_S;
    /**< Sinapsi between consecutive hidden layers.*/
    sinapsi_hid_t hid_S[MAX_HID_NUM-1];
    /**< Sinapsi from last hidden layer to output one.*/
    sinapsi_out_t out_S;

    layer_in_t  in_L;                   /**< Input layer.*/
    layer_hid_t hid_L[MAX_HID_NUM];     /**< Hidden layers.*/
    layer_out_t out_L;                  /**< Output layer.*/

    int num_hidden;     /**< Number of hidden layers of the model.*/
} network_t;

/**< Model container. */
static network_t neural_network[3]; 

/**< Actual active model. */
static network_target active_net;

/**< Mapping between output neuron of the network and character. */
static const char digits_map[DIGIT_OUTPUT_SIZE] = 
                        { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};

static const char letters_map[LETTER_OUTPUT_SIZE] = 
                            { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                              'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                              'U', 'V', 'W', 'X', 'Y', 'Z'};

static const char mixed_map[MIXED_OUTPUT_SIZE] =  
                        { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                          'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e', 
                          'f', 'g', 'h', 'n', 'q', 'r', 't'};

/**
* GLOBAL DATA
*/

network_target requested_model;      /**< Requested active model. */
data_network_t nn_result;            /**< Result of the computation. */

/**
* GLOBAL MUTEX
*/

pthread_mutex_t actual_model_mutex;     /**< Active model.*/

/**
* LOCAL FUNCTION DEFINITION
*/

/**
* @brief Initialization of all structs of digits model.
*
* This fuction assign every size to each struct of the digits model.
*
*/
static void init_digits_net() {
    neural_network[DIGITS].num_hidden   = HID_DIGITS;
    neural_network[LETTERS].num_hidden  = HID_LET_MIX;
    neural_network[MIXED].num_hidden    = HID_LET_MIX;

    neural_network[DIGITS].in_S.card_in     = INPUT_SIZE;
    neural_network[DIGITS].hid_S[0].card_in = DIGIT_HID_SIZE_1;
    neural_network[DIGITS].out_S.card_in    = DIGIT_HID_SIZE_2;

    neural_network[DIGITS].in_S.card_out        = DIGIT_HID_SIZE_1;
    neural_network[DIGITS].hid_S[0].card_out    = DIGIT_HID_SIZE_2;
    neural_network[DIGITS].out_S.card_out       = DIGIT_OUTPUT_SIZE;

    neural_network[DIGITS].in_L.num_neuron      = INPUT_SIZE;
    neural_network[DIGITS].hid_L[0].num_neuron  = DIGIT_HID_SIZE_1;
    neural_network[DIGITS].hid_L[1].num_neuron  = DIGIT_HID_SIZE_2;
    neural_network[DIGITS].out_L.num_neuron     = DIGIT_OUTPUT_SIZE;
}

/**
* @brief Initialization of all structs of letters model.
*
* This fuction assign every size to each struct of the letters model.
*
*/
static void init_letters_net() {
    neural_network[LETTERS].in_S.card_in     = INPUT_SIZE;
    neural_network[LETTERS].hid_S[0].card_in = LET_HID_SIZE;
    neural_network[LETTERS].hid_S[1].card_in = LET_HID_SIZE;
    neural_network[LETTERS].out_S.card_in    = LET_HID_SIZE;

    neural_network[LETTERS].in_S.card_out        = LET_HID_SIZE;
    neural_network[LETTERS].hid_S[0].card_out    = LET_HID_SIZE;
    neural_network[LETTERS].hid_S[1].card_out    = LET_HID_SIZE;
    neural_network[LETTERS].out_S.card_out       = LETTER_OUTPUT_SIZE;

    neural_network[LETTERS].in_L.num_neuron      = INPUT_SIZE;
    neural_network[LETTERS].hid_L[0].num_neuron  = LET_HID_SIZE;
    neural_network[LETTERS].hid_L[1].num_neuron  = LET_HID_SIZE;
    neural_network[LETTERS].hid_L[2].num_neuron  = LET_HID_SIZE;
    neural_network[LETTERS].out_L.num_neuron     = LETTER_OUTPUT_SIZE;
}

/**
* @brief Initialization of all structs of mixed model.
*
* This fuction assign every size to each struct of the mixed model.
*
*/
static void init_mixed_net() {
    neural_network[MIXED].in_S.card_in     = INPUT_SIZE;
    neural_network[MIXED].hid_S[0].card_in = MIX_HID_SIZE;
    neural_network[MIXED].hid_S[1].card_in = MIX_HID_SIZE;
    neural_network[MIXED].out_S.card_in    = MIX_HID_SIZE;

    neural_network[MIXED].in_S.card_out        = MIX_HID_SIZE;
    neural_network[MIXED].hid_S[0].card_out    = MIX_HID_SIZE;
    neural_network[MIXED].hid_S[1].card_out    = MIX_HID_SIZE;
    neural_network[MIXED].out_S.card_out       = MIXED_OUTPUT_SIZE;

    neural_network[MIXED].in_L.num_neuron      = INPUT_SIZE;
    neural_network[MIXED].hid_L[0].num_neuron  = MIX_HID_SIZE;
    neural_network[MIXED].hid_L[1].num_neuron  = MIX_HID_SIZE;
    neural_network[MIXED].hid_L[2].num_neuron  = MIX_HID_SIZE;
    neural_network[MIXED].out_L.num_neuron     = MIXED_OUTPUT_SIZE;
}

/**
* @brief Loading of weights and bias of input sinapsi.
*
* The file from which the weigths are loaded must have a specific pattern.
* Each weights to the same outgoing neuron are separeted by a '_' .After them 
* ther is the value of the bias between two '\n', then other weights follow in 
* the same pattern. At the end of the file there is the accurancy of the model
* and the relative accurancy for each symbol. If the pattern is not followed, 
* the function return an ERROR code.
*
* @param  fp is the file descriptor whic contain the weights
* @param  target specificy which model must be loaded {DIGITS, LETTERS, MIXED}
* @return an int to notify if the loading is done correctly or not
*/
static int load_input_sinapsi(FILE* fp, network_target target) {

    int i, j;
    int char_count = 0;     /**< Counter for the characters read. */
    char ch;                /**< Character read. */
    char string[10] = "";   /**< String to contain the characters read. */

    for (i = 0; i < neural_network[target].in_S.card_out; ++i) {

        for (j = 0; j < neural_network[target].in_S.card_in; ++j) {

            ch = fgetc(fp);

            while(ch != '_' && ch != '\n') {
                string[char_count] = ch;
                char_count++;
                if (char_count > 10)
                    return ERROR;
                ch = fgetc(fp);
            }

            string[char_count] = '\0';
            neural_network[target].in_S.weights[i][j] = atof(string);

            char_count = 0;
        }

        while((ch = fgetc(fp)) != '\n') {
            string[char_count] = ch;
            char_count++;
            if (char_count > 10)
                return ERROR;
        }  

        string[char_count] = '\0';
        neural_network[target].in_S.bias[i] = atof(string);

        char_count = 0;
    }

    return SUCCESS;
}

/**
* @brief Loading of weights and bias of hiddens sinapsi.
*
* The file from which the weigths are loaded must have a specific pattern.
* Each weights to the same outgoing neuron are separeted by a '_' .After them 
* ther is the value of the bias between two '\n', then other weights follow in 
* the same pattern. At the end of the file there is the accurancy of the model
* and the relative accurancy for each symbol. If the pattern is not followed, 
* the function return an ERROR code.
*
* @param  fp is the file descriptor whic contain the weights
* @param  target specificy which model must be loaded {DIGITS, LETTERS, MIXED}
* @return an int to notify if the loading is done correctly or not
*/
static int load_hidden_sinapsi(FILE* fp, network_target target) {

    int i, j, k;
    int char_count = 0;     /**< Counter for the characters read. */
    char ch;                /**< Character read. */
    char string[10];        /**< String to contain the characters read. */

    for (k = 0; k < neural_network[target].num_hidden-1; ++k) {

        for (i = 0; i < neural_network[target].hid_S[k].card_out; ++i) {

            for (j = 0; j < neural_network[target].hid_S[k].card_in; ++j) {

                ch = fgetc(fp);

                while(ch != '_' && ch != '\n') {
                    string[char_count] = ch;
                    char_count++;
                    ch = fgetc(fp);
                    if (char_count > 10)
                        return ERROR;
                }

                string[char_count] = '\0';
                neural_network[target].hid_S[k].weights[i][j] = atof(string);

                char_count = 0;
            }

            while((ch = fgetc(fp)) != '\n') {
                string[char_count] = ch;
                char_count++;
                if (char_count > 10)
                    return ERROR;
            }  

            string[char_count] = '\0';
            neural_network[target].hid_S[k].bias[i] = atof(string);

            char_count = 0;
        }
    }

    return SUCCESS;
}

/**
* @brief Loading of weights and bias of output sinapsi.
*
* The file from which the weigths are loaded must have a specific pattern.
* Each weights to the same outgoing neuron are separeted by a '_' .After them 
* ther is the value of the bias between two '\n', then other weights follow in 
* the same pattern. At the end of the file there is the accurancy of the model
* and the relative accurancy for each symbol. If the pattern is not followed, 
* the function return an ERROR code.
*
* @param  fp is the file descriptor whic contain the weights
* @param  target specificy which model must be loaded {DIGITS, LETTERS, MIXED}
* @return an int to notify if the loading is done correctly or not
*/
static int load_out_sinapsi(FILE* fp, network_target target) {

    int i, j;
    int char_count = 0;     /**< Counter for the characters read. */
    char ch;                /**< Character read. */
    char string[10];        /**< String to contain the characters read. */

    for (i = 0; i < neural_network[target].out_S.card_out; ++i) {

        for (j = 0; j < neural_network[target].out_S.card_in; ++j) {

            ch = fgetc(fp);

            while(ch != '_' && ch != '\n') {
                string[char_count] = ch;
                char_count++;
                if (char_count > 10)
                    return ERROR;
                ch = fgetc(fp);
            }

            string[char_count] = '\0';
            neural_network[target].out_S.weights[i][j] = atof(string);

            char_count = 0;
        }

        while((ch = fgetc(fp)) != '\n') {
            string[char_count] = ch;
            char_count++;
            if (char_count > 10)
                return ERROR;
        } 

        string[char_count] = '\0';
        neural_network[target].out_S.bias[i] = atof(string);

        char_count = 0;
    }

    return SUCCESS;
}

/**
* @brief Hidden Activation function of the neural network.
*
* It computes the logistic function of the input value.
*
* @param  z input value for the logistic function computation
* @return logistic function of the value in input
*/
static float logistic_function(float z) {
    return 1 / (1 + exp(-z));
}

/**
* @brief Output Activation function of the neural network.
*
* It computes the softmax function of the input value, it is used only in the
* output layer of the neural network.
*
* @param  z input value for the softmax function computation
* @param  sum of all exponatial of ouput z_value
* @param  max of all exponatial of ouput z_value, is neede to avoid the 
*         explosion of the exp
* @return softmax function of the value in input
*/
static float softmax(float z, float sum, float max) {
    return (exp(z + max) / (sum) );
}


/**
* @brief Feed forward result from input layer to first hidden one.
*
* For each neuron in the first hidden layer, the fuction computes the weighted
* sum of the incoming neurons based on the connection value plus the related
* bias and assign it to the z_value of the relative outgoing neuron. Then it 
* compute the logistic function (activation function) of the z_value and assign 
* it to the act_value of the relative outgoing neuron.
*/
static void propagate_from_in_layer() {

    int i, j, k;
    float sum_up;       /**< Weighted sum of all incoming neurons. */

    for (i = 0; i < neural_network[active_net].in_S.card_out; ++i) {

        sum_up=0;
        
        /**< The sum_up is computed as:. */
        /**< sum_up = [sum of ( weight * activation value )] + bias. */
        for (j = 0; j < neural_network[active_net].in_S.card_in; ++j) 
            sum_up += neural_network[active_net].in_S.weights[i][j] * 
                                neural_network[active_net].in_L.act_value[j];    
        
        sum_up += neural_network[active_net].in_S.bias[i];

        neural_network[active_net].hid_L[0].z_value[i] = sum_up;
        neural_network[active_net].hid_L[0].act_value[i] = 
                                                    logistic_function(sum_up);
    }
    
    return;
}

/**
* @brief Feed forward result from hidden layers.
*
* If there are more than 1 hidden layer, for each of (except for the last one) 
* them the fuction computes the weighted sum of the incoming neurons based on 
* the connection value plus the related bias and assign it to the z_value of 
* the relative outgoing neuron. Then it compute the logistic function 
* (activation function) of the z_value and assign it to the act_value of the 
* relative outgoing neuron.
*/
static void propagate_into_hid_layer() {

    int i, j, k;
    float sum_up;       /**< Weighted sum of all incoming neurons. */

    for (k = 0; k < neural_network[active_net].num_hidden-1; ++k) {

        /**< The sum_up is computed as:. */
        /**< sum_up = [sum of ( weight * activation value )] + bias. */
        for (i = 0; i < neural_network[active_net].hid_S[k].card_out; ++i) {

            sum_up=0;
            
            for (j = 0; j < neural_network[active_net].hid_S[k].card_in; ++j) 
                sum_up += neural_network[active_net].hid_S[k].weights[i][j] * 
                            neural_network[active_net].hid_L[k].act_value[j];    
            
            sum_up += neural_network[active_net].hid_S[k].bias[i];

            neural_network[active_net].hid_L[k+1].z_value[i] = sum_up;
            neural_network[active_net].hid_L[k+1].act_value[i] = 
                                                    logistic_function(sum_up);
        }
    }

    return;
}

/**
* @brief Feed forward result from the last hidden layer to the output one.
*
* For each neuron in the output layer, the fuction computes the weighted
* sum of the incoming neurons based on the connection value plus the related
* bias and assign it to the z_value of the relative outgoing neuron. Then it 
* compute the logistic function (activation function) of the z_value and assign 
* it to the act_value of the relative outgoing neuron using the softmax 
* function.
*/
static void propagate_to_out_layer() {

    int i, j, k;
    float sum_up;               /**< Weighted sum of all incoming neurons. */
    float sum_softmax = 0;      /**< Sum of exp( z_value + max(z_value) ). */
    float max_softmax = 0;      /**< Max of all z_value. */

    int hid_num = neural_network[active_net].num_hidden;

    for (i = 0; i < neural_network[active_net].out_S.card_out; ++i) {

        sum_up=0;

        /**< The sum_up is computed as:. */
        /**< sum_up = [sum of ( weight * activation value )] + bias. */
        for (j = 0; j < neural_network[active_net].out_S.card_in; ++j) 
            sum_up += neural_network[active_net].out_S.weights[i][j] * 
                    neural_network[active_net].hid_L[hid_num-1].act_value[j];    
        
        sum_up += neural_network[active_net].out_S.bias[i];

        neural_network[active_net].out_L.z_value[i] = sum_up;
        if (max_softmax < sum_up) {
            max_softmax = sum_up;
        }

    }

    for (i = 0; i < neural_network[active_net].out_S.card_out; ++i) {
        sum_softmax += exp(neural_network[active_net].out_L.z_value[i] +
                                                                max_softmax);
    }

    for (i = 0; i < neural_network[active_net].out_S.card_out; ++i) {
        neural_network[active_net].out_L.act_value[i] = softmax(
                                neural_network[active_net].out_L.z_value[i], 
                                sum_softmax, 
                                max_softmax);
    }

    return;
}

/**
* GLOBAL FUNCTIONS
*/

/**
* @brief Initialize all 3 models.
*
* Using the function defined before, this function load all the weights 
* and bias of all models after initialize the structs of all of them. 
* It also active the DIGITS one.
*
* @param  change specificy which model must be active {DIGITS, LETTERS, MIXED}
*/
int init_networks() {

    /**< File descriptor. */
    FILE *fp;
    int result;

    /**< Initilize all 3 different model structures. */
    init_digits_net();
    init_letters_net();
    init_mixed_net();

    fp = fopen(digits_filename, "r");

    if (fp == NULL)
        return NN_ERROR_NO_FILE;

    /**< Load weights and bias of the DIGITS model. */
    result = load_input_sinapsi(fp, DIGITS);
    if (result == ERROR)
        return NN_ERROR_READING_FILE;

    result = load_hidden_sinapsi(fp, DIGITS);
    if (result == ERROR)
        return NN_ERROR_READING_FILE;

    result = load_out_sinapsi(fp, DIGITS);
    if (result == ERROR)
        return NN_ERROR_READING_FILE;

    fclose(fp);

    fp = fopen(letters_filename, "r");

    /**< Load weights and bias of the LETTERS model. */
    result = load_input_sinapsi(fp, LETTERS);
    if (result == ERROR)
        return NN_ERROR_READING_FILE;
    result = load_hidden_sinapsi(fp, LETTERS);
    if (result == ERROR)
        return NN_ERROR_READING_FILE;
    result = load_out_sinapsi(fp, LETTERS);
    if (result == ERROR)
        return NN_ERROR_READING_FILE;

    fclose(fp);

    fp = fopen(mixed_filename, "r");

    /**< Load weights and bias of the MIXED model. */
    result = load_input_sinapsi(fp, MIXED);
    if (result == ERROR)
        return NN_ERROR_READING_FILE;
    result = load_hidden_sinapsi(fp, MIXED);
    if (result == ERROR)
        return NN_ERROR_READING_FILE;
    result = load_out_sinapsi(fp, MIXED);
    if (result == ERROR)
        return NN_ERROR_READING_FILE;

    fclose(fp);

    active_net = DIGITS;

    pthread_mutex_init(&actual_model_mutex, NULL);

    return SUCCESS;
};

/**
* @brief Compute the output of the active neural network.
*
* Utilizing the fuction defined before, this function compute the output of
* the neural network corresponding to the input image. It will use
* the requested network model (protected by a mutex), feeding the image to its 
* input layer and fowarding it until the output one. The result is passed 
* filling a specific global struct (protected by a mutex) containg the 
* character recognized by the network with the relative percentage.
*
*/

void recognize_character(BITMAP* image) {

    int i, j;           /**< Loop counter. */

    float max_prob = 0;         /**< Max value of the resulting prob. */
    int max_prob_index = -1;    /**< Neuron with the max prob. */

    /**< Read the requested active model. */
    pthread_mutex_lock(&actual_model_mutex);
    active_net = requested_model;
    pthread_mutex_unlock(&actual_model_mutex);

    /**< Fill the input layer of the active model. */
    for (i = 0; i < INPUT_DIM; ++i) {
        for (j = 0; j < INPUT_DIM; ++j) {            
            if (getpixel(image, i, j) == BLACK) {
                neural_network[active_net].in_L.z_value[i*INPUT_DIM + j]   = 1;
                neural_network[active_net].in_L.act_value[i*INPUT_DIM + j] = 1;
            }
            else {
                neural_network[active_net].in_L.z_value[i*INPUT_DIM + j]   = 0;
                neural_network[active_net].in_L.act_value[i*INPUT_DIM + j] = 0;
            }
            
            
        }
    }

    propagate_from_in_layer();

    propagate_into_hid_layer();

    propagate_to_out_layer();

    /**< Search the max probability among all output neuron. */
    for (i = 0; i <  neural_network[active_net].out_L.num_neuron; ++i) {
        if (max_prob < neural_network[active_net].out_L.act_value[i]) {
            max_prob = neural_network[active_net].out_L.act_value[i];
            max_prob_index = i;
        }
    }

    /**< Write the output of the network in the global varible. */
    switch(active_net) {
        case DIGITS:
            if(max_prob_index >= 0) {
                nn_result.rec_char = digits_map[max_prob_index];
                nn_result.prob = max_prob * 100;
            }
            break;
        case LETTERS:
            if(max_prob_index >= 0) {
                nn_result.rec_char = letters_map[max_prob_index];
                nn_result.prob = max_prob * 100;
            }
            break;
        case MIXED:
            if(max_prob_index >= 0) {
                nn_result.rec_char = mixed_map[max_prob_index];
                nn_result.prob = max_prob * 100;
            }
            break;
        default:
            break;
    }
}