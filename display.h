#ifndef DISPLAY_H
#define DISPLAY_H

/**
* @file display.h
* @author Gianluca D'Amico
* @brief File containing displaying routines
*
* HANDLING USER INTERFACE: It display all the contents of the application.
*
* Firstly, draw a fixed skeleton structure on a first video page with all fixed
* contents (lines shapes etc...), the read all result computed by other tasks 
* and display it on the screen. All functionalities are handled with Allegro
* libabry, in particulare the display is managed with the video paging.
*
*/

#include "common.h"
#include "nn_handler.h"

/**
* GLOBAL CONSTANTS
*/

/**< Constant related to the properties texture. */
#define PROP_HEIGHT 20
#define PROP_WIDTH 140
#define PROP_MRG 20
#define PROP_OFFSET 70

/**< Constant related MLP result and input. */
#define IN_OUT_LABEL_MRG 10

/**< Constant related to arrow and equal figures. */
#define ARROW_LENGHT 20
#define ARROW_WIDTH 12
#define ARROW_HEIGHT 4
#define ARROW_MRG 6
#define EQUAL_LENGHT 10
#define EQUAL_HEIGHT 6
#define EQUAL_MRG 12

/**< Constant related to buttons. */
#define BTN_HEIGHT 20
#define BTN_WIDTH 65
#define BTN_MRG 10
#define MODEL_MRG 5
#define MODEL_LENGHT 170

/**
* RETURN CONSTANT
*/

#define DISPLAY_SUCCESS                 0
#define DISPLAY_ERROR_NO_FONT_FILE      1
#define DISPLAY_ERROR_CREATE_BITMAP     2
#define DISPLAY_ERROR_SHOW_VIDEO        3

/**
* GLOBAL STRUCTURE
*/

/**< Struct that identify actual input and output.*/
typedef struct {
    BITMAP *ROI;     /**< ROI extracted from the captured image. */
    BITMAP *input_image;        /**< Input image fed to the MLP. */

    int image_radius;           /**< ROI radius. */

    data_network_t result;      /**< Corresponding MLP result. */
} display_network_t;

/**< Struct that identify position and dimension of the ROI. */
typedef struct {
    int centerX;
    int centerY;

    int radius;
} sqr_center_t;

/**< Struct that identify extracted ROI with the related radius dim. */
typedef struct {
    BITMAP* image;

    int radius;
} ROI_t;

/**
* GLOBAL DATA STRUCTURES
*/

extern display_network_t display_nn_data[2];    /**< MLP data. */
extern sqr_center_t ROI_dim;                    /**< ROI dimensions. */
extern ROI_t extracted_ROI;                     /**< Extracted ROI image. */
extern int current_result;                      /**< Last MLP result.*/

/**
* GLOBAL MUTEX
*/

extern pthread_mutex_t current_result_mutex;    /**< MLP data mutex.*/
extern pthread_mutex_t ROI_dim_mutex;           /**< ROI dim and pos mutex.*/
extern pthread_mutex_t ROI_image_mutex;         /**< ROI mutex.*/

/**
* GLOBAL FUNCTIONS
*/

/**< Initilize all structure and parameters used by the display task. */
int init_display();

/**< Screen drawing routine of the display task. */
int draw_display();

/**< Deallocate the memory used by the display task. */
void free_display();


#endif