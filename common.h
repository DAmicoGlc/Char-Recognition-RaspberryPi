#ifndef COMMON_H
#define COMMON_H

/**
* @file nn_handler.h
* @author Gianluca D'Amico
* @brief File containing common define macros
*
* COMMON CONSTANTS
*
* In this file there are the common constants shared from each files 
* contained in the project.
*
*/

/**
* PTASK SCHEDULING CONSTANTS
*/

#define PERIOD_US   40          /**< Period of user task.*/
#define DLINE_US    40          /**< Deadline of user task.*/
#define PRIO_US     10          /**< Priority of user task.*/

#define PERIOD_NN   500         /**< Period of nerual network task.*/
#define DLINE_NN    200         /**< Deadline of nerual network task.*/
#define PRIO_NN     70          /**< Priority of nerual network task.*/

#define PERIOD_CAM  40          /**< Period of camera component task.*/
#define DLINE_CAM   40          /**< Deadline of camera component task.*/
#define PRIO_CAM    90          /**< Priority of camera component task.*/

#define PERIOD_DIS  40          /**< Period of display screen task.*/
#define DLINE_DIS   40          /**< Deadline of display screen task.*/
#define PRIO_DIS    50          /**< Priority of display screen task.*/

/**
* CAMERA COMPONENT CONSTANTS
*/

#define MONOCHROME              1   /**< Camera color mode. */
#define VIDEO_FRAME_RATE_NUM    25  /**< Numbero of frame per second. */

/**
* COLOR CONSTANTS
*/

#define WHITE   makecol(255, 255, 255)
#define BLACK   makecol(0, 0, 0)
#define GREEN   makecol(0, 255, 0)
#define RED     makecol(255, 0, 0)

/**
* DISPLAY CONSTANTS
*/

#define WIN_HEIGHT  320     /**< Height of the screen window. */
#define WIN_WIDTH   720     /**< Width of the screen window. */

#define CAM_HEIGHT  240     /**< Camera display height. */
#define CAM_WIDTH   320     /**< Camera display width. */
#define CAM_MRG_TOP 30      /**< Camera display margin from top. */

/**
* MODEL BUTTON DISPLAY CONSTANT
*/

/**< Display depth of models buttons. */
#define BTN_Y       (CAM_HEIGHT + CAM_MRG_TOP + BTN_MRG)

/**< Display vertical position of DIGITS button. */
#define BTN_DIG_X   (CAM_WIDTH + MODEL_MRG + MODEL_LENGHT)

/**< Display vertical position of LETTERS button. */
#define BTN_LET_X   (BTN_DIG_X + BTN_WIDTH + BTN_MRG)

/**< Display vertical position of MIXED button. */
#define BTN_MIX_X   (BTN_DIG_X + 2 * BTN_WIDTH + 2 * BTN_MRG)

/**
* REGION OF INTREST CONSTANTS
*/

#define ROI_MAX     224 /**< Max length of ROI. */
#define ROI_MIN     56  /**< Min length of ROI. */
#define ROI_MRG     20  /**< Display left margin of ROI. */
#define ROI_DEPTH   3   /**< Thickness of ROI in the showed camera frame. */

/**
* CAMERA PROPERTY CONSTANTS
*/

#define INIT_CONTRAST   0   
#define INIT_BRIGHTNESS 50
#define INIT_SATURATION 0
#define INIT_SHARPNESS  0

/**
* NEURAL NETWORK CONSTANTS
*/

#define INPUT_DIM 28 /**< Height of input image in the neural network. */

/**< Enumeration of the different models. */
typedef enum {
    DIGITS = 0, 
    LETTERS, 
    MIXED
} network_target;

/**< Enumeration of operation. */
typedef enum
{
    INCR = 0,
    DECR
} operation;

/**
* ERROR CONSTANTS
*/


#endif
