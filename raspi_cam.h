#ifndef RASPI_CAM_H
#define RASPI_CAM_H

////////////////////////////////////////////////////////////////////////////////
//
// Many source code lines are copied from RaspiVid.c
// Copyright (c) 2012, Broadcom Europe Ltd
// 
// Lines have been copied from GITHUB project 
// https://github.com/robidouille/robidouille of Emil Valkov
// 
////////////////////////////////////////////////////////////////////////////////

/**
* @file raspi_cam.h
* @author Gianluca D'Amico
* @brief File containing RaspBerry Camera handling functions
*
* HANDLING CAMERA FUNCTIONS: It manages all the functions needed to capture
* video frame from the raspberry camera.
*
* Utilizing the non static function, it is possible to create all the structure
* to initilize the image frame buffer from the camera. In this file the Allegro
* 4 library is utilize to show the captured images. 
*
*/

#include "common.h"
#include "display.h"
#include <pthread.h>

/**
* RETURN CONSTANT
*/
#define CAM_SUCCESS 0
#define CAM_ERROR   1

/**
* GLOBAL STRUCT
*/

/**< Over write of the raspivid state struct. See userland library */
typedef struct _RASPIVID_STATE RASPIVID_STATE;

/**< Basic camera configuration information. */
typedef struct {
    int width;              
    int height;             
    int bitrate;            
    int framerate;          
    int monochrome;			
} RASPIVID_CONFIG;

/**< Capturing state struct. */
typedef struct {
    RASPIVID_STATE * pState;
} raspi_cam_capture;

/**< Enumaration for the basic camera information. */
enum {
    RPI_CAP_PROP_FRAME_WIDTH    = 3,
    RPI_CAP_PROP_FRAME_HEIGHT   = 4,
    RPI_CAP_PROP_FPS            = 5,
    RPI_CAP_PROP_MONOCHROME		= 19,
    RPI_CAP_PROP_BITRATE		= 37   
};

/**< Enumaration for the basic capture property. */
typedef enum {
    CONTRAST = 0,
    BRIGHTNESS,
    SATURATION,
    SHARPNESS
} cam_property;

/**
* GLOBAL DATA
*/

extern unsigned char capture_buffer[CAM_HEIGHT * CAM_WIDTH];

extern int contrast_value;     /**< Global contrast value.*/
extern int brightness_value;   /**< Global brightness value.*/
extern int saturation_value;   /**< Global saturation value.*/
extern int sharpness_value;    /**< Global sharpness value.*/

/**
* GLOBAL MUTEX
*/

extern pthread_mutex_t capture_buffer_mutex;
extern pthread_mutex_t contrast_mutex;
extern pthread_mutex_t brightness_mutex;
extern pthread_mutex_t saturation_mutex;
extern pthread_mutex_t sharpness_mutex;


/**
* GLOBAL FUNCTIONS
*/

/**< Create camera component. */
int raspi_cam_create_camera_capture(RASPIVID_CONFIG* config);

/**< Release camera component. */
void raspi_cam_release_capture();

/**< Retrive frame image captured by the camera. */
void raspi_cam_query_frame();

#endif