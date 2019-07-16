////////////////////////////////////////////////////////////
//
// Many source code lines are copied from RaspiVid.c
// Copyright (c) 2012, Broadcom Europe Ltd
// 
// Lines have been copied from GITHUB project 
// https://github.com/robidouille/robidouille of Emil Valkov
// 
/////////////////////////////////////////////////////////////

/**
* @file raspi_cam.c
* @author Gianluca D'Amico
* @brief File containing RaspBerry Camera handling functions
*
* HANDLING CAMERA FUNCTIONS: It manages all the functions needed to capture
* video frame from the raspberry camera.
*
* Utilizing the non static function, it is possible to create all the structure
* to initilize the image frame buffer from the camera. In this file the Allegro
* 4 library is utilize to show the captured images. The important functions are:
*
* - video_buffer_callback: handle the video capturing, in particular it will 
*           copy the image caputered from the buffer to the BITMAP image 
*           destination utilized with allegro;
* 
* - raspi_cam_get_capture_property: retrive main property of the capturing mode;
*
* - raspi_cam_set_capture_property: change some campturing property;
*
* - raspi_cam_create_camera_capture: create and initilize the campera component
*           and all related structre, in particulare it allocates memory for 
*           the image destination in wich the buffer from the camera is copied
*           (e.g. this allocation must be changed if you want to use a 
*           different library from Allegro);
* 
* - raspi_cam_release_capture: release the memory allocate for all the camera 
*           component and also for the destination image;
*
* - raspi_cam_query_frame: allow the copy of the buffer video frame in the 
*           global image structure.
*
* @note If you want to use a different library from Allegro, you have to change:
*   - video_buffer_callback;
*   - raspi_cam_create_camera_capture;
*   - raspi_cam_release_capture;
*   - raspi_cam_query_frame
*   - Global variable acquired_image.
*
*/

/**
* STANDARD LIBRARIES
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory.h>
#include <semaphore.h>
#include <pthread.h>

/**
* ALLEGRO LIBRARY
*/

#include <allegro.h>

/**
* USERLAND LIBRARIES
*/

#include "time.h"

#include "bcm_host.h"
#include "interface/vcos/vcos.h"

#include "interface/mmal/mmal.h"
#include "interface/mmal/mmal_logging.h"
#include "interface/mmal/mmal_buffer.h"
#include "interface/mmal/util/mmal_util.h"
#include "interface/mmal/util/mmal_util_params.h"
#include "interface/mmal/util/mmal_default_components.h"
#include "interface/mmal/util/mmal_connection.h"

#include "RaspiCamControl.h"

/**
* PROJECT LIBRARY
*/

#include "raspi_cam.h"

/**
* GLOBAL DATA
*/

// BITMAP* captured_image; /**< Global image captured by camera.*/
// aquired_image_t acquired_image; /**< Global ROI of captured image. */

unsigned char capture_buffer[CAM_HEIGHT * CAM_WIDTH];

int contrast_value;     /**< Global contrast value.*/
int brightness_value;   /**< Global brightness value.*/
int saturation_value;   /**< Global saturation value.*/
int sharpness_value;    /**< Global sharpness value.*/

/**
* GLOBAL MUTEX
*/

pthread_mutex_t capture_buffer_mutex;

// pthread_mutex_t capture_mutex;
// pthread_mutex_t acquire_mutex;

pthread_mutex_t contrast_mutex;
pthread_mutex_t brightness_mutex;
pthread_mutex_t saturation_mutex;
pthread_mutex_t sharpness_mutex;

/**
* LOCAL DATA
*/

static int contrast_local;
static int brightness_local;
static int saturation_local;
static int sharpness_local;

/**< Camera number to use - we only have one camera, indexed from 0. */
#define CAMERA_NUMBER 0

/**< Standard port setting for the camera component. */
#define MMAL_CAMERA_PREVIEW_PORT 0
#define MMAL_CAMERA_VIDEO_PORT 1
#define MMAL_CAMERA_CAPTURE_PORT 2

/**< Video format information. */
#define VIDEO_FRAME_RATE_DEN 1

/**< Video render needs at least 2 buffers. */
#define VIDEO_OUTPUT_BUFFERS_NUM 3

/**< Max bitrate we allow for recording. */
static const int MAX_BITRATE = 30000000; // 30Mbits/s

/**< Capture struct.*/
static raspi_cam_capture capture; 

/**
* LOCAL STRUCT
*/

/**< Structure containing all state information for the current run. */
typedef struct _RASPIVID_STATE {
    int finished;
    int width;            	/**< Requested width of image. */
    int height;           	/**< requested height of image. */
    int bitrate;          	/**< Requested bitrate. */
    int framerate;        	/**< Requested frame rate (fps). */
    int monochrome;			/**< Capture in grey only (2x faster). */
    int immutableInput;     /**< Flag to specify whether encoder works in */
                            /**< place or creates a new buffer. Result is */
                            /**< preview can display either the camera */ 
                            /**< output or the encoder output. */
    RASPICAM_CAMERA_PARAMETERS camera_parameters; /**< Camera setup param.*/

    MMAL_COMPONENT_T *camera_component;  /**< Pointer to the camera.*/
    MMAL_COMPONENT_T *encoder_component; /**< Pointer to the encoder.*/

    MMAL_POOL_T *video_pool;    /**< Pointer to the pool of buffers used by */
                                /**< encoder output port. */

    VCOS_SEMAPHORE_T capture_sem;      /**< Semaphore variable for capturing.*/
    VCOS_SEMAPHORE_T capture_done_sem; /**< Semaphore variable for capturing.*/
   
} RASPIVID_STATE;

/**
* LOCAL FUNCTIONS
*/

/**
* @brief Set default capture parameters.
*
* This function set all the parameter needed to the camera component with 
* default values. It also set the global variable of the main property of the 
* capturing mode.
*
* @param state Pointer to state control struct
*/
static void default_status(RASPIVID_STATE *state) {

    if (!state) {
        vcos_assert(0);
        return;
    }

    memset(state, 0, sizeof(RASPIVID_STATE));

    state->finished         = 0;
    state->width 			= 320;      /**< use a multiple of 320. */
    state->height 			= 240;		/**< use a multiple of 240. */
    state->bitrate 			= 17000000; /**< Depends on resolution. */
    state->framerate 		= VIDEO_FRAME_RATE_NUM;
    state->immutableInput 	= 1;
    state->monochrome 		= 0;		/**< Grey = 1, Color = 0. */
    
    raspicamcontrol_set_defaults(&state->camera_parameters);

    /**< Initialization of global variable.*/

    /**< Default contrast value.*/
    contrast_value = state->camera_parameters.contrast;

    /**< Default brightness value.*/
    brightness_value = state->camera_parameters.brightness;

    /**< Default saturation value.*/
    saturation_value = state->camera_parameters.saturation;

    /**< Default sharpness value.*/
    sharpness_value  = state->camera_parameters.sharpness;
}

/**
* @brief Buffer header callback function for video
*
* This function manages he buffer pool of captured images. It also copies
* the buffer into destination image.
*
* @param port Pointer to port from which callback originated
* @param buffer mmal buffer header pointer
*/
static void video_buffer_callback(MMAL_PORT_T *port, 
                                        MMAL_BUFFER_HEADER_T *buffer) {

    int i,j;
    int color, r, g, b, grey;   /**< Color variables of the captured frame .*/
    int w, h;                   /**< Widht and height of the captured frame .*/
    int x_1, x_2, y_1, y_2;
    int radius;

    MMAL_BUFFER_HEADER_T *new_buffer;
    RASPIVID_STATE * state = (RASPIVID_STATE *)port->userdata;

    if (state) {
        if (state->finished) {
            vcos_semaphore_post(&state->capture_done_sem);
            return;
        }
        if (buffer->length) {
            mmal_buffer_header_mem_lock(buffer);

            pthread_mutex_lock(&capture_buffer_mutex);

            memcpy(capture_buffer, buffer->data, 
                CAM_WIDTH * CAM_HEIGHT * sizeof(unsigned char));
            
            pthread_mutex_unlock(&capture_buffer_mutex);

            vcos_semaphore_post(&state->capture_done_sem);
            vcos_semaphore_wait(&state->capture_sem);

            mmal_buffer_header_mem_unlock(buffer);
        }
        else {
        	vcos_log_error("buffer null");
        }
    }
    else {
        vcos_log_error("Received a encoder buffer callback with no state");
    }

    /**< Release buffer back to the pool. */
    mmal_buffer_header_release(buffer);

    /**< And send one back to the port (if still open). */
    if (port->is_enabled) {
        MMAL_STATUS_T status;

        new_buffer = mmal_queue_get(state->video_pool->queue);

        if (new_buffer)
            status = mmal_port_send_buffer(port, new_buffer);

        if (!new_buffer || status != MMAL_SUCCESS)
            vcos_log_error("Unable to return a buffer to the encoder port");
    }
}


/**
* @biref Create the camera component and set up its configuration.
*
* @param state Pointer to state control struct
* @return 0 if failed, pointer to component if successful
*/
static MMAL_COMPONENT_T *create_camera_component(RASPIVID_STATE *state) {
    MMAL_COMPONENT_T *camera = 0;
    MMAL_ES_FORMAT_T *format;
    MMAL_PORT_T *video_port = NULL;
    MMAL_STATUS_T status;

    /**< Create the component. */
    status = mmal_component_create(MMAL_COMPONENT_DEFAULT_CAMERA, &camera);

    if (status != MMAL_SUCCESS) {
        vcos_log_error("Failed to create camera component");
        
        if (camera)
            mmal_component_destroy(camera);

       return 0;
    }
    
    if (!camera->output_num) {
        vcos_log_error("Camera doesn't have output ports");
        
        if (camera)
            mmal_component_destroy(camera);

        return 0;
	}
	
    video_port = camera->output[MMAL_CAMERA_VIDEO_PORT];
    
    /**< Set up the camera configuration. */
	{
        MMAL_PARAMETER_CAMERA_CONFIG_T cam_config = {
            { MMAL_PARAMETER_CAMERA_CONFIG, sizeof(cam_config) },
            .max_stills_w = state->width,
            .max_stills_h = state->height,
            .stills_yuv422 = 0,
            .one_shot_stills = 0,
            .max_preview_video_w = state->width,
            .max_preview_video_h = state->height,
            .num_preview_video_frames = 3,
            .stills_capture_circular_buffer_height = 0,
            .fast_preview_resume = 0,
            .use_stc_timestamp = MMAL_PARAM_TIMESTAMP_MODE_RESET_STC
        };
        mmal_port_parameter_set(camera->control, &cam_config.hdr);
    }

    /**< Set the encode format on the video  port. */
    format = video_port->format;
    if (state->monochrome) {
        format->encoding_variant = MMAL_ENCODING_I420;
        format->encoding = MMAL_ENCODING_I420;
    }
    else {
        format->encoding =
            mmal_util_rgb_order_fixed(video_port) ? 
            MMAL_ENCODING_BGR24 : MMAL_ENCODING_RGB24;
        format->encoding_variant = 0;
    }

    format->es->video.width = state->width;
    format->es->video.height = state->height;
    format->es->video.crop.x = 0;
    format->es->video.crop.y = 0;
    format->es->video.crop.width = state->width;
    format->es->video.crop.height = state->height;
    format->es->video.frame_rate.num = state->framerate;
    format->es->video.frame_rate.den = VIDEO_FRAME_RATE_DEN;

    status = mmal_port_format_commit(video_port);
    if (status) {
        vcos_log_error("camera video format couldn't be set");
        
        if (camera)
          mmal_component_destroy(camera);

        return 0;
    }
    
    /**< PR : plug the callback to the video port. */
    status = mmal_port_enable(video_port, video_buffer_callback);
    if (status) {
        vcos_log_error("camera video callback2 error");
        
        if (camera)
            mmal_component_destroy(camera);

        return 0;
    }

    /**< Ensure there are enough buffers to avoid dropping frames. */
    if (video_port->buffer_num < VIDEO_OUTPUT_BUFFERS_NUM)
        video_port->buffer_num = VIDEO_OUTPUT_BUFFERS_NUM;


    /**< PR : create pool of message on video port. */
    MMAL_POOL_T *pool;
    video_port->buffer_size = video_port->buffer_size_recommended;
    video_port->buffer_num = video_port->buffer_num_recommended;
    pool = mmal_port_pool_create(video_port, video_port->buffer_num, 
                                                    video_port->buffer_size);
    if (!pool) {
        vcos_log_error("Failed to create buffer header pool for video \
                                                                output port");
    }
    state->video_pool = pool;

    /**< Enable component. */
    status = mmal_component_enable(camera);

    if (status) {
        vcos_log_error("camera component couldn't be enabled");

        if (camera)
            mmal_component_destroy(camera);

        return 0;
    }

    raspicamcontrol_set_all_parameters(camera, &state->camera_parameters);

    raspicamcontrol_set_rotation(camera, 270);

    state->camera_component = camera;

    contrast_local      = INIT_CONTRAST;
    brightness_local    = INIT_BRIGHTNESS;
    saturation_local    = INIT_SATURATION;
    sharpness_local     = INIT_SHARPNESS;

    pthread_mutex_init(&capture_buffer_mutex, NULL);
    // pthread_mutex_init(&capture_mutex, NULL);
    // pthread_mutex_init(&acquire_mutex, NULL);

    pthread_mutex_init(&contrast_mutex, NULL);
    pthread_mutex_init(&brightness_mutex, NULL);
    pthread_mutex_init(&saturation_mutex, NULL);
    pthread_mutex_init(&sharpness_mutex, NULL);
    return camera;  
}

/**
* @brief Destroy the camera component
*
* @param state Pointer to state control struct
*/
static void destroy_camera_component(RASPIVID_STATE *state) {
    if (state->camera_component) {
        mmal_component_destroy(state->camera_component);
        state->camera_component = NULL;
    }
}


/**
* @brief Destroy the encoder component
*
* @param state Pointer to state control struct
*/
static void destroy_encoder_component(RASPIVID_STATE *state) {
    /**<  Get rid of any port buffers first. */
    if (state->video_pool) {
        mmal_port_pool_destroy(state->encoder_component->output[0], 
                                                            state->video_pool);
    }
}

/**
* @brief Connect two specific ports together
*
* @param output_port Pointer the output port
* @param input_port Pointer the input port
* @param Pointer to a mmal connection pointer, reassigned if function successful
* @return Returns a MMAL_STATUS_T giving result of operation
*/
static MMAL_STATUS_T connect_ports(MMAL_PORT_T *output_port, 
                MMAL_PORT_T *input_port, MMAL_CONNECTION_T **connection) {

    MMAL_STATUS_T status;

    status =  mmal_connection_create(connection, output_port, input_port, 
                                    MMAL_CONNECTION_FLAG_TUNNELLING | 
                                    MMAL_CONNECTION_FLAG_ALLOCATION_ON_INPUT);

    if (status == MMAL_SUCCESS) {
        status =  mmal_connection_enable(*connection);
        if (status != MMAL_SUCCESS)
            mmal_connection_destroy(*connection);
    }

    return status;
}

/**
* @brief Checks if specified port is valid and enabled, then disables it.
*
* @param port Pointer the port
*/
static void check_disable_port(MMAL_PORT_T *port) {
    if (port && port->is_enabled)
        mmal_port_disable(port);
}

/**
* @brief Retrive capture property.
*
* This function return the main property of the actual capture mode. For each 
* property there is a Enum that represent it.
*
* @param property_id    Property to return
* @return               Value of requested property.
*/
static double raspi_cam_get_capture_property(int property_id) {
    double property;    /**< Property requested  . */
  
    switch(property_id) {
        case RPI_CAP_PROP_FRAME_HEIGHT:
            property = capture.pState->height;
            break;
        case RPI_CAP_PROP_FRAME_WIDTH:
            property = capture.pState->width;
            break;
        case RPI_CAP_PROP_FPS:
            property = capture.pState->framerate;
            break;
        case RPI_CAP_PROP_MONOCHROME:
            property = capture.pState->monochrome;
            break;
        case RPI_CAP_PROP_BITRATE:
            property = capture.pState->bitrate;
            break;

        default:
            property = 0;
    }
    return property;
}

/**
* @brief Set capture property.
*
* This function set some property of the capture mode using functions defined 
* in the file RaspiCamControl.c:
* - Sharpness;
* - Brightness;
* - Contrast;
* - Saturation.
*
* @param property_id    Property to set
* @param op             {INCR, DECR} to increase or decrease the property by 5.
* @return               0 if successful, non-zero if parameters is out of range
*
* @note See RaspiCamControl on Userland library to change other property.
*/
static int raspi_cam_set_capture_property(cam_property property_id, int value) {
    int retval = 0; /**< Indicate failure. */

    switch(property_id) {
        case CONTRAST:
            retval = raspicamcontrol_set_contrast( 
                        capture.pState->camera_component, value);
            break;
        case 1:
            retval = raspicamcontrol_set_brightness(
                        capture.pState->camera_component, value);
            break;
        case 2:
            retval = raspicamcontrol_set_saturation(
                        capture.pState->camera_component, value);
            break;
        case 3:
            retval = raspicamcontrol_set_sharpness(
                        capture.pState->camera_component, value);
            break;

        default:
            retval = 0;
            break;
    }

    return retval;
}

/**
* GLOBAL FUNCTIONS
*/

/**
* @brief Create the camera component.
*
* This function create the camera component to capture video frame from the
* camera. It utilize the static fucntion create_camera_component().
*
* @param config     Struct containg configuration data
*/
int raspi_cam_create_camera_capture(RASPIVID_CONFIG* config) {

    int i;
    int num;    /**< Queue lenght of mmal frame pool. */
    int w, h;   /**< Width and height of the capturing frame. */

    /**< Our main data storage vessel... */
    RASPIVID_STATE * state = (RASPIVID_STATE*)malloc(sizeof(RASPIVID_STATE));
    capture.pState = state;

    MMAL_STATUS_T status = -1;
    MMAL_PORT_T *camera_video_port = NULL;

    default_status(state);

    if (config != NULL)	{
        if (config->width != 0)
            state->width = config->width;
        if (config->height != 0)
            state->height = config->height;
        if (config->bitrate != 0)
            state->bitrate = config->bitrate;
        if (config->framerate != 0)
            state->framerate = config->framerate;
        if (config->monochrome != 0)
            state->monochrome = config->monochrome;
    }

    w = state->width;
    h = state->height;

    // captured_image = create_bitmap(w, h);
    // acquired_image.image = create_bitmap(ROI_MAX, ROI_MAX);

    vcos_semaphore_create(&state->capture_sem, "Capture-Sem", 0);
    vcos_semaphore_create(&state->capture_done_sem, "Capture-Done-Sem", 0);

    /**< Create camera. */
    if (!create_camera_component(state)) {
        vcos_log_error("%s: Failed to create camera component", __func__);
        raspi_cam_release_capture();
        return CAM_ERROR;
    }

    camera_video_port = state->camera_component->
                                            output[MMAL_CAMERA_VIDEO_PORT];

    /**< Assign data to use for callback. */
    camera_video_port->userdata = (struct MMAL_PORT_USERDATA_T *)state;

    /**< Start capture. */
    if (mmal_port_parameter_set_boolean(camera_video_port, 
                                MMAL_PARAMETER_CAPTURE, 1) != MMAL_SUCCESS) {
        vcos_log_error("%s: Failed to start capture", __func__);
        raspi_cam_release_capture();
        return CAM_ERROR;
    }

    /**< Send all the buffers to the video port. */
    num = mmal_queue_length(state->video_pool->queue);
    for (i = 0; i < num; i++) {
        MMAL_BUFFER_HEADER_T *buffer = 
                                    mmal_queue_get(state->video_pool->queue);

        if (!buffer)
            vcos_log_error("Unable to get a required buffer %d \
                                                        from pool queue", i);

        if (mmal_port_send_buffer(camera_video_port, buffer)!= MMAL_SUCCESS)
            vcos_log_error("Unable to send a buffer to encoder \
                                                        output port (%d)", i);
    }

    vcos_semaphore_wait(&state->capture_done_sem);
    return CAM_SUCCESS;
}

/**
* @brief Release camera component.
*
* This function destroy the camera component, finishing the viedo stream 
* of frame. It will also release the memory allocated for the component 
* calling the static function destroy_camera_component().
*
*/
void raspi_cam_release_capture() {
    RASPIVID_STATE * state = capture.pState;

    /**< Unblock the callback. */
    state->finished = 1;
    vcos_semaphore_post(&state->capture_sem);
    vcos_semaphore_wait(&state->capture_done_sem);

    vcos_semaphore_delete(&state->capture_sem);
    vcos_semaphore_delete(&state->capture_done_sem);

    if (state->camera_component)
        mmal_component_disable(state->camera_component);

    destroy_camera_component(state);

    free(state);
}

/**
* @brief Execute capture of video frame.
*
* This function execute the capture of the video frame by the camera component, 
* so that the buffer video frame will be copied into the global image. 
*
*
* @note See video_buffer_callback for the copy phase.
*/
void raspi_cam_query_frame() {
    RASPIVID_STATE * state;
    int property_change;

    state = capture.pState;
    vcos_semaphore_post(&state->capture_sem);
    vcos_semaphore_wait(&state->capture_done_sem);

    pthread_mutex_lock(&contrast_mutex);
    property_change = contrast_value;
    pthread_mutex_unlock(&contrast_mutex);

    if (property_change != contrast_local) {
        raspi_cam_set_capture_property(CONTRAST, property_change);
        contrast_local = property_change;
    }

    pthread_mutex_lock(&brightness_mutex);
    property_change = brightness_value;
    pthread_mutex_unlock(&brightness_mutex);

    if (property_change != brightness_local) {
        raspi_cam_set_capture_property(BRIGHTNESS, property_change);
        brightness_local = property_change;
    }

    pthread_mutex_lock(&saturation_mutex);
    property_change = saturation_value;
    pthread_mutex_unlock(&saturation_mutex);

    if (property_change != saturation_local) {
        raspi_cam_set_capture_property(SATURATION, property_change);
        saturation_local = property_change;
    }

    pthread_mutex_lock(&sharpness_mutex);
    property_change = sharpness_value;
    pthread_mutex_unlock(&sharpness_mutex);

    if (property_change != sharpness_local) {
        raspi_cam_set_capture_property(SHARPNESS, property_change);
        sharpness_local = property_change;
    }       

    return;
}
