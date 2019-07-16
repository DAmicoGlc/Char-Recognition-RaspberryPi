/**
* @file hand_written_recognition.c
* @author Gianluca D'Amico
* @brief Main core
*
* It handles the creation, activation and destruction of all tasks involved.
* It also initilize all the structure needed. 
* The Camera task will caputer the image, from it a ROI is extracte by the 
* display task. The NN task will then shrink the ROI and feed the MLP to
* compute the resulting recognized character. The display task will show 
* the camera previez, the extracted ROI, the input fed to the MLP and the 
* resulting character. It also show some properties of the camera module, and
* the current active model of the MLP: THe user task manages the interaction 
* with the user, who can change the cam properties, the active model, the 
* dimension of the ROI and its position.
*
*/

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <allegro.h>

#include "user.h"
#include "display.h"
#include "raspi_cam.h"
#include "nn_handler.h"
#include "common.h"
#include "ptask_handler.h"

/**
* LOCAL CONSTANTS
*/

#define SUCCESS 0
#define ERROR   1

/**
* GLOBAL DATA
*/

/**< Variable to check if the user press the ESC key */
/* to close the application. */
int completed = 0;                  

/**
* GLOBAL MUTEX
*/

pthread_mutex_t completed_mutex;

/**
* LOCAL VARIABLE
*/

/**< images needed to the nn task */
BITMAP* local_input;
BITMAP* local_acquired;

/**
* LOCAL FUCNTION
*/

/**< Initilize the allegro settings and the task parameters. */
int init();

/**< Task handling routines */
void *display_task(void *arg);
void * user_task(void * arg);
void * nn_task(void * arg);
void * cam_task(void * arg);

/**< Error checking */
void display_error(int return_value);
void cam_error(int return_value);
void nn_error(int return_value);

/**
* @brief Initialization
*
* Call the init function of all task and creates all the task envolved in the
* application.
*
* @return 0 on SUCCESS, ERROR CODE otherwise
*/
int init() {

    int error; /**< Error variable */

    /**< Camera module configuration settings */
    RASPIVID_CONFIG* config = (RASPIVID_CONFIG*)malloc(sizeof(RASPIVID_CONFIG));
    
    config->width       = CAM_WIDTH;
    config->height      = CAM_HEIGHT;
    config->bitrate     = 0;            /**< Leave as default */
    config->framerate   = VIDEO_FRAME_RATE_NUM;
    config->monochrome  = MONOCHROME;

    /**< Allegro init */
    allegro_init();

    set_color_depth(32);
    set_gfx_mode(GFX_AUTODETECT_WINDOWED, 
                                WIN_WIDTH, WIN_HEIGHT, 
                                WIN_WIDTH, 2*WIN_HEIGHT);
    clear_to_color(screen, WHITE);

    install_keyboard();
    install_mouse();

    enable_hardware_cursor();
    show_mouse(screen);

    /**< Display task init */
    error = init_display();
    display_error(error);
    if (error != DISPLAY_SUCCESS) {
        allegro_exit();
        return ERROR;
    }

    /**< Cam task init */
    error = raspi_cam_create_camera_capture(config);
    cam_error(error);
    if (error != CAM_SUCCESS) {
        allegro_exit();
        return ERROR;
    }

    free(config);

    /**< Allocate memory for the local images */
    local_acquired  = create_bitmap(ROI_MAX, ROI_MAX);
    local_input     = create_bitmap(INPUT_DIM, INPUT_DIM);

    /**< NN task init */
    error = init_networks();
    nn_error(error);
    if (error != NN_SUCCESS) {
        allegro_exit();
        return ERROR;
    }

    /**< Init the conclusion mutex */
    pthread_mutex_init(&completed_mutex, NULL);

    /**< Creates tasks */
    task_create(cam_task, PERIOD_CAM, DLINE_CAM, PRIO_CAM);
    task_create(nn_task, PERIOD_NN, DLINE_NN, PRIO_NN);
    task_create(display_task, PERIOD_DIS , DLINE_DIS, PRIO_DIS);
    task_create(user_task, PERIOD_US, DLINE_US, PRIO_US);

    return SUCCESS;
}

/**
* @brief Display routine
*
* Call draw display function defined in display.c, drawing all the content on
* the screen
*
*/
void * display_task(void * arg)
{
    int end = 0; /**< Local value of conclusion variable*/

    /**< Get its own ID*/
    const int id = get_task_index(arg);

    /**< Set activation instant*/
    set_activation(id);

    while (!end) {
        /**< Check conclusion variable*/
        pthread_mutex_lock(&completed_mutex);
        end = completed;
        pthread_mutex_unlock(&completed_mutex);

        /**< Fill the screen. */
        draw_display();

        /**< Check deadline miss. */
        if (deadline_miss(id)) {   
            printf("%d) deadline missed! Display %d\n", id);
        }

        /**< Wait untill next activation. */
        wait_for_activation(id);
    }

    return NULL;
}

/**
* @brief User routine
*
* Check user interactions from keyboard or mouse
*
*/
void * user_task(void * arg)
{
    int key;     /**< Pressend key*/
    int end = 0; /**< Local value of conclusion variable*/

    /**< Get its own ID*/
    const int id = get_task_index(arg);
    /**< Set activation instant*/
    set_activation(id);

    while (!end) {
        /**< Check conclusion variable*/
        pthread_mutex_lock(&completed_mutex);
        end = completed;
        pthread_mutex_unlock(&completed_mutex);

        /**< Check mouse touch*/
        if(mouse_b & 1)
            mouse_touch();

        /**< Check key pressed*/
        if (keypressed()) {
            key = readkey() >> 8;
            end = key_pressed(key);
        }

        /**< If ESC is pressend conclude the application*/
        if (end) {
            pthread_mutex_lock(&completed_mutex);
            completed = end;
            pthread_mutex_unlock(&completed_mutex);
        }

        /**< Check deadline miss. */
        if (deadline_miss(id)) {   
            printf("%d) deadline missed! User\n", id);
        }
        
        /**< Wait untill next activation. */
        wait_for_activation(id);
    }
    return NULL;
}

/**
* @brief Cam routine
*
* Signal the request of a new video frame to the camera module.
*
*/
void * cam_task(void * arg)
{
    int end = 0; /**< Local value of conclusion variable*/

    /**< Get its own ID*/
    const int id = get_task_index(arg);
    /**< Set activation instant*/
    set_activation(id);

    while (!end) {
        /**< Check conclusion variable*/
        pthread_mutex_lock(&completed_mutex);
        end = completed;
        pthread_mutex_unlock(&completed_mutex);

        /**< Signal the fram acquisition*/
        raspi_cam_query_frame();
        
        /**< Check deadline miss. */
        if (deadline_miss(id)) {   
            printf("%d) deadline missed! Cam\n", id);
        }
        
        /**< Wait untill next activation. */
        wait_for_activation(id);

    }

    return NULL;
}

/**
* @brief NN routine
*
* Get the ROI, shrink it and feed the MLP to comput the resulting character.
*
*/
void * nn_task(void * arg)
{
    int end = 0; /**< Local value of conclusion variable*/

    /**< Get its own ID*/
    const int id = get_task_index(arg);
    /**< Set activation instant*/
    set_activation(id);

    int local_radius = 0; /**< Local radius of the ROI*/
    /**< Index of the array in which the taks have to write*/
    int index_result = 1; 

    while (!end) {
        /**< Check conclusion variable*/
        pthread_mutex_lock(&completed_mutex);
        end = completed;
        pthread_mutex_unlock(&completed_mutex);

        /**< Get the extracted ROI by the display task*/
        pthread_mutex_lock(&ROI_image_mutex);

        local_radius = extracted_ROI.radius;

        /**< Copy it in the local image*/
        blit(extracted_ROI.image, local_acquired, 0, 0, 0, 0, 
            2 * local_radius, 2 * local_radius);

        pthread_mutex_unlock(&ROI_image_mutex);

        /**< Shrink the ROI to fit the input image*/
        stretch_blit(local_acquired, local_input, 
                            0, 0, 2 * local_radius, 2 * local_radius,
                            0, 0, INPUT_DIM, INPUT_DIM);

        /**< Compute the MLP result*/
        recognize_character(local_input);

        /**< Copy the result in the global struct*/
        blit(local_acquired, display_nn_data[index_result].ROI, 0, 0, 0, 0, 
                                            2 * local_radius, 2 * local_radius);
        blit(local_input, display_nn_data[index_result].input_image, 0, 0, 0, 0, 
                                                        INPUT_DIM, INPUT_DIM);
        
        display_nn_data[index_result].result.rec_char = nn_result.rec_char;
        display_nn_data[index_result].result.prob     = nn_result.prob;

        display_nn_data[index_result].image_radius = local_radius;

        /**< Update the current global index result*/
        pthread_mutex_lock(&current_result_mutex);
        current_result = (current_result + 1) % 2;
        pthread_mutex_unlock(&current_result_mutex);

        /**< Update the current local index result*/
        index_result = (index_result + 1) % 2;
        
        /**< Check deadline miss. */
        if (deadline_miss(id)) {   
            printf("%d) deadline missed! NN\n", id);
        }
        
        /**< Wait untill next activation. */
        wait_for_activation(id);
    }

    return NULL;
}

/**
* @brief Display errors
*
* Check if the display task return some error.
*/
void display_error(int return_value)
{
    int error = 0;

    switch (return_value)
    {
    case DISPLAY_ERROR_NO_FONT_FILE:
        fprintf(stderr, "DISPLAY_ERROR_NO_FONT_FILE ERROR - %s\n",
                "Font file missed or nto supported!");
        error++;
        break;
    case DISPLAY_ERROR_CREATE_BITMAP:
        fprintf(stderr, "DISPLAY_ERROR_CREATE_BITMAP ERROR - %s\n",
                "Error while creating a BITMAP in display task!");
        error++;
        break;
    case DISPLAY_ERROR_SHOW_VIDEO:
        fprintf(stderr, "DISPLAY_ERROR_SHOW_VIDEO ERROR - %s\n",
                "Error during video page showing, maybe video paging is \
                 not supported!");
        error++;
        break;
    default:
        break;
    }

    if (error) {
        pthread_mutex_lock(&completed_mutex);
        completed = 1;
        pthread_mutex_unlock(&completed_mutex);
    }
}

/**
* @brief Cam errors
*
* Check if the cam task return some error.
*/
void cam_error(int return_value) {
    if (return_value == CAM_ERROR) {
        fprintf(stderr, "CAM ERROR - %s\n",
                "Something go wrong with the camera module!");

        pthread_mutex_lock(&completed_mutex);
        completed = 1;
        pthread_mutex_unlock(&completed_mutex);
    }
}

/**
* @brief NN errors
*
* Check if the NN task return some error.
*/
void nn_error(int return_value) {
    int error = 0;

    switch (return_value) {
        case NN_ERROR_NO_FILE:
            fprintf(stderr, "NN_ERROR_NO_FILE ERROR - %s\n",
                    "Error opnening weigths file!");
            error++;
            break;
        case NN_ERROR_READING_FILE:
            fprintf(stderr, "NN_ERROR_READING_FILE ERROR - %s\n",
                    "Error while loading weights values from file!");
            error++;
            break;
        default:
        break;
    }

    if (error) {
        pthread_mutex_lock(&completed_mutex);
        completed = 1;
        pthread_mutex_unlock(&completed_mutex);
    }
}

/**
* @brief Main core
*
* Initilize all the structures and tasks. Wait that the tasks finish their 
* routing (user press ESC). Realease all memory allocated and return.
*
*/
int main()
{
    int error = 0;
    /**< Initilize. */
    error = init();
    if (error == ERROR)
        return 0;
    /**< Run. */
    wait_tasks();

    /**< Free local images. */
    destroy_bitmap(local_acquired);
    destroy_bitmap(local_input);

    /**< Free all structures of the display task. */
    free_display();

    /**< Realese the camera module. */
    raspi_cam_release_capture();

    allegro_exit();
    return 0;
}
