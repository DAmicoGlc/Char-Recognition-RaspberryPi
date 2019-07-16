/**
* @file display.c
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

#include <stdio.h>
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <allegro.h>

#include "display.h"
#include "raspi_cam.h"
#include "nn_handler.h"

/**
* LOCAL CONSTANTS
*/

#define FONT_TIT    "titleFont.pcx"
#define FONT_NOR    "normalFont.pcx"

/**
* LOCAL DATA STRUCTURES
*/

/**< Properties names. */
static const char property[4][12] = {"Contrast",
                                     "Brightness",
                                     "Saturation",
                                     "Sharpness"};
/**< Properties values. */
static char property_value[4][4] = {"", "", "", ""};

/**< FONT variables. */
static FONT *normal_font; 
static FONT *title_font;

/**< Coordinates of different contents in the screen. */
static int x_ROI;
static int x_input;
static int x_arrow;
static int x_equal;
static int x_center_output;
static int prop_x_1, prop_x_2, prop_y_1, prop_y_2;
static sqr_center_t input_center;
static sqr_center_t ROI_acquired_center;

/**< BITMAP containing the buffer acquired by the camera. */
static BITMAP* captured_image;

/**<  */
static int current_page = 0;
static BITMAP *video_page[2];

/**
* GLOBAL DATA STRUCTURES
*/

display_network_t display_nn_data[2];   /**< MLP data. */
sqr_center_t ROI_dim;                   /**< ROI dimensions. */
ROI_t extracted_ROI;                    /**< Extracted ROI image. */
int current_result = 0;                 /**< Last MLP result.*/

/**
* GLOBAL MUTEX
*/

pthread_mutex_t current_result_mutex;   /**< MLP data mutex.*/
pthread_mutex_t ROI_dim_mutex;          /**< ROI dim and pos mutex.*/
pthread_mutex_t ROI_image_mutex;        /**< ROI mutex.*/

/**
* LOCAL FUNCTIONS
*/

/**
* @brief Draw an arrow.
*
* Draw an arrow in a specif position in the video page passed as parameter.
*
* @param page is the target video page.
* @param x1 is the x coordinate where the arrow starts.
* @param y1 is the y coordinate where the arrow body starts.
* @param y2 is the y coordinate where the arrow body finishes.
* @param lenght is the arrow body length.
* @param width is the arrow width.
* @param height is the arrow head height.
* @param color is the arrow color.
*/
static void draw_arrow (BITMAP *page, int x1, int y1, int y2,
                                 int length, int width, int height, int color) {
    /**< Arrow body.*/
    fastline(page, x1, y1, x1 + width, y1, color);
    fastline(page, x1, y2, x1 + width, y2, color);
    /**< Arrow head.*/
    triangle(page, x1 + width, y1 - height, 
                     x1 + width, y2 + height, 
                     x1 + length, y1 + (y2-y1)/2, color);
}

/**
* @brief Draw an equal symbol.
*
* Draw an arrow in a specif position in the video page passed as parameter.
*
* @param page is the target video page.
* @param x1 is the x coordinate where the equal starts.
* @param y1 is the y coordinate where the euqal body starts.
* @param y2 is the y coordinate where the equal body finishes.
* @param width is the equal width.
* @param color is the equal color.
*/
static void draw_equal(BITMAP *page, int x1, int y1, int y2, 
                                                        int width, int color) {
    fastline(page, x1, y1, x1 + width, y1, color);
    fastline(page, x1, y2, x1 + width, y2, color);
}

/**
* @brief Draw the screen skeleton structure.
*
* Draw sections of the screen that don't change during the execution of the
* application, e.g. titles, properties name, lines etc.
*
* @param page is the target video page.
*/
static void draw_fixed(BITMAP *page)
{

    /**< Camera preview title. */
    textout_centre_ex(page, title_font, "Camera Preview", CAM_WIDTH / 2,
                      10, RED, WHITE);

    /**< ROI title. */
    textout_centre_ex(page, title_font, "Acquired Image",
                      ROI_acquired_center.centerX, 10, RED, WHITE);

    /**< Network input title. */
    textout_centre_ex(page, title_font, "Input", input_center.centerX, 10,
                      RED, WHITE);
    /**< Network output title. */
    textout_centre_ex(page, title_font, "Output", x_center_output, 10,
                      RED, WHITE);

    /**< Properties label. */
    textout_ex(page, normal_font, property[CONTRAST], PROP_MRG,
               CAM_HEIGHT + CAM_MRG_TOP + 10, BLACK, WHITE);
    textout_ex(page, normal_font, property[BRIGHTNESS], PROP_MRG,
               CAM_HEIGHT + CAM_MRG_TOP + PROP_HEIGHT + 10,
               BLACK, WHITE);
    textout_ex(page, normal_font, property[SATURATION],
               2 * PROP_MRG + PROP_WIDTH,
               CAM_HEIGHT + CAM_MRG_TOP + 10, BLACK, WHITE);
    textout_ex(page, normal_font, property[SHARPNESS],
               2 * PROP_MRG + PROP_WIDTH,
               CAM_HEIGHT + CAM_MRG_TOP + PROP_HEIGHT + 10,
               BLACK, WHITE);

    /**< Select Model title. */
    textout_ex(page, title_font, "Select Active Model:",
               CAM_WIDTH + MODEL_MRG,
               CAM_HEIGHT + CAM_MRG_TOP + 15,
               RED, WHITE);

    /**< Arrow between ROI and input. */
    draw_arrow(page, x_arrow, ROI_acquired_center.centerY - ARROW_HEIGHT / 2,
               ROI_acquired_center.centerY + ARROW_HEIGHT / 2,
               ARROW_LENGHT, ARROW_WIDTH, ARROW_HEIGHT, BLACK);

    /**< Equal between input and output of the network. */
    draw_equal(page, x_equal, input_center.centerY + EQUAL_HEIGHT / 2,
               input_center.centerY - EQUAL_HEIGHT / 2,
               EQUAL_LENGHT, 4);

    /**< Draw model boxes. */
    rect(page, BTN_DIG_X, BTN_Y, BTN_DIG_X + BTN_WIDTH,
         BTN_Y + BTN_HEIGHT, BLACK);
    rect(page, BTN_LET_X, BTN_Y, BTN_LET_X + BTN_WIDTH,
         BTN_Y + BTN_HEIGHT, BLACK);
    rect(page, BTN_MIX_X, BTN_Y, BTN_MIX_X + BTN_WIDTH,
         BTN_Y + BTN_HEIGHT, BLACK);

    /**< Separetor captured-acquired. */
    fastline(page, CAM_WIDTH, 0,
             CAM_WIDTH, CAM_HEIGHT + CAM_MRG_TOP + 2 * PROP_HEIGHT + 5, BLACK);
    /**< Separetor acquired-selectModel. */
    fastline(page, 0, CAM_HEIGHT + CAM_MRG_TOP,
             WIN_WIDTH, CAM_HEIGHT + CAM_MRG_TOP, BLACK);
    /**< Separetor title-images. */
    fastline(page, 0, CAM_MRG_TOP, WIN_WIDTH, CAM_MRG_TOP, BLACK);
}

/**
* @brief Initialize all data structures used by the display task.
*
* Initialize all structure needed by the display task, from the video page to
* the mutex needed for the shared data structure. It also initilize the 
* coordinates of each content displayed on the screen.
*
* @return 0 if Succes, ERROR code otherwise
*/
int init_display() {

    /**< Allocate memory for the video memory pages. */
    video_page[0] = create_video_bitmap(SCREEN_W, SCREEN_H);
    if (video_page[0] == NULL)
        return DISPLAY_ERROR_CREATE_BITMAP;
    
    video_page[1] = create_video_bitmap(SCREEN_W, SCREEN_H);
    if (video_page[1] == NULL)
        return DISPLAY_ERROR_CREATE_BITMAP;

    /**< Color video page to full white. */
    clear_to_color(video_page[0], WHITE);
    clear_to_color(video_page[1], WHITE);

    /**< Load fonts. */
    normal_font = load_font(FONT_NOR, NULL, NULL);
    if (normal_font == NULL)
        return DISPLAY_ERROR_NO_FONT_FILE;

    title_font = load_font(FONT_TIT, NULL, NULL);
    if (title_font == NULL)
        return DISPLAY_ERROR_NO_FONT_FILE;

    /**< Set the ROI dimension and position to default values. */
    ROI_dim.centerX = CAM_WIDTH / 2;
    ROI_dim.centerY = CAM_MRG_TOP + CAM_HEIGHT / 2;
    ROI_dim.radius = ROI_MAX / 2;

    /**< Set the ROI extracted to default values. */
    ROI_acquired_center.centerX = 2 * ROI_dim.centerX + ROI_MRG + ROI_MAX / 2;
    ROI_acquired_center.centerY = ROI_dim.centerY;
    ROI_acquired_center.radius = ROI_MAX / 2;

    /**< Set the x coordinate of the input image. */
    x_input = ROI_acquired_center.centerX + ROI_MAX / 2 +
              ARROW_MRG + ARROW_LENGHT + IN_OUT_LABEL_MRG;

    /**< Set the input images dimension and position to default values. */
    input_center.centerX = x_input + INPUT_DIM / 2;
    input_center.centerY = ROI_dim.centerY;
    input_center.radius = INPUT_DIM / 2;

    /**< Set the x coordinate of the arrow. */
    x_arrow = ROI_acquired_center.centerX + ROI_MAX / 2 + ARROW_MRG;

    /**< Set the x coordinate of the equal. */
    x_equal = input_center.centerX + input_center.radius + EQUAL_MRG;

    /**< Set the x coordinate of the center of the output. */
    x_center_output = x_equal + EQUAL_LENGHT +
                        (WIN_WIDTH - x_equal - EQUAL_LENGHT) / 2;

    /**< Set the x and y coordinates of the properties. */
    prop_x_1 = PROP_MRG + PROP_OFFSET;
    prop_x_2 = 2 * PROP_MRG + PROP_WIDTH + PROP_OFFSET;
    prop_y_1 = CAM_HEIGHT + CAM_MRG_TOP + 10;
    prop_y_2 = CAM_HEIGHT + CAM_MRG_TOP + PROP_HEIGHT + 10;

    /**< Initilize the mutex variable. */
    pthread_mutex_init(&ROI_dim_mutex, NULL);
    pthread_mutex_init(&current_result_mutex, NULL);
    pthread_mutex_init(&ROI_image_mutex, NULL);

    /**< Allocate memory for the ROI image. */
    extracted_ROI.image = create_bitmap(ROI_MAX, ROI_MAX);
    if (extracted_ROI.image == NULL)
        return DISPLAY_ERROR_CREATE_BITMAP;
    extracted_ROI.radius = ROI_MAX / 2;

    /**< Allocate memory for the MLP data images. */
    display_nn_data[0].ROI = create_bitmap(ROI_MAX, ROI_MAX);
    if (display_nn_data[0].ROI == NULL)
        return DISPLAY_ERROR_CREATE_BITMAP;

    display_nn_data[0].input_image = create_bitmap(INPUT_DIM, INPUT_DIM);
    if (display_nn_data[0].input_image == NULL)
        return DISPLAY_ERROR_CREATE_BITMAP;

    display_nn_data[1].ROI = create_bitmap(ROI_MAX, ROI_MAX);
    if (display_nn_data[1].ROI == NULL)
        return DISPLAY_ERROR_CREATE_BITMAP;

    display_nn_data[1].input_image = create_bitmap(INPUT_DIM, INPUT_DIM);
    if (display_nn_data[1].input_image == NULL)
        return DISPLAY_ERROR_CREATE_BITMAP;

    /**< Color MLP data images to full white. */
    clear_to_color(display_nn_data[0].ROI, WHITE);
    clear_to_color(display_nn_data[0].input_image, WHITE);

    clear_to_color(display_nn_data[1].ROI, WHITE);
    clear_to_color(display_nn_data[1].input_image, WHITE);

    display_nn_data[0].image_radius = 0;

    /**< Initilize the MLP data results. */
    display_nn_data[0].result.rec_char = '\0';
    display_nn_data[0].result.prob = 0;

    /**< Allocate memory for the captured image. */
    captured_image = create_bitmap(CAM_WIDTH, CAM_HEIGHT);
    if (captured_image == NULL)
        return DISPLAY_ERROR_CREATE_BITMAP;

    return DISPLAY_SUCCESS;
}

/**
* @brief Displaying routine.
*
* Update all the contens in the screen, drawing the captured video frame, 
* the results of the MLP, the properties of the camera and the active model.
*
* @return 0 if Succes, ERROR code otherwise
*/
int draw_display() {

    /**< Local variable of global values needed to reduce criticl sections. */
    int new_acquired_image = 1;    /**< Index of MLP data. */
    int acq_radius_local;          /**< Local radius of ROI. */
    char rec_char[2], rec_prob[6]; /**< MLP result. */
    network_target green_model;    /**< Active model. */
    int actual_property;           /**< Cam properties. */

    /**< Default button color. */
    int model_color[3] = {BLACK, BLACK, BLACK};

    int color;                          /**< Captured buffer pixel color. */
    int x_1, x_2, y_1, y_2, diameter;   /**< Coordinates and dim of ROI. */

    int white_color = WHITE;
    int black_color = BLACK;

    int i, j;                           /**< Loop counter. */

    int show_video_result;              /**< Returning result of Show_video. */

    /**< Auxiliar pointer. */
    BITMAP *display = video_page[current_page];
    clear_to_color(display, WHITE);

    /**< Draw the skeleton structure. */
    draw_fixed(display);

    /**< Access the frame buffer and copy the pixels values in the */
    /*  capture BITMAP imasge. */
    pthread_mutex_lock(&capture_buffer_mutex);

    for (i = 0; i < CAM_HEIGHT; ++i) {
        for (j = 0; j < CAM_WIDTH; ++j) {
            color = (capture_buffer[i * CAM_WIDTH + j] >= 120) 
                                                ? white_color : black_color;
            putpixel(captured_image, j, i, color);
        }
    }

    pthread_mutex_unlock(&capture_buffer_mutex);

    /**< Save the ROI dimension and position.*/
    pthread_mutex_lock(&ROI_dim_mutex);

    x_1 = ROI_dim.centerX - ROI_dim.radius;
    x_2 = ROI_dim.centerX + ROI_dim.radius;
    y_1 = ROI_dim.centerY - ROI_dim.radius;
    y_2 = ROI_dim.centerY + ROI_dim.radius;
    diameter = ROI_dim.radius * 2;

    pthread_mutex_unlock(&ROI_dim_mutex);

    /**< Copy the captured image in the display page.*/
    blit(captured_image, display, 0, 0, 0, CAM_MRG_TOP, CAM_WIDTH, CAM_HEIGHT);

    /**< Draw the ROI sqaure on the camera preview.*/
    for (i = 1; i <= ROI_DEPTH; ++i)
        rect(display, x_1 - i, y_1 - i,
                x_2 + i, y_2 + i, RED);

    /**< Copy the extracted ROI and its radius in the global variable.*/
    pthread_mutex_lock(&ROI_image_mutex);

    blit(captured_image, extracted_ROI.image, x_1, y_1 - CAM_MRG_TOP,
            0, 0, diameter, diameter);

    extracted_ROI.radius = diameter / 2;

    pthread_mutex_unlock(&ROI_image_mutex);

    /**< Access the current result of the MLP.*/
    pthread_mutex_lock(&current_result_mutex);

    acq_radius_local = display_nn_data[current_result].image_radius;

    /**< Display the acquired ROI image. */
    blit(display_nn_data[current_result].ROI, display, 0, 0,
            ROI_acquired_center.centerX - acq_radius_local,
            ROI_acquired_center.centerY - acq_radius_local,
            2 * acq_radius_local, 2 * acq_radius_local);

    /**< Display the input image. */
    blit(display_nn_data[current_result].input_image, display, 0, 0,
            input_center.centerX - input_center.radius,
            input_center.centerY - input_center.radius,
            2 * input_center.radius, 2 * input_center.radius);

    /**< Recognized character. */
    rec_char[0] = display_nn_data[current_result].result.rec_char;

    /**< Percentage of recognition. */
    sprintf(rec_prob, "%2.2f", display_nn_data[current_result].result.prob);

    pthread_mutex_unlock(&current_result_mutex);

    /**< Highlight the ROI.*/
    rect(display,
            ROI_acquired_center.centerX - acq_radius_local,
            ROI_acquired_center.centerY - acq_radius_local,
            ROI_acquired_center.centerX + acq_radius_local,
            ROI_acquired_center.centerY + acq_radius_local,
            RED);

    rec_char[1] = '\0';
    rec_prob[4] = '%';
    rec_prob[5] = '\0';

    /**< Write the MLP result.*/
    textout_centre_ex(display, normal_font, rec_char,
                        x_center_output,
                        input_center.centerY - 20, BLACK, WHITE);

    textout_centre_ex(display, normal_font, rec_prob,
                        x_center_output,
                        input_center.centerY + 20, BLACK, WHITE);

    /**< Load capturing property. */
    pthread_mutex_lock(&contrast_mutex);
    actual_property = contrast_value;
    pthread_mutex_unlock(&contrast_mutex);

    sprintf(property_value[CONTRAST], "%d", actual_property);
    /**< Contrast value. */
    textout_ex(display, normal_font, property_value[CONTRAST],
                prop_x_1, prop_y_1, BLACK, WHITE);

    pthread_mutex_lock(&brightness_mutex);
    actual_property = brightness_value;
    pthread_mutex_unlock(&brightness_mutex);

    sprintf(property_value[BRIGHTNESS], "%d", actual_property);
    /**< Brightness value. */
    textout_ex(display, normal_font, property_value[BRIGHTNESS],
                prop_x_1, prop_y_2, BLACK, WHITE);

    pthread_mutex_lock(&saturation_mutex);
    actual_property = saturation_value;
    pthread_mutex_unlock(&saturation_mutex);

    sprintf(property_value[SATURATION], "%d", actual_property);
    /**< Saturation value. */
    textout_ex(video_page[current_page], normal_font, property_value[SATURATION],
                prop_x_2, prop_y_1, BLACK, WHITE);

    pthread_mutex_lock(&sharpness_mutex);
    actual_property = sharpness_value;
    pthread_mutex_unlock(&sharpness_mutex);

    sprintf(property_value[SHARPNESS], "%d", actual_property);
    /**< Sharpness value. */
    textout_ex(display, normal_font, property_value[SHARPNESS],
                prop_x_2, prop_y_2, BLACK, WHITE);

    /**< Check the active model and make it green. */
    pthread_mutex_lock(&actual_model_mutex);
    green_model = requested_model;
    pthread_mutex_unlock(&actual_model_mutex);

    model_color[green_model] = GREEN;

    /**< Draw text boxes. */
    textout_centre_ex(display, title_font, "DIGITS", BTN_DIG_X + BTN_WIDTH / 2,
                        BTN_Y + 5, model_color[DIGITS], WHITE);

    textout_centre_ex(display, title_font, "LETTERS", BTN_LET_X + BTN_WIDTH / 2,
                        BTN_Y + 5, model_color[LETTERS], WHITE);

    textout_centre_ex(display, title_font, "MIXED", BTN_MIX_X + BTN_WIDTH / 2,
                        BTN_Y + 5, model_color[MIXED], WHITE);

    /**< Show the current video page on the screen. */
    show_video_result = show_video_bitmap(display);
    if (show_video_result != 0)
        return DISPLAY_ERROR_SHOW_VIDEO;

    /**< Update the current video page. */
    current_page = (current_page + 1) % 2;

    return DISPLAY_SUCCESS;
}

/**
* @brief Deallocate memory allocated.
*/
void free_display() {
    destroy_font(normal_font);
    destroy_font(title_font);

    destroy_bitmap(extracted_ROI.image);
    destroy_bitmap(display_nn_data[0].ROI);
    destroy_bitmap(display_nn_data[1].ROI);
    destroy_bitmap(display_nn_data[0].input_image);
    destroy_bitmap(display_nn_data[1].input_image);
    destroy_bitmap(captured_image);
}
