/**
* @file user.c
* @author Gianluca D'Amico
* @brief File containing user interaction function
*
* HANDLING USER INTERACTION: It manages all possible interaction between user
* and application.
*
* Keyboard interaction:
*   - 'X': decrease contranst of the camera component;
*   - 'C': increase contranst of the camera component;
*
*   - 'V': decrease brightness of the camera component;
*   - 'B': increase brightness of the camera component;
*
*   - 'D': decrease saturation of the camera component;
*   - 'F': increase saturation of the camera component;
*
*   - 'A': decrease sharpness of the camera component;
*   - 'S': increase sharpness of the camera component;
*
*   - ARROWS: move the ROI on the captured image;
*   - '+': increase dimension of ROI, moving it in the center;
*   - '-': decrease dimension of ROI, moving it in the center;
*
*   - 'ESC': close the application.
*
* Mouse interaction: it is possible to click on the model button to change 
* the active model of the neural network.
*
*/

#include <allegro.h>

#include "raspi_cam.h"
#include "nn_handler.h"
#include "display.h"
#include "user.h"


/**
* GLOBAL FUNCTIONS
*/

/**
* @brief Handle keyboard interactions.
*
* - 'X': decrease contranst of the camera component;
* - 'C': increase contranst of the camera component;
*
* - 'V': decrease brightness of the camera component;
* - 'B': increase brightness of the camera component;
*
* - 'D': decrease saturation of the camera component;
* - 'F': increase saturation of the camera component;
*
* - 'A': decrease sharpness of the camera component;
* - 'S': increase sharpness of the camera component;
*
* - ARROWS: move the ROI on the captured image;
* - '+': increase dimension of ROI, moving it in the center;
* - '-': decrease dimension of ROI, moving it in the center;
*
* - 'ESC': close the application.
*
* @param key Keyboard key pressed by the user.
* @return 1 if ESC is pressed, 0 otherwise.
*/
int key_pressed(int key) {

    int end = 0;        /**< Variable to check if ESC is pressed. */

    int property_local; /**< Local property value. */

    switch(key) 
    {
        case KEY_ESC:
            end++;
            break;
        case KEY_X:
            /**< Lock the global contrast value of the camera and
             * save it in the local variable. */
            pthread_mutex_lock(&contrast_mutex);
            property_local = contrast_value;
            pthread_mutex_unlock(&contrast_mutex);

            /**< The decrement is of 5 unit, so check if it is grater or equal 
             * to 5 before decrease it, otherwise make no action. */
            if (property_local >= 5) {
                property_local -= 5;

                /**< Lock the global contrast value of the camera and
                * update it. */
                pthread_mutex_lock(&contrast_mutex);
                contrast_value = property_local;
                pthread_mutex_unlock(&contrast_mutex);
            }
            break;
        case KEY_C:
            /**< Lock the global contrast value of the camera and
             * save it in the local variable. */
            pthread_mutex_lock(&contrast_mutex);
            property_local = contrast_value;
            pthread_mutex_unlock(&contrast_mutex);
            if (property_local <= 95) {
                property_local += 5;
                pthread_mutex_lock(&contrast_mutex);
                contrast_value = property_local;
                pthread_mutex_unlock(&contrast_mutex);
            }
            break;
        case KEY_V:
            /**< Lock the global contrast value of the camera and
             * save it in the local variable. */
            pthread_mutex_lock(&brightness_mutex);
            property_local = brightness_value;
            pthread_mutex_unlock(&brightness_mutex);
            if (property_local >= 5) {
                property_local -= 5;
                pthread_mutex_lock(&brightness_mutex);
                brightness_value = property_local;
                pthread_mutex_unlock(&brightness_mutex);
            }
            break;
        case KEY_B:
            /**< Lock the global contrast value of the camera and
             * save it in the local variable. */
            pthread_mutex_lock(&brightness_mutex);
            property_local = brightness_value;
            pthread_mutex_unlock(&brightness_mutex);
            if (property_local <= 95) {
                property_local += 5;
                pthread_mutex_lock(&brightness_mutex);
                brightness_value = property_local;
                pthread_mutex_unlock(&brightness_mutex);
            }
            break;
        case KEY_D:
            /**< Lock the global contrast value of the camera and
             * save it in the local variable. */
            pthread_mutex_lock(&saturation_mutex);
            property_local = saturation_value;
            pthread_mutex_unlock(&saturation_mutex);
            if (property_local >= 5) {
                property_local -= 5;
                pthread_mutex_lock(&saturation_mutex);
                saturation_value = property_local;
                pthread_mutex_unlock(&saturation_mutex);
            }
            break;
        case KEY_F:
            /**< Lock the global contrast value of the camera and
             * save it in the local variable. */
            pthread_mutex_lock(&saturation_mutex);
            property_local = saturation_value;
            pthread_mutex_unlock(&saturation_mutex);
            if (property_local <= 95) {
                property_local += 5;
                pthread_mutex_lock(&saturation_mutex);
                saturation_value = property_local;
                pthread_mutex_unlock(&saturation_mutex);
            }
            break;
        case KEY_A:
            /**< Lock the global contrast value of the camera and
             * save it in the local variable. */
            pthread_mutex_lock(&sharpness_mutex);
            property_local = sharpness_value;
            pthread_mutex_unlock(&sharpness_mutex);
            if (property_local >= 5) {
                property_local -= 5;
                pthread_mutex_lock(&sharpness_mutex);
                sharpness_value = property_local;
                pthread_mutex_unlock(&sharpness_mutex);
            }
            break;
        case KEY_S:
            /**< Lock the global contrast value of the camera and
             * save it in the local variable. */
            pthread_mutex_lock(&sharpness_mutex);
            property_local = sharpness_value;
            pthread_mutex_unlock(&sharpness_mutex);
            if (property_local <= 95) {
                property_local += 5;
                pthread_mutex_lock(&sharpness_mutex);
                sharpness_value = property_local;
                pthread_mutex_unlock(&sharpness_mutex);
            }
            break;
        case KEY_LEFT:
            pthread_mutex_lock(&ROI_dim_mutex);
            if ((ROI_dim.centerX - 2 - ROI_DEPTH) - 
                                            ROI_dim.radius >= 0)
                ROI_dim.centerX -= 2;
            pthread_mutex_unlock(&ROI_dim_mutex);
            break;
        case KEY_RIGHT:
            pthread_mutex_lock(&ROI_dim_mutex);
            if ((ROI_dim.centerX + 2 + ROI_DEPTH) + 
                                    ROI_dim.radius <= CAM_WIDTH)
                ROI_dim.centerX += 2;
            pthread_mutex_unlock(&ROI_dim_mutex);
            break;
        case KEY_UP:
            pthread_mutex_lock(&ROI_dim_mutex);
            if ((ROI_dim.centerY - 2 - ROI_DEPTH) - 
                                    ROI_dim.radius >= CAM_MRG_TOP)
                ROI_dim.centerY -= 2;
            pthread_mutex_unlock(&ROI_dim_mutex);
            break;
        case KEY_DOWN:
            pthread_mutex_lock(&ROI_dim_mutex);
            if ((ROI_dim.centerY + 2 + ROI_DEPTH) + 
                        ROI_dim.radius <= CAM_HEIGHT + CAM_MRG_TOP)
                ROI_dim.centerY += 2;
            pthread_mutex_unlock(&ROI_dim_mutex);
            break;
        case KEY_PLUS_PAD:
        case 65:
            pthread_mutex_lock(&ROI_dim_mutex);
            if (ROI_dim.radius < ROI_MAX/2) {
                ROI_dim.radius    *= 2;
            }
            ROI_dim.centerX = CAM_WIDTH/2;
            ROI_dim.centerY = CAM_MRG_TOP + CAM_HEIGHT/2;
            pthread_mutex_unlock(&ROI_dim_mutex);

            break;
        case KEY_MINUS_PAD:
        case 61:
            pthread_mutex_lock(&ROI_dim_mutex);
            if (ROI_dim.radius > ROI_MIN/2) {
                ROI_dim.radius    /= 2;
            }
            ROI_dim.centerX = CAM_WIDTH/2;
            ROI_dim.centerY = CAM_MRG_TOP + CAM_HEIGHT/2;
            pthread_mutex_unlock(&ROI_dim_mutex);
            break;
        default:
            break;
    }

    return end;
}

void mouse_touch() {
    int i;

    if (mouse_y >= BTN_Y && mouse_y <= (BTN_Y + BTN_HEIGHT) ) {

        if (mouse_x >= BTN_DIG_X && mouse_x <= (BTN_DIG_X + BTN_WIDTH)) {
            pthread_mutex_lock(&actual_model_mutex);
            requested_model = DIGITS;
            pthread_mutex_unlock(&actual_model_mutex);
        }

        if (mouse_x >= BTN_LET_X && mouse_x <= (BTN_LET_X + BTN_WIDTH)) {
            pthread_mutex_lock(&actual_model_mutex);
            requested_model = LETTERS;
            pthread_mutex_unlock(&actual_model_mutex);
        }

        if (mouse_x >= BTN_MIX_X && mouse_x <= (BTN_MIX_X + BTN_WIDTH)) {
            pthread_mutex_lock(&actual_model_mutex);
            requested_model = MIXED;
            pthread_mutex_unlock(&actual_model_mutex);
        }
    }
}