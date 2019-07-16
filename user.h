#ifndef USER_H
#define USER_H

/**
* @file user.h
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

#include "common.h"

/**
* GLOBAL FUNCTIONS
*/

/**< Handle keyboard interactions.*/
int key_pressed(int key);

/**< Handle mouse interactions.*/
void mouse_touch();

#endif