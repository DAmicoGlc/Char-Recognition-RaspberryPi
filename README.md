# Char-Recognition-RaspberryPi
MLP network on a Raspberry Pi with camera module to recognize Hand written character!

# Dependencies
- Allegro 4 (https://www.allegro.cc/files/?v=4.4)
- Userland (https://github.com/raspberrypi/userland.git /git/raspberrypi/userland)

# Install & Run
```bash
sudo apt install liballegro4-dev
sudo git clone https://github.com/raspberrypi/userland.git /git/raspberrypi/userland
cd /git/raspberrypi/userland
./buildme
sudo git clone https://github.com/DAmicoGlc/Char-Recognition-RaspberryPi charRecognition
cd charRecognition/
mkdir objs
make
./hand_written_recognition
```

# User interaction

| Key          | Action                 |
| ------------ | ---------------------- |
| Arrow Up     | Move the ROI up        |
| Arrow Down   | Move the ROI down      |
| Arrow Left   | Move the ROI left      |
| Arrow Right  | Move the ROI right     |
| +            | Increase ROI dimension |
| -            | Decrease ROI dimension |
| c            | Increase Contrast      |
| x            | Decrease Contrast      |
| b            | Increase Brightness    |
| v            | Decrease Brightness    |
| s            | Increase Sharpness     |
| a            | Decrease Sharpness     |
| f            | Increase Saturation    |
| d            | Decrease Saturation    |

# Stand alone MLP
It is a training and testing files using the EMNIST (Cohen G., Afshar S., Tapson J., \& van Schaik A. (2017). EMNIST: an extension of MNIST to handwritten letters. https://www.nist.gov/node/1298471/emnist-dataset) datasets. To compile and run it:
```bash
cd NN
make
./NN h n...n i b l e m
```
Where:
-h = (# of hidden layer) 
-n...n = (# of neuron in first hidden layer)...(# of neuron in last hidden layer) 
-i = (Training iteration, e.g. Epoches) 
-b = (Size of mini batch to compute sthocastic gradient) 
-l = (Learning rate of training phase) 
-e = (epsilon of Error in training phase) 
-m = (momentum of training phase)
All parameter must be grather than 0. The learning rate must be also less than 1.

Example for a 2 hidden layer whit 64 neurons each:
```bash
./NN 2 64 64 100 10 0.5 0.2 0.9
```

Notice that the size of the input and output must be changed in the source code, depending on the datasets!
## Note
If the installation of userland returns error about "mmal_status_to_int" not found, try to past the following code:
```bash
int mmal_status_to_int(MMAL_STATUS_T status)
{
   if (status == MMAL_SUCCESS)
      return 0;
   else
   {
      switch (status)
      {
      case MMAL_ENOMEM :
         vcos_log_error("Out of memory");
         break;
      case MMAL_ENOSPC :
         vcos_log_error("Out of resources (other than memory)");
         break;
      case MMAL_EINVAL:
         vcos_log_error("Argument is invalid");
         break;
      case MMAL_ENOSYS :
         vcos_log_error("Function not implemented");
         break;
      case MMAL_ENOENT :
         vcos_log_error("No such file or directory");
         break;
      case MMAL_ENXIO :
         vcos_log_error("No such device or address");
         break;
      case MMAL_EIO :
         vcos_log_error("I/O error");
         break;
      case MMAL_ESPIPE :
         vcos_log_error("Illegal seek");
         break;
      case MMAL_ECORRUPT :
         vcos_log_error("Data is corrupt \attention FIXME: not POSIX");
         break;
      case MMAL_ENOTREADY :
         vcos_log_error("Component is not ready \attention FIXME: not POSIX");
         break;
      case MMAL_ECONFIG :
         vcos_log_error("Component is not configured \attention FIXME: not POSIX");
         break;
      case MMAL_EISCONN :
         vcos_log_error("Port is already connected ");
         break;
      case MMAL_ENOTCONN :
         vcos_log_error("Port is disconnected");
         break;
      case MMAL_EAGAIN :
         vcos_log_error("Resource temporarily unavailable. Try again later");
         break;
      case MMAL_EFAULT :
         vcos_log_error("Bad address");
         break;
      default :
         vcos_log_error("Unknown status error");
         break;
      }

      return 1;
   }
}
```
In the file RaspiCamControl.c of the userland library and:
```bash
int mmal_status_to_int(MMAL_STATUS_T status);
```
In the related header file RaspiCamControl.h.
