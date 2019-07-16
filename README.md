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
If the installation of userland returns error about 
