#EasyDLROS, a new framework for easy deployment of pre-trained DL models on robots.
All the experiments are conducted on a Nvidia TX2 device, which is ideal for industrial robots and medical equipment. A Logitech C925e webcam have been attached to the TX2 devics capture images.

To run DL models using this framework first, the following preconditions need to be fulfilled.

##Preconditions
 1. Install Jetpack >4.2.2 from https://developer.nvidia.com/embedded/jetpack

 2. Install jetson-inference project: 
 
        $ cd ~
        $ sudo apt-get install git cmake
        $ git clone --recursive https://github.com/dusty-nv/jetson-inference
        $ cd jetson-inference
        $ mkdir build
        $ cd build
        $ cmake ../
        $ make
        $ sudo make install

 3. ROS Melodic
 
        $ sudo apt-get install ros-melodic-image-transport
        $ sudo apt-get install ros-melodic-image-publisher
        $ sudo apt-get install ros-melodic-vision-msgs

 4. Catkin Workspace

    - Create a Catkin workspace (~/catkin_ws) using the steps mentioned in the following tutorial:
    - http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment#Create_a_ROS_Workspace

 5. Navigate into the catkin workspace and clone and build ros_deep_learning:

        $ cd ~/catkin_ws/src
        $ git clone https://github.com/dusty-nv/ros_deep_learning
        $ cd ../
        $ catkin_make

6. Store trained model (.param and .json file), sysset.txt and live_image_recognition.py file in a  directory.
7. Update image path in live_image_recognition.py file if necessary.


##How to run DL inference on a live video stream
In this section, image classification will be done when one image is captured by the robot camera. This is a live event. Based on the classification results the robot takes a future decision. The decision could be to stop, turn, or continue the robot.


 1. Make sure that roscore is running:
 
     - roscore

 2. Capture image data and stream the data from the image node to the deep learning node:

    - Open another terminal and run "rosrun cv_camera cv_camera_node"
    - cv_camera_node will capture image messages from the camera installed in a robot.

 3. Run DL model and classify images

    - open a new terminal and run "python live_image_recognition.py image:=/cv_camera/image_raw"
    - live_image_recognition.py will receive image data from the camera node, makes image data suitable for the deep learning model, and finally classify the image.
    - Classification results will be published after every classification
    
 4. Display the results

    - Open another terminal and run "rostopic echo /result"
    - image classification result will be shown on the terminal in a one-second interval


## How to run DL inference on images captured/stored by the camera node
 
If a robot is not capable to run DL inference onboard then, that robot can use the camera node to store the images and then share the images with larger computing architecture to perform image classification. In this section, image classification will be done on images captured and stored by the camera.


 1. Make sure that roscore is running:
 
     - roscore

 2. Capture image data and stream the data from the image node to the deep learning node:

    - Open another terminal and run "rosrun cv_camera cv_camera_node"
    - cv_camera_node will capture image messages from the camera installed in a robot.
    
 3. Convert image message data to cv formate and store image in a local hard drive:
    - open a new terminal and run "python live_image_recognition.py image:=/cv_camera/image_raw"
    - update base_path in image_capture.py file if necessary

 4. Run DL model and classify images

    - open a new terminal and run "python saved_image_recognition.py image:=/cv_camera/image_raw"
    - saved_image_recognition.py will use saved images for the input of DL model and finally classify the image.
    - Classification results will be published after every classification
    
 5. Display the results

    - Open another terminal and run "rostopic echo /result"
    - image classification result will be shown on the terminal in a one-second interval


## Measure DL model accuracy on images captured by robot camera node

The accuracy_measure_mxnet.py can measure the accuracy of a DL model. This script uses images that are stored on a hard drive. This script also reports the average run time per image. 