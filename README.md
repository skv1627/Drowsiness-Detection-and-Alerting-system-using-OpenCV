# Drowsiness and Yawning Detection using Facial Landmarks

This repository contains a Python script for detecting drowsiness and yawning in real time using a webcam and facial landmarks analysis. It leverages computer vision libraries and the dlib library for face detection and facial landmark prediction.

## Getting Started

These instructions will help you set up and run the code on your local machine.

### Prerequisites

- Python (3.7 or higher)
- Required Python libraries: pygame, scipy, imutils, dlib, cv2 (OpenCV)

### Installing

1. Clone the repository to your local machine:  https://github.com/skv1627/Drowsiness-Detection-and-Alerting-system-using-OpenCV


2. Install the required Python libraries if you haven't already:
 
   pygame, scipy, imutils, dlib, cv2 (OpenCV)


### Running the Code

1. Execute the main script:  Driver Drowsiness Detection.py


2. The script will use your webcam to detect drowsiness and yawning in real time.

### Results
a. Detecting Drowsiness State of the Driver
![Screenshot (291)](https://github.com/skv1627/Drowsiness-Detection-and-Alerting-system-using-OpenCV/assets/146156111/fd32f171-724c-48e5-b4fa-e6fb3b9da132)

b. Detecting the Yawning State of the Driver
![Screenshot (292)](https://github.com/skv1627/Drowsiness-Detection-and-Alerting-system-using-OpenCV/assets/146156111/2a32e4b3-f386-4683-98da-ed822f7e94fa)

c. Detecting whether the Driver is actively looking at the road or not

![Screenshot (293)](https://github.com/skv1627/Drowsiness-Detection-and-Alerting-system-using-OpenCV/assets/146156111/3ae02c90-7871-4d77-ad1e-fdd621694d42)


### Model Architecture

The code uses the dlib library's pre-trained facial landmark predictor ("shape_predictor_68_face_landmarks.dat") to identify facial landmarks, including those for the eyes and mouth. It calculates the Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) to detect drowsiness and yawning.

### Training Paradigms

- The model uses the Histogram of Oriented Gradients (HOG) face detector for face detection.
- It calculates EAR based on the Euclidean distances between specific eye landmarks.
- It calculates MAR based on the Euclidean distances between specific mouth landmarks.
- Thresholds and consecutive frames criteria are used to trigger alarms for drowsiness and yawning.


## Acknowledgments

- The code makes use of various open-source libraries and utilities.
- Inspiration for this project came from the need for real-time drowsiness and yawning detection, particularly for driver monitoring systems.

## Authors
## Team TechRoos

- Swarna Kumar Vusa  (https://github.com/skv1627)
- Sandeep Chanda  (https://github.com/chsandeep8)
- Kalyan Kumar Sukasi (https://github.com/kalyansukasi)
- Archana Vootukuru ( https://github.com/archu1012)

   
   
  


