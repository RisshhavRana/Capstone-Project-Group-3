# Capstone-Project---Group-3

## Drowsiness Detection in Truck Drivers
 
This repository contains Python code for drowsiness detection and gaze analysis using computer vision techniques and the dlib library. The code can be used to monitor a person's eye and head movements to determine their drowsiness level and gaze direction.

* Requirements:<br />
Before running the code, make sure you have the following installed:<br />
1. Python (>= 3.11)<br />
2. OpenCV (cv2)<br />
3. NumPy<br />
4. pandas<br />
5. dlib<br />
6. imutils<br />
7. pygame

* Setup<br />
Clone the repository to your local machine using Git:<br />
git clone https://github.com/your-username/your-repo-name.git

* Install the required Python packages using pip:<br />
pip install opencv-python numpy pandas dlib imutils pygame

* Download the shape_predictor_68_face_landmarks.dat file from the dlib model repository and place it in the same directory as the code.

* Usage
  Run the below Python script:<br />
  python drowsiness_detection.py


* How does code works?<br />
The code uses your computer's camera to capture live video frames and performs the following tasks:<br />
1. Detects faces in the video stream using the dlib face detector.<br />
2. Detects facial landmarks (e.g., eye corners, mouth corners) using the dlib shape predictor.<br />
3. Calculates the Eye Aspect Ratio (EAR) to detect blinking.<br />
4. Determines if the person is yawning based on the Mouth Aspect Ratio (MAR).<br />
5. Detects nodding based on the head's pitch angle.<br />
6. Estimates the direction of the gaze (left, right, or center) based on the position of the eyes.<br />
7. Monitors the driver's status and saves the status and timestamps in an Excel file (drowsy_data.xlsx).<br />

* The driver's status is categorized as follows:<br />
 1. Active: The driver is attentive and not showing any signs of drowsiness.<br />
 2. Drowsy: The driver is slightly drowsy or fatigued.<br />
 3. Sleeping: The driver's eyes are closed for an extended period, indicating possible sleepiness.<br />
 4. Yawning: The driver is yawning, which might indicate fatigue.<br />
 5. Gazing Left/Right: The driver's gaze is focused to the left or right, which might indicate distraction.<br />
 6. Looking Forward: The driver's gaze is looking forward, indicating attentiveness.<br />
 7. The code will display the driver's status and other relevant information on the video stream.<br />

* Notes:<br />
1. For yawning detection, the code uses the Mouth Aspect Ratio (MAR) to determine if the mouth is open widely, resembling a yawn.<br />
2. The Gaze Detection function estimates the gaze direction based on the ratio of sclera (white part of the eyes) visible on the left and right sides.<br />
3. The code will also play an alarm sound if the driver is detected as drowsy or distracted.<br />
4. Please use this code responsibly and avoid using it while driving. It's essential to prioritize safety on the roads.<br />
