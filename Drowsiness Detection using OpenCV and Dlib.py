import cv2
import numpy as np
import pandas as pd
import dlib
from imutils import face_utils
import pygame
import time

# Initializing the camera and taking the instance
cap = cv2.VideoCapture('HarshilNight2.mp4')

# Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Status marking for current state
sleep = 0
drowsy = 0
active = 0
Right_Gaze = 0
Left_Gaze = 0
Centre_Gaze = 0
nodding = 0
status = ""
status_data = []
color = (0, 0, 0)
alarm_active = False
yawning_status = 0
 

###########################################################
#    Defining Function to Compute Distance b/w 2 Points   #

def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist


###########################################################
#       Defining Function to Detect Blinking of Eye       #

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    # Checking the eyes blink frequency/duration
    if ratio > 0.28:
        return 2
    elif ratio > 0.18 and ratio <= 0.28:
        return 1
    else:
        return 0


###########################################################
#     Defining Function for Yawn Detection Using MAR      #

# def yawn(a, b, c, d, e, f, g, h):
#     up = compute(b, e) + compute(c, f) + compute(d, g)
#     down = compute(a, h)
#     ratio_yawn = up / (2.0 * down)

def yawn(mouth_landmarks):
    a = compute(mouth_landmarks[3], mouth_landmarks[9])
    b = compute(mouth_landmarks[2], mouth_landmarks[10])
    c = compute(mouth_landmarks[4], mouth_landmarks[8])
    d = compute(mouth_landmarks[0], mouth_landmarks[6])
    ratio_yawn = (a + b + c) / (2.0 * d)

    # Checking if the driver yawned
    if ratio_yawn > 0.75:
        return 1
    else:
        return 0

###########################################################
#         Defining Function to Compute EAR Ratio          #

def compute_ear(eye_landmarks):
    a = compute(eye_landmarks[1], eye_landmarks[5])
    b = compute(eye_landmarks[2], eye_landmarks[4])
    c = compute(eye_landmarks[0], eye_landmarks[3])
    ear = (a + b) / (2.0 * c)
    return ear


###########################################################
#         Defining Function to Compute MAR Ratio          #

def compute_mar(mouth_landmarks):
    a = compute(mouth_landmarks[3], mouth_landmarks[9])
    b = compute(mouth_landmarks[2], mouth_landmarks[10])
    c = compute(mouth_landmarks[4], mouth_landmarks[8])
    d = compute(mouth_landmarks[0], mouth_landmarks[6])
    mar = (a + b + c) / (2.0 * d)
    return mar


#############################################################################
#       Defining Function to get Ratios of Eye For Gaze Detection           #

# Splitting the eye in two parts and find out in which of the two parts there is more sclera visible.

def get_gaze_ratio(eye_points, facial_landmarks):
    # Gaze detection
    eye_region = np.array([(facial_landmarks[eye_points[0]][0], facial_landmarks[eye_points[0]][1]),
                                (facial_landmarks[eye_points[1]][0], facial_landmarks[eye_points[1]][1]),
                                (facial_landmarks[eye_points[2]][0], facial_landmarks[eye_points[2]][1]),
                                (facial_landmarks[eye_points[3]][0], facial_landmarks[eye_points[3]][1]),
                                (facial_landmarks[eye_points[4]][0], facial_landmarks[eye_points[4]][1]),
                                (facial_landmarks[eye_points[5]][0], facial_landmarks[eye_points[5]][1])], np.int32)

    height, width, _ = frame.shape

    # creating the mask to extract exactly the inside of the left eye and exclude all the surroundings.

    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    # Extract the eye from the face and put it on its own window.
    # Only we need to keep in mind that we can only cut out rectangular shapes from the image,
    # so we take all the extremes points of the eyes to get the rectangle.
    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])
    gray_eye = eye[min_y: max_y, min_x: max_x]

    # Threshold to separate iris and pupil from the white part of the eye.
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

    # Divide the eye into 2 parts: left_side and right_side.
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white

    return gaze_ratio

# Defining Functions to get Mid Points and Ratios of Eye For Gaze Detection #
#############################################################################


##########################################################################
##            Defining Function to Play and Stop Alarm                  ##

def play_alarm_sound():
    global alarm_active
    if not alarm_active:
        pygame.mixer.music.load("Alarm.wav")
        pygame.mixer.music.play()
        alarm_active = True

def stop_alarm_sound():
    global alarm_active
    if alarm_active:
        pygame.mixer.music.stop()
        alarm_active = False

pygame.init()
pygame.mixer.init()

##            Defining Function to Play and Stop Alarm                  ##
##########################################################################


##########################################################################
# Initializing the skip_interval to process every 3rd frame
skip_interval = 2

##########################################################################

##########################################################################
##                       Setting Up the Frame                           ##

while True:
    _, frame = cap.read()

         # Apply frame skipping
    if (int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1) % skip_interval != 0:
        continue  # Skip this frame

    flipped_frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    


    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = flipped_frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        shape = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        mouth_landmarks = landmarks[48:68]
        lips_yawn = yawn(mouth_landmarks)
        
##                       Setting Up the Frame                           ##
##########################################################################


##########################################################################
##                     Code for NODDING Detection                       ##

        # Extract the image points from the facial landmarks
        image_points = np.array([
            (shape.part(30).x, shape.part(30).y),     # Nose tip
            (shape.part(8).x, shape.part(8).y),       # Chin
            (shape.part(36).x, shape.part(36).y),     # Left eye left corner
            (shape.part(45).x, shape.part(45).y),     # Right eye right corner
            (shape.part(48).x, shape.part(48).y),     # Left Mouth corner
            (shape.part(54).x, shape.part(54).y)      # Right mouth corner
        ], dtype="double")

        # Define the 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        # Define the camera matrix
        focal_length = flipped_frame.shape[1]
        center = (flipped_frame.shape[1] / 2, flipped_frame.shape[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        # Define the distortion coefficients
        dist_coeffs = np.zeros((4, 1))

        # Solve the PnP problem
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        # Extract the pitch angle from the rotation vector
        pitch_angle_degrees = float(rotation_vector[0])
        pitch_angle_degrees = abs(round(pitch_angle_degrees, 1))

        if pitch_angle_degrees < 3:
            nodding += 1
            if nodding < 45:
                node_status = "Nodding!"
                 # Append the Sleeping status and timestamp to the list
                status_data.append([node_status, time.strftime("%Y-%m-%d %H:%M:%S")])
                cv2.putText(flipped_frame, node_status, (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

            elif nodding >= 45:
                node_status = "Nodding! Feeling Drowsy!!!"
                # Append the Sleeping status and timestamp to the list
                status_data.append([node_status, time.strftime("%Y-%m-%d %H:%M:%S")])    
                cv2.putText(flipped_frame, node_status, (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                play_alarm_sound()
            else:
                stop_alarm_sound()
                
        else:
            nodding = 0
                
        df = pd.DataFrame(status_data, columns=['Status', 'Timestamp'])
        df.to_excel('drowsy_data.xlsx', index=False)
            
        # Display Nod Values in frame
        cv2.putText(flipped_frame,"Nod Angle: {:.1f}".format(pitch_angle_degrees), (460, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 2)

##                     Code for NODDING Detection                       ##
##########################################################################


#########################################################################
##   Code for Detecting Active, Sleeping, Drowsy and Yawning Status    ##

        if left_blink == 0 and right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            yawning_status = 0
            if sleep > 20 and sleep < 40:
                status = "EYES CLOSED !!!"
                color = (0, 128, 0)
                cv2.putText(flipped_frame, "EYES CLOSED !!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                # Append the Sleeping status and timestamp to the list
                status_data.append([status, time.strftime("%Y-%m-%d %H:%M:%S")])
                

            elif sleep >= 40:
                status = "SLEEPING! WAKE UP!"
                color = (0, 0, 255)
                cv2.putText(flipped_frame, "SLEEPING! WAKE UP!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                status_data.append([status, time.strftime("%Y-%m-%d %H:%M:%S")])

                play_alarm_sound()
            else:
                stop_alarm_sound()
                # Append the Sleeping status and timestamp to the list
                

        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            drowsy += 1
            active = 0
            yawning_status = 0
            if drowsy > 25 and drowsy < 80: # Frames
                status = "Drowsy !"
                color = (0, 128, 0)
                cv2.putText(flipped_frame, "Drowsy !", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                # Append the Sleeping status and timestamp to the list
                status_data.append([status, time.strftime("%Y-%m-%d %H:%M:%S")])
                

            elif drowsy >= 80:
                status = "Drowsy! WAKE UP!"
                color = (0, 0, 255)
                cv2.putText(flipped_frame, "Drowsy! WAKE UP!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                # Append the Sleeping status and timestamp to the list
                status_data.append([status, time.strftime("%Y-%m-%d %H:%M:%S")])

                play_alarm_sound()
            else:
                stop_alarm_sound()
                


        elif lips_yawn == 1:
            sleep = 0
            active = 0
            drowsy = 0
            yawning_status += 1
            if yawning_status >= 1:  # Adjust the threshold here
                status = "Yawning !!"
                color = (0, 0, 255)
                cv2.putText(flipped_frame, 'Yawning !!', (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                # Append the Sleeping status and timestamp to the list
                status_data.append([status, time.strftime("%Y-%m-%d %H:%M:%S")])

                play_alarm_sound()
            else:
                stop_alarm_sound()
                

        else:
            drowsy = 0
            sleep = 0
            active += 1
            yawning_status = 0

            if active > 1:
                status = "Active :)"
                color = (0, 128, 0)
                cv2.putText(flipped_frame, "Active :)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        df = pd.DataFrame(status_data, columns=['Status', 'Timestamp'])
        df.to_excel('drowsy_data.xlsx', index=False)

##   Code for Detecting Active, Sleeping, Drowsy and Yawning Status    ##
#########################################################################


######################################################################### 
##                     Code for Gazing Detection                       ##
         

        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

        if gaze_ratio <= 0.75:
            Right_Gaze = 0
            Left_Gaze += 1
            Centre_Gaze = 0
            if Left_Gaze <= 45:
                gaze_status = "Gazing Left"
                color = (255,0,0)
                # Append the Sleeping status and timestamp to the list
                status_data.append([gaze_status, time.strftime("%Y-%m-%d %H:%M:%S")])
                cv2.putText(flipped_frame, gaze_status, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            elif Left_Gaze > 45:
                gaze_status = "Distracted Looking Left!"
                color = (0, 0, 255)
                # Append the Sleeping status and timestamp to the list
                status_data.append([gaze_status, time.strftime("%Y-%m-%d %H:%M:%S")])
                cv2.putText(flipped_frame, gaze_status, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                play_alarm_sound()
            else:
                stop_alarm_sound()
        
        elif gaze_ratio >= 2.1:
            Right_Gaze += 1
            Left_Gaze = 0
            Centre_Gaze = 0
            if Right_Gaze <= 45:
                gaze_status = "Gazing Right"

                color = (255,0,0)
                # Append the Sleeping status and timestamp to the list
                status_data.append([gaze_status, time.strftime("%Y-%m-%d %H:%M:%S")])
                cv2.putText(flipped_frame, gaze_status, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            elif Right_Gaze > 45:
                gaze_status = "Distracted Looking Right!"
                color = (0, 0, 255)
                # Append the Sleeping status and timestamp to the list
                status_data.append([gaze_status, time.strftime("%Y-%m-%d %H:%M:%S")])
                cv2.putText(flipped_frame, gaze_status, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8,color, 2)
                play_alarm_sound()
            else:
                stop_alarm_sound()

        else:
            Right_Gaze = 0
            Left_Gaze = 0
            Centre_Gaze = 1
            gaze_status = "Looking Forward"
            color = (255,0,0)
            # # Append the Sleeping status and timestamp to the list
            # status_data.append([gaze_status, time.strftime("%Y-%m-%d %H:%M:%S")])

            cv2.putText(flipped_frame, gaze_status, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        df = pd.DataFrame(status_data, columns=['Driver Status', 'Timestamp'])
        df.to_excel('drowsy_data.xlsx', index=False)            
        
##                     Code for Gazing Detection                       ##
#########################################################################


#########################################################################
##                 Displaying MAR and EAR Values                       ##

        left_eye_landmarks = landmarks[36:42]
        right_eye_landmarks = landmarks[42:48]
        mouth_landmarks = landmarks[48:68]

        left_ear = compute_ear(left_eye_landmarks)
        right_ear = compute_ear(right_eye_landmarks)
        average_ear = (left_ear + right_ear) / 2.0

        mouth_mar = compute_mar(mouth_landmarks)

        cv2.putText(flipped_frame, "EAR: {:.2f}".format(average_ear), (460, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 0, 128), 2)
        cv2.putText(flipped_frame, "MAR: {:.2f}".format(mouth_mar), (460, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 0, 128), 2)

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

##                 Displaying MAR and EAR Values                       ##
#########################################################################


#########################################################################
##                  Getting the Video Frame Per Second                 ##
    framespersecond= int(cap.get(cv2.CAP_PROP_FPS))
    framespersecond = str(framespersecond)
    cv2.putText(flipped_frame,f"FPS: {framespersecond}", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,128,0), 2)

#########################################################################
##                    Getting the Video Frame                          ##

    cv2.imshow("Frame", flipped_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

##                    Getting the Video Frame                          ##
#########################################################################