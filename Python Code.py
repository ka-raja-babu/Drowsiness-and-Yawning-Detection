#NEGLIGENCE DETECTION(DROWSINESSS AND YAWNING)

#Import required libraries
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os

#Define function for alarm
def alarm(msg):
    global alarm_drowsy
    global alarm_yawn
    global saying

    if alarm_drowsy:
        print('call')
        saying = True
        s = 'espeak "' + msg + '"'
        os.system(s)
        saying = False

    if alarm_yawn:
        print('call')
        saying = True
        s = 'espeak "' + msg + '"'
        os.system(s)
        saying = False

#Define function for EAR
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

#Define function for average EAR of left and right eye
def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

#Define function for MAR
def mouth_aspect_ratio(shape):
    A = dist.euclidean(shape[61],shape[67])+ dist.euclidean(shape[62],shape[66])+dist.euclidean(shape[63],shape[65])
    B =dist.euclidean(shape[60],shape[64])
    mar = A/(3.0 * B)
    return mar

#Setting camera
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

#Define ear and mar threshold
ear_threshold = 0.18
mar_threshold = 0.40
#Define alarm status for drowsy and yawn
alarm_drowsy = False
alarm_yawn = False
#Defining saying variable so that alarm do not mix
saying = False

#Defining function for face detection and landmark prediction
print("-> Loading the predictor and detector...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#Starting video streaming
print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:
    #Reading frame and converting it into grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detecting face region
    rects = detector(gray, 0)

    for rect in rects:
        #Predicting 68 landmarks of face
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        #Calculating EAR
        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]

        # Drawing countour around eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Calculating MAR
        mar = mouth_aspect_ratio(shape)

        #Drawing contour around mouth
        mouth = shape[48:60]
        cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)

        if ear < ear_threshold:
                if alarm_drowsy == False and saying == False:
                    alarm_drowsy = True
                    t = Thread(target=alarm, args=('Drowsy...Negligence...Detected',))
                    t.deamon = True
                    t.start()

                cv2.putText(frame, "DROWSY !!...NEGLIGENCE", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            alarm_drowsy = False

        if (mar > mar_threshold):
                if alarm_yawn == False and saying == False:
                    alarm_yawn = True
                    t = Thread(target=alarm, args=('Yawning...Negligence...Detected',))
                    t.deamon = True
                    t.start()

                cv2.putText(frame, "YAWNING !!...NEGLIGENCE", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            alarm_yawn = False


        #Displaying EAR and MAR ratio
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    #Displaying frame output
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF

    #Quitting program by pressing "q"
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
