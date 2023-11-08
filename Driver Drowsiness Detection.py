#importing all the required libraries
import pygame
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2

# Initialize pygame mixer
pygame.mixer.init()

def sound_alarm(path):
    # Play an alarm sound
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

def eye_aspect_ratio(eye):
    # Compute the Euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute the Euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear

def mouth_aspect_ratio(mouth):
    # Compute the Euclidean distances between the three sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[13], mouth[19])  # Upper lip
    B = dist.euclidean(mouth[14], mouth[18])  # Upper lip
    C = dist.euclidean(mouth[15], mouth[17])  # Upper lip

    D = dist.euclidean(mouth[12], mouth[16])  # Lower lip

    # Compute the mouth aspect ratio
    mar = (A + B + C) / (3.0 * D)

    return mar

# Constants for blink detection
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48

# Constants for yawning detection
MOUTH_AR_THRESH = 0.5  # Adjust the threshold (lower value for more sensitivity)
MOUTH_AR_CONSEC_FRAMES = 20  # Adjust the number of consecutive frames

# Constants for face detection
NO_FACE_ALARM_FRAMES = 60  # Set the threshold for no face detection

# Initialize frame counter and alarm status
COUNTER = 0
ALARM_ON = False

# Initialize mouth counter and mouth aspect ratio
mouthCOUNTER = 0
MOUTH_AR = 0

# Initialize face counter
no_face_counter = 0

# Initialize dlib's face detector (HOG-based) and create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define the indexes of the facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Define the indexes of the facial landmarks for the mouth
(mStart, mEnd) = (61, 68)  # Adjust the indices for the inner mouth landmarks

# Start the video stream
vs = VideoStream(src=1).start()
time.sleep(1.0)

# Initialize previous facial landmarks
prev_shape = None

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 0)
    
    if len(rects) == 0:
        no_face_counter += 1
        if no_face_counter >= NO_FACE_ALARM_FRAMES:
            if not ALARM_ON:
                ALARM_ON = True
                sound_alarm("alarm.wav")
                print("No face detected for a period. Trigger alarm.")
    else:
        no_face_counter = 0
        if ALARM_ON:
            ALARM_ON = False

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio) calculation
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Calculate mouth aspect ratio
        mouth = shape[mStart:mEnd]
        MAR = mouth_aspect_ratio(mouth)

        # Check for drowsiness
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    sound_alarm("alarm.wav")
                    print("Drowsiness detected. Trigger alarm.")
        else:
            COUNTER = 0

        # Check for yawning
        if MAR > MOUTH_AR_THRESH:
            mouthCOUNTER += 1
            if mouthCOUNTER >= MOUTH_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    sound_alarm("alarm.wav")
                    print("Yawning detected. Trigger alarm.")
        else:
            mouthCOUNTER = 0

        # Draw EAR and MAR on the frame
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(MAR), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw facial landmarks
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
