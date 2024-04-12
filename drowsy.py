from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
from playsound import playsound

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Define thresholds and parameters
thresh = 0.27
frame_check = 18
escalation_threshold = 5  # Number of consecutive frames before escalating alert level

# Define alert levels and corresponding actions
alert_levels = {
    1: {"threshold": 10, "action": lambda: playsound("assets/mild_alert.mp3")},
    2: {"threshold": 20, "action": lambda: playsound("assets/medium_alert.mp3")},
    3: {"threshold": float('inf'), "action": lambda: playsound("assets/intense_alert.mp3")}
}

# Initialize variables
frame_count = 0
drowsy_frames = 0
alert_level = 1

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap=cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        if ear < thresh:
            drowsy_frames += 1
            if drowsy_frames >= escalation_threshold:
                frame_count += 1
                if frame_count >= alert_levels[alert_level]["threshold"]:
                    alert_levels[alert_level]["action"]()
                    alert_level += 1
                    frame_count = 0
        else:
            drowsy_frames = 0
            frame_count = 0
            alert_level = 1
            
        alert_level = min(alert_level, len(alert_levels))  # Cap alert level to maximum defined
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
