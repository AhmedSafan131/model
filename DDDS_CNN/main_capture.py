import cv2
import os
import numpy as np
from keras.models import load_model
from pygame import mixer
import time

mixer.init()
# sound = mixer.Sound('DDDS_CNN/alarm.wav') # alarm sound
sound = mixer.Sound('Transfer_learning/openyoureyes.wav') # please open your eyes sound

face_cascade = cv2.CascadeClassifier('DDDS_CNN/haar cascade files/haarcascade_frontalface_alt.xml')
leye_cascade = cv2.CascadeClassifier('DDDS_CNN/haar cascade files/haarcascade_lefteye_2splits.xml')
reye_cascade = cv2.CascadeClassifier('DDDS_CNN/haar cascade files/haarcascade_righteye_2splits.xml')

model = load_model('models/cnncat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
alarm_on = False  # Variable to track if the alarm is currently playing
rpred = [99]
lpred = [99]

close_time = 0  # Variable to track when the eyes were closed
alarm_delay = 2  # Time delay before activating the alarm (in seconds)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2] 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and eyes
    faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye_cascade.detectMultiScale(gray)
    right_eye = reye_cascade.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)

    # Check right eye for drowsiness
    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y+h, x:x+w]
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = np.argmax(model.predict(r_eye), axis=-1)
        break

    # Check left eye for drowsiness
    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y+h, x:x+w]
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = np.argmax(model.predict(l_eye), axis=-1)
        break

    # Activate alarm if eyes are closed for more than the delay time
    if rpred[0] == 0 or lpred[0] == 0:
        if close_time == 0:
            close_time = time.time()  # Start timer when eyes close
        if time.time() - close_time >= alarm_delay:
            cv2.putText(frame, "Sleepy!", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            if not alarm_on:
                try:
                    sound.play(-1)  # Play sound indefinitely
                    alarm_on = True
                except Exception as e:
                    print("Error playing sound:", e)
    else:
        # Turn off alarm if eyes are open
        close_time = 0  # Reset timer when eyes open
        if alarm_on:
            sound.stop()
            alarm_on = False
        cv2.putText(frame, "Alert", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
