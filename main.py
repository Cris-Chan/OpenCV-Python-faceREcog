import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # get width and height of capture
    width = cap.get(3)
    height = cap.get(4)
    midX = width/2
    midY = height/2
    # Capture frame-by-frame
    ret, frame = cap.read()
    # convert to grey scale to run algorithm on
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # set font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Facial recog cascades
    face_cascade = cv2.CascadeClassifier(
        # Replace with the path to your facial recog cascade
        '/usr/local/Cellar/opencv/4.1.0_2/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(
        # Replace with the path to your eye recog cascade
        '/usr/local/Cellar/opencv/4.1.0_2/share/opencv4/haarcascades/haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(
        gray, 1.3, 5)  # run algorithm on single frame

    for (x, y, w, h) in faces:
        string = "(" + str(x) + "," + str(y) + ")"
        direction = "?"

        if x < midX-70:  # decides movements face should make to reach center
            direction = "MOVE LEFT"
        else:
            direction = "MOVE RIGHT"

        cv2.putText(frame, string, (0, 100), font,
                    2, [0, 255, 255], 2)
        cv2.putText(frame, direction, (0, 180), font,
                    2, [0, 255, 255], 2)
        cv2.putText(frame, "Faces: "+str(len(faces)), (0, 260), font,
                    2, [0, 255, 255], 2)

        img = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)  # detect eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    # cv2.imshow('gray', gray)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
