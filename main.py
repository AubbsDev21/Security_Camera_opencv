import time
import datetime

try:
    import cv2
except:
    print("please pip install opencv-python")

## Gets and opens your web cam
capture = cv2.VideoCapture(0)

recording = True

## Setting a classifier and poniting to the training models that detehced faces and bodies
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
## Making sure the video size is the same as the capture size
frame_size = (int(capture.get(3)), int(capture.get(4)))
vid_format = cv2.VideoWriter_fourcc(*"mp4v")
vid_output = cv2.VideoWriter("video.mp4", vid_format, 20, frame_size)

## While webcam is open we are going to read a frame by frame from our camarea
while True:
    _, frame = capture.read()
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale, 1.2, 5)
    bodies = body_cascade.detectMultiScale(grayscale, 1.2, 5)

    if len(faces) + len(bodies) > 0:
        recording = True
    vid_output.write(frame)
 ## This will draw the rectangles on the color scale for the data it get from the gray scale
    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (79, 105, 213), 3)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('s'):
        break
#save vid
vid_output.release()
capture.release()
cv2.destroyAllWindows()