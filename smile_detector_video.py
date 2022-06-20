import cv2
from keras.models import load_model
import numpy as np

blue = (255, 0, 0)

video_capture = cv2.VideoCapture(0)

# load the haar cascades face and smile detectors
face_detector = cv2.CascadeClassifier("haar_cascade/haarcascade_frontalface_default.xml")
model = load_model("model")

# loop over the frames
while True:
    # get the next frame from the video and convert it to grayscale
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # apply our face detector to the grayscale frame
    faces = face_detector.detectMultiScale(gray, 1.1, 8)
    
    # go through the face bounding boxes 
    for (x, y, w, h) in faces:
        # draw a rectangle around the face on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), blue, 2)
        # extract the face from the grayscale image
        roi = gray[y:y + h, x:x + w]

        # Applying CLAHE to the face ROI
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        roi_clahe = clahe.apply(roi)
        
        roi = cv2.resize(roi_clahe, (32, 32))
        roi = roi / 255.0
        # add a new axis to the image
        # previous shape: (32, 32), new shape: (1, 32, 32)
        roi = roi[np.newaxis, ...]
        # apply the smile detector to the face roi
        prediction = model.predict(roi)[0]
        label = "Smiling" if prediction >= 0.5 else "Not Smiling"

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, blue, 2)
         
                
    cv2.imshow("CLAHE", roi_clahe)
    cv2.imshow('Frame', frame)
   
    # wait for 1 milliseconde and if the q key is pressed, we break the loop
    if cv2.waitKey(1) == ord('q'):
        break
    
# release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
