from cv2 import cv2

#To load pre-trained frontal face data from haar cascade algorithm--opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
trained_eye_data = cv2.CascadeClassifier('haarcascade_eye.xml')
trained_smile_data = cv2.CascadeClassifier('haarcascade_smile.xml')

# To choose image for detecting faces and read its array
#img1 = cv2.imread('test2.jpg')
# Capture video
webcam = cv2.VideoCapture(0)

while True:

    #Read current frame/picture
    successful_frame_read, frame = webcam.read()

    # Must convert images to grey-scale 
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img, scaleFactor=1.04, minNeighbors=5, minSize=(20, 20))
    eye_coordinates  = trained_eye_data.detectMultiScale(grayscaled_img, scaleFactor= 1.5, minNeighbors=6, minSize=(4, 4))
    smile_coordinates  = trained_smile_data.detectMultiScale(grayscaled_img, scaleFactor= 3.5, minNeighbors=15, minSize=(20, 20))
    #print(face_coordinates)

    # Draw a rectangle around the face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
    # Draw a rectangle arounf the eyes
    for (ex, ey, ew, eh) in eye_coordinates:
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)
    # Rectangle around smile
    for (xx, yy, ww, hh) in smile_coordinates:
        cv2.rectangle(frame, (xx, yy), (xx + ww, yy + hh), (255, 0, 0), 1)
        

    # To show in an app
    cv2.imshow('Face Detector', frame)
    # Pause execution of progrmam
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

webcam.release()
cv2.destroyAllWindows()














print("Code Completed")
