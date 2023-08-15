import cv2
import streamlit as st

face_cascade = cv2.CascadeClassifier('C:/Users/rania/OneDrive/Bureau/Checkpoints/face detection/haarcascade_frontalface_default.xml')

def detect_faces(rectangle_color, min_neighbors, scale_factor):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    while True:
        # Read the frames from the webcam
      ret, frame = cap.read()
        # Convert the frames to grayscale
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the faces using the face cascade classifier
      faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
      
        # Draw rectangles around the detected faces
      for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, 2)
        # Display the frames
      cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)
        
      if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('Save your Face image:', frame)
        st.success("Image saved successfully as 'detected_face.jpg")

      # Exit the loop when 'q' is pressed
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
          
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write('This algorithm will allow you to detect faces from your webcam by drawing a rectangle !')
    st.write("Press the button below to start detecting faces from your webcam")
    rectangle_color=st.color_picker("Choose a color for your rectangle", "#FF0000")
    rectangle_color = tuple(int(rectangle_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    min_neighbors=st.slider('choose a minNeighbor parameter:', 1, 10, 5 )
    scale_factor = st.slider('Choose scaleFactor parameter:', 1.01, 1.5, 1.1)
    # Add a button to start detecting faces
    if st.button("Detect Faces"):
      detect_faces(rectangle_color, min_neighbors, scale_factor)
      

if __name__ == "__main__":
  app()