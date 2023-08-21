import cv2
import streamlit as st
from PIL import ImageColor


path = r'C:/Users/rania/OneDrive/Bureau/Checkpoints/face detection/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(path)


def detect_faces(color, minNeighbors, scaleFactor):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if gray.dtype != "uint8":
            gray = gray.astype("uint8")
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        for idx, (x, y, w, h) in enumerate(faces):
            r, g, b = ImageColor.getcolor(color, "RGB")
            bgr_color = (b, g, r)
            cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, 2)
            
            # Crop and save the detected face
            face = frame[y:y+h, x:x+w]
            cv2.imwrite(f"detected_face_{idx}.jpg", face)
            
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write("Press the button below to start detecting faces from your webcam")
    color = st.color_picker("Select a color", "#FF5733")
    minNeighbors = st.slider('minNeighbors', min_value=5, max_value=10)
    scaleFactor = st.slider('scaleFactor', min_value=1.01, max_value=3.0)
    if st.button("Detect Faces"):
        detect_faces(color, minNeighbors, scaleFactor)

if __name__ == "__main__":
    app()


