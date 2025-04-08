import streamlit as st
import cv2
import face_recognition
import json
import numpy as np
import time
import os

# Set page config
st.set_page_config(page_title="Face Segmentation", layout="wide")

def load_known_faces():
    """
    Load existing face encodings from JSON file
    """
    if os.path.exists('C:/Users/igalp/MultiMediaProject/known_faces.json'):
        with open('C:/Users/igalp/MultiMediaProject/known_faces.json', 'r') as f:
            data = json.load(f)
            # Convert string representations back to numpy arrays
            return {name: np.array(encoding) for name, encoding in data.items()}
    return {}

def process_frame(frame, known_faces, show_names):
    """
    Process frame to detect and classify faces
    """
    # Resize frame for better performance
    frame = cv2.resize(frame, (640, 480))
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Find all face locations
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    # Process each face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Add padding to make rectangle bigger
        padding = 20
        top = max(0, top - padding)
        right = min(frame.shape[1], right + padding)
        bottom = min(frame.shape[0], bottom + padding)
        left = max(0, left - padding)
        
        # Compare with known faces
        matches = []
        for name, known_encoding in known_faces.items():
            match = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=0.6)[0]
            if match:
                matches.append(name)
        
        # Draw rectangle and label
        if matches:
            # Ally (green)
            color = (0, 255, 0)
            label = f"Ally: {matches[0]}" if show_names else "Ally"
           
        else:
            # Enemy (red)
            color = (0, 0, 255)
            label = "Enemy"
            
        
        # Draw rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Draw label
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
    
    return frame

def run_detection():
    """
    Run the face detection and classification
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not access webcam")
        return
    
    # Load known faces
    known_faces = load_known_faces()
    if not known_faces:
        st.warning("No known faces found in database. Please add some allies first.")
        return
    
    # Create a placeholder for the video feed
    frame_placeholder = st.empty()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Could not read frame from webcam")
                break
            
            # Process frame
            processed_frame = process_frame(frame, known_faces, st.session_state.get('show_names', False))
            
            # Display the frame
            frame_placeholder.image(processed_frame, channels="BGR")
            
            # Add a small delay
            time.sleep(0.03)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    
    finally:
        cap.release()

def main():
    st.title("Face Segmentation - Ally/Enemy Detection")
    st.write("Real-time face detection and classification")
    
    # Add show names checkbox
    st.session_state.show_names = st.checkbox("Show names", value=True)
    
    # Add start button
    if st.button("Start Detection"):
        run_detection()

if __name__ == "__main__":
    main()

# To run:
# streamlit run face_segmentation.py 