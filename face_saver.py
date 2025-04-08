import streamlit as st
import cv2
import face_recognition
import json
import os
import numpy as np
import time

# Set page config
st.set_page_config(page_title="Face Saver", layout="wide")

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

def save_known_faces(known_faces):
    """
    Save face encodings to JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    data = {name: encoding.tolist() for name, encoding in known_faces.items()}
    with open('C:/Users/igalp/MultiMediaProject/known_faces.json', 'w') as f:
        json.dump(data, f)
    return True

def process_frame(frame):
    """
    Detect and encode face in the frame
    """
    # Convert BGR to RGB (face_recognition uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Find all face locations
    face_locations = face_recognition.face_locations(rgb_frame)
    
    if len(face_locations) == 0:
        return None, "No face detected"
    elif len(face_locations) > 1:
        return None, "Too many faces detected"
    
    # Get face encodings
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    return face_encodings[0], None

def main():
    st.title("Face Saver - Add New Ally")
    st.write("Add new ally faces to the database")
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    # Initialize session state for webcam
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False
    
    # Load existing known faces
    known_faces = load_known_faces()
    
    with col1:
        st.subheader("Webcam Feed")
        
        # Webcam control buttons
        if not st.session_state.webcam_active:
            if st.button("Start Webcam"):
                st.session_state.webcam_active = True
                st.rerun()
        else:
            if st.button("Stop Webcam"):
                st.session_state.webcam_active = False
                st.rerun()
        
        # Create a placeholder for the video feed
        frame_placeholder = st.empty()
        
        # Only run webcam if active
        if st.session_state.webcam_active:
            # Initialize webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Error: Could not access webcam")
                st.session_state.webcam_active = False
            else:
                try:
                    # Display a few frames
                    for _ in range(10):  # Show 10 frames
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Error: Could not read frame from webcam")
                            break
                        
                        # Display the frame
                        frame_placeholder.image(frame, channels="BGR")
                        time.sleep(0.03)
                    
                    # Store the last frame for saving
                    st.session_state.last_frame = frame
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                
                finally:
                    cap.release()
    
    with col2:
        # Input for person's name
        person_name = st.text_input("Enter person's name:")
        
        # Display current database status
        st.subheader("Database Status")
        if known_faces:
            st.write(f"✅ Database contains {len(known_faces)} known faces:")
            for name in known_faces.keys():
                st.write(f"- {name}")
        else:
            st.write("❌ No faces in database yet")
        
        # Save button with confirmation
        if st.button("Save as Ally", type="primary"):
            if not person_name:
                st.error("Please enter a name")
                return
            
            if not st.session_state.webcam_active or 'last_frame' not in st.session_state:
                st.error("Please start the webcam first")
                return
            
            # Get the last captured frame
            frame = st.session_state.last_frame
            
            # Process the frame
            face_encoding, error = process_frame(frame)
            
            if error:
                st.error(error)
                return
            
            # Add to known faces
            known_faces[person_name] = face_encoding
            if save_known_faces(known_faces):
                st.success(f"✅ Successfully saved face as {person_name}")
                
                # Show confirmation with animation
                st.balloons()
                
                # Display updated database status
                st.subheader("Updated Database Status")
                st.write(f"✅ Database now contains {len(known_faces)} known faces:")
                for name in known_faces.keys():
                    st.write(f"- {name}")
            else:
                st.error("Failed to save face to database")
        
        # Add a button to view the JSON file
        if st.button("View Database File"):
            if os.path.exists('C:/Users/igalp/MultiMediaProject/known_faces.json'):
                with open('C:/Users/igalp/MultiMediaProject/known_faces.json', 'r') as f:
                    data = json.load(f)
                    st.json(data)
            else:
                st.warning("Database file does not exist yet")

if __name__ == "__main__":
    main()

# To run:
# streamlit run face_saver.py 