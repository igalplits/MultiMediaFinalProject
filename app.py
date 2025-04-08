import streamlit as st
import cv2
import time
from ultralytics import YOLO
import numpy as np

# Set page config for better layout
st.set_page_config(page_title="Handgun Segmentation", layout="wide")

def load_model():
    """
    Load the YOLOv8 model for segmentation
    """
    try:
        model = YOLO("C:/Users/igalp/MultiMediaProject/models/best12.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_frame(frame, model):
    """
    Process a single frame with the YOLOv8 model
    """
    # Resize frame for better performance
    frame = cv2.resize(frame, (640, 480))
    
    # Run inference
    results = model(frame, task='segment')
    
    # Get the annotated frame
    annotated_frame = results[0].plot()
    
    return annotated_frame

def run_webcam():
    """
    Run the webcam feed and perform real-time segmentation
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Error: Could not access webcam")
        return
    
    # Load model
    model = load_model()
    if model is None:
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
            processed_frame = process_frame(frame, model)
            
            # Convert BGR to RGB for Streamlit
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame
            frame_placeholder.image(processed_frame, channels="RGB")
            
            # Add a small delay to prevent CPU overload
            time.sleep(0.03)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    
    finally:
        # Release resources
        cap.release()

def main():
    st.title("Real-time Handgun Segmentation")
    st.write("This app performs real-time segmentation of handguns using YOLOv8")
    
    # Add start button
    if st.button("Start Webcam Segmentation"):
        run_webcam()

if __name__ == "__main__":
    main()

# To run the app:
# streamlit run app.py 