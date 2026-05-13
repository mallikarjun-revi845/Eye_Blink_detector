import cv2
import time
import argparse
import logging
import numpy as np
from typing import Tuple

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BlinkDetector:
    """
    A professional-grade Eye Blink Detector using OpenCV Haar Cascades.
    Provides robust OOP structure, FPS tracking, and blink counting without 
    external dependencies that conflict with your environment.
    """

    def __init__(self, consecutive_frames: int = 2):
        """
        Initializes the BlinkDetector.

        Args:
            consecutive_frames (int): Number of consecutive frames eyes must be missing to count as a blink.
        """
        self.consecutive_frames = consecutive_frames
        
        # Initialize Haar Cascades
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
        except Exception as e:
            logging.error(f"Error loading cascades: {e}")
            raise
        
        # State variables
        self.blink_counter = 0
        self.eyes_closed_frames = 0
        self.prev_time = 0.0

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Processes a single video frame to detect faces, eyes, and blinks.

        Args:
            frame (np.ndarray): The BGR image frame from the camera.

        Returns:
            np.ndarray: The annotated frame with blink status, counter, and FPS.
        """
        # FPS Calculation
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = current_time

        # Convert to Grayscale for Haar Cascades
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        status_text = "No Face Detected"
        status_color = (0, 255, 255) # Yellow

        # Detect Faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))
        
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Extract the Region of Interest (ROI) for eyes
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
                # Detect Eyes
                eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.3, 5, minSize=(50, 50))

                # Blink Logic: Less than 2 eyes detected indicates closed eyes
                if len(eyes) >= 2:
                    status_text = "Eyes Open"
                    status_color = (0, 255, 0) # Green
                    
                    # If eyes were closed for enough frames before opening, count a blink
                    if self.eyes_closed_frames >= self.consecutive_frames:
                        self.blink_counter += 1
                        logging.info(f"Blink detected! Total blinks: {self.blink_counter}")
                    
                    self.eyes_closed_frames = 0
                    
                    # Draw rectangles around eyes
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)
                else:
                    status_text = "Blink Detected!"
                    status_color = (0, 0, 255) # Red
                    self.eyes_closed_frames += 1

        # Overlay Statistics on the frame
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Blinks: {self.blink_counter}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        return frame

    def run(self, camera_index: int = 0) -> None:
        """
        Starts the video capture and blink detection loop.
        
        Args:
            camera_index (int): The index of the camera to use.
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logging.error(f"Failed to open camera with index {camera_index}.")
            return

        logging.info("Starting Eye Blink Detector. Press 'q' to quit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.warning("Failed to grab frame from camera. Exiting...")
                    break
                
                # Process the frame
                annotated_frame = self.process_frame(frame)
                
                # Display the result
                cv2.imshow("Professional Blink Detector (Haar Cascade)", annotated_frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("Exit requested by user.")
                    break
        finally:
            # Cleanup resources
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Professional Eye Blink Detector.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--frames", type=int, default=1, help="Consecutive frames for blink detection (default: 1)")
    
    args = parser.parse_args()
    
    detector = BlinkDetector(consecutive_frames=args.frames)
    detector.run(camera_index=args.camera)
