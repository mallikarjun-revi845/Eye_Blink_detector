# Eye_Blink_detector
👁️ Real-time Eye Blink Detector built with Python, OpenCV, and MediaPipe Face Mesh. Uses Eye Aspect Ratio (EAR) to accurately detect blinks through webcam input. Features facial landmark tracking, FPS monitoring, blink counter, and customizable detection settings for drowsiness detection and AI-based monitoring systems.

🚀 Features
Real-time blink detection using webcam
Eye Aspect Ratio (EAR) calculation
MediaPipe Face Mesh integration
FPS monitoring
Blink counter display
Lightweight and fast performance
Adjustable EAR threshold and frame sensitivity
Professional object-oriented Python implementation
🛠️ Technologies Used
Python
OpenCV
MediaPipe
NumPy
📌 How It Works

The system detects facial landmarks using MediaPipe Face Mesh and extracts eye landmark points. It then calculates the Eye Aspect Ratio (EAR) to determine whether the eyes are open or closed. If the EAR drops below a predefined threshold for consecutive frames, a blink is detected.
