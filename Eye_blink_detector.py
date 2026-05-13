import cv2
import time
import winsound  # For sound alert on Windows (use playsound for Mac/Linux)

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

blink_count = 0
eyes_closed = False
last_blink_time = 0
emoji = "😎"  # Default emoji

start_time = time.time()

while True:
    ret, img = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 1, 1)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
    eyes_detected = False

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_face = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5, minSize=(30, 30))
        if len(eyes) >= 2:
            eyes_detected = True
            break

    # --- Blink Detection Logic ---
    if eyes_detected:
        cv2.putText(img, "Eyes Open 😃", (50, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)
        emoji = "😃"
        if eyes_closed:
            blink_count += 1
            winsound.Beep(1000, 150)  # short beep
            last_blink_time = time.time()
            eyes_closed = False
    else:
        cv2.putText(img, "Eyes Closed 😴", (50, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 2)
        emoji = "😴"
        eyes_closed = True

    # --- Fun Display ---
    elapsed_time = time.time() - start_time
    blink_rate = blink_count / (elapsed_time / 60) if elapsed_time > 10 else 0  # blinks per minute

    cv2.putText(img, f"Total Blinks: {blink_count}", (50, 120),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 0), 2)

    cv2.putText(img, f"Blink Rate: {blink_rate:.1f} /min", (50, 170),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 200, 100), 2)

    # Add emoji reaction
    cv2.putText(img, emoji, (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)

    # Draw a progress-style bar for fun
    bar_width = int((blink_count % 10) * 50)
    cv2.rectangle(img, (50, 200), (50 + bar_width, 230), (0, 200, 255), -1)
    cv2.putText(img, "Blink Energy Bar", (50, 260), cv2.FONT_HERSHEY_PLAIN, 1.2, (200, 255, 255), 2)

    cv2.imshow("👁️ Smart Blink Tracker 👁️", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"✅ Session ended. Total blinks detected: {blink_count}")
