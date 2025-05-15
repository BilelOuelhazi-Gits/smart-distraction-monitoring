import cv2
import time
from pymongo import MongoClient
import mediapipe as mp

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['attention_db']
collection = db['distraction_time']

# Parameters
SCALE_FACTOR = 1.2
MIN_NEIGHBORS = 6
MIN_FACE_ABSENCE_DURATION = 2  # seconds
HAND_ON_MOUTH_COOLDOWN = 2  # seconds
TOO_CLOSE_THRESHOLD = 200  # pixels
POSTURE_ALERT_COOLDOWN = 5  # seconds between alerts
POSTURE_DISPLAY_DURATION = 3  # seconds display time
EYE_MOTION_THRESHOLD = 10
BLINK_WINDOW = 10  # seconds
PHONE_HOLD_THRESHOLD = 500  # pixels area

# State variables
distraction_time = 0
not_looking_start = None
face_absent = False
hand_on_mouth_count = 0
last_hand_touch_time = 0
hand_motion_detected = False
prev_gray = None
last_mouth_coords = None
last_face_time = 0
last_known_points = None
last_posture_alert_time = 0
posture_display_until = 0
blink_count = 0
blink_start_time = time.time()
last_blink_time = 0
eye_detected_last_frame = True
blink_pending = False
phone_held_start = None
holding_phone = False
phone_holding_duration = 0
total_phone_usage_time = 0
phone_detection_start_time = None

# MediaPipe setup for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

print("✅ System started. Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera frame not received.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_diff = None

    if prev_gray is not None:
        frame_diff = cv2.absdiff(prev_gray, gray)
    prev_gray = gray.copy()

    current_time = time.time()

    # === Phone detection (purple object) ===
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_purple = (125, 50, 50)
    upper_purple = (150, 255, 255)
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    contours, _ = cv2.findContours(mask_purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    phone_detected = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > PHONE_HOLD_THRESHOLD:
            phone_detected = True
            cv2.drawContours(frame, [cnt], -1, (255, 0, 255), 2)
            break

    if not holding_phone:
        if phone_detected:
            if phone_detection_start_time is None:
                phone_detection_start_time = current_time
            elif current_time - phone_detection_start_time >= 1:  # Require 1 second of detection
                phone_held_start = current_time
                holding_phone = True
                print(f"[📱] Phone held confirmed at {time.strftime('%H:%M:%S')}")
        else:
            phone_detection_start_time = None
    else:
        if not phone_detected:
            phone_holding_duration = current_time - phone_held_start
            total_phone_usage_time += phone_holding_duration
            holding_phone = False
            phone_detection_start_time = None

            print(f"[⏱️] Phone was held for {phone_holding_duration:.2f} seconds.")
            print("📥 Phone usage logged to MongoDB.")
            print(f"📊 Total phone usage so far: {total_phone_usage_time:.2f} seconds.")

            # Update MongoDB with phone usage
            collection.update_one(
                {"user": "test_user"},
                {
                    "$inc": {
                        "total_phone_usage_time": round(phone_holding_duration, 2)
                    }
                },
                upsert=True
            )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        if not_looking_start is None:
            not_looking_start = current_time
            print("[!] Face not detected. Starting distraction timer...")

        if last_mouth_coords and current_time - last_face_time <= 2 and frame_diff is not None:
            mx1, my1, mx2, my2 = last_mouth_coords
            motion_region = frame_diff[my1:my2, mx1:mx2]
            if motion_region.size > 0:
                motion_score = cv2.mean(motion_region)[0]
                if motion_score > 20:  # Increase sensitivity for hand motion detection
                    if not hand_motion_detected and (current_time - last_hand_touch_time > HAND_ON_MOUTH_COOLDOWN):
                        hand_on_mouth_count += 1
                        last_hand_touch_time = current_time
                        hand_motion_detected = True
                        print(f"[🤚] Hand on mouth detected (motion score: {motion_score}). Count = {hand_on_mouth_count}")
                else:
                    hand_motion_detected = False

        if current_time - not_looking_start >= MIN_FACE_ABSENCE_DURATION:
            distraction_duration = current_time - not_looking_start
            distraction_time += distraction_duration
            not_looking_start = None

            print(f"[⚠️] Distracted for {distraction_duration:.2f} seconds.")
            print("↪️  Updating MongoDB...")

            # Update MongoDB with distraction time and hand on mouth count
            collection.update_one(
                {"user": "test_user"},
                {
                    "$inc": {
                        "distraction_time": round(distraction_duration, 2),
                        "hand_on_mouth_count": hand_on_mouth_count
                    }
                },
                upsert=True
            )
            print("✅ MongoDB updated.\n")
        face_absent = True

    else:
        if face_absent:
            print("[✔️] Face detected again. Resuming tracking.\n")
        not_looking_start = None
        face_absent = False

        for (x, y, w, h) in faces:
            # Defining key facial landmarks
            eye_left = (x + int(w * 0.3), y + int(h * 0.4))
            eye_right = (x + int(w * 0.7), y + int(h * 0.4))
            nose = (x + int(w * 0.5), y + int(h * 0.6))
            mouth = (x + int(w * 0.5), y + int(h * 0.75))
            chin = (x + int(w * 0.5), y + h)

            # More points around the face perimeter
            left_cheek = (x + int(w * 0.2), y + int(h * 0.6))
            right_cheek = (x + int(w * 0.8), y + int(h * 0.6))
            jawline_left = (x, y + int(h * 0.8))
            jawline_right = (x + w, y + int(h * 0.8))
            forehead = (x + int(w * 0.5), y)

            # List of points to be drawn
            face_points = [eye_left, eye_right, nose, mouth, chin, left_cheek, right_cheek, jawline_left, jawline_right,
                           forehead]

            # Draw points for all the key facial features
            color = (0, 0, 255)  # Red color for the points
            for point in face_points:
                cv2.circle(frame, point, 5, color, -1)  # Draw a circle at each point

            # Optional: Draw a bounding box around the face (to highlight the detected face region)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle around the face

            # Handle mouth tracking and hand-on-mouth detection
            last_known_points = (
            eye_left, eye_right, nose, mouth, chin, left_cheek, right_cheek, jawline_left, jawline_right, forehead)
            last_mouth_coords = (x, y, x + w, y + h)
            last_face_time = current_time

            if w > TOO_CLOSE_THRESHOLD:
                if current_time - last_posture_alert_time > POSTURE_ALERT_COOLDOWN:
                    last_posture_alert_time = current_time
                    posture_display_until = current_time + POSTURE_DISPLAY_DURATION

                    # Update MongoDB with posture alerts
                    collection.update_one(
                        {"user": "test_user"},
                        {"$push": {"posture_alerts": time.strftime("%Y-%m-%d %H:%M:%S")}},
                        upsert=True
                    )
                    print("[📏] Posture alert: too close to the camera.")

            if frame_diff is not None:
                motion_region = frame_diff[y:y + h, x:x + w]
                if motion_region.size > 0:
                    motion_score = cv2.mean(motion_region)[0]
                    if motion_score > 15:
                        if not hand_motion_detected and (current_time - last_hand_touch_time > HAND_ON_MOUTH_COOLDOWN):
                            hand_on_mouth_count += 1
                            last_hand_touch_time = current_time
                            hand_motion_detected = True
                            print(f"[🤚] Hand on mouth detected. Count = {hand_on_mouth_count}")
                    else:
                        hand_motion_detected = False

            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 1)

            if len(eyes) >= 1:
                if blink_pending:
                    blink_count += 1
                    last_blink_time = current_time
                    blink_pending = False
                    print(f"[👁️] Blink detected. Total = {blink_count}")
                eye_detected_last_frame = True
            else:
                if eye_detected_last_frame:
                    blink_pending = True
                eye_detected_last_frame = False

            # Hand tracking with MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw blue sky tracing around the hand
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
                                              mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))

    if current_time - blink_start_time > BLINK_WINDOW:
        print(f"[📊] Blinks in last 10s: {blink_count}")
        blink_count = 0
        blink_start_time = current_time

    # Update MongoDB with blink count
    collection.update_one(
        {"user": "test_user"},
        {"$set": {"blink_count": blink_count}},
        upsert=True
    )

    # Display the current information on the screen
    cv2.putText(frame, f"Distractions: {int(distraction_time)}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Hand on Mouth: {hand_on_mouth_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Blinks (10s): {blink_count}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if holding_phone:
        duration = int(current_time - phone_held_start)
        cv2.putText(frame, f"Phone Held: {duration}s", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 100, 255), 2)

    if current_time < posture_display_until:
        cv2.putText(frame, "⚠️ Fix posture", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Smart Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n🛑 Exiting system.")
        break

cap.release()
cv2.destroyAllWindows()