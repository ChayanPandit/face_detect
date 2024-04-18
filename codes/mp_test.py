import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection()
face_mesh = mp_face_mesh.FaceMesh()

# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Function to detect face, draw landmarks, and estimate face coverage
def process_frame(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detection_results = face_detection.process(image_rgb)
    mesh_results = face_mesh.process(image_rgb)

    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:
            # Draw facial landmarks
            mp.solutions.drawing_utils.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

            # Assuming landmarks 0 and 17 are around the forehead and 13 and 14 are around the chin
            # These numbers can vary based on the model's landmark schema
            forehead = (face_landmarks.landmark[10].x, face_landmarks.landmark[10].y)
            chin = (face_landmarks.landmark[152].x, face_landmarks.landmark[152].y)
            nose_tip = (face_landmarks.landmark[4].x, face_landmarks.landmark[4].y)

            # Estimate face coverage by comparing distances
            forehead_to_chin = calculate_distance(forehead, chin)
            forehead_to_nose = calculate_distance(forehead, nose_tip)

            if forehead_to_nose / forehead_to_chin < 0.5:  # Threshold to determine coverage
                cv2.putText(image, "Face Covered", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, "Face Uncovered", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return image

# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = process_frame(frame)
    cv2.imshow('MediaPipe Face Detection and Coverage Estimation', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to break
        break

cap.release()
cv2.destroyAllWindows()
