import cv2
import mediapipe as mp
from deepface import DeepFace


def draw_face_rectangle(frame, face_rect):
    x, y, w, h = int(face_rect.xmin), int(face_rect.ymin), int(face_rect.width), int(face_rect.height)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_pose_landmarks(frame, landmarks, pose_connections):
    for connection in pose_connections:
        start_point, end_point = connection
        start_landmark = landmarks.landmark[start_point]
        end_landmark = landmarks.landmark[end_point]
        start_x, start_y = int(start_landmark.x * frame.shape[1]), int(start_landmark.y * frame.shape[0])
        end_x, end_y = int(end_landmark.x * frame.shape[1]), int(end_landmark.y * frame.shape[0])
        cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)


def main():
    mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            face_results = mp_face_detection.process(frame_rgb)
            if face_results.detections:
                for detection in face_results.detections:
                    draw_face_rectangle(frame, detection.location_data.relative_bounding_box)

            pose_results = pose.process(frame_rgb)
            if pose_results.pose_landmarks:
                draw_pose_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            result = DeepFace.analyze(frame_rgb, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']
            dominant_emotion_confidence = result[0]['emotion'][dominant_emotion]

            cv2.putText(frame, f'Dominant Emotion: {dominant_emotion}', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 0), 2)
            cv2.putText(frame, f'Confidence: {dominant_emotion_confidence:.2f}%', (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(result)
        except ValueError as e:
            print(f"Error: {e}")

        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


