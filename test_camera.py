import cv2
import mediapipe as mp
import time

def main():
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # 0 targets your built-in laptop camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the built-in laptop camera.")
        print("Check Windows Settings -> Privacy & Security -> Camera to ensure desktop access is ON.")
        return

    # Set frame dimensions to 640x480 for smooth performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = 0
    print("Camera active. Press 'q' on your keyboard while focusing on the camera window to exit.")

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as detector:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Mirror view for natural laptop camera feel
            frame = cv2.flip(frame, 1)

            # Convert BGR (OpenCV) to RGB (MediaPipe)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb_frame)

            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time

            # Draw face detection boxes and landmark points
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(frame, detection)

            # Draw live FPS
            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Smart Attendance - Milestone 1 Test", frame)

            # Press 'q' to close the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()