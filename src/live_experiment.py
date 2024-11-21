import cv2
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf
import matplotlib.patches as patches


# Load the pre-trained model
model = tf.keras.models.load_model('/Users/khangbuiphuoc/Study/Computer_Science/ComputerVision/DrowsinessDetection/DrowsinessTracking/SimpleModel.keras') 
img_size = (128, 128)  # Model input size

# Edit the video path
video_path = "/Users/khangbuiphuoc/Study/Computer_Science/ComputerVision/DrowsinessDetection/Drowsiness_Val.mp4"
cap = cv2.VideoCapture(video_path)

# Function to get bounding boxes for eyes
def get_eye_bounding_boxes(left_eye, right_eye, box_size=128):
    left_x, left_y = left_eye
    right_x, right_y = right_eye

    left_bbox = [int(left_x - box_size // 2), int(left_y - box_size // 2),
                 int(left_x + box_size // 2), int(left_y + box_size // 2)]
    right_bbox = [int(right_x - box_size // 2), int(right_y - box_size // 2),
                  int(right_x + box_size // 2), int(right_y + box_size // 2)]

    return left_bbox, right_bbox

# Real-time drowsiness detection function
def process_real_time(model, detector, closed_frame_threshold=10):
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Approximate FPS if unavailable
    closed_frames = 0  # Counter for consecutive closed-eye frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB (MTCNN expects RGB)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces and landmarks
        detections = detector.detect_faces(image_rgb)

        eye_closed = False  # Flag to check if eyes are closed in the current frame

        for detection in detections:
            # Extract face bounding box
            x, y, width, height = detection['box']
            x, y = max(0, x), max(0, y)  # Ensure bounding box is within frame bounds

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)

            # Extract eye landmarks
            keypoints = detection['keypoints']
            left_eye = keypoints['left_eye']
            right_eye = keypoints['right_eye']

            # Get bounding boxes for eyes
            left_bbox, right_bbox = get_eye_bounding_boxes(left_eye, right_eye)
            for (x1, y1, x2, y2) in [left_bbox, right_bbox]:
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Process each eye region
            for bbox in [left_bbox, right_bbox]:
                x1, y1, x2, y2 = bbox
                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                eye_region = frame[y1:y2, x1:x2]

                # Preprocess eye region
                if eye_region.size > 0:
                    eye_region_resized = cv2.resize(eye_region, img_size)  # Resize to model input size
                    eye_region_gray = cv2.cvtColor(eye_region_resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                    eye_region_normalized = eye_region_gray / 255.0  # Normalize
                    eye_region_input = np.expand_dims(eye_region_normalized, axis=(0, -1))  # Add batch & channel dims

                    # Predict eye state (1 = open, 0 = closed)
                    prediction = model.predict(eye_region_input)
                    if prediction[0] < 0.5:  # Threshold for closed eyes
                        eye_closed = True
                        break  # Exit early if any eye is closed

        # Handle drowsiness detection
        if eye_closed:
            closed_frames += 1
            if closed_frames > closed_frame_threshold:
                cv2.putText(frame, "DROWSY!", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 7)
        else:
            closed_frames = 0  # Reset counter if eyes are open

        # Display the video frame with annotations
        cv2.imshow("Drowsiness Detection", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

"""
def process_video(video_path, output_path, model, detector, closed_frame_threshold=30):
    
    # Processes a video to detect drowsiness by analyzing eye closure frames using MTCNN and a CNN model.

    # Args:
    #     video_path (str): Path to the input video.
    #     output_path (str): Path to save the processed video.
    #     model: Trained CNN model for eye state classification.
    #     detector: MTCNN face and landmark detector.
    #     closed_frame_threshold (int): Threshold of consecutive frames with closed eyes to trigger an alert.
    
    cap = cv2.VideoCapture(video_path)
    closed_frames = 0  # Count of consecutive frames with closed eyes
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second of the video

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)), 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert frame to RGB (MTCNN expects RGB images)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces and landmarks
        detections = detector.detect_faces(image_rgb)

        eye_closed = False  # Flag to check if any eye is closed in the current frame
        alert_frames = 0  # Tracks how long the alert is displayed
        show_alert = False  # Flag to control alert visibility

        for detection in detections:
            # Extract face bounding box
            x, y, width, height = detection['box']
            x, y = max(0, x), max(0, y)  # Ensure bounding box is within frame bounds

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)

            # Extract eye landmarks
            keypoints = detection['keypoints']
            left_eye = keypoints['left_eye']
            right_eye = keypoints['right_eye']

            # Define bounding boxes for eyes
            left_bbox, right_bbox = get_eye_bounding_boxes(left_eye, right_eye)
            for (x1, y1, x2, y2) in [left_bbox, right_bbox]:
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)    
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Extract and preprocess eye regions
            for bbox in [left_bbox, right_bbox]:
                x1, y1, x2, y2 = bbox
                eye_region = frame[y1:y2, x1:x2]

                # Preprocess eye region for model prediction
                eye_region_resized = cv2.resize(eye_region, (128, 128))  # Resize to model input size
                eye_region_gray = cv2.cvtColor(eye_region_resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                eye_region_normalized = eye_region_gray / 255.0  # Normalize
                eye_region_input = np.expand_dims(eye_region_normalized, axis=(0, -1))  # Add batch and channel dims

                # Predict eye state (1 = open, 0 = closed)
                prediction = model.predict(eye_region_input)
                if prediction[0] < 0.5:  # Threshold for closed eyes
                    eye_closed = True
                    break  # Exit early if any eye is closed

        # Handle drowsiness detection based on eye state
        if eye_closed:
            closed_frames += 1
            closed_time = closed_frames / fps  # Calculate time in seconds
            # Trigger alert if closed time exceeds the threshold
            if closed_frames > closed_frame_threshold and not show_alert:
                show_alert = True
                alert_frames = 0  # Reset alert duration counter
        else:
            closed_frames = 0  # Reset closed frame counter if eyes are open
            show_alert = False  # Reset alert if no drowsiness detected
            
            
        # Display the alert for a fixed duration
        if show_alert:
            alert_frames += 1
            if alert_frames <= 60:
                cv2.putText(frame, "DROWSY!", (50, 250), 
                            cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 0, 255), 4)
                cv2.putText(frame, f"Closed Time: {closed_frames / fps:.2f}s", (50, 400), 
                            cv2.FONT_HERSHEY_SIMPLEX, 4.5, (255, 255, 255), 3)
            else:
                show_alert = False  # Stop showing the alert after 2 seconds

            

        # Write the processed frame to output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
"""

# Initialize MTCNN detector
detector = MTCNN()

# Run the real-time detection
process_real_time(model, detector)

output_path = "/Users/khangbuiphuoc/Study/Computer_Science/ComputerVision/DrowsinessDetection/Drowsiness_Val_output.avi"  # Output file path
# process_video(video_path, output_path, model, detector)
