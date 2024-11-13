# import streamlit as st
# from PIL import Image
# from ultralytics import YOLO
# import numpy as np
# import os
#
# # Load the YOLOv8 model
# model = YOLO(r'C:\Users\utsav\OneDrive\Desktop\B. Tech\5th semester\ML\projects\SGP\yolov8_model\yolov8m_model.pt')
#
# # Streamlit title
# st.title('Marine Plastic Detection')
#
# # Upload image file
# uploaded_file = st.file_uploader("Choose an image...", type="jpg")
#
# if uploaded_file is not None:
#     # Load the image
#     image = Image.open(uploaded_file)
#
#     # Display the uploaded image
#     st.image(image, caption='Uploaded Image.', use_column_width=True)
#     st.write("Classifying...")
#
#     # Perform detection
#     results = model(image)
#
#     # Access the first result from the list
#     result = results[0]
#
#     # Display the results using pandas dataframe (bounding boxes, labels, confidence scores)
#     st.write(result.boxes.xyxy)  # Coordinates of bounding boxes
#     st.write(result.boxes.conf)  # Confidence scores
#     st.write(result.names)  # Class names
#
#     # Plot the results (returns numpy array)
#     annotated_image = result.plot()
#
#     # Convert the numpy array (annotated_image) to a PIL image
#     annotated_pil_image = Image.fromarray(np.uint8(annotated_image))
#
#     # Create an 'output' folder if it doesn't exist
#     output_dir = 'output'
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Save the annotated image to the 'output' directory
#     save_path = os.path.join(output_dir, 'annotated_image.jpg')
#     annotated_pil_image.save(save_path)
#
#     # Display the saved image with bounding boxes
#     st.image(save_path, caption='Detected Image with Bounding Boxes', use_column_width=True)
# import streamlit as st
# from PIL import Image
# from ultralytics import YOLO
# import numpy as np
# import os
#
# # Define the paths for your models
# model_paths = {
#     "   YOLO V8 NANO": r'C:\Users\utsav\OneDrive\Desktop\B. Tech\5th semester\ML\projects\SGP\yolov8_model\yolov8_model.pt',
#     "YOLO V8 MEDIUM": r'C:\Users\utsav\OneDrive\Desktop\B. Tech\5th semester\ML\projects\SGP\yolov8_model\yolov8m_model.pt',
#     "YOLO V8 SMALL": r'C:\Users\utsav\OneDrive\Desktop\B. Tech\5th semester\ML\projects\SGP\yolov8_model\yolov8small_model.pt',
#     "YOLO V10 NANO": r'C:\Users\utsav\OneDrive\Desktop\B. Tech\5th semester\ML\projects\SGP\yolov8_model\yolov10n.pt',
#     "YOLO V10 SMALL": r'C:\Users\utsav\OneDrive\Desktop\B. Tech\5th semester\ML\projects\SGP\yolov8_model\yolov10small_model.pt',
# }
#
# # Streamlit title
# st.title('Marine Plastic Detection')
#
# # Dropdown to select model
# model_choice = st.selectbox("Choose a YOLOv8 model:", list(model_paths.keys()))
#
# # Load the selected model
# model = YOLO(model_paths[model_choice])
#
# # Upload image file
# uploaded_file = st.file_uploader("Choose an image...", type="jpg")
#
# if uploaded_file is not None:
#     # Load the image
#     image = Image.open(uploaded_file)
#
#     # Display the uploaded image
#     st.image(image, caption='Uploaded Image.', use_column_width=True)
#     st.write("Classifying...")
#
#     # Perform detection
#     results = model(image)
#
#     # Access the first result from the list
#     result = results[0]
#
#     # Display the results using pandas dataframe (bounding boxes, labels, confidence scores)
#     st.write(result.boxes.xyxy)  # Coordinates of bounding boxes
#     st.write(result.boxes.conf)  # Confidence scores
#     st.write(result.names)  # Class names
#
#     # Plot the results (returns numpy array)
#     annotated_image = result.plot()
#
#     # Convert the numpy array (annotated_image) to a PIL image
#     annotated_pil_image = Image.fromarray(np.uint8(annotated_image))
#
#     # Create an 'output' folder if it doesn't exist
#     output_dir = 'output'
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Save the annotated image to the 'output' directory
#     save_path = os.path.join(output_dir, 'annotated_image.jpg')
#     annotated_pil_image.save(save_path)
#
#     # Display the saved image with bounding boxes
#     st.image(save_path, caption='Detected Image with Bounding Boxes', use_column_width=True)
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import os
import tempfile
import cv2

# Define the paths for your models
model_paths = {
    "YOLO V8 NANO": r'C:\Users\utsav\OneDrive\Desktop\B. Tech\5th semester\ML\projects\SGP\yolov8_model\yolov8_model.pt',
    "YOLO V8 MEDIUM": r'C:\Users\utsav\OneDrive\Desktop\B. Tech\5th semester\ML\projects\SGP\yolov8_model\yolov8m_model.pt',
    "YOLO V8 SMALL": r'C:\Users\utsav\OneDrive\Desktop\B. Tech\5th semester\ML\projects\SGP\yolov8_model\yolov8small_model.pt',
     "YOLO V10 NANO": r'C:\Users\utsav\OneDrive\Desktop\B. Tech\5th semester\ML\projects\SGP\yolov8_model\yolov10nano_model-2 (1).pt',
    "YOLO V10 SMALL": r'C:\Users\utsav\OneDrive\Desktop\B. Tech\5th semester\ML\projects\SGP\yolov8_model\yolov10small_model.pt',
}

# Streamlit title
st.title('Marine Pollution Detection')

# Dropdown to select model
model_choice = st.selectbox("Choose a YOLOv8 model:", list(model_paths.keys()))

# Load the selected model
model = YOLO(model_paths[model_choice])

# Option to choose between image or video input
input_type = st.radio("Choose input type", ('Image', 'Video'))

# Image detection
if input_type == 'Image':
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Classifying...")

        # Perform detection
        results = model(image)

        # Access the first result from the list
        result = results[0]

        # Display the results using pandas dataframe (bounding boxes, labels, confidence scores)
        st.write(result.boxes.xyxy)  # Coordinates of bounding boxes
        st.write(result.boxes.conf)  # Confidence scores
        st.write(result.names)  # Class names

        # Plot the results (returns numpy array)
        annotated_image = result.plot()

        # Convert the numpy array (annotated_image) to a PIL image
        annotated_pil_image = Image.fromarray(np.uint8(annotated_image))

        # Create an 'output' folder if it doesn't exist
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)

        # Save the annotated image to the 'output' directory
        save_path = os.path.join(output_dir, 'annotated_image.jpg')
        annotated_pil_image.save(save_path)

        # Display the saved image with bounding boxes
        st.image(save_path, caption='Detected Image with Bounding Boxes', use_column_width=True)

# Video detection
elif input_type == 'Video':
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        # Open the video using OpenCV
        video_capture = cv2.VideoCapture(video_path)

        # Create an 'output' folder if it doesn't exist
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)

        # Define video writer to save the annotated video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
        out_path = os.path.join(output_dir, 'annotated_video.mp4')
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

        stframe = st.empty()  # Placeholder for displaying video frames

        # Process video frame by frame
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            # Convert frame to RGB (YOLOv8 expects RGB format)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform detection on the frame
            results = model(frame_rgb)

            # Annotate the frame with detection results
            annotated_frame = results[0].plot()

            # Convert back to BGR for OpenCV compatibility and write to output
            annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            out.write(annotated_frame_bgr)

            # Display the frame in Streamlit app
            stframe.image(annotated_frame)

        # Release resources
        video_capture.release()
        out.release()

        # Display download link for the annotated video
        st.video(out_path, format="video/mp4")
        st.write(f"Annotated video saved at: {out_path}")
