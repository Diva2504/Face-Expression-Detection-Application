
import cv2
import numpy as np
import streamlit as st
from tensorflow import keras
from keras.models import load_model
from keras.utils import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

# classes name
classes_dict = {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happiness', 5: 'neutral', 6: 'sadness',
                7: 'surprise'}

# load model
model_classifier = load_model('Model-3-FER-PLUS-epoch-100.h5')

# load face frame
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


class FaceExpression(VideoTransformerBase):
    def __init__(self):
        self.capture_frame = None

    def transform(self, frame):
        # convert frame to arraynum
        img = frame.to_ndarray(format="bgr24")
        # convert from bgr to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(0, 255, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi_input = roi_gray.astype('float') / 255.0
                roi_input = img_to_array(roi_input)
                roi_input = np.expand_dims(roi_input, axis=0)
                prediction = model_classifier.predict(roi_input)[0]
                maxindex = int(np.argmax(prediction))
                finalout = classes_dict[maxindex]
                accuracy = prediction[maxindex]
                output_label = str(finalout)
                output_accuracy = f'Accuracy: {accuracy:.2f}'
            label_position = (x, y)
            accuracy_position = (x, y + h + 20)
            cv2.putText(img, output_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, output_accuracy, accuracy_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return img

    def capture_image(self):
        # Display the captured frame with bounding boxes and classification results
        if self.capture_frame is not None:
            st.image(self.capture_frame, channels="BGR", caption="Captured Image", use_column_width=True)
            st.write("Image captured!")


# Function to upload image and perform classification
def upload_and_classify():
    allowed_image_formats = ["jpg", "jpeg", "png", "webp", "heic"]

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp", "heic"])

    if uploaded_file is not None:
        # Convert the uploaded image to numpy array
        image = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, 1)

        # Convert the uploaded image to grayscale
        uploaded_image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Initialize the variables
        faces = face_cascade.detectMultiScale(uploaded_image_gray, scaleFactor=1.3, minNeighbors=5)

        # Check if any face is detected
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Draw bounding box around detected face
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Perform emotion classification
                roi_gray = uploaded_image_gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi_input = roi_gray.astype('float') / 255.0
                    roi_input = img_to_array(roi_input)
                    roi_input = np.expand_dims(roi_input, axis=0)
                    prediction = model_classifier.predict(roi_input)[0]
                    maxindex = int(np.argmax(prediction))
                    finalout = classes_dict[maxindex]
                    accuracy = prediction[maxindex]
                    output_label = str(finalout)
                    output_accuracy = f'Accuracy: {accuracy:.2f}'

                    # Display the label and accuracy on the bounding box
                    label_position = (x, y - 10)
                    accuracy_position = (x, y + h + 20)
                    cv2.putText(image, output_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, output_accuracy, accuracy_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                                2)

        # Display the uploaded image with bounding boxes and classification results
        st.image(image, channels="BGR")
        if len(faces) > 0:
            st.write(f"Number of Faces Detected: {len(faces)}")
            for i, (x, y, w, h) in enumerate(faces):
                st.write(f"Face {i + 1} - Expression Prediction: {output_label}, {output_accuracy}")
        else:
            st.write("No faces detected.")

def capture_and_classify():
    captured_file = st.camera_input("take picture")
    if captured_file is not None:
        # Convert the uploaded image to numpy array
        image = np.asarray(bytearray(captured_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, 1)

        # Convert the uploaded image to grayscale
        uploaded_image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Initialize the variables
        faces = face_cascade.detectMultiScale(uploaded_image_gray, scaleFactor=1.3, minNeighbors=5)

        # Check if any face is detected
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Draw bounding box around detected face
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Perform emotion classification
                roi_gray = uploaded_image_gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi_input = roi_gray.astype('float') / 255.0
                    roi_input = img_to_array(roi_input)
                    roi_input = np.expand_dims(roi_input, axis=0)
                    prediction = model_classifier.predict(roi_input)[0]
                    maxindex = int(np.argmax(prediction))
                    finalout = classes_dict[maxindex]
                    accuracy = prediction[maxindex]
                    output_label = str(finalout)
                    output_accuracy = f'Accuracy: {accuracy:.2f}'

                    # Display the label and accuracy on the bounding box
                    label_position = (x, y - 10)
                    accuracy_position = (x, y + h + 20)
                    cv2.putText(image, output_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, output_accuracy, accuracy_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                                2)

        # Display the uploaded image with bounding boxes and classification results
        st.image(image, channels="BGR")
        if len(faces) > 0:
            st.write(f"Number of Faces Detected: {len(faces)}")
            for i, (x, y, w, h) in enumerate(faces):
                st.write(f"Face {i + 1} - Expression Prediction: {output_label}, {output_accuracy}")
        else:
            st.write("No faces detected.")

def main():
    # Face Analysis Application #
    st.title("Face Expression Detection Application")
    activiteis = ["Home", "Webcam Face Detection", "Upload Image", "Capture Image", "About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """ Developed by I Gede Diva Dwijayana    
            Email : vaadivv@gmail.com """)
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face Emotion detection application using OpenCV, Custom CNN model and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The application has two functionalities.

                 1. Real time face detection using web cam feed.

                 2. Real time face emotion recognization.

                 3. upload an image for emotion classification.
                 
                 4. capture from camera.

                 """)
    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=FaceExpression)

    elif choice == "Upload Image":
        st.header("Upload Image")
        upload_and_classify()

    elif choice == "Capture Image":
        st.header("Capture Image")
        capture_and_classify()

    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1 = """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    face expression detection application using OpenCV, Custom Trained CNN model and Streamlit.</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                             		<div style="background-color:#98AFC7;padding:10px">
                             		<h4 style="color:white;text-align:center;">This Application is developed by I Gede Diva Dwijayana using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose.</h4>
                             		<h4 style="color:white;text-align:center;">Thanks for Using</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass


if __name__ == "__main__":
    main()
