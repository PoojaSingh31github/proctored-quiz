import streamlit as st
import tensorflow as tf
import cv2
import json
import numpy as np

# Load the quiz questions
with open('question.json') as f:
    questions = json.load(f)

# Load the Teachable Machine model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open('labels.txt', 'r') as f:
    labels = f.read().strip().split('\n')

def predict(frame):
    input_shape = input_details[0]['shape']
    input_size = (input_shape[1], input_shape[2])
    
    frame_resized = cv2.resize(frame, input_size)
    frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)  # Ensure 3 channels
    input_data = np.expand_dims(frame_resized, axis=0).astype(np.float32) / 255.0
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output_data), output_data

def main():
    st.title("AI-Powered Proctored Quiz Application")

    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
        st.session_state.score = 0
        st.session_state.alerts = 0
        st.session_state.timer = 300  # 5 minutes timer

    # Timer countdown
    if st.session_state.timer > 0:
        st.session_state.timer -= 1
    else:
        st.write("Time's up!")
        st.stop()

    st.write(f"Time remaining: {st.session_state.timer} seconds")

    # Display current question
    question = questions[st.session_state.current_question]
    st.write(question["question"])
    options = question["options"]
    answer = st.radio("Choose your answer:", options)

    # Webcam setup
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()
    if ret:
        st.image(frame, channels="BGR")

        # Predict user attention
        prediction, output_data = predict(frame)
        predicted_label = labels[prediction]

        if predicted_label != "Looking at Screen":
            st.session_state.alerts += 1
            st.warning(f"Alert {st.session_state.alerts}: Please stay focused on the screen!")

        if st.session_state.alerts >= 3:
            st.error("Too many alerts. The quiz is now stopped.")
            st.stop()

    video_capture.release()

    # Check answer
    if st.button("Submit Answer"):
        if answer == question["answer"]:
            st.session_state.score += 1

        st.session_state.current_question += 1
        if st.session_state.current_question >= len(questions):
            st.write(f"Quiz finished! Your score: {st.session_state.score}/{len(questions)}")
            st.stop()

    st.write(f"Question {st.session_state.current_question + 1} of {len(questions)}")

if __name__ == "__main__":
    main()
