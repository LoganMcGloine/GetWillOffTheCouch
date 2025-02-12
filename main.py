import cv2
import numpy as np
import tensorflow.lite as tflite
import pyttsx3  # For text-to-speech yelling

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load the TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path="model_unquant.tflite")

interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("labels.txt", "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

# Open the Raspberry Pi camera
camera = cv2.VideoCapture(0)  # 0 is the default camera ID

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    if not ret:
        break

    # Preprocess the image
    input_size = input_details[0]['shape'][1:3]  # Get input dimensions
    # Resize with proper interpolation
    resized_frame = cv2.resize(frame, (input_size[1], input_size[0]), 
                             interpolation=cv2.INTER_CUBIC)
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    # Normalize to [-1, 1] instead of [0, 1]
    normalized_frame = (np.float32(rgb_frame) - 127.5) / 127.5
    # Add batch dimension
    normalized_frame = np.expand_dims(normalized_frame, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], normalized_frame)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    # Print input shape and predictions for debugging
    print(f"Input shape: {normalized_frame.shape}")
    print(f"Input range: {normalized_frame.min():.2f} to {normalized_frame.max():.2f}")
    for i, confidence in enumerate(predictions):
        print(f"Class {class_labels[i]}: {confidence:.3f}")

    # Get the most confident prediction
    max_index = np.argmax(predictions)
    class_name = class_labels[max_index]
    confidence = predictions[max_index]

    # Check if it's the "Dog on Couch" class
    if class_name == "Dog on Couch" and confidence > 0.8:
        print(f"Dog detected on couch! Confidence: {confidence:.2f}")
        engine.say("Get off the couch!")
        engine.runAndWait()

    # Display the frame (optional)
    cv2.putText(frame, f"{class_name}: {confidence:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Camera Feed", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
camera.release()
cv2.destroyAllWindows()
