from model.model import ColorizationNet, colorize_and_test
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import torch
import os
import numpy as np
import base64
from PIL import Image
import io
import cv2
import base64
from flask import Flask, send_from_directory

# Initialize Flask app
app = Flask(__name__)
CORS(app)
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the model once during initialization
MODEL_PATH = "backend\second_trained_model.pth"
print(MODEL_PATH)
if os.path.exists(MODEL_PATH):
    model = ColorizationNet().to(device)
    model.load_state_dict(torch.load('backend\second_trained_model.pth',map_location=device))
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully!")
else:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Convert image to base64
def image_to_base64(arr):
    # Ensure the array is of type uint8
        if arr.dtype != np.uint8 or arr.min() < 0 or arr.max() > 255:
            # Normalize the array to the range [0, 255]
            arr = arr - arr.min()  # Shift the data to start from 0
            arr_max = arr.max()
            if arr_max > 0:
                arr = (arr / arr_max) * 255
            arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        
        # Convert the NumPy array to a PIL Image
        img = Image.fromarray(arr, mode='RGB')

        # Save the image to a bytes buffer in PNG format
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)

        # Read the bytes and encode to base64
        img_bytes = buffer.read()
        img_base64 = base64.b64encode(img_bytes)

        # Convert base64 bytes to string
        img_base64_str = img_base64.decode('utf-8')

        return img_base64_str

def numpy_to_pillow(np_array):
    """
    Converts a NumPy array to a Pillow Image.
    
    Args:
        np_array (np.ndarray): Input NumPy array representing an image.
    
    Returns:
        PIL.Image.Image: Pillow Image object.
    """
    # Ensure the array is uint8
    if np_array.dtype != np.uint8:
        np_array = np.clip(np_array, 0, 255).astype(np.uint8)

    # Convert NumPy array to Pillow Image
    try:
        return Image.fromarray(np_array)
    except ValueError as e:
        raise ValueError(f"Invalid array shape: {np_array.shape}. Ensure it's (H, W) for grayscale or (H, W, 3) for RGB.") from e


@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Open the uploaded image as a PIL Image
        print("IMAGE PARSED TO FLASK")
        image = Image.open(file).convert("RGB")
        image_array = np.array(image)
        print(image_array.size)
        # Process the image using the model
        result_image = colorize_and_test(model,  image)
        print(result_image.shape)
        print(type(result_image))

        
        print("RESULT IMAGE COLORIZE AND TEST FUNC")
        # Convert the numpy array back to a PIL Image

        # Resize result_image to match image_array dimensions
        result_image_resized = cv2.resize(result_image, (image_array.shape[1], image_array.shape[0]))
        # Convert the processed image to base64 for transmission to frontend
        img_base64 = image_to_base64(result_image_resized)
        #print(img_base64)

        # Return the processed image and other relevant information
        result = {
            "prediction": "Colorized",  # This could be customized based on your model
            "accuracy": 95,  # Example value; update according to your model
            "RGB": "N/A",  # If applicable, replace with RGB values if necessary
            "processed_image": img_base64
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
