# main.py
# To run this Flask API:
# 1. Install dependencies: pip install Flask requests numpy Pillow
# 2. Save this code as a Python file (e.g., app.py).
# 3. Save the mock_skin_analysis.py file in the same directory.
# 4. Run from your terminal: python app.py

from flask import Flask, request, jsonify
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import json
import traceback
import os

# Import the mock analysis function from the other file
# In your actual implementation, you would import your real 'analyze_image_all_regions'
from skin_analysis import analyze_image_all_regions

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Helper Functions ---

def process_image_from_url(url: str) -> np.ndarray:
    """
    Downloads an image from a given URL, converts it to RGB,
    and returns it as a NumPy array.

    Args:
        url: The URL of the image to process.

    Returns:
        A NumPy array representing the image.
        
    Raises:
        requests.exceptions.RequestException: If the image download fails.
        IOError: If the image data is corrupt or not a valid image.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return np.array(image)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image from {url}: {e}")
        raise
    except IOError as e:
        print(f"Error processing image data from {url}: {e}")
        raise


# --- API Endpoints ---

@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    """
    The main API endpoint to trigger skin analysis.
    It expects a JSON payload with 'imageUrl' and returns the
    analysis result directly.
    """
    print("Received a new analysis request.")

    # 1. Get and validate the incoming JSON data
    try:
        data = request.get_json()
        if not data or 'imageUrl' not in data:
            return jsonify({"error": "Invalid request. 'imageUrl' is required."}), 400
        
        image_url = data['imageUrl']
        print(f"Processing image from URL: {image_url}")

    except Exception as e:
        return jsonify({"error": "Failed to parse request JSON.", "details": str(e)}), 400

    try:
        # 2. Process the image from the URL
        image_np = process_image_from_url(image_url)

        # 3. Perform the image analysis (using the imported function)
        analysis_result = analyze_image_all_regions(image_np)

        print("Analysis complete. Returning results.")
        
        # 4. Return the analysis result as a JSON response
        return jsonify(analysis_result), 200

    except Exception as e:
        # Catch any exception during processing, from image download to analysis
        print("An error occurred during the analysis process.")
        print(traceback.format_exc())
        return jsonify({"error": "An internal error occurred.", "details": str(e)}), 500


@app.route('/', methods=['GET'])
def index():
    """A simple health-check endpoint."""
    return jsonify({"status": "API is running"}), 200


# --- Main execution block ---
if __name__ == '__main__':
    # Runs the Flask app. 'debug=True' provides detailed error pages
    # and reloads the server automatically when you save changes.
    # For production, use a proper WSGI server like Gunicorn or uWSGI.
    app.run(debug=True, port=int(os.environ.get("PORT", 5001)))