import os
from flask import Flask, request, jsonify, render_template
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define possible diseases
disease_classes = ["healthy grape leaf", "infected grape leaf"]

@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    """Handle image upload and prediction."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]

    # Save the file
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Process the image
    image = Image.open(file_path).convert("RGB")
    inputs = processor(text=disease_classes, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    max_prob_index = probs.argmax()

    # Get prediction
    predicted_disease = disease_classes[max_prob_index]
    predicted_prob = probs[0][max_prob_index].item()

    return jsonify({
        "predicted_disease": predicted_disease,
        "probability": predicted_prob
    })

if __name__ == "__main__":
    app.run(debug=True)
