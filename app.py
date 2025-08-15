# app.py
from flask import Flask, request, jsonify, render_template, url_for
from backend import analyze_waste, configure_genai, SUGGESTIONS
from PIL import Image
import io

app = Flask(__name__)

# Replace with your actual API key
API_KEY = "API_KEY_HERE"
configure_genai(API_KEY)

@app.route("/")
def home():
    return render_template("frontend.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    
    try:
        # Read the image file
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Get prediction
        predicted_label, confidence, probabilities, explanation = analyze_waste(image)
        
        return jsonify({
            "prediction": predicted_label,
            "confidence": round(confidence * 100, 2),
            "suggestion": SUGGESTIONS[predicted_label],
            "explanation": explanation,
            "probabilities": {label: round(prob * 100, 2) for label, prob in probabilities.items()}
        })
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
