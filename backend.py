# backend.py
import google.generativeai as genai
import json
import io
from PIL import Image

# Categories for waste classification
LABELS = [
    "battery", "biological", "cardboard", "clothes", "glass",
    "metal", "paper", "plastic", "shoes", "trash"
]

# Disposal suggestions for each category
SUGGESTIONS = {
    "battery": "Take to battery recycling points. Contains hazardous materials - never dispose in regular trash!",
    "biological": "Dispose in organic waste bins. Can be composted to create nutrient-rich soil.",
    "cardboard": "Flatten and place in recycling bin. Remove any tape or metal staples.",
    "clothes": "Donate if in good condition, or take to textile recycling points.",
    "glass": "Clean and separate by color. Take to glass recycling containers.",
    "metal": "Clean and crush if possible. Check if local recycling accepts the type of metal.",
    "paper": "Keep dry and clean. Bundle together for paper recycling.",
    "plastic": "Clean, check recycling number, and sort according to local guidelines.",
    "shoes": "Donate wearable shoes. For worn-out shoes, check specialty recycling programs.",
    "trash": "If not recyclable or reusable, dispose in general waste as last resort."
}

# Configure Gemini API
def configure_genai(api_key):
    genai.configure(api_key=api_key)

# Main function to analyze waste image
def analyze_waste(pil_image):
    try:
        # Convert PIL Image to bytes
        with io.BytesIO() as bio:
            pil_image.save(bio, format='PNG')
            image_bytes = bio.getvalue()

        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = """
        Analyze this waste item image and classify it into ONE of these categories:
        battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash

        Provide ONLY a JSON response in this exact format:
        {
            "category": "category_name",
            "confidence": 0.95,
            "explanation": "brief explanation"
        }
        """

        response = model.generate_content([
            prompt,
            {"mime_type": "image/png", "data": image_bytes}
        ])
        response.resolve()

        try:
            # Extract the JSON part from the response
            json_str = response.text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:-3]  # Remove ```json and ``` markers
            
            result = json.loads(json_str)
            
            category = result["category"].lower()
            confidence = float(result["confidence"])
            explanation = result["explanation"]

            # Generate probabilities
            probabilities = {label: 0.1 for label in LABELS}
            probabilities[category] = confidence

            # Normalize probabilities
            total = sum(probabilities.values())
            probabilities = {k: v/total for k, v in probabilities.items()}

            return category, confidence, probabilities, explanation

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing response: {response.text}")
            raise ValueError(f"Invalid response format: {str(e)}")

    except Exception as e:
        print(f"Error in analyze_waste: {str(e)}")
        raise e

