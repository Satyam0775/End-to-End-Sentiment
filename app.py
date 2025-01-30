from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Sentiment Analysis API!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the review text from the request
        data = request.json
        review_text = data.get("review_text", "")

        # Validate input
        if not review_text:
            return jsonify({"error": "No review_text provided"}), 400

        # Preprocess and predict
        review_vectorized = vectorizer.transform([review_text])
        prediction = model.predict(review_vectorized)

        # Return the result
        sentiment = "positive" if prediction[0] == 1 else "negative"
        return jsonify({"sentiment_prediction": sentiment})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
