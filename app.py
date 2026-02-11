from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("agrimind_crop_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return "AgriMind AI Crop Recommendation API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = [
            data["N"],
            data["P"],
            data["K"],
            data["temperature"],
            data["humidity"],
            data["ph"],
            data["rainfall"]
        ]

        final_features = np.array(features).reshape(1, -1)
        prediction = model.predict(final_features)

        return jsonify({
            "recommended_crop": prediction[0]
        })

    except Exception as e:
        return jsonify({"error": str(e)})

    import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


