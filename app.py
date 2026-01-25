from flask import Flask, jsonify, request
from model import DockerSklearnClassifier

app = Flask(__name__)

# Train model at startup (in production, load from saved file)
classifier = DockerSklearnClassifier()
classifier.train()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "DockerSklearn API"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body: {"features": [float, float, float, float]}
    Returns churn probability.
    """
    data = request.get_json() or {}
    features = data.get("features")

    if not isinstance(features, list) or len(features) != 4:
        return (
            jsonify(
                {
                    "error": "Must provide 'features' as list of exactly 4 floats",
                    "example": {"features": [0.1, -1.2, 3.4, 0.0]},
                }
            ),
            400,
        )

    try:
        proba = classifier.predict_proba(features)
        return jsonify({"churn_probability": proba}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify(
        {
            "model_type": "LogisticRegression",
            "input_features": 4,
            "output": "churn_probability",
            "service": "DockerSklearn API",
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
