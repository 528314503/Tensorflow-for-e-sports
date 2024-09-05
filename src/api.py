from flask import Flask, request, jsonify
from inference_service import InferenceService

app = Flask(__name__)
inference_service = None  # This should be initialized with the loaded model

@app.route('/recommend', methods=['POST'])
def recommend_events():
    data = request.json
    user_ids = data['user_ids']
    available_event_ids = data['available_event_ids']
    recommendations = inference_service.recommend_events(user_ids, available_event_ids)
    return jsonify(recommendations)

@app.route('/predict_price', methods=['POST'])
def predict_price():
    data = request.json
    event_features = data['event_features']
    optimal_price = inference_service.predict_optimal_price(event_features)
    return jsonify({'optimal_price': float(optimal_price[0])})

if __name__ == '__main__':
    app.run(debug=True)