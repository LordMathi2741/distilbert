from flask import Flask, request, jsonify
from validation.test import test_inference
import json
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
   try:
        data = request.get_json()
        reviews = data.get('reviews', [])
        results = []
        for review in reviews:
               payload = {
                    "category": review.get("category", ""),
                    "rating": review.get("rating", 0),
                    "text" : review.get("text", "")
               }
               print(payload)
               result = test_inference(json.dumps(payload))
               results.append(result)
        result = results
        return jsonify({'result': result})
   except Exception as e:
        return jsonify({'error': str(e)}), 400

app.run(port=5000)  