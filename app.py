from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# تحميل الـ preprocessor والموديل
preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('model.pkl')

# إعداد تطبيق Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "Real Estate Ensemble Price Prediction API is Running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # تحويل الداتا إلى DataFrame
        input_df = pd.DataFrame([data])

        # المعالجة بنفس الـ preprocessor الذي دربناه
        input_processed = preprocessor.transform(input_df)

        # التنبؤ
        y_pred_log = model.predict(input_processed)
        y_pred = np.expm1(y_pred_log)

        return jsonify({'predicted_price': float(y_pred[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
