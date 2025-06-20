from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

# تحميل الـ preprocessor والموديل
preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('model.pkl')

# إعداد تطبيق Flask
app = Flask(__name__)
CORS(app)

# نقطة البداية (home)
@app.route('/')
def home():
    return "✅ Real Estate Ensemble Price Prediction API is Running."

# مسار التنبؤ
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # تحويل البيانات إلى DataFrame
        input_df = pd.DataFrame([data])

        # تطبيق المعالجة المسبقة
        input_processed = preprocessor.transform(input_df)

        # إجراء التنبؤ
        y_pred_log = model.predict(input_processed)
        y_pred = np.expm1(y_pred_log)  # Inverse of log1p

        return jsonify({'predicted_price': float(y_pred[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

# لتشغيل الخادم محليًا فقط
if __name__ == '__main__':
    app.run(debug=True)
