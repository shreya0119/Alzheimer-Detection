import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppresses all TF info logs


from flask import Flask, render_template, request
import os
from model_utils import predict_alzheimer

app = Flask(__name__)

# Configure folder for uploaded MRI images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists on your laptop
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        # 1. Save the file locally
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # 2. Get the diagnosis from your .h5 model
        model_path = 'models/alzheimer_pro_weights.weights.h5'
        label, confidence = predict_alzheimer(filepath, model_path)

        # 3. Show the result (Simple version for now)
        return f"""
        <div style="text-align: center; padding: 50px; font-family: sans-serif;">
            <h1>Diagnosis: {label}</h1>
            <h3>Confidence: {confidence}%</h3>
            <hr>
            <a href="/">Go Back and Try Another Scan</a>
        </div>
        """


if __name__ == '__main__':
    app.run(debug=True)