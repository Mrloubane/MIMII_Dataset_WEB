import os
import logging
import time
import concurrent.futures
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import numpy as np
import librosa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()
# Initialize Flask app
app = Flask(__name__)
CORS(app)

app.config['DEBUG']= os.environ.get('FLASK_DEBUG')
# Configure Upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define thresholds for each model
THRESHOLD_ID00 = 10.21
THRESHOLD_ID02 = 12  
THRESHOLD_ID04 = 11.83 
THRESHOLD_ID06 = 11.36  

# Build model (Autoencoder) functionll
def keras_model():
    inputLayer = Input(shape=(640,))
    h = Dense(64, activation="relu")(inputLayer)
    h = Dense(64, activation="relu")(h)
    h = Dense(8, activation="relu")(h)
    h = Dense(64, activation="relu")(h)
    h = Dense(64, activation="relu")(h)
    h = Dense(640, activation=None)(h)
    return Model(inputs=inputLayer, outputs=h)

# Load the four models
model_id00 = keras_model()
model_id00.load_weights('model_valve_id_00_6_dB_valve.weights.h5')  # Path to the valve ID00 model

model_id02 = keras_model()
model_id02.load_weights('model_valve_id_02_6_dB_valve.weights.h5')  # Path to the valve ID02 model

model_id04 = keras_model()
model_id04.load_weights('model_valve_id_04_6_dB_valve.weights.h5')  # Path to the valve ID04 model

model_id06 = keras_model()
model_id06.load_weights('model_valve_id_06_6_dB_valve.weights.h5')  # Path to the valve ID06 model

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def file_to_vector_array(file_name, n_mels=128, frames=5, n_fft=1024, hop_length=512, power=2.0):
    dims = n_mels * frames
    y, sr = librosa.load(file_name, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power)
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + np.finfo(float).eps)
    
    vectorarray_size = len(log_mel_spectrogram[0, :]) - frames + 1
    if vectorarray_size < 1:
        return np.empty((0, dims), float)

    vectorarray = np.zeros((vectorarray_size, dims), float)
    for t in range(frames):
        vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T
    return vectorarray

# Function to process each chunk in parallel
def process_chunk(chunk, model):
    predictions = model.predict(chunk, batch_size=len(chunk))
    errors = np.mean(np.square(chunk - predictions), axis=1)
    return errors

# Audio analysis function
def analyze_audio(filepath):
    data = file_to_vector_array(filepath)
    chunk_size = 100  # Process 100 vectors at a time
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    errors_id00 = []
    errors_id02 = []
    errors_id04 = []
    errors_id06 = []

    # Use ThreadPoolExecutor to parallelize processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Process chunks in parallel for all four models
        futures_id00 = [executor.submit(process_chunk, chunk, model_id00) for chunk in chunks]
        futures_id02 = [executor.submit(process_chunk, chunk, model_id02) for chunk in chunks]
        futures_id04 = [executor.submit(process_chunk, chunk, model_id04) for chunk in chunks]
        futures_id06 = [executor.submit(process_chunk, chunk, model_id06) for chunk in chunks]

        # Collect the results
        for future in concurrent.futures.as_completed(futures_id00):
            errors_id00.extend(future.result())
        for future in concurrent.futures.as_completed(futures_id02):
            errors_id02.extend(future.result())
        for future in concurrent.futures.as_completed(futures_id04):
            errors_id04.extend(future.result())
        for future in concurrent.futures.as_completed(futures_id06):
            errors_id06.extend(future.result())

    avg_error_id00 = np.mean(errors_id00)
    avg_error_id02 = np.mean(errors_id02)
    avg_error_id04 = np.mean(errors_id04)
    avg_error_id06 = np.mean(errors_id06)

    # Determine the result based on thresholds
    if avg_error_id00 < THRESHOLD_ID00:
        result = "This is a valve, ID: 00 (Normal)"
    elif avg_error_id02 < THRESHOLD_ID02:
        result = "This is a valve, ID: 02 (Normal)"
    elif avg_error_id04 < THRESHOLD_ID04:
        result = "This is a valve, ID: 04 (Normal)"
    elif avg_error_id06 < THRESHOLD_ID06:
        result = "This is a valve, ID: 06 (Normal)"
    else:
        result = "This is abnormal data for all valves."

    return result, avg_error_id00, avg_error_id02, avg_error_id04, avg_error_id06

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    start_time = time.time()
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result, avg_error_id00, avg_error_id02, avg_error_id04, avg_error_id06 = analyze_audio(filepath)
        analysis_time = round(time.time() - start_time, 2)
        
        return render_template('result.html', result=result, filename=filename, 
                               error_id00=avg_error_id00, error_id02=avg_error_id02, 
                               error_id04=avg_error_id04, error_id06=avg_error_id06, 
                               analysis_time=analysis_time)

# Start the Flask server
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run()
