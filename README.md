
# MIMII Dataset Web Application

This repository contains a Flask-based web application that analyzes audio files to detect anomalies in industrial machine sounds using the MIMII dataset. The application utilizes pre-trained autoencoder models to classify the audio data and provide feedback on the state of different valves.

## Features

- **Audio File Upload:** Upload `.wav`, `.mp3`, or `.ogg` files for analysis.
- **Autoencoder Models:** Use pre-trained autoencoder models to detect normal or abnormal machine sounds in valve audio data.
- **Parallel Processing:** Speed up analysis with multi-threading using Python's `concurrent.futures`.
- **User-Friendly Web Interface:** An easy-to-use Flask web interface for audio analysis and result visualization.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Mrloubane/MIMII_Dataset_WEB.git
cd MIMII_Dataset_WEB
```

### 2. Install Dependencies

Make sure you have Python 3.7+ installed. Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the root directory to configure the Flask application:

```bash
FLASK_DEBUG=True
```

### 4. Download Pre-Trained Models

Ensure that the pre-trained model weights for each valve are placed in the root directory:

- `model_valve_id_00_6_dB_valve.weights.h5`
- `model_valve_id_02_6_dB_valve.weights.h5`
- `model_valve_id_04_6_dB_valve.weights.h5`
- `model_valve_id_06_6_dB_valve.weights.h5`

### 5. Create Uploads Folder

Create a directory named `uploads/` where the uploaded audio files will be saved:

```bash
mkdir uploads
```

### 6. Run the Flask App

Start the Flask server locally:

```bash
python app.py
```

Once the server is running, open your browser and go to `http://127.0.0.1:5000` to access the application.

## Usage

1. Upload an audio file (`.wav`, `.mp3`, `.ogg`) through the web interface.
2. The app will analyze the file using pre-trained autoencoder models.
3. Results, including the classification of the valve state and error metrics, will be displayed on the results page.

## Project Structure

```
MIMII_Dataset_WEB/
│
├── app.py                     # Main Flask application
├── templates/                 # HTML templates for Flask
│   ├── index.html             # Home page
│   ├── info.html              # Info page
│   └── result.html            # Result display page
│
├── uploads/                   # Directory for uploaded audio files
├── requirements.txt           # Python dependencies
├── README.md                  # Project README file
└── .env                       # Environment configuration (not included in repo)
```

## Audio Processing Pipeline

1. **Audio Upload:** The user uploads an audio file in `.wav`, `.mp3`, or `.ogg` format.
2. **Feature Extraction:** The application uses `librosa` to extract mel-spectrogram features from the audio.
3. **Model Prediction:** Pre-trained autoencoder models process the audio features, calculating reconstruction errors to classify the data as normal or abnormal for different valves.
4. **Results:** The application compares reconstruction errors with predefined thresholds to determine the state of the valves and presents the results to the user.

## Models

The application uses four pre-trained autoencoder models, one for each valve:

- Valve ID: **00**
- Valve ID: **02**
- Valve ID: **04**
- Valve ID: **06**

### Model Training

The models were trained on the MIMII dataset, which consists of industrial machine sounds. The models were trained to recognize normal sounds and flag abnormal ones based on reconstruction errors.

## Dependencies

- Flask
- TensorFlow/Keras
- NumPy
- Librosa
- Flask-CORS
- Python-dotenv
- Concurrent.futures

To install dependencies, simply run:

```bash
pip install -r requirements.txt
```

## Contributing

We welcome contributions! Please submit an issue or a pull request if you have suggestions or improvements for this project.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Mrloubane/MIMII_Dataset_WEB/commit/20002499ecc49cf250bdf07961973154f945ac4f) file for details.
