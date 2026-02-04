




# !pip -q install flask flask-cors pyngrok joblib tensorflow python-multipart

"""
WBC Image Classification - Flask (Google Colab)
Model: TF encoder (feature extractor) + scikit KNN
Input: Image upload
Output: predicted class + confidence + per-class probabilities
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
import logging
import time
import os

# OPTIONAL: NGROK (google colab)
USE_NGROK = True  # False if running locally

if USE_NGROK:
    try:
        from pyngrok import ngrok
    except ImportError:
        os.system("pip -q install pyngrok")
        from pyngrok import ngrok

# -------------------- LOGGING --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wbc-image-api")

# -------------------- CONFIG --------------------
NGROK_TOKEN = ""  # ngrok auth 
CLASS_NAMES = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
IMG_SIZE = (224, 224)

ENCODER_PATH = Path("/content/drive/MyDrive/projects/wbc-cnn-knn-classifier/models/encoder/feature_encoder.keras")
KNN_PATH = Path("/content/drive/MyDrive/projects/wbc-cnn-knn-classifier/models/knn/knn_best.joblib")

# -------------------- PREPROCESS --------------------
def preprocess_for_inference(img_bytes: bytes) -> tf.Tensor:
    """
    Match training preprocessing:
    """
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])

    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)

    return tf.expand_dims(img, axis=0)  # (1, 224, 224, 3)

# -------------------- MODEL WRAPPER --------------------
class WBCPredictor:
    def __init__(self, encoder_path: Path, knn_path: Path):
        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder not found: {encoder_path.resolve()}")
        if not knn_path.exists():
            raise FileNotFoundError(f"KNN not found: {knn_path.resolve()}")

        logger.info("Loading encoder: %s", str(encoder_path))
        self.encoder = tf.keras.models.load_model(str(encoder_path))

        logger.info("Loading KNN: %s", str(knn_path))
        self.knn = joblib.load(str(knn_path))

        logger.info("✅ Models loaded successfully")

    def predict_bytes(self, img_bytes: bytes) -> dict:
        x = preprocess_for_inference(img_bytes)

        feats = self.encoder(x, training=False)   # (1, D)
        feats_np = feats.numpy()

        # predict_proba returns shape (1, n_classes)
        proba = self.knn.predict_proba(feats_np)[0]
        pred_idx = int(np.argmax(proba))

        return {
            "prediction": CLASS_NAMES[pred_idx],
            "confidence": float(np.max(proba)),
            "class_id": pred_idx,
            "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, proba)}
        }

# -------------------- FLASK APP --------------------
app = Flask(__name__)
CORS(app)

try:
    predictor = WBCPredictor(ENCODER_PATH, KNN_PATH)
except Exception as e:
    logger.error("Failed to load models: %s", e)
    predictor = None

# -------------------- HTML UI --------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>WBC Image Classifier | AI-Powered Blood Cell Analysis</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      padding: 20px;
      color: #2d3748;
    }

    .container {
      max-width: 1200px;
      margin: 0 auto;
    }

    /* Header */
    .header {
      text-align: center;
      color: white;
      margin-bottom: 40px;
      padding: 40px 20px;
    }

    .header h1 {
      font-size: 2.5rem;
      font-weight: 700;
      margin-bottom: 10px;
      text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .header p {
      font-size: 1.1rem;
      opacity: 0.95;
      font-weight: 300;
    }

    .header .badge {
      display: inline-block;
      background: rgba(255,255,255,0.2);
      backdrop-filter: blur(10px);
      padding: 8px 16px;
      border-radius: 20px;
      margin-top: 15px;
      font-size: 0.9rem;
      font-weight: 500;
    }

    /* Main Content */
    .main-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      margin-bottom: 24px;
    }

    @media (max-width: 900px) {
      .main-grid {
        grid-template-columns: 1fr;
      }
      .header h1 {
        font-size: 1.8rem;
      }
    }

    .card {
      background: white;
      border-radius: 16px;
      padding: 28px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.15);
      transition: transform 0.2s, box-shadow 0.2s;
    }

    .card:hover {
      transform: translateY(-2px);
      box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    }

    .card h3 {
      font-size: 1.3rem;
      font-weight: 600;
      margin-bottom: 20px;
      color: #1a202c;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .card h3::before {
      content: '';
      display: inline-block;
      width: 4px;
      height: 24px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      border-radius: 2px;
    }

    /* Upload Section */
    .upload-zone {
      border: 3px dashed #cbd5e0;
      border-radius: 12px;
      padding: 40px 20px;
      text-align: center;
      cursor: pointer;
      transition: all 0.3s;
      position: relative;
      background: #f7fafc;
    }

    .upload-zone:hover {
      border-color: #667eea;
      background: #edf2f7;
    }

    .upload-zone.dragover {
      border-color: #667eea;
      background: #e6f2ff;
      transform: scale(1.02);
    }

    .upload-icon {
      font-size: 3rem;
      margin-bottom: 10px;
    }

    .upload-zone input[type="file"] {
      position: absolute;
      opacity: 0;
      width: 100%;
      height: 100%;
      cursor: pointer;
      top: 0;
      left: 0;
    }

    .upload-text {
      font-size: 1rem;
      color: #4a5568;
      font-weight: 500;
    }

    .upload-hint {
      font-size: 0.875rem;
      color: #718096;
      margin-top: 8px;
    }

    button {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      border: none;
      padding: 14px 32px;
      font-size: 1rem;
      font-weight: 600;
      border-radius: 10px;
      cursor: pointer;
      width: 100%;
      margin-top: 16px;
      transition: all 0.3s;
      box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    button:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }

    button:active {
      transform: translateY(0);
    }

    button:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
    }

    /* Preview */
    .preview-container {
      width: 100%;
      height: 350px;
      background: #f7fafc;
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      border: 2px solid #e2e8f0;
    }

    .preview-container img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      border-radius: 8px;
    }

    .preview-placeholder {
      text-align: center;
      color: #a0aec0;
      font-size: 0.95rem;
    }

    /* Message */
    .message {
      padding: 12px 16px;
      border-radius: 8px;
      margin-top: 12px;
      font-size: 0.9rem;
      text-align: center;
      font-weight: 500;
    }

    .message.info {
      background: #ebf8ff;
      color: #2c5282;
      border: 1px solid #bee3f8;
    }

    .message.error {
      background: #fff5f5;
      color: #c53030;
      border: 1px solid #feb2b2;
    }

    .message.success {
      background: #f0fff4;
      color: #276749;
      border: 1px solid #9ae6b4;
    }

    /* Results */
    .result-header {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 20px 24px;
      border-radius: 12px 12px 0 0;
      margin: -28px -28px 24px -28px;
    }

    .result-header h3 {
      color: white;
      margin: 0;
    }

    .result-header h3::before {
      display: none;
    }

    .prediction-summary {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }

    .stat-box {
      background: #f7fafc;
      padding: 16px;
      border-radius: 10px;
      border-left: 4px solid #667eea;
    }

    .stat-label {
      font-size: 0.85rem;
      color: #718096;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 6px;
    }

    .stat-value {
      font-size: 1.4rem;
      font-weight: 700;
      color: #1a202c;
    }

    /* Probability Cards */
    .prob-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
    }

    .prob-card {
      background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
      border-radius: 12px;
      padding: 18px;
      border: 2px solid #e2e8f0;
      transition: all 0.3s;
    }

    .prob-card:hover {
      border-color: #667eea;
      transform: translateY(-3px);
      box-shadow: 0 8px 20px rgba(102, 126, 234, 0.15);
    }

    .prob-card.winner {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      border-color: #667eea;
      box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }

    .prob-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }

    .prob-name {
      font-weight: 600;
      font-size: 1rem;
    }

    .prob-percent {
      font-size: 1.1rem;
      font-weight: 700;
      padding: 4px 10px;
      background: rgba(255,255,255,0.2);
      border-radius: 6px;
    }

    .prob-card.winner .prob-percent {
      background: rgba(255,255,255,0.3);
    }

    .prob-bar-container {
      height: 8px;
      background: rgba(0,0,0,0.1);
      border-radius: 4px;
      overflow: hidden;
    }

    .prob-card.winner .prob-bar-container {
      background: rgba(255,255,255,0.2);
    }

    .prob-bar {
      height: 100%;
      background: #667eea;
      border-radius: 4px;
      transition: width 0.6s ease-out;
      width: 0%;
    }

    .prob-card.winner .prob-bar {
      background: white;
    }

    /* Info Note */
    .info-note {
      background: #edf2f7;
      border-left: 4px solid #667eea;
      padding: 12px 16px;
      border-radius: 6px;
      font-size: 0.9rem;
      color: #4a5568;
      margin-top: 16px;
    }

    .info-note code {
      background: white;
      padding: 2px 6px;
      border-radius: 4px;
      font-family: 'Courier New', monospace;
      font-size: 0.85rem;
      color: #667eea;
    }

    /* Loading Animation */
    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    .loading {
      display: inline-block;
      width: 20px;
      height: 20px;
      border: 3px solid rgba(255,255,255,0.3);
      border-radius: 50%;
      border-top-color: white;
      animation: spin 0.8s linear infinite;
    }

    /* Fade in animation */
    @keyframes fadeIn {
      from {
        opacity: 0;
        transform: translateY(10px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    .fade-in {
      animation: fadeIn 0.5s ease-out;
    }
  </style>
</head>
<body>
  <div class="container">
    <!-- Header -->
    <div class="header">
      <h1>🔬 White Blood Cell Classifier</h1>
      <p>AI-Powered Blood Cell Analysis Using Deep Learning</p>
      <div class="badge">CNN Feature Extractor + k-NN Classifier</div>
    </div>

    <!-- Main Grid -->
    <div class="main-grid">
      <!-- Upload Card -->
      <div class="card">
        <h3>📤 Upload Image</h3>
        <div class="upload-zone" id="dropZone">
          <div class="upload-icon">📁</div>
          <div class="upload-text">Click to upload or drag & drop</div>
          <div class="upload-hint">Supports JPG, PNG, JPEG • Max 10MB</div>
          <input id="fileInput" type="file" accept="image/*" />
        </div>
        <button id="predictBtn" onclick="predict()" disabled>
          🚀 Analyze Image
        </button>
        <div class="info-note">
          <strong>API Endpoint:</strong> <code>POST /api/predict</code>
        </div>
        <div id="message"></div>
      </div>

      <!-- Preview Card -->
      <div class="card">
        <h3>👁️ Image Preview</h3>
        <div class="preview-container">
          <div class="preview-placeholder" id="placeholder">
            <div style="font-size: 3rem; margin-bottom: 10px;">🖼️</div>
            <div>No image selected</div>
          </div>
          <img id="preview" style="display: none;" alt="Preview" />
        </div>
      </div>
    </div>

    <!-- Results Card -->
    <div class="card" id="resultsCard" style="display: none;">
      <div class="result-header">
        <h3>📊 Analysis Results</h3>
      </div>

      <div class="prediction-summary" id="summaryBox"></div>

      <h4 style="margin-bottom: 16px; color: #2d3748; font-weight: 600;">Class Probabilities</h4>
      <div class="prob-grid" id="probGrid"></div>
    </div>
  </div>

  <script>
    const fileInput = document.getElementById('fileInput');
    const dropZone = document.getElementById('dropZone');
    const preview = document.getElementById('preview');
    const placeholder = document.getElementById('placeholder');
    const predictBtn = document.getElementById('predictBtn');
    const message = document.getElementById('message');
    const resultsCard = document.getElementById('resultsCard');
    const summaryBox = document.getElementById('summaryBox');
    const probGrid = document.getElementById('probGrid');

    let selectedFile = null;

    // File input change
    fileInput.addEventListener('change', (e) => {
      handleFile(e.target.files[0]);
    });

    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
      dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      handleFile(e.dataTransfer.files[0]);
    });

    function handleFile(file) {
      if (!file || !file.type.startsWith('image/')) {
        showMessage('Please select a valid image file', 'error');
        return;
      }

      selectedFile = file;
      const url = URL.createObjectURL(file);
      preview.src = url;
      preview.style.display = 'block';
      placeholder.style.display = 'none';
      predictBtn.disabled = false;
      message.innerHTML = '';
      resultsCard.style.display = 'none';
    }

    function showMessage(text, type = 'info') {
      message.innerHTML = `<div class="message ${type}">${text}</div>`;
    }

    async function predict() {
      if (!selectedFile) {
        showMessage('Please select an image first', 'error');
        return;
      }

      predictBtn.disabled = true;
      predictBtn.innerHTML = '<span class="loading"></span> Analyzing...';
      showMessage('Running AI inference...', 'info');
      resultsCard.style.display = 'none';

      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
        const response = await fetch('/api/predict', {
          method: 'POST',
          body: formData
        });

        const data = await response.json();

        if (data.error) {
          showMessage(data.error, 'error');
          return;
        }

        // Show results
        displayResults(data);
        showMessage('Analysis complete!', 'success');

      } catch (error) {
        showMessage('Error: ' + error.message, 'error');
      } finally {
        predictBtn.disabled = false;
        predictBtn.innerHTML = '🚀 Analyze Image';
      }
    }

    function displayResults(data) {
      // Summary boxes
      summaryBox.innerHTML = `
        <div class="stat-box">
          <div class="stat-label">Predicted Class</div>
          <div class="stat-value">${data.prediction}</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Confidence</div>
          <div class="stat-value">${(data.confidence * 100).toFixed(1)}%</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Inference Time</div>
          <div class="stat-value">${data.inference_time_ms}ms</div>
        </div>
      `;

      // Probability cards
      const probs = Object.entries(data.probabilities)
        .sort((a, b) => b[1] - a[1]);

      probGrid.innerHTML = '';

      probs.forEach(([name, prob], index) => {
        const isWinner = index === 0;
        const percent = (prob * 100).toFixed(1);

        const card = document.createElement('div');
        card.className = `prob-card ${isWinner ? 'winner' : ''} fade-in`;
        card.style.animationDelay = `${index * 0.1}s`;
        card.innerHTML = `
          <div class="prob-header">
            <div class="prob-name">${isWinner ? '🏆 ' : ''}${name}</div>
            <div class="prob-percent">${percent}%</div>
          </div>
          <div class="prob-bar-container">
            <div class="prob-bar" style="width: ${percent}%"></div>
          </div>
        `;
        probGrid.appendChild(card);
      });

      resultsCard.style.display = 'block';
      resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  </script>
</body>
</html>
"""

# -------------------- ROUTES --------------------
@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": predictor is not None,
        "encoder_path": str(ENCODER_PATH),
        "knn_path": str(KNN_PATH),
        "classes": CLASS_NAMES
    })

@app.route("/api/predict", methods=["POST"])
def predict_api():
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "Image file required (field name: 'file')"}), 400

    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "Empty file"}), 400

    start = time.time()

    try:
        img_bytes = f.read()
        out = predictor.predict_bytes(img_bytes)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    elapsed_ms = round((time.time() - start) * 1000, 2)

    out.update({
        "filename": f.filename,
        "img_size": list(IMG_SIZE),
        "inference_time_ms": elapsed_ms,
        "model": "Encoder + KNN"
    })
    return jsonify(out)

# -------------------- ENTRY POINT (WITH NGROK HOOK) --------------------
if __name__ == "__main__":
    print("=" * 80)
    print("WBC Image Classification Server")
    print("Classes:", CLASS_NAMES)
    print("=" * 80)

    if USE_NGROK:
        ngrok.set_auth_token(NGROK_TOKEN)
        try:
            ngrok.kill()
        except Exception:
            pass

        tunnel = ngrok.connect(5000, "http")
        print("Public URL:", tunnel.public_url)

    app.run(host="0.0.0.0", port=5000, debug=False)

