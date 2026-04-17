from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import base64
from io import BytesIO
from PIL import Image

from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

app = Flask(__name__)
CORS(app)

MODEL_PATH = "Aaron_ModelKlasifikasi3.h5"
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

labels = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
    
    
]

LAST_CONV_LAYER = "Conv_1"

# ---------------------------
# PREPROCESS IMAGE
# ---------------------------
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.array(img).astype(np.float32)
    return np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# ---------------------------
# GRAD-CAM
# ---------------------------
def gradcam(img_array):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(LAST_CONV_LAYER).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

    return heatmap.numpy(), class_idx.numpy(), predictions.numpy()[0]

# ---------------------------
# API ROUTE
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img = Image.open(file).convert("RGB")

    img_array = preprocess_image(img)
    heatmap, class_idx, probs = gradcam(img_array)

    all_predictions = {
        labels[i]: float(probs[i] * 100)
        for i in range(len(labels))
    }

    # Overlay heatmap
    img_np = np.array(img.resize((224, 224)))
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    # Convert to base64
    _, buffer = cv2.imencode(".png", overlay)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "predicted_class": labels[class_idx],
        "confidence": float(probs[class_idx] * 100),
        "all_predictions": all_predictions,
        "gradcam_image": img_base64
    })

# @app.route("/predict", methods=["POST"])
# def predict():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["file"]
#     image = Image.open(file).convert("RGB")
#     image = image.resize((224, 224))

#     img_array = np.array(image) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     preds = model.predict(img_array)[0]

#     top_idx = int(np.argmax(preds))

#     return jsonify({
#         "predicted_class": labels[top_idx],
#         "confidence": float(preds[top_idx] * 100),
#         "all_predictions": {
#             labels[i]: float(preds[i] * 100)
#             for i in range(len(labels))
#         }
#     })

if __name__ == "__main__":
    app.run(debug=True)