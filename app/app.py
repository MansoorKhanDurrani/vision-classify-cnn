import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)

# --- Load model once at startup ---
device = torch.device("cpu")
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

model = mobilenet_v2()
model.classifier[1] = nn.Linear(model.last_channel, 6)
model.load_state_dict(torch.load("models/transfer_mobilenet.pth", map_location=device))
model = model.to(device)
model.eval()

# Same normalization used during training
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Model loaded successfully.")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    img_tensor = transform(image).unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)

    return jsonify({
        "class": class_names[predicted_idx.item()],
        "confidence": round(confidence.item() * 100, 2)
    })


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "API is running", "classes": class_names})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)