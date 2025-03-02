# app.py
import os
from flask import Flask, render_template, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import io
import json

app = Flask(__name__)

# Load pre-trained model
model = models.resnet18(pretrained=True)
model.eval()

# ImageNet class labels
with open('imagenet_classes.json', 'r') as f:
    class_idx = json.load(f)
    
# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
        
    # Read and transform the image
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = outputs.max(1)
        
    # Get class name
    idx = predicted.item()
    class_name = class_idx[idx]
    
    # Get top 5 predictions
    probs, indices = torch.topk(torch.nn.functional.softmax(outputs, dim=1), 5)
    top_predictions = [
        {'class': class_idx[idx.item()], 'probability': prob.item()} 
        for idx, prob in zip(indices[0], probs[0])
    ]
    
    return jsonify({
        'prediction': class_name,
        'top_predictions': top_predictions
    })

if __name__ == '__main__':
    app.run(debug=True)