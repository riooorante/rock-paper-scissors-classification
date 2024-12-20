import os
from flask import Flask, request, render_template
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = models.alexnet(pretrained=False)
model.classifier[6] = nn.Linear(4096, 3)

state_dict = torch.load(r'D:\Computer-Vision\rock-scissorsp-paper\result_train_model\alexnet_fine_tuned.pth')
model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        class_names = ['Paper', 'Rock', 'Scissors']

        prediction = predict_image(filepath)
        return f"Kelas: {class_names[prediction]}"

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
