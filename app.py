from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("plant_model.h5")

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Change these names to match your dataset classes
classes = [
    "Alstonia Scholaris (P2)",
    "Arjun (P1)",
    "Bael (P4)",
    "Basil (P8)",
    "Chinar (P11)",
    "Guava (P3)",
    "Jamun (P5)",
    "Jatropha (P6)",
    "Lemon (P10)",
    "Mango (P0)",
    "Pomegranate (P9)",
    "Pongamia Pinnata (P7)"
]

def predict_plant(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    return classes[index]

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["file"]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        output = predict_plant(filepath)
        return render_template("index.html", prediction=output, file_path=filepath)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)


