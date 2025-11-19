from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your trained model
model = load_model("plant_model.h5")

def predict_plant(img_path):
    # Load image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    # Put your class names in order as used in training
    classes = ["Alstonia Scholaris (P2)", "Arjun (P1)", "Bael (P4)", "Basil (P8)","Chinar (P11)","Gauva (P3)","Jamun (P5)","Jatropha (P6)","Lemon (P10)","Mango (P0)","Pomegranate (P9)","Pongamia Pinnata (P7)"]

    print("Predicted:", classes[class_index])

# Enter the image path here
predict_plant("test.jpg")
