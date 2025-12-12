from flask import Flask, send_from_directory
from config import Config
from extensions import mongo, login_manager, mail
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
mongo.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.init_app(app)
mail.init_app(app)

# ------------------ BLUEPRINTS ------------------

# Import blueprints
from routes.auth_routes import auth_bp
from routes.admin_routes import admin_bp
from routes.user_routes import user_bp
from routes.main_routes import main_bp  # ✅ Main blueprint for index/homepage

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(admin_bp)
app.register_blueprint(user_bp, url_prefix="/user")
app.register_blueprint(main_bp)

# ------------------ ROUTES ------------------

# Serve favicon.ico to avoid 404 errors in browser
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

# ------------------ MAIN ------------------

if __name__ == '__main__':
    app.run(debug=True)
import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd

# Load CSV files
disease_info = pd.read_csv('disease.csv', encoding='cp1252')
supplement_info = pd.read_csv('supp.csv', encoding='cp1252')

# Load model safely (state dict)
model = CNN.CNN(39)
model.load_state_dict(torch.load("model_1.pt", map_location=torch.device('cpu')))
model.eval()

# Make sure uploads folder exists
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def prediction(image_path):
    image = Image.open(image_path).convert("RGB")   # Ensure 3 channels
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image).unsqueeze(0)  # shape: (1,3,224,224)
    with torch.no_grad():
        output = model(input_data)
    output = output.detach().numpy()  # Minor fix: convert tensor to numpy
    index = np.argmax(output)
    return index

app = Flask(__name__)

@app.route('/')
def main_page():
    return redirect('/home')

@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index1')
def ai_engine_page():
    return render_template('index1.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        if image.filename == "":
            return "No file selected", 400
        file_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(file_path)

        # Run prediction
        pred = prediction(file_path)

        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]

        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]

        return render_template(
            'submit.html',
            title=title,
            desc=description,
            prevent=prevent,
            image_url=image_url,
            pred=pred,
            sname=supplement_name,
            simage=supplement_image_url,
            buy_link=supplement_buy_link,
            uploaded_image_url = f"/static/uploads/{image.filename}"
        )
    return redirect('/home')

@app.route('/market')
def market():
    return render_template(
        'market.html',
        supplement_image=list(supplement_info['supplement image']),
        supplement_name=list(supplement_info['supplement name']),
        disease=list(disease_info['disease_name']),
        buy=list(supplement_info['buy link'])
    )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)




