# 🧠 Brain Tumor Detection App

This is a Flask-based web application for detecting brain tumors from MRI images using a trained machine learning model. The model is saved as a `.pkl` file and used to make predictions through a simple web interface.

---

## 🚀 Features

- Upload MRI images to detect brain tumors
- Uses a pre-trained machine learning model
- Built with Python and Flask
- Easy to run locally

---

## 📁 Project Structure


brain-tumor-detection/
│
├── static/ # Static files (CSS, JS, images)
├── templates/ # HTML templates (index.html)
├── model/ # Folder containing model.pkl
│ └── model.pkl
├── app.py # Main Flask application
├── requirements.txt # Python dependencies
└── README.md # Project documentation



---

## 🔧 Installation & Running Locally

Follow these steps to set up and run the app on your local machine:

### 1. Clone the Repository



    git clone https://github.com/your-username/brain-tumor-detection.git
    cd brain-tumor-detection
    
    
    
    python -m venv venv
    # Activate virtual environment:
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    
    pip install -r requirements.txt
    
    python app.py
    
    http://127.0.0.1:5000/


🧠 Model
The model used for prediction is stored in the model/model.pkl file. Make sure the file path in your app.py matches this location.

📸 How to Use
    Open the web app in your browser.
    
    Upload an MRI image (JPG/PNG).
    
    Click the Predict button.
    
    View the prediction result (e.g., Tumor detected / No tumor).

✅ Requirements
Make sure Python 3.6+ is installed.

The required Python packages are listed in requirements.txt. Common packages include:

    -Flask
    -numpy
    -pandas
    -scikit-learn
    -opencv-python

