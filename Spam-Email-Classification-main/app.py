from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import pickle
import string
import nltk
from nltk.stem import PorterStemmer
from pymongo import MongoClient
import hashlib
from datetime import datetime
from PIL import Image
import io
import os

app = Flask(__name__)
app.secret_key = '1c8073775dbc85a92ce20ebd44fd6a4fd832078f59ef16ec'

ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

client = MongoClient("mongodb://localhost:27017/")
db = client["smc"]
users_col = db["users"]
history_col = db["history"]

# Try to load pytesseract
try:
    import pytesseract
    # Common Tesseract install path on Windows
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def transform_text(text):
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stop_words and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

def classify(text):
    transformed = transform_text(text)
    vec = tfidf.transform([transformed])
    result = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = int(round(max(proba) * 100))
    prediction = "Spam" if result == 1 else "Not Spam"
    return prediction, confidence

def save_history(username, message, prediction):
    history_col.insert_one({
        "username": username,
        "message": message[:300],
        "prediction": prediction,
        "timestamp": datetime.now()
    })

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/index')
def index():
    if 'user' not in session:
        return redirect(url_for('signin'))
    return render_template('index.html', username=session['user'])

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('signin'))
    input_sms = request.form.get('message', '').strip()
    if not input_sms:
        flash('Please enter a message to classify.', 'warning')
        return redirect(url_for('index'))
    prediction, confidence = classify(input_sms)
    save_history(session['user'], input_sms, prediction)
    return render_template('result.html', prediction=prediction, message=input_sms, confidence=confidence)

@app.route('/predict_ajax', methods=['POST'])
def predict_ajax():
    if 'user' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    input_sms = request.form.get('message', '').strip()
    if not input_sms:
        return jsonify({'error': 'Empty message'}), 400
    prediction, confidence = classify(input_sms)
    save_history(session['user'], input_sms, prediction)
    return jsonify({'prediction': prediction, 'confidence': confidence})

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'user' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    if not OCR_AVAILABLE:
        return jsonify({'error': 'OCR not available. Please install Tesseract-OCR from: https://github.com/UB-Mannheim/tesseract/wiki'}), 400

    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        import pytesseract
        img = Image.open(io.BytesIO(file.read()))
        extracted_text = pytesseract.image_to_string(img).strip()
        if not extracted_text:
            return jsonify({'error': 'Could not extract text from image. Please try a clearer screenshot.'}), 400
        prediction, confidence = classify(extracted_text)
        save_history(session['user'], extracted_text, prediction)
        return jsonify({'prediction': prediction, 'confidence': confidence, 'extracted_text': extracted_text[:500]})
    except Exception as e:
        return jsonify({'error': f'OCR failed: {str(e)}'}), 500

@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('signin'))
    records = list(history_col.find(
        {"username": session['user']},
        {"_id": 0, "message": 1, "prediction": 1, "timestamp": 1}
    ).sort("timestamp", -1).limit(20))
    return render_template('history.html', records=records, username=session['user'])

@app.route('/signin')
def signin():
    if 'user' in session:
        return redirect(url_for('index'))
    return render_template('signin.html')

@app.route('/signup', methods=['GET'])
def signup():
    return render_template('signup.html')

@app.route('/register', methods=['POST'])
def register():
    full_name = request.form['full_name'].strip()
    username = request.form['username'].strip()
    email = request.form['email'].strip().lower()
    phone = request.form['phone'].strip()
    password = request.form['password']
    confirm_password = request.form['confirm_password']

    if password != confirm_password:
        flash('Passwords do not match.', 'danger')
        return redirect(url_for('signup'))
    if users_col.find_one({"email": email}):
        flash('Email already registered. Please sign in.', 'danger')
        return redirect(url_for('signup'))
    if users_col.find_one({"username": username}):
        flash('Username already taken.', 'danger')
        return redirect(url_for('signup'))

    users_col.insert_one({
        "full_name": full_name, "username": username,
        "email": email, "phone": phone,
        "password": hash_password(password)
    })
    flash('Registration successful! Please sign in.', 'success')
    return redirect(url_for('signin'))

@app.route('/login', methods=['POST'])
def login():
    email = request.form['email'].strip().lower()
    password = request.form['password']
    remember_me = request.form.get('remember_me')
    user = users_col.find_one({"email": email, "password": hash_password(password)})
    if user:
        session['user'] = user["username"]
        if remember_me:
            session.permanent = True
        return redirect(url_for('index'))
    flash('Invalid email or password.', 'danger')
    return redirect(url_for('signin'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
