import os
from flask import Flask, render_template, request, redirect
from utils import predict_single_image

app = Flask(__name__)
# Ensure uploads folder uses an absolute path based on this file’s directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    file = request.files.get('image')
    if not file or file.filename == '':
        return redirect('/')

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)

    prediction = predict_single_image(image_path)

    # Best-effort cleanup
    try:
        os.remove(image_path)
    except OSError:
        pass

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    # Bind to 0.0.0.0 so it’s reachable if needed; change port if occupied
    app.run(host='0.0.0.0', port=5000, debug=True)
