from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
with open('xgb_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

# Define routes
@app.route('/')
def home():
    return render_template('index.html')  # HTML form for user input

@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from form
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(final_features)[0]
    result = "Malignant" if prediction == 1 else "Benign"
    
    return render_template('index.html', prediction_text=f"The tumor is likely to be {result}")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
