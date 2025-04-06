from flask import Flask, render_template, request, jsonify, redirect, url_for
import csv
import os
import time
import threading

app = Flask(__name__)

# Flag to track if processing is complete
processing_complete = False
prediction_result = None

def process_data_and_generate_prediction(data):
    global processing_complete, prediction_result
    
    # Simulate processing time (replace with your actual processing logic)
    time.sleep(5)  # Simulating a 5-second process
    
    # Write prediction to file (your actual prediction logic goes here)
    prediction = calculate_prediction(data)  # Replace with your prediction function
    
    # Save prediction to file
    with open('static/txt/predictedPrice.txt', 'w') as f:
        f.write(str(prediction))
    
    # Update global variables
    prediction_result = prediction
    processing_complete = True

def calculate_prediction(data):
    # Placeholder for your prediction logic
    # For demonstration, just return a simple calculation
    return float(data.get('year', 2020)) * 100 - float(data.get('mileage', 0)) * 0.05

@app.route('/')
def home():
    return render_template('nasaLanding.html')

@app.route('/valuationForm')
def valuation_form():
    # Reset processing flags when accessing the form
    global processing_complete, prediction_result
    processing_complete = False
    prediction_result = None
    
    # Ensure the prediction file doesn't exist when starting a new valuation
    try:
        os.remove('static/txt/predictedPrice.txt')
    except FileNotFoundError:
        pass
        
    return render_template('valuationForm.html')

@app.route('/loadingpage')
def loadingpage():
    return render_template('loadingpage.html')

@app.route('/check_status')
def check_status():
    """API endpoint to check if processing is complete"""
    if processing_complete:
        return jsonify({"status": "complete", "redirect": "/results"})
    else:
        return jsonify({"status": "processing"})

@app.route('/results')
def results():
    # Check if prediction file exists
    try:
        with open('static/txt/predictedPrice.txt', 'r') as f:
            predicted_price = f.read().strip()
        return render_template('results.html', predicted_price=predicted_price)
    except FileNotFoundError:
        # If file doesn't exist, redirect to valuation form
        return redirect(url_for('valuation_form'))

@app.route('/submit_data', methods=['POST'])
def submit_data():
    global processing_complete
    
    # Reset processing flag
    processing_complete = False
    
    # Get the JSON data from the request
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    # Define the CSV file path
    csv_file_path = 'car_valuation_data.csv'

    try:
        # Check if the CSV file exists, and write the headers only if it doesn't
        file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()  # Write headers if the file is new
            writer.writerow(data)  # Write the data row
        
        # Start processing in a background thread
        thread = threading.Thread(target=process_data_and_generate_prediction, args=(data,))
        thread.daemon = True
        thread.start()
        
        # Return response immediately
        return jsonify({"message": "Data submitted successfully", "redirect": "/loadingpage"}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure directory exists
    os.makedirs('static/txt', exist_ok=True)
    app.run(debug=True)