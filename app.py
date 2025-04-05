from flask import Flask, render_template, request, jsonify
import csv

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('nasaLanding.html')

@app.route('/valuationForm')
def valutationForm():
    return render_template('valuationForm.html')

@app.route('/submit_data', methods=['POST'])
def submit_data():
    # Get the JSON data from the request
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    # Define the CSV file path
    csv_file_path = 'car_valuation_data.csv'

    # Write JSON data to the CSV file
    try:
        # Check if the CSV file exists, and write the headers only if it doesn't
        file_exists = False
        try:
            with open(csv_file_path, 'r'):
                file_exists = True
        except FileNotFoundError:
            file_exists = False

        with open(csv_file_path, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()  # Write headers if the file is new
            writer.writerow(data)  # Write the data row

        return jsonify({"message": "Data successfully saved to CSV"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)