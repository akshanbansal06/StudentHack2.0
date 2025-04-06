import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
import threading
import time
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

app = Flask(__name__)

# Timeout handler for long-running tasks
def handler():
    raise Exception("Model training timeout")

# Timeout class
class Timeout:
    def __init__(self, seconds):
        self.timer = threading.Timer(seconds, handler)

    def start(self):
        self.timer.start()

    def cancel(self):
        self.timer.cancel()

# Homepage
@app.route('/')
def home():
    return "🚗 Welcome to the Car Price Prediction API"

# Train the model
@app.route('/train', methods=['POST'])
def train_model():
    timeout = Timeout(600)  # 10 minutes
    timeout.start()

    try:
        print("Loading dataset...")
        train_df = pd.read_csv('train.csv')
        print("Dataset loaded.")

        def preprocess_data(df):
            df.ffill(inplace=True)
            df = pd.get_dummies(df, drop_first=True)
            return df

        train_df = preprocess_data(train_df)

        X_train = train_df.drop('Price', axis=1)
        y_train = train_df['Price']

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        pca = PCA(n_components=50)
        X_train_pca = pca.fit_transform(X_train_scaled)

        model = LinearRegression()
        start_time = time.time()
        model.fit(X_train_pca, y_train)
        end_time = time.time()

        y_pred = model.predict(X_train_pca)
        mse = mean_squared_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)

        # Save model artifacts for reuse
        train_df['predicted_price'] = y_pred
        train_df.to_csv('predicted_values.csv', index=False)

        print(f"✅ Model trained in {end_time - start_time:.2f}s | MSE: {mse:.2f} | R²: {r2:.4f}")
        return jsonify({"message": "Model trained successfully", "mse": mse, "r2": r2}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        timeout.cancel()

# Predict future car prices (depreciation)
@app.route('/predict', methods=['POST'])
def predict_future_price():
    try:
        train_df = pd.read_csv('train.csv')
        df = train_df.copy()
        df.ffill(inplace=True)
        df = pd.get_dummies(df, drop_first=True)

        X = df.drop('Price', axis=1)
        y = df['Price']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=50)
        X_pca = pca.fit_transform(X_scaled)

        model = LinearRegression()
        model.fit(X_pca, y)

        input_data = request.get_json()

        future_years = list(range(2025, 2031))
        predictions = []

        for future_year in future_years:
            car = input_data.copy()
            car['Year'] = future_year

            input_df = pd.DataFrame([car])
            full_df = pd.concat([train_df, input_df], ignore_index=True)
            full_df.ffill(inplace=True)
            full_df = pd.get_dummies(full_df, drop_first=True)

            # Align with training columns
            for col in X.columns:
                if col not in full_df.columns:
                    full_df[col] = 0
            full_df = full_df[X.columns]

            scaled = scaler.transform(full_df)
            reduced = pca.transform(scaled)
            future_pred = model.predict(reduced)[-1]

            predictions.append({
                "year": future_year,
                "predicted_price": round(float(future_pred), 2)
            })

        return jsonify({"future_predictions": predictions}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Plot depreciation chart based on /predict output
@app.route('/plot', methods=['POST'])
def plot_depreciation():
    try:
        input_data = request.get_json()

        # Call predict endpoint internally
        with app.test_client() as client:
            response = client.post('/predict', json=input_data)
            result = response.get_json()
            predictions = result["future_predictions"]

        years = [p["year"] for p in predictions]
        prices = [p["predicted_price"] for p in predictions]

        plt.figure(figsize=(8, 5))
        plt.plot(years, prices, marker='o', label='Predicted Depreciation')
        plt.title('Car Price Depreciation Over Time')
        plt.xlabel('Year')
        plt.ylabel('Predicted Price ($)')
        plt.grid(True)
        plt.legend()
        os.makedirs('static', exist_ok=True)
        plt.savefig('static/depreciation.png')
        plt.close()

        return jsonify({"message": "Graph generated", "path": "/static/depreciation.png"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Start server
if __name__ == '__main__':
    app.run(debug=True)
