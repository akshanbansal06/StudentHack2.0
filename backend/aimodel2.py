import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
import signal
import time
import matplotlib.pyplot as plt
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to handle timeout
def handler(signum, frame):
    raise Exception("Model training timeout")

# Set the timeout signal
signal.signal(signal.SIGALRM, handler)
signal.alarm(600)  # Timeout after 600 seconds (10 minutes)

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")
    return directory

def preprocess_data(df, train_mode=False, reference_columns=None):
    """
    Preprocess the data with improved handling of columns alignment
    
    Args:
        df: DataFrame to process
        train_mode: Whether this is training data (True) or prediction data (False)
        reference_columns: Columns to align with (for prediction data)
        
    Returns:
        Processed DataFrame and column names (if train_mode)
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Handle missing values
    df.ffill(inplace=True)
    
    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)
    
    # Align columns with the training set if in prediction mode
    if not train_mode and reference_columns is not None:
        # Get the intersection of columns (excluding target if present)
        pred_columns = [col for col in reference_columns if col != 'Price']
        # Reindex to match training columns
        for col in pred_columns:
            if col not in df.columns:
                df[col] = 0  # Add missing columns with default value
        
        # Make sure columns are in the same order
        df = df[pred_columns]
    
    if train_mode:
        return df, df.columns
    else:
        return df

def calculate_depreciation(values, years, rate_of_depreciation=0.15):
    """Calculate car depreciation over multiple years"""
    result = [values]
    for _ in range(1, years + 1):
        values = values * (1 - rate_of_depreciation)
        result.append(values)
    return result

def generate_depreciation_graph(values, start_year, plot_path):
    """Generate and save depreciation graph"""
    years = list(range(start_year, start_year + len(values)))
    
    plt.figure(figsize=(10, 6))
    plt.plot(years, values, marker='o', linewidth=2, color='#1f77b4')
    plt.title('Vehicle Depreciation Forecast', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Predicted Value ($)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the graph
    plt.savefig(plot_path)
    logger.info(f"Plot saved to: {plot_path}")
    plt.close()  # Close the figure to free memory

def main():
    # Define paths for saving model, scaler, PCA, and output
    model_dir = ensure_directory_exists("models")
    output_dir = ensure_directory_exists("output")
    txt_dir = ensure_directory_exists("static/txt")
    plot_dir = ensure_directory_exists("static/plot")
    
    model_path = os.path.join(model_dir, "car_price_model.joblib")
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    pca_path = os.path.join(model_dir, "pca.joblib")
    predictions_path = os.path.join(output_dir, "predicted_values.csv")
    txt_output_path = os.path.join(txt_dir, "predictedPrice.txt")
    plot_path = os.path.join(plot_dir, "valuation.png")
    
    try:
        # Load the dataset
        logger.info("Loading datasets...")
        train_df = pd.read_csv('train.csv')
        logger.info(f"Dataset loaded successfully with {train_df.shape[0]} rows and {train_df.shape[1]} columns.")
    except FileNotFoundError as e:
        logger.error(f"Error loading dataset: {e}")
        return

    try:
        # Split data into training and validation sets
        logger.info("Splitting data into training and validation sets...")
        X = train_df.drop('Price', axis=1, errors='ignore')  # In case Price doesn't exist
        y = train_df['Price'] if 'Price' in train_df.columns else None
        
        if y is None:
            logger.error("Target variable 'Price' not found in the dataset.")
            return
            
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Recombine for preprocessing
        train_data = X_train.copy()
        train_data['Price'] = y_train
        
        val_data = X_val.copy()
        val_data['Price'] = y_val
        
        logger.info(f"Training set: {X_train.shape[0]} samples, Validation set: {X_val.shape[0]} samples")
    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        return

    try:
        # Preprocess the data
        logger.info("Preprocessing training data...")
        train_processed, train_columns = preprocess_data(train_data, train_mode=True)
        
        # Separate features and target
        X_train = train_processed.drop('Price', axis=1)
        y_train = train_processed['Price']
        
        # Preprocess validation data using training columns as reference
        logger.info("Preprocessing validation data...")
        val_processed = preprocess_data(val_data, train_mode=False, reference_columns=train_columns)
        X_val = val_processed.drop('Price', axis=1)
        y_val = val_processed['Price']
        
        logger.info("Data preprocessing completed successfully.")
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return

    try:
        # Handle NaNs in target variables
        logger.info("Handling NaNs in target variables...")
        nan_count_before = y_train.isnull().sum()
        if nan_count_before > 0:
            logger.warning(f"Found {nan_count_before} NaN values in y_train")
            y_train.fillna(y_train.mean(), inplace=True)
        
        nan_count_after = y_train.isnull().sum()
        logger.info(f"NaNs in y_train after handling: {nan_count_after}")
        
        # Check for NaNs in X_train
        nan_count_x = X_train.isnull().sum().sum()
        if nan_count_x > 0:
            logger.warning(f"Found {nan_count_x} NaN values in X_train")
            X_train.fillna(X_train.mean(), inplace=True)
            
        logger.info("NaN handling completed successfully.")
    except Exception as e:
        logger.error(f"Error during NaN handling: {e}")
        return

    try:
        # Normalize/scale the data
        logger.info("Normalizing/scaling the data...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Save the scaler
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
    except Exception as e:
        logger.error(f"Error during normalization/scaling: {e}")
        return

    try:
        # Apply PCA for dimensionality reduction
        logger.info("Applying PCA for dimensionality reduction...")
        # Choose number of components that explain 95% of variance
        pca = PCA(n_components=0.95)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)
        
        # Save the PCA model
        joblib.dump(pca, pca_path)
        
        logger.info(f"PCA reduced dimensions from {X_train_scaled.shape[1]} to {X_train_pca.shape[1]} features")
        logger.info(f"PCA explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    except Exception as e:
        logger.error(f"Error during PCA: {e}")
        return

    try:
        # Initialize and train the model
        logger.info("Training the model...")
        start_time = time.time()
        model = LinearRegression()
        model.fit(X_train_pca, y_train)
        training_time = time.time() - start_time
        
        # Save the model
        joblib.dump(model, model_path)
        
        logger.info(f"Model trained successfully in {training_time:.2f} seconds and saved to {model_path}")
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return
    finally:
        # Disable the alarm
        signal.alarm(0)

    try:
        # Make predictions on the training set
        logger.info("Making predictions on training data...")
        y_train_pred = model.predict(X_train_pca)
        
        # Make predictions on the validation set
        logger.info("Making predictions on validation data...")
        y_val_pred = model.predict(X_val_pca)
        
        # Handle NaNs in predictions if any
        if np.isnan(y_train_pred).any():
            logger.warning("NaN values found in training predictions. Replacing with mean.")
            mean_pred = np.nanmean(y_train_pred)
            y_train_pred = np.nan_to_num(y_train_pred, nan=mean_pred)
            
        if np.isnan(y_val_pred).any():
            logger.warning("NaN values found in validation predictions. Replacing with mean.")
            mean_pred = np.nanmean(y_val_pred)
            y_val_pred = np.nan_to_num(y_val_pred, nan=mean_pred)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return

    try:
        # Evaluate the model
        logger.info("Evaluating the model...")
        
        # Training metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        
        # Validation metrics
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        logger.info(f'Training Set - MSE: {train_mse:.2f}, R²: {train_r2:.4f}')
        logger.info(f'Validation Set - MSE: {val_mse:.2f}, R²: {val_r2:.4f}')
        
        # Check for overfitting
        if train_r2 - val_r2 > 0.1:
            logger.warning("Possible overfitting detected. Training R² is significantly higher than Validation R².")
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        return

    try:
        # Feature importance analysis (for the original features through PCA)
        logger.info("Analyzing feature importance through PCA components...")
        
        # Get the most important components
        components = pca.components_
        feature_names = X_train.columns
        
        # Get top 5 features for the first principal component
        pc1 = components[0]
        top_features_idx = np.abs(pc1).argsort()[-5:][::-1]
        top_features = [(feature_names[i], pc1[i]) for i in top_features_idx]
        
        logger.info("Top 5 features in first principal component:")
        for feature, weight in top_features:
            logger.info(f"  {feature}: {weight:.4f}")
    except Exception as e:
        logger.error(f"Error during feature importance analysis: {e}")

    try:
        # Predict future values and calculate depreciation
        logger.info("Calculating depreciation for all cars...")
        
        # Create output DataFrame starting with original data
        output_df = pd.DataFrame()
        output_df['ID'] = train_df['ID'] if 'ID' in train_df.columns else pd.Series(range(len(train_df)))
        output_df['Original_Price'] = y_train.values  # Original prices
        output_df['Predicted_Price'] = y_train_pred  # Predicted prices
        
        # Calculate future values for years 1-5
        years_to_predict = 5
        current_year = 2025
        
        # Calculate depreciation for each car
        standard_depreciation_rate = 0.15  # 15% per year
        
        for i, price in enumerate(y_train_pred):
            future_values = calculate_depreciation(price, years_to_predict, standard_depreciation_rate)
            # First value is current price, so future values start from index 1
            for year in range(1, years_to_predict + 1):
                output_df[f'Year_{current_year + year}'] = future_values[year]
        
        # Calculate total depreciation over 5 years
        output_df['Depreciation_5yr'] = output_df['Predicted_Price'] - output_df[f'Year_{current_year + years_to_predict}']
        output_df['Depreciation_Percentage'] = (output_df['Depreciation_5yr'] / output_df['Predicted_Price'] * 100).round(2)
        
        # Save the predictions
        output_df.to_csv(predictions_path, index=False)
        logger.info(f"Predictions and depreciation saved to {predictions_path}")
    except Exception as e:
        logger.error(f"Error during depreciation calculation: {e}")

    try:
        # Process specific car example
        logger.info("Processing example car...")
        
        # Example car with more realistic attributes matching expected columns
        car_attributes = {
            'ID': 12345678,
            'Prod. year': 2020,
            'Cylinders': 4.0,
            'Airbags': 6
        }
        
        # Add any categorical variables that were in the training data
        categorical_columns = ['Levy_1011', 'Color']
        if 'Levy_1011' in X_train.columns or any(col.startswith('Levy_1011_') for col in X_train.columns):
            car_attributes['Levy_1011'] = False
        
        # Add a specific color - ensure it's one that was in the training data
        if any(col.startswith('Color_') for col in X_train.columns):
            color_columns = [col for col in X_train.columns if col.startswith('Color_')]
            if color_columns:
                # Choose Blue if it exists, otherwise the first color
                blue_col = next((col for col in color_columns if 'Blue' in col), color_columns[0])
                color_name = blue_col.replace('Color_', '')
                car_attributes['Color'] = color_name
        
        # Create DataFrame for the car
        car_df = pd.DataFrame([car_attributes])
        
        # Process the car data
        car_processed = preprocess_data(car_df, train_mode=False, reference_columns=train_columns)
        
        # Ensure all expected columns are present
        missing_cols = set(X_train.columns) - set(car_processed.columns)
        for col in missing_cols:
            car_processed[col] = 0
        
        # Align columns with the training data
        car_processed = car_processed[X_train.columns]
        
        # Scale the data
        car_scaled = scaler.transform(car_processed)
        
        # Apply PCA
        car_pca = pca.transform(car_scaled)
        
        # Make prediction
        car_price = model.predict(car_pca)[0]
        logger.info(f"Predicted price for example car: ${car_price:.2f}")
        
        # Save the predicted price to a text file
        with open(txt_output_path, "w") as file:
            file.write(f"{car_price:.2f}")
        logger.info(f"Predicted price saved to {txt_output_path}")
        
        # Calculate future values
        current_year = 2025
        years_to_predict = 5
        future_values = [car_price]
        for year in range(1, years_to_predict + 1):
            future_value = car_price * ((1 - standard_depreciation_rate) ** year)
            future_values.append(future_value)
        
        # Generate depreciation graph
        generate_depreciation_graph(future_values, current_year, plot_path)
        
    except Exception as e:
        logger.error(f"Error during example car prediction: {e}")

if __name__ == "__main__":
    main()