import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
import signal
import time

# Function to handle timeout
def handler(signum, frame):
    raise Exception("Model training timeout")

# Set the timeout signal
signal.signal(signal.SIGALRM, handler)
signal.alarm(600)  # Timeout after 600 seconds (10 minutes)

def main():
    try:
        # Load the dataset
        print("Loading datasets...")
        train_df = pd.read_csv('train.csv')
        print("Datasets loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        return

    # Preprocess the data
    def preprocess_data(df, columns=None):
        # Handle missing values
        df.ffill(inplace=True)

        # Encode categorical variables
        df = pd.get_dummies(df, drop_first=True)
        
        # Align columns with the training set
        if columns is not None:
            df = df.reindex(columns=columns, fill_value=0)
        
        return df

    try:
        print("Preprocessing training data...")
        train_df = preprocess_data(train_df)
        train_columns = train_df.columns  # Save the columns for later use
        print("Training data preprocessed successfully.")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return

    try:
        # Define features and target variable
        print("Defining features and target variable...")
        X_train = train_df.drop('Price', axis=1)
        y_train = train_df['Price']
        print("Features and target variable defined successfully.")
        print(f"Size of X_train: {X_train.shape}")
        print(f"Size of y_train: {y_train.shape}")
        print("Sample of X_train:")
        print(X_train.head())
        print("Sample of y_train:")
        print(y_train.head())

        # Handle NaNs in target variables
        print("Handling NaNs in target variables...")
        print(f"NaNs in y_train before handling: {y_train.isnull().sum()}")
        y_train.fillna(y_train.mean(), inplace=True)
        print(f"NaNs in y_train after handling: {y_train.isnull().sum()}")
        print("NaNs in target variables handled successfully.")

        # Normalize/scale the data
        print("Normalizing/scaling the data...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        print("Data normalized/scaled successfully.")

        # Dimensionality reduction with PCA
        print("Applying PCA for dimensionality reduction...")
        pca = PCA(n_components=50)
        X_train_pca = pca.fit_transform(X_train_scaled)
        print(f"Size of X_train after PCA: {X_train_pca.shape}")

        # Check for NaNs in X_train_pca, y_train
        print("Checking for NaNs in scaled datasets and target variables...")
        if pd.isnull(X_train_pca).any():
            print("NaNs found in X_train_pca")
        if pd.isnull(y_train).any():
            print("NaNs found in y_train")

        if pd.isnull(X_train_pca).any() or pd.isnull(y_train).any():
            raise ValueError("NaN values found in scaled datasets or target variables")
        print("No NaNs found in scaled datasets or target variables.")
    except Exception as e:
        print(f"Error during normalization/scaling or NaN check: {e}")
        return

    try:
        # Initialize and train the model
        print("Initializing the model...")
        model = LinearRegression()
        print("Training the model...")
        start_time = time.time()
        model.fit(X_train_pca, y_train)
        end_time = time.time()
        print(f"Model trained successfully in {end_time - start_time} seconds.")
    except Exception as e:
        print(f"Error during model training: {e}")
        return
    finally:
        # Disable the alarm
        signal.alarm(0)

    try:
        # Make predictions on the training set
        print("Making predictions...")
        y_pred = model.predict(X_train_pca)
        print("Predictions made successfully.")
        print(f"Sample of predictions: {y_pred[:5]}")

        # Handle NaNs in predictions
        if pd.isnull(y_pred).any():
            print("NaN values found in predictions. Replacing NaNs with the mean of predictions.")
            y_pred = pd.Series(y_pred).fillna(y_pred.mean()).values
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    try:
        # Evaluate the model on the training set
        print("Evaluating the model...")
        mse = mean_squared_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)

        print(f'Mean Squared Error: {mse}')
        print(f'R-squared: {r2}')
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return

    # Function to predict car value 5 years later and calculate depreciation
    def predict_future_value(current_value, rate_of_depreciation=0.15):
        future_value = current_value * ((1 - rate_of_depreciation) ** 5)
        depreciation = current_value - future_value
        return future_value, depreciation

    try:
        # Predict future values for all cars in the training set and calculate depreciation
        print("Predicting future values and calculating depreciation...")
        train_df['predicted_price'] = y_pred
        train_df['future_value'] = train_df['predicted_price'].apply(lambda x: predict_future_value(x)[0])
        train_df['depreciation'] = train_df['predicted_price'].apply(lambda x: predict_future_value(x)[1])
        print("Future values and depreciation predicted successfully.")
        print("Sample of future values and depreciation:")
        print(train_df[['predicted_price', 'future_value', 'depreciation']].head())

        # Save the results to a CSV file
        output_path = 'predicted_values.csv'
        train_df.to_csv(output_path, index=False)
        print(f"Predicted values and depreciation saved to '{output_path}'.")
    except Exception as e:
        print(f"Error during prediction or saving results: {e}")

if __name__ == "__main__":
    main()