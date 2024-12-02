from flask import Flask, request, jsonify
import pandas as pd
import traceback
import joblib
import numpy as np
import sys
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)  # This allows all origins by default

# If you want to allow specific origins:
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}})

# Load pre-trained model, scaler, encoder, and columns
model_path = 'rf_model.pkl'
scaler_path = 'scaler.pkl'
encoder_path = 'encoder.pkl'
columns_path = 'model_columns_group2.pkl'

# Load the files
rf_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
encoder = joblib.load(encoder_path)
model_columns = joblib.load(columns_path)

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Parse input data from the request form
        input_data = [x for x in request.form.values()]
        print("Input Data: ", input_data)

        # Convert input into a DataFrame
        input_df = pd.DataFrame(
            [input_data],
            columns=[
                'PRIMARY_OFFENCE', 'OCC_DOW', 'REPORT_DOW', 'HOOD_158',
                'BIKE_MAKE', 'BIKE_TYPE', 'BIKE_COLOUR', 'BIKE_COST', 
                'LOCATION_TYPE', 'PREMISES_TYPE'
            ]
        )
        print("Input DataFrame:\n", input_df)

        # Numerical and categorical features
        num_features = ['BIKE_COST']
        cat_features = ['PRIMARY_OFFENCE', 'OCC_DOW', 'REPORT_DOW', 'HOOD_158', 
                        'BIKE_MAKE', 'BIKE_TYPE', 'BIKE_COLOUR', 'LOCATION_TYPE', 'PREMISES_TYPE']

        # Fill missing values
        input_df[num_features] = input_df[num_features].fillna(0)  # Default for numeric
        input_df[cat_features] = input_df[cat_features].fillna('UNKNOWN')  # Default for categorical

        # Transform numerical features
        num_data = scaler.transform(input_df[num_features])
        print(f"Numerical feature shape: {num_data.shape}")

        # One-hot encode categorical features
        cat_data = encoder.transform(input_df[cat_features])
        print(f"Categorical feature shape: {cat_data.shape}")

        # Combine numerical and categorical features
        input_transformed = np.hstack((num_data, cat_data))
        input_transformed_df = pd.DataFrame(input_transformed)
        print(f"Combined feature shape: {input_transformed.shape}")
        
        input_transformed_df.columns = list(encoder.get_feature_names_out(
            ['PRIMARY_OFFENCE', 'OCC_DOW', 'REPORT_DOW', 'HOOD_158', 
             'BIKE_MAKE', 'BIKE_TYPE', 'BIKE_COLOUR', 'LOCATION_TYPE', 'PREMISES_TYPE']
        )) + ['BIKE_COST']
        
        aligned_input_df = input_transformed_df.reindex(columns=model_columns, fill_value=0)

        # Make predictions
        prediction = rf_model.predict(aligned_input_df)
        prediction_proba = rf_model.predict_proba(aligned_input_df)

        # Prepare response
        response = {
            'predicted_class': 'Recovered' if prediction[0] == 1 else 'Stolen',
            'prediction_probability': prediction_proba[0].tolist()
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': traceback.format_exc()})

if __name__ == '__main__':
    try:
        port = int(sys.argv[1])  # Command-line input for port
    except:
        port = 12345  # Default port if not provided

    print("Starting Flask app...")
    app.run(debug=False, port=port)
