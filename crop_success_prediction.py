import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# Load the dataset (make sure your file path is correct)
file_path = r'crop_success_prediction.py'
data = pd.read_csv(r'C:\Users\raghunath\OneDrive\Desktop\geoharvest ml\district_wise_crop_success.csv')

# Preprocess the data
label_encoder_state = LabelEncoder()
label_encoder_crop = LabelEncoder()

# Encoding the categorical variables (state and crop)
data['state_encoded'] = label_encoder_state.fit_transform(data['state'])
data['crop_encoded'] = label_encoder_crop.fit_transform(data['crop'])

# Features and target variable
X = data[['state_encoded', 'crop_encoded']]
y = data['success_rate']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction function with error handling and feature names
def predict_success_rate(state_name, crop_name):
    # Normalize input (strip spaces and convert to lowercase for matching)
    state_name = state_name.strip().lower()
    crop_name = crop_name.strip().lower()

    # Convert state and crop classes to lowercase for comparison
    states_lower = [s.lower() for s in label_encoder_state.classes_]
    crops_lower = [c.lower() for c in label_encoder_crop.classes_]

    if state_name not in states_lower:
        st.error(f"Error: State '{state_name}' not found in the training data.")
        return
    if crop_name not in crops_lower:
        st.error(f"Error: Crop '{crop_name}' not found in the training data.")
        return

    # Transform normalized inputs to the encoded values
    state_encoded = label_encoder_state.transform([label_encoder_state.classes_[states_lower.index(state_name)]])[0]
    crop_encoded = label_encoder_crop.transform([label_encoder_crop.classes_[crops_lower.index(crop_name)]])[0]

    # Add feature names to the input for prediction
    input_data = pd.DataFrame([[state_encoded, crop_encoded]], columns=['state_encoded', 'crop_encoded'])

    # Predict the success rate
    prediction = model.predict(input_data)[0]
    return round(prediction * 100, 2)  # Return success rate as percentage


# Streamlit app layout with custom styling
st.markdown(
    """
    <style>
    body {
        background-color: #CDE5D0;
    }
    .stApp {
        background-color: #CDE5D0;
    }
    h1 {
        color: #2F4F4F;
        text-align: center;
        font-family: 'Arial', sans-serif;
        font-size: 3em;
        margin-bottom: 0.5em;
    }
    .main-box {
        background-color: #66CDAA;
        border-radius: 20px;
        padding: 2em;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
    }
    .input-field {
        font-size: 1.5em;
        padding: 0.5em;
        border-radius: 10px;
        margin-bottom: 1em;
        border: 1px solid #2F4F4F;
    }
    .predict-button {
        background-color: #2F4F4F;
        color: white;
        border-radius: 10px;
        padding: 0.75em;
        font-size: 1.5em;
        border: none;
        cursor: pointer;
        text-align: center;
    }
    .success {
        font-size: 2em;
        color: #006400;
        text-align: center;
    }
    .error {
        color: red;
        font-size: 1.5em;
        text-align: center;
    }
    .footer {
        text-align: center;
        margin-top: 2em;
        color: #2F4F4F;
        font-size: 1.2em;
    }
    </style>
    """, unsafe_allow_html=True
)

# Custom HTML for the title and input section
st.markdown("<h1>🌿 Crop Success Rate Prediction 🌱</h1>", unsafe_allow_html=True)

# Input form in a styled box
st.markdown('<div class="main-box">', unsafe_allow_html=True)

state_input = st.text_input('Enter state name:', help="Type the name of the state")
crop_input = st.text_input('Enter crop name:', help="Type the name of the crop")

# Add custom buttons and outputs
if st.button('🌾 Predict Success Rate', key='predict', help="Click to predict success rate"):
    if state_input and crop_input:
        predicted_success_rate = predict_success_rate(state_input, crop_input)
        if predicted_success_rate:
            st.markdown(f"<p class='success'>Predicted success rate for {crop_input.capitalize()} in {state_input.capitalize()}: {predicted_success_rate}% 🌱</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='error'>Please enter both state and crop names.</p>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Footer with info
st.markdown("<p class='footer'>🌀 Powered by RandomForest Machine Learning Model 🌀</p>", unsafe_allow_html=True)

# Input fields with custom HTML and styling
state_input = st.text_input('Enter state name:', help="Type the name of the state").strip()
crop_input = st.text_input('Enter crop name:', help="Type the name of the crop").strip()

# Prediction button and output
if st.button('Predict Success Rate'):
    if state_input and crop_input:
        predicted_success_rate = predict_success_rate(state_input, crop_input)
        if predicted_success_rate:
            st.markdown(f"<p class='success'>Predicted success rate for {crop_input.capitalize()} in {state_input.capitalize()}: {predicted_success_rate}%</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='error'>Please enter both state and crop names.</p>", unsafe_allow_html=True)
