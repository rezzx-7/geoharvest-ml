import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings
import base64

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# Step 1: Load the dataset
df = pd.read_csv(r'C:\Users\raghunath\OneDrive\Desktop\geoharvest ml\commodities_data.csv')

# Step 3: Data preprocessing
# Replace states with numerical mapping
state_mapping = {'Maharashtra': 1}
df['state'] = df['state'].replace(state_mapping)

# Drop unwanted columns
df.drop(columns=['arrival_date'], inplace=True)

# Step 4: Apply OneHotEncoder and LabelEncoder
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
le = LabelEncoder()

# Apply Label Encoding to 'grade' column
df['grade'] = le.fit_transform(df['grade'])

# OneHotEncoding 'market', 'district', and 'commodity'
x1 = ohe.fit_transform(df[['market', 'district', 'commodity']])

# Combine features for training
x = np.hstack((df[['state', 'grade', 'min_price', 'max_price']].values, x1))

# Step 5: Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, df['modal_price'], test_size=0.2, random_state=42)

# Step 6: Train RandomForestRegressor model
reg = RandomForestRegressor(bootstrap=True, n_estimators=500, random_state=42)
reg.fit(x_train, y_train)

# Step 7: Predict and evaluate
y_pred = reg.predict(x_test)


# Step 8: Function to predict modal price based on user input
def predict_modal_price(district, market, commodity):
    # Preprocess user inputs
    district_encoded = ohe.transform([[district, market, commodity]])  # One-hot encode the inputs
    input_data = np.hstack((df[['state', 'grade', 'min_price', 'max_price']].values[0], district_encoded[0]))  # Combine the encoded data with other features

    # Predict modal price
    predicted_price = reg.predict([input_data])
    return predicted_price


# Step 9: Streamlit app with custom HTML and CSS

def set_background(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
# Inject CSS code to change the background img


# Custom CSS Styling
st.markdown("""
    <style>
 
    .st-emotion-cache-13k62yr {
    background: #f5f5dc;
    color: Black;
            }

    .st-emotion-cache-6qob1r {
    position: relative;
    height: 100%;
    width: 100%;
    overflow: overlay;
    background: #50d8af;
}

    .st-emotion-cache-kgpedg {
    display: flex;
    -webkit-box-pack: justify;
    justify-content: space-between;
    -webkit-box-align: start;
    align-items: start;
    padding: calc(1.375rem) 1.5rem 1.5rem;
    background: #50d8af;
}

    .st-emotion-cache-h4xjwg {
    position: fixed;
    top: 0px;
    left: 0px;
    right: 0px;
    height: 3.75rem;
    background: #f5f5dc;
    outline: none;
    z-index: 999990;
    display: block;
}             
    
            
    /* Center the title */
    .title {
            text-align: center;
            color: black;
            font-family: 'Arial', sans-serif;
            font-size: 36px;
            
        }
    .header{
            color:black;}
        
    .st-emotion-cache-1f3w014 {
    vertical-align: middle;
    overflow: hidden;
    color:black;
    fill: currentcolor;
    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    font-size: 1.5rem;
    width: 1.5rem;
    height: 1.5rem;
    flex-shrink: 0;
}
    .st-emotion-cache-1gwvy71 {
    padding: 0px 1.5rem 6rem;
    background: #50d8af;
}
    .st-bc {
    color: rgb(0 0 0);
}
    .st-bw {
    caret-color: rgb(0 0 0);
    background: white;
}
    .st-emotion-cache-lpgk4i {
    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    -webkit-box-pack: center;
    justify-content: center;
    font-weight: 400;
    padding: 0.25rem 0.75rem;
    border-radius: 0.5rem;
    min-height: 2.5rem;
    margin: 0px;
    line-height: 1.6;
    color: inherit;
    width: auto;
    user-select: none;
    background-color: #f5f5dc;
    border: 1px solid rgba(250, 250, 250, 0.2);
}
        
        /* Style the input boxes */
        .stTextInput>div>input {
            border: 2px solid #2E86C1;
            border-radius: 5px;
            padding: 10px;
            
           
        .p, ol, ul, dl {
    margin: 0px 0px 1rem;
    padding: 0px;
    font-size: 1rem;
    font-weight: 400;
    color: black;
}
        }

        /* Style the result box */
        .result-box {
            background-color: #F4F6F7;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #BFC9CA;
            text-align: center;
            color: black;
            font-size: 22px;
        }

        /* Success price styling */
        .price-success {
            font-size: 28px;
            font-weight: bold;
            color: #27AE60;
        }

       .image(
            border-size:2px;
            )

    </style>
""", unsafe_allow_html=True)

image_path = r'C:\Users\raghunath\OneDrive\Desktop\geoharvest ml\pexels-pixabay-461960.jpg'
set_background(image_path)
# Header with enhanced HTML styling
st.markdown("""<h2 class='title'>🌾 CROP PRICE PREDICTOR 🌾</h2>
            <p>Our project GEO-Harvest develops a model that tries to predict crop yields based on environmental and climatic data. We analyze historical data using machine learning techniques to provide accurate predictions that can help farmers make better decisions, use resources wisely, and improve food production.
 Crop yield prediction and crop success in specific locations is the key to efficient and sustainable farming.This model can also help guess when there might be too little or too much of a crop, which lets people plan markets and set prices better.
</p>""", unsafe_allow_html=True)

st.image(r'C:\Users\raghunath\OneDrive\Desktop\geoharvest ml\pexels-pixabay-461960.jpg', use_column_width=True)
# Sidebar inputs
st.sidebar.header("ENTER INPUT DETAILS")
district = st.sidebar.text_input("ENTER DISTRICT").strip().lower()
market = st.sidebar.text_input("ENTER MARKET").strip().lower()
commodity = st.sidebar.text_input("ENTER CROP").strip().lower()

# Predict the price when the user clicks the button
if st.sidebar.button("PREDICTED PRICE"):
    try:
        predicted_price = predict_modal_price(district, market, commodity)

        # Display the result with enhanced formatting
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center; color: #50d8af;'>Predicted Price</h3>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class='result-box'>
                <p><strong>CROP:</strong> {commodity.capitalize()}<br>
                <strong>MARKET PRICE:</strong> {market.capitalize()}<br>
                <strong>DISTRICT:</strong> {district.capitalize()}<br></p>
                <p2 class='price-success'> {predicted_price[0]:.2f}Rs per Quintal </h2>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        # Handle errors gracefully with enhanced formatting
        st.error(f"Error: {e}")
        st.markdown(f"<p style='color: red; font-size: 18px; text-align: center;'>Something went wrong. Please check the input values!</p>", unsafe_allow_html=True)


