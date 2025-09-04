import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the training data
# Note: In a real scenario, replace 'train.csv' with the actual file path.
# Since the data is provided in the query as a document, you can save it to a file or use io.StringIO with the full content.
df_train = pd.read_csv('train.csv')

# Preprocess the data: Create TotalBath feature
df_train['TotalBath'] = df_train['FullBath'] + 0.5 * df_train['HalfBath']

# Select features and target
features = ['GrLivArea', 'BedroomAbvGr', 'TotalBath']
X = df_train[features]
y = df_train['SalePrice']

# Split data for validation (optional, but good practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Optional: Evaluate the model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
st.sidebar.write(f"Model RMSE on test set: ${rmse:,.2f}")

# Streamlit UI
st.title("House Price Prediction App")
st.write("Predict the sale price of a house based on square footage, number of bedrooms, and number of bathrooms.")

# User inputs
sqft = st.number_input("Square Footage (Living Area)", min_value=500, max_value=10000, value=1500, step=100)
bedrooms = st.number_input("Number of Bedrooms", min_value=0, max_value=10, value=3, step=1)
bathrooms = st.number_input("Number of Bathrooms (e.g., 2.5 for 2 full + 1 half)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)

# Prediction button
if st.button("Predict Price"):
    # Prepare input data
    input_data = np.array([[sqft, bedrooms, bathrooms]])
    prediction = model.predict(input_data)[0]
    
    st.success(f"Predicted House Price: ${prediction:,.2f}")

# Additional info
st.sidebar.title("About the Model")
st.sidebar.write("This is a simple linear regression model trained on the Ames Housing dataset.")
st.sidebar.write("Features used:")
st.sidebar.write("- GrLivArea: Above grade (ground) living area square feet")
st.sidebar.write("- BedroomAbvGr: Bedrooms above grade")
st.sidebar.write("- TotalBath: Total bathrooms (full + 0.5 * half)")
st.sidebar.write("Target: SalePrice")

# Run the app with: streamlit run this_file.py
