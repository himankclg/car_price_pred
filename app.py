import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(page_title="Car Price Predictor", layout="wide")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main {padding-top: 2rem;}
    [data-testid="stMetricValue"] {font-size: 1.8rem;}
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('car_prediction_model2.pkl')
    except FileNotFoundError:
        return None

model = load_model()

# --- HEADER ---
st.title("🚗 Car Price Prediction System")

if model is None:
    st.error("Model file not found! Please run your training script first.")
    st.stop()

# --- SIDEBAR INPUTS ---
st.sidebar.header("Vehicle Specifications")

# Grouping inputs for better UX
year = st.sidebar.slider('Manufacturing Year', 2000, 2018, 2005)
present_price = st.sidebar.number_input('Showroom Price (Lakhs)', 0.5, 100.0, 10.0)
kms_driven = st.sidebar.number_input('Total Kilometers Driven', 0, 1000000, 30000)

st.sidebar.markdown("---")
fuel_type = st.sidebar.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG'])
seller_type = st.sidebar.selectbox('Seller Type', ['Dealer', 'Individual'])
transmission = st.sidebar.selectbox('Transmission', ['Manual', 'Automatic'])
owner = st.sidebar.selectbox('Previous Owners', [0, 1, 2, 3])

# Logic for car age
car_age = 2026 - year # Updated to current year

# Action Button
predict_btn = st.sidebar.button("Calculate Value", type="primary", use_container_width=True)

# --- MAIN CONTENT ---
if predict_btn:
    # 1. Encoding
    fuel_encoded = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}[fuel_type]
    seller_encoded = {'Dealer': 0, 'Individual': 1}[seller_type]
    transmission_encoded = {'Manual': 0, 'Automatic': 1}[transmission]
    
    # 2. Prediction
    input_data = pd.DataFrame({
        'Year': [year],
        'Present_Price': [present_price],
        'Kms_Driven': [kms_driven],
        'Fuel_Type': [fuel_encoded],
        'Seller_Type': [seller_encoded],
        'Transmission': [transmission_encoded],
        'Owner': [owner]
    })
    
    predicted_price = max(0, model.predict(input_data)[0]) # Ensure no negative price
    
    # 3. Calculations
    depreciation = present_price - predicted_price
    
    # --- DISPLAY RESULTS ---
    st.subheader("Valuation Summary")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated Market Price", f"₹{predicted_price:.2f} L")
    m2.metric("Original Price", f"₹{present_price:.2f} L")
    m3.metric("Depreciation", f"₹{depreciation:.2f} L", delta=f"-{(depreciation/present_price)*100:.1f}%", delta_color="inverse")

    st.markdown("---")

    # Layout for Gauge and Summary
    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        st.write("### Price Analysis")
        # Visual range indicator
        low, high = predicted_price * 0.95, predicted_price * 1.05
        st.success(f"**Fair Market Range:** ₹{low:.2f}L - ₹{high:.2f}L")
        
        # Summary Table
        st.write("**Vehicle Details:**")
        summary_df = pd.DataFrame({
            "Attribute": ["Age", "Mileage", "Transmission", "Fuel"],
            "Value": [f"{car_age} Years", f"{kms_driven:,} km", transmission, fuel_type]
        })
        st.table(summary_df)

    with col_right:
        # Minimalist Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_price,
            number={'prefix': "₹", 'suffix': "L", 'font': {'size': 40}},
            gauge={
                'axis': {'range': [0, present_price]},
                'bar': {'color': "#e74c3c"},
                'steps': [
                    {'range': [0, present_price * 0.5], 'color': "#f4f4f4"},
                    {'range': [present_price * 0.5, present_price], 'color': "#e8f5e9"}
                ],
            }
        ))
        fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

else:
    # Clean landing state
    st.info("👈 Enter car details in the sidebar to generate a price report.")
    st.image("https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?auto=format&fit=crop&q=80&w=1000", use_container_width=True)