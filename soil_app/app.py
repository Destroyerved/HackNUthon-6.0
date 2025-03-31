import streamlit as st
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Add the current directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set page configuration
st.set_page_config(
    page_title="Digital Farming Platform - Soil Analysis",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    /* Navbar Styles */
    .navbar {
        background-color: #1a1a1a;
        padding: 1rem 2rem;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }

    .navbar-brand {
        color: #68d391;
        font-size: 1.5rem;
        font-weight: bold;
        text-decoration: none;
    .stTitle {
        color: #ffffff;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stAlert {
        background-color: #2d5a27;
        border-left: 4px solid #68d391;
    }
    .css-1d391kg {
        background-color: #ffffff;
    }
    .farm-card {
        background: #1a1a1a;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border: 1px solid #333;
        border-left: 4px solid #68d391;
    }
    .farm-header {
        color: #ffffff;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .farm-metric {
        background: #1a1a1a;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #333;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        color: #68d391;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
    }
    .metric-delta {
        color: #48bb78;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0;
    }
    .farm-button {
        background: linear-gradient(135deg, #3d7535, #2d5a27);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
    }
    .farm-button:hover {
        background: linear-gradient(135deg, #2d5a27, #3d7535);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .crop-icon {
        font-size: 2rem;
        margin-right: 0.5rem;
        color: #68d391;
    }
    .project-card {
        background: #1a1a1a;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border: 1px solid #333;
        border-left: 4px solid #68d391;
    }
    .card-title {
        color: #ffffff;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .feature-list {
        color: #ffffff;
        list-style-type: none;
        padding-left: 0;
    }
    .feature-list li {
        margin: 0.5rem 0;
        color: #e2e8f0;
        padding-left: 1.5rem;
        position: relative;
    }
    .feature-list li:before {
        content: "🌱";
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Sidebar
    st.sidebar.title("Digital Farming")
    page = st.sidebar.radio("Navigation", 
        ["Home", "Soil Analysis", "Crop Management", "Crop Disease Detection", "Results", "About"])

    if page == "Home":
        show_home_page()
    elif page == "Soil Analysis":
        show_soil_analysis()
    elif page == "Crop Management":
        show_crop_management()
    elif page == "Crop Disease Detection":
        show_crop_disease_detection()
    elif page == "Results":
        show_results()
    else:
        show_about()

def show_home_page():
    st.title("🌾 Digital Farming Platform")
    st.markdown("### Smart Soil Analysis System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="farm-card">
            <h3 class="farm-header">Soil Analysis Features:</h3>
            <ul style="list-style-type: none; padding-left: 0;">
                <li>🌿 Nutrient Levels</li>
                <li>💧 Moisture Content</li>
                <li>🌡️ Temperature</li>
                <li>⚡ Conductivity</li>
                <li>🧪 pH Levels</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="farm-card">
            <h3 class="farm-header">Farm Benefits:</h3>
            <ul style="list-style-type: none; padding-left: 0;">
                <li>📈 Increased Crop Yield</li>
                <li>💧 Optimal Water Usage</li>
                <li>🌱 Better Soil Health</li>
                <li>💰 Cost Reduction</li>
                <li>🌍 Sustainable Farming</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Farm Statistics Dashboard
    st.subheader("📊 Farm Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="farm-metric">
            <div class="metric-title">Average Yield</div>
            <div class="metric-value">+25%</div>
            <div class="metric-delta">↑5%</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="farm-metric">
            <div class="metric-title">Water Savings</div>
            <div class="metric-value">30%</div>
            <div class="metric-delta">↑8%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="farm-metric">
            <div class="metric-title">Soil Health Score</div>
            <div class="metric-value">8.5/10</div>
            <div class="metric-delta">↑0.5</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="farm-metric">
            <div class="metric-title">Cost Reduction</div>
            <div class="metric-value">20%</div>
            <div class="metric-delta">↑3%</div>
        </div>
        """, unsafe_allow_html=True)

def show_soil_analysis():
    st.title("🌱 Soil Analysis")
    
    # Field Information
    st.markdown("""
    <div class="farm-card">
        <h3 class="farm-header">Field Information</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        field_id = st.text_input("Field ID", placeholder="Enter field ID")
        area = st.number_input("Area (acres)", min_value=0.0, value=1.0)
        crop_type = st.selectbox("Current/Planned Crop", 
            ["Wheat", "Rice", "Corn", "Soybeans", "Cotton", "Other"])
    
    with col2:
        location = st.text_input("Location", placeholder="Enter field location")
        last_harvest = st.date_input("Last Harvest Date")
    
    # Soil Data Upload
    st.markdown("""
    <div class="farm-card">
        <h3 class="farm-header">Soil Data Upload</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Spectral Data")
        spectral_file = st.file_uploader("Upload Soil Spectrometer Data", 
            type=["csv"], help="Upload soil spectral analysis data in CSV format")
        
    with col2:
        st.subheader("Sensor Data")
        sensor_file = st.file_uploader("Upload Sensor Data", 
            type=["csv"], help="Upload soil sensor data in CSV format")
    
    if st.button("Analyze Soil"):
        with st.spinner("Analyzing soil composition..."):
            time.sleep(2)
            st.success("Soil analysis completed!")

def show_crop_management():
    st.title("Crop Management")
    
    # Analysis Type
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Soil Health Assessment", "Nutrient Management", "Irrigation Planning"]
    )
    
    # Analysis Parameters
    st.header("Analysis Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        depth = st.slider("Soil Depth (cm)", 0, 100, 30)
        season = st.selectbox("Growing Season", 
            ["Spring", "Summer", "Fall", "Winter"])
    
    with col2:
        irrigation = st.selectbox("Irrigation System",
            ["Drip", "Sprinkler", "Flood", "None"])
        fertilizer = st.multiselect("Applied Fertilizers",
            ["Nitrogen", "Phosphorus", "Potassium", "Organic"])
    
    # Start Analysis
    if st.button("Generate Recommendations"):
        with st.spinner("Analyzing data..."):
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.03)
            
            st.success("Analysis Complete!")
            
            # Display Recommendations
            st.subheader("Recommendations")
            st.markdown("""
            1. **Soil Amendments:**
               - Add organic matter
               - Adjust pH levels
            
            2. **Irrigation Schedule:**
               - Morning: 6:00 AM - 7:00 AM
               - Evening: 6:00 PM - 7:00 PM
            
            3. **Fertilizer Application:**
               - Nitrogen: 40 kg/ha
               - Phosphorus: 20 kg/ha
               - Potassium: 30 kg/ha
            """)

def show_crop_disease_detection():
    st.title("🌾 Crop Disease Detection")
    
    # Introduction
    st.markdown("""
    <div style="padding: 1.5rem; background: linear-gradient(135deg, rgba(18, 18, 32, 0.95), rgba(27, 27, 50, 0.95)); 
                border-radius: 12px; margin-bottom: 2rem; border: 1px solid rgba(0, 255, 163, 0.1); 
                box-shadow: 0 4px 20px rgba(0, 255, 163, 0.15);">
        Our AI-powered crop disease detection system helps farmers identify and manage plant diseases early, 
        preventing crop loss and ensuring better yields. Upload images of your crops for instant analysis and recommendations.
    </div>
    """, unsafe_allow_html=True)
    
    # Disease Detection Interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="project-card">
            <h3 class="card-title">📸 Upload Crop Images</h3>
            <p style="color: #00FFA3; font-weight: 600; margin: 1rem 0;">Supported Crops:</p>
            <ul class="feature-list">
                <li>Wheat</li>
                <li>Rice</li>
                <li>Corn</li>
                <li>Tomatoes</li>
                <li>Potatoes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader("Upload crop image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            # Analysis options
            st.markdown("""
            <div class="project-card">
                <h3 class="card-title">🔍 Analysis Options</h3>
                <ul class="feature-list">
                    <li>Disease Detection</li>
                    <li>Severity Assessment</li>
                    <li>Treatment Recommendations</li>
                    <li>Prevention Strategies</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Analyze Disease"):
                with st.spinner("Analyzing crop health..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        time.sleep(0.03)
                    
                    st.success("Analysis Complete!")
                    
                    # Display results
                    st.markdown("""
                    <div class="project-card">
                        <h3 class="card-title">📊 Analysis Results</h3>
                        <p style="color: #00FFA3; font-weight: 600; margin: 1rem 0;">Detected: Leaf Blight</p>
                        <p style="color: #E0E0E0;">Severity: Moderate</p>
                        <p style="color: #E0E0E0;">Confidence: 92%</p>
                        
                        <p style="color: #00FFA3; font-weight: 600; margin: 1rem 0;">Recommended Actions:</p>
                        <ul class="feature-list">
                            <li>Apply fungicide treatment</li>
                            <li>Improve air circulation</li>
                            <li>Monitor moisture levels</li>
                            <li>Remove infected leaves</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="project-card">
            <h3 class="card-title">📈 Disease Statistics</h3>
            <div style="height: 300px; background: linear-gradient(135deg, rgba(18, 18, 32, 0.95), rgba(27, 27, 50, 0.95)); 
                        border-radius: 12px; display: flex; align-items: center; justify-content: center; margin: 1rem 0;">
                <div style="text-align: center;">
                    <span style="font-size: 4rem; filter: drop-shadow(0 0 10px rgba(0, 255, 163, 0.5));">📊</span>
                </div>
            </div>
            <ul class="feature-list">
                <li>Real-time disease tracking</li>
                <li>Historical data analysis</li>
                <li>Regional disease patterns</li>
                <li>Prevention effectiveness</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Disease Prevention Tips
        st.markdown("""
        <div class="project-card">
            <h3 class="card-title">💡 Prevention Tips</h3>
            <ul class="feature-list">
                <li>Regular crop inspection</li>
                <li>Proper spacing between plants</li>
                <li>Balanced nutrition</li>
                <li>Water management</li>
                <li>Sanitation practices</li>
                <li>Resistant varieties</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def show_results():
    st.title("Analysis Results")
    
    # Field Selection
    field_id = st.selectbox(
        "Select Field",
        ["FIELD001", "FIELD002", "FIELD003"]  # Demo IDs
    )
    
    # Results Dashboard
    st.header(f"Results for Field {field_id}")
    
    # Soil Health Indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Nutrient Levels")
        st.metric("Nitrogen", "High", "↑")
        st.metric("Phosphorus", "Medium", "→")
        st.metric("Potassium", "Low", "↓")
    
    with col2:
        st.subheader("Physical Properties")
        st.metric("Moisture", "23%", "↑2%")
        st.metric("Temperature", "22°C", "↓1°C")
    
    with col3:
        st.subheader("Chemical Properties")
        st.metric("pH Level", "6.8", "optimal")
        st.metric("Organic Matter", "4%", "↑0.5%")
    
    # Detailed Report
    st.header("Detailed Report")
    st.markdown("""
    #### Soil Health Assessment:
    - Overall health score: 8.5/10
    - Good organic matter content
    - Balanced pH levels
    
    #### Recommendations:
    1. Increase potassium levels
    2. Maintain current irrigation schedule
    3. Monitor nitrogen levels
    """)

def show_about():
    st.title("About Digital Farming Platform")
    
    st.markdown("""
    ### Smart Soil Analysis System
    
    This platform is part of the Reve Sponsored Track at Nirma University Hackathon, 
    focusing on digital farming and soil health management using AI technology.
    
    #### Technology Stack:
    - 🧠 PyTorch for AI models
    - 🔍 Spectral analysis
    - 📊 Advanced analytics
    - 🌡️ IoT sensor integration
    - 📈 Real-time monitoring
    
    #### Key Features:
    1. Comprehensive Analysis:
        - Soil composition
        - Nutrient levels
        - Moisture content
    2. Smart Recommendations:
        - Crop selection
        - Irrigation planning
        - Fertilizer optimization
    3. Sustainability Focus:
        - Resource optimization
        - Environmental protection
        - Sustainable practices
    
    #### Support:
    For technical support or inquiries:
    - 📧 support@digital-farming.ai
    - 🌐 github.com/digital-farming
    """)

if __name__ == "__main__":
    main() 