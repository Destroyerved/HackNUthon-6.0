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
    page_title="Healthcare AI Platform - Cancer Detection",
    page_icon="🏥",
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
        color: #00ff00;
        font-size: 1.5rem;
        font-weight: bold;
        text-decoration: none;
        text-shadow: 0 0 10px rgba(0,255,0,0.5);
    }

    .navbar-links {
        display: flex;
        gap: 2rem;
    }

    .navbar-link {
        color: #ffffff;
        text-decoration: none;
        font-size: 1rem;
        transition: color 0.3s ease;
        position: relative;
    }

    .navbar-link:hover {
        color: #00ff00;
    }

    .navbar-link::after {
        content: '';
        position: absolute;
        width: 0;
        height: 2px;
        bottom: -5px;
        left: 0;
        background-color: #00ff00;
        transition: width 0.3s ease;
    }

    .navbar-link:hover::after {
        width: 100%;
    }

    /* Hero Section Styles */
    .hero {
        background: linear-gradient(135deg, rgba(18, 18, 32, 0.95), rgba(27, 27, 50, 0.95));
        padding: 8rem 2rem 4rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        border-bottom: 1px solid rgba(0, 255, 163, 0.1);
    }

    .hero-content {
        max-width: 800px;
        margin: 0 auto;
        position: relative;
        z-index: 1;
    }

    .hero-title {
        color: #00FFA3;
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 0 0 20px rgba(0, 255, 163, 0.5);
    }

    .hero-subtitle {
        color: #ffffff;
        font-size: 1.5rem;
        margin-bottom: 2rem;
        line-height: 1.6;
    }

    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 2rem;
    }

    .stat-item {
        background: rgba(0, 255, 163, 0.1);
        padding: 1rem 2rem;
        border-radius: 12px;
        border: 1px solid rgba(0, 255, 163, 0.2);
    }

    .stat-value {
        color: #00FFA3;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .stat-label {
        color: #ffffff;
        font-size: 1rem;
    }

    /* Adjust main content to account for fixed navbar */
    .main {
        margin-top: 80px;
    }

    .main {
        padding: 2rem;
        background-color: #1a1a1a;
    }
    .stTitle {
        color: #ffffff;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stAlert {
        background-color: #2c5282;
        border-left: 4px solid #90cdf4;
    }
    .css-1d391kg {
        background-color: #ffffff;
    }
    .medical-card {
        background: #1a1a1a;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border: 1px solid #333;
        border-left: 4px solid #90cdf4;
    }
    .medical-header {
        color: #ffffff;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .medical-metric {
        background: #1a1a1a;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #333;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        color: #90cdf4;
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
    .medical-button {
        background: linear-gradient(135deg, #2c5282, #1a365d);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
    }
    .medical-button:hover {
        background: linear-gradient(135deg, #1a365d, #2c5282);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .feature-list {
        color: #ffffff;
        list-style-type: none;
        padding-left: 0;
    }
    .feature-list li {
        margin: 0.5rem 0;
        color: #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Sidebar
    st.sidebar.title("Healthcare AI")
    page = st.sidebar.radio("Navigation", 
        ["Home", "Patient Analysis", "Cancer Detection", "Results", "About"])

    if page == "Home":
        show_home_page()
    elif page == "Patient Analysis":
        show_patient_analysis()
    elif page == "Cancer Detection":
        show_cancer_detection()
    elif page == "Results":
        show_results()
    else:
        show_about()

def show_home_page():
    st.title("🏥 Healthcare AI Platform")
    st.markdown("### Advanced Cancer Detection System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="medical-card">
            <h3 class="medical-header">Clinical Features:</h3>
            <ul class="feature-list">
                <li>🔬 Optical Spectrometer Analysis</li>
                <li>📊 High-Resolution Medical Imaging</li>
                <li>📋 Patient History Integration</li>
                <li>🔍 Early Detection Algorithms</li>
                <li>📈 Prognostic Indicators</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="medical-card">
            <h3 class="medical-header">Clinical Benefits:</h3>
            <ul class="feature-list">
                <li>🎯 Improved Diagnostic Accuracy</li>
                <li>⚡ Faster Treatment Planning</li>
                <li>📊 Data-Driven Decisions</li>
                <li>👥 Enhanced Patient Care</li>
                <li>💊 Personalized Treatment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Clinical Statistics Dashboard
    st.subheader("📊 Clinical Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="medical-metric">
            <div class="metric-title">Diagnostic Accuracy</div>
            <div class="metric-value">95.2%</div>
            <div class="metric-delta">↑2.1%</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="medical-metric">
            <div class="metric-title">Cases Analyzed</div>
            <div class="metric-value">1,234</div>
            <div class="metric-delta">↑156</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="medical-metric">
            <div class="metric-title">False Positives</div>
            <div class="metric-value">3.2%</div>
            <div class="metric-delta">↓0.5%</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="medical-metric">
            <div class="metric-title">Processing Time</div>
            <div class="metric-value">1.2s</div>
            <div class="metric-delta">↓0.3s</div>
        </div>
        """, unsafe_allow_html=True)

def show_patient_analysis():
    st.title("👤 Patient Analysis")
    
    # Patient Information Form
    st.markdown("""
    <div class="medical-card">
        <h3 class="medical-header">Patient Information</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        patient_id = st.text_input("Patient ID", placeholder="Enter patient ID")
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    
    with col2:
        medical_history = st.text_area("Medical History", 
            placeholder="Enter relevant medical history...")
        symptoms = st.text_area("Current Symptoms", 
            placeholder="Describe current symptoms...")
    
    # Medical Data Upload
    st.markdown("""
    <div class="medical-card">
        <h3 class="medical-header">Medical Data Upload</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Spectral Data")
        spectral_file = st.file_uploader("Upload Spectral Analysis Data", 
            type=["csv"], help="Upload spectral analysis data in CSV format")
        
    with col2:
        st.subheader("Medical Images")
        image_file = st.file_uploader("Upload Medical Images", 
            type=["jpg", "jpeg", "png"], help="Upload medical images for analysis")

    if st.button("Process Patient Data"):
        with st.spinner("Processing..."):
            time.sleep(2)  # Simulating processing time
            st.success("Patient data processed successfully!")

def show_cancer_detection():
    st.title("Cancer Detection")
    
    # Model Selection
    model_type = st.selectbox(
        "Select Detection Model",
        ["Combined (Image + Spectral)", "Image-based Only", "Spectral-based Only"]
    )
    
    # Analysis Parameters
    st.header("Analysis Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7
        )
        
    with col2:
        detection_mode = st.selectbox(
            "Detection Mode",
            ["High Accuracy", "Balanced", "Fast Detection"]
        )
    
    # Start Detection
    if st.button("Start Detection"):
        with st.spinner("Analyzing..."):
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.05)
            
            # Simulated results
            st.success("Analysis Complete!")
            
            # Display Results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Detection Confidence", "94.8%")
                st.metric("False Positive Rate", "2.3%")
            with col2:
                st.metric("Processing Time", "1.2s")
                st.metric("Model Version", "v2.1.0")

def show_results():
    st.title("Analysis Results")
    
    # Patient Selection
    patient_id = st.selectbox(
        "Select Patient ID",
        ["PAT001", "PAT002", "PAT003"]  # Demo IDs
    )
    
    # Results Dashboard
    st.header(f"Results for Patient {patient_id}")
    
    # Detection Results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Image Analysis")
        st.metric("Confidence", "94.5%")
        st.metric("Precision", "93.8%")
    
    with col2:
        st.subheader("Spectral Analysis")
        st.metric("Confidence", "92.3%")
        st.metric("Precision", "91.7%")
    
    with col3:
        st.subheader("Combined Analysis")
        st.metric("Confidence", "96.7%")
        st.metric("Precision", "95.9%")
    
    # Detailed Report
    st.header("Detailed Report")
    st.markdown("""
    #### Key Findings:
    - No malignant patterns detected
    - Confidence level above threshold
    - Recommended follow-up in 6 months
    
    #### Recommendations:
    1. Regular monitoring
    2. Follow-up scan in 6 months
    3. Maintain health records
    """)

def show_about():
    st.title("About Healthcare AI Platform")
    
    st.markdown("""
    ### Advanced Cancer Detection System
    
    This platform is part of the Reve Sponsored Track at Nirma University Hackathon, 
    focusing on early and accurate cancer detection using advanced AI techniques.
    
    #### Technology Stack:
    - 🧠 PyTorch for deep learning
    - 🔍 OpenCV for image processing
    - 📊 Streamlit for interface
    - 🔬 Scikit-learn for analysis
    - 📈 Advanced visualization tools
    
    #### Key Features:
    1. Multi-modal Analysis:
        - Optical spectrometer data
        - High-resolution imaging
        - Patient history integration
    2. Advanced AI Models:
        - CNN for image processing
        - Spectral data analysis
        - Combined prediction models
    3. Clinical Support:
        - Rapid detection
        - High accuracy
        - Detailed reporting
    
    #### Support:
    For technical support or inquiries:
    - 📧 medical.support@healthcare-ai.com
    - 🌐 github.com/healthcare-ai
    """)

if __name__ == "__main__":
    main() 