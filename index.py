import streamlit as st
import subprocess
import os
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="AI Solutions Portfolio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with modern styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Main styles */
    .main {
        font-family: 'Inter', sans-serif;
        padding: 2rem;
        color: #E0E0E0;
        background-color: #0A0A0F;
        margin-top: 80px;
    }
    
    /* Header styles */
    .stTitle {
        color: #00FFA3;
        font-weight: 700;
        font-size: 2.5rem !important;
        margin-bottom: 1rem !important;
        text-shadow: 0 0 10px rgba(0, 255, 163, 0.5);
    }
    
    .subtitle {
        color: #B4B4B4;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Card styles */
    .project-card {
        background: linear-gradient(135deg, rgba(18, 18, 32, 0.95), rgba(27, 27, 50, 0.95));
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 255, 163, 0.15);
        margin: 1rem 0;
        transition: all 0.3s ease;
        border: 1px solid rgba(0, 255, 163, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .project-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 255, 163, 0.2);
        border: 1px solid rgba(0, 255, 163, 0.3);
    }
    
    .card-title {
        color: #00FFA3;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
        text-shadow: 0 0 10px rgba(0, 255, 163, 0.3);
    }
    
    .feature-list {
        list-style-type: none;
        padding-left: 0;
    }
    
    .feature-list li {
        padding: 8px 0;
        color: #E0E0E0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .feature-list li:before {
        content: "•";
        color: #00FFA3;
        font-weight: bold;
        font-size: 1.2em;
        text-shadow: 0 0 5px rgba(0, 255, 163, 0.5);
    }
    
    /* Button styles */
    .stButton button {
        width: 100%;
        background: linear-gradient(45deg, #00FFA3, #00B8FF) !important;
        color: #0A0A0F !important;
        padding: 0.75rem !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        text-shadow: none !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 20px rgba(0, 255, 163, 0.4);
    }
    
    /* Metrics styles */
    .metric-card {
        background: linear-gradient(135deg, rgba(18, 18, 32, 0.95), rgba(27, 27, 50, 0.95));
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 255, 163, 0.15);
        text-align: center;
        border: 1px solid rgba(0, 255, 163, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00FFA3;
        text-shadow: 0 0 10px rgba(0, 255, 163, 0.3);
    }
    
    /* Image container */
    div[data-testid="stImage"] {
        display: flex;
        justify-content: center;
        padding: 1rem 0;
    }
    
    div[data-testid="stImage"] img {
        max-width: 100%;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Tech stack styles */
    .tech-card {
        background: linear-gradient(135deg, rgba(18, 18, 32, 0.95), rgba(27, 27, 50, 0.95));
        padding: 1.5rem;
        border-radius: 12px;
        height: 100%;
        border: 1px solid rgba(0, 255, 163, 0.1);
        box-shadow: 0 4px 20px rgba(0, 255, 163, 0.15);
    }
    
    .tech-title {
        color: #00FFA3;
        font-weight: 600;
        margin-bottom: 1rem;
        text-shadow: 0 0 10px rgba(0, 255, 163, 0.3);
    }
    
    /* Footer styles */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #B4B4B4;
        border-top: 1px solid rgba(0, 255, 163, 0.1);
        margin-top: 3rem;
    }

    /* Custom background for project icons */
    .project-icon-bg {
        height: 300px;
        background: linear-gradient(135deg, rgba(18, 18, 32, 0.95), rgba(27, 27, 50, 0.95));
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 1rem 0;
        border: 1px solid rgba(0, 255, 163, 0.1);
        box-shadow: inset 0 0 30px rgba(0, 255, 163, 0.1);
    }

    /* Override Streamlit's default background */
    .stApp {
        background-color: #0A0A0F;
    }

    .stMarkdown {
        color: #E0E0E0;
    }

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
    </style>
""", unsafe_allow_html=True)

def main():
    # Header Section
    st.markdown('<h1 class="stTitle">🧬 AI Solutions Portfolio</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Reve Sponsored Track - Nirma University Hackathon</p>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div style="padding: 1.5rem; background: linear-gradient(135deg, rgba(18, 18, 32, 0.95), rgba(27, 27, 50, 0.95)); 
                border-radius: 12px; margin-bottom: 2rem; border: 1px solid rgba(0, 255, 163, 0.1); 
                box-shadow: 0 4px 20px rgba(0, 255, 163, 0.15);">
        Welcome to our cutting-edge AI solutions showcase! We present two innovative applications that leverage advanced artificial intelligence 
        to revolutionize healthcare and agriculture. Our solutions combine state-of-the-art technology with practical applications to address 
        real-world challenges.
    </div>
    """, unsafe_allow_html=True)
    
    # Projects Section
    st.markdown('<h2 style="color: #00FFA3; margin-bottom: 1.5rem; text-shadow: 0 0 10px rgba(0, 255, 163, 0.3);">🚀 Featured Projects</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="project-card">
            <h3 class="card-title">🏥 Healthcare AI Platform</h3>
            <div class="project-icon-bg">
                <div style="text-align: center;">
                    <span style="font-size: 5rem; filter: drop-shadow(0 0 10px rgba(0, 255, 163, 0.5));">👨‍⚕️</span>
                    <br/>
                    <span style="font-size: 4rem; filter: drop-shadow(0 0 10px rgba(0, 255, 163, 0.5));">🔬</span>
                </div>
            </div>
            <p style="color: #00FFA3; font-weight: 600; margin: 1rem 0; text-shadow: 0 0 10px rgba(0, 255, 163, 0.3);">Cancer Detection System</p>
            <ul class="feature-list">
                <li>Advanced image analysis with deep learning</li>
                <li>Real-time spectral data processing</li>
                <li>Secure patient history integration</li>
                <li>AI-powered diagnostic assistance</li>
                <li>Comprehensive medical reporting</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Healthcare App"):
            try:
                subprocess.Popen(["python", "-m", "streamlit", "run", "healthcare_app/app.py"])
            except Exception as e:
                st.error(f"Error launching Healthcare App: {str(e)}")
    
    with col2:
        st.markdown("""
        <div class="project-card">
            <h3 class="card-title">🌾 Digital Farming Platform</h3>
            <div style="height: 300px; background: linear-gradient(45deg, rgba(0, 204, 153, 0.1), rgba(74, 144, 226, 0.1)); 
                        border-radius: 12px; display: flex; align-items: center; justify-content: center; margin: 1rem 0;">
                <span style="font-size: 4rem;">🌱</span>
            </div>
            <p style="color: #0A2540; font-weight: 600; margin: 1rem 0;">Smart Soil Analysis System</p>
            <ul class="feature-list">
                <li>AI-powered soil composition analysis</li>
                <li>Real-time nutrient monitoring</li>
                <li>Smart crop recommendations</li>
                <li>Automated irrigation optimization</li>
                <li>Predictive yield analytics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🌱 Launch Soil Analysis App"):
            try:
                subprocess.Popen(["python", "-m", "streamlit", "run", "soil_app/app.py"])
            except Exception as e:
                st.error(f"Error launching Soil Analysis App: {str(e)}")
    
    # Technology Stack
    st.markdown('<h2 style="color: #0A2540; margin: 3rem 0 1.5rem;">🛠️ Technology Stack</h2>', unsafe_allow_html=True)
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown("""
        <div class="tech-card">
            <h4 class="tech-title">🧠 AI & ML</h4>
            <ul class="feature-list">
                <li>PyTorch Deep Learning</li>
                <li>Advanced Computer Vision</li>
                <li>Spectral Analysis</li>
                <li>Neural Networks</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tech_col2:
        st.markdown("""
        <div class="tech-card">
            <h4 class="tech-title">⚡ Backend</h4>
            <ul class="feature-list">
                <li>Python Ecosystem</li>
                <li>Streamlit Framework</li>
                <li>Real-time Processing</li>
                <li>Data Analytics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tech_col3:
        st.markdown("""
        <div class="tech-card">
            <h4 class="tech-title">🔐 Infrastructure</h4>
            <ul class="feature-list">
                <li>Cloud Architecture</li>
                <li>Data Security</li>
                <li>Scalable Systems</li>
                <li>Performance Monitoring</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Project Impact
    st.markdown('<h2 style="color: #0A2540; margin: 3rem 0 1.5rem;">📊 Project Impact</h2>', unsafe_allow_html=True)
    impact_col1, impact_col2 = st.columns(2)
    
    with impact_col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #0A2540;">Healthcare Impact</h3>
            <p class="metric-value">95% Accuracy</p>
            <p style="color: #00CC99;">↑ 15% Improvement</p>
            <ul class="feature-list">
                <li>Enhanced early detection</li>
                <li>Faster diagnosis time</li>
                <li>Better patient outcomes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with impact_col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #0A2540;">Agricultural Impact</h3>
            <p class="metric-value">30% Yield Increase</p>
            <p style="color: #00CC99;">↑ 25% Efficiency</p>
            <ul class="feature-list">
                <li>Resource optimization</li>
                <li>Sustainable practices</li>
                <li>Cost effectiveness</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Developed for Reve Sponsored Track - Nirma University Hackathon</p>
        <p style="color: #718096;">© {year} All Rights Reserved</p>
    </div>
    """.format(year=datetime.now().year), unsafe_allow_html=True)

if __name__ == "__main__":
    main() 