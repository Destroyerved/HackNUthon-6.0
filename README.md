# Cancer Detection using Machine Learning

This project focuses on developing machine learning models for early cancer detection using optical spectrometer data and digital images. The project is part of the Reve Sponsored Track at the Nirma University Hackathon.

## Project Structure

```
├── data/                   # Directory for storing datasets
├── src/                    # Source code
│   ├── data_processing/    # Data preprocessing scripts
│   ├── models/            # Model architecture definitions
│   └── utils/             # Utility functions
├── notebooks/             # Jupyter notebooks for analysis
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset and place it in the `data/` directory

## Features

- Data preprocessing for optical spectrometer data
- Image processing for digital camera images
- Multiple model architectures for cancer detection
- Model evaluation and visualization tools
- Cross-validation and hyperparameter tuning

## Usage

1. Data Preprocessing:
```bash
python src/data_processing/preprocess.py
```

2. Model Training:
```bash
python src/models/train.py
```

3. Model Evaluation:
```bash
python src/models/evaluate.py
```

## Model Architecture

The project implements multiple model architectures:
- CNN for image-based cancer detection
- Neural Network for spectral data analysis
- Ensemble methods combining both modalities

## Performance Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 