import numpy as np
import pandas as pd
import cv2
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        
    def load_spectral_data(self, file_path):
        """Load and preprocess spectral data"""
        try:
            data = pd.read_csv(file_path)
            # Assuming the first column is the target variable (cancer type)
            X = data.iloc[:, 1:]
            y = data.iloc[:, 0]
            
            # Scale the features
            X_scaled = self.scaler.fit_transform(X)
            
            return X_scaled, y
        except Exception as e:
            print(f"Error loading spectral data: {e}")
            return None, None
    
    def load_image_data(self, image_dir):
        """Load and preprocess image data"""
        images = []
        labels = []
        
        try:
            for cancer_type in os.listdir(image_dir):
                cancer_path = os.path.join(image_dir, cancer_type)
                if os.path.isdir(cancer_path):
                    for img_name in os.listdir(cancer_path):
                        img_path = os.path.join(cancer_path, img_name)
                        img = cv2.imread(img_path)
                        if img is not None:
                            # Resize image to standard size
                            img = cv2.resize(img, (224, 224))
                            # Normalize pixel values
                            img = img / 255.0
                            images.append(img)
                            labels.append(cancer_type)
            
            return np.array(images), np.array(labels)
        except Exception as e:
            print(f"Error loading image data: {e}")
            return None, None
    
    def prepare_data(self, spectral_file, image_dir):
        """Prepare both spectral and image data for training"""
        X_spectral, y_spectral = self.load_spectral_data(spectral_file)
        X_image, y_image = self.load_image_data(image_dir)
        
        if X_spectral is None or X_image is None:
            return None, None, None, None
        
        # Split the data
        X_spectral_train, X_spectral_test, y_spectral_train, y_spectral_test = train_test_split(
            X_spectral, y_spectral, test_size=0.2, random_state=42
        )
        
        X_image_train, X_image_test, y_image_train, y_image_test = train_test_split(
            X_image, y_image, test_size=0.2, random_state=42
        )
        
        return (X_spectral_train, X_spectral_test, y_spectral_train, y_spectral_test,
                X_image_train, X_image_test, y_image_train, y_image_test)

if __name__ == "__main__":
    # Example usage
    data_dir = "data"
    preprocessor = DataPreprocessor(data_dir)
    
    # Replace with actual file paths
    spectral_file = os.path.join(data_dir, "spectral_data.csv")
    image_dir = os.path.join(data_dir, "images")
    
    data = preprocessor.prepare_data(spectral_file, image_dir)
    if data is not None:
        print("Data preprocessing completed successfully!") 