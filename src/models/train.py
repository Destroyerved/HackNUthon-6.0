import os
import sys
import numpy as np
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.preprocess import DataPreprocessor
from model import CancerDetectionModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_training_history(losses, title):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 1, 1)
    plt.plot(losses, label='Training Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/{title.lower().replace(" ", "_")}_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, title):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'{title} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'results/{title.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.close()

def main():
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Initialize data preprocessor
    data_dir = "data"
    preprocessor = DataPreprocessor(data_dir)
    
    # Load and preprocess data
    spectral_file = os.path.join(data_dir, "spectral_data.csv")
    image_dir = os.path.join(data_dir, "images")
    
    data = preprocessor.prepare_data(spectral_file, image_dir)
    if data is None:
        print("Error: Failed to prepare data")
        return
    
    (X_spectral_train, X_spectral_test, y_spectral_train, y_spectral_test,
     X_image_train, X_image_test, y_image_train, y_image_test) = data
    
    # Initialize model
    spectral_input_size = X_spectral_train.shape[1]  # Get the size of spectral input
    num_classes = len(np.unique(y_spectral_train))
    model = CancerDetectionModel(spectral_input_size, num_classes)
    
    # Train models
    print("Training image model...")
    model.train_image_model(X_image_train, y_image_train, epochs=20)
    
    print("Training spectral model...")
    model.train_spectral_model(X_spectral_train, y_spectral_train, epochs=20)
    
    print("Training combined model...")
    model.train_combined_model(X_image_train, X_spectral_train, y_spectral_train, epochs=20)
    
    # Evaluate models
    results = model.evaluate_models(X_image_test, X_spectral_test, y_spectral_test)
    
    # Print results
    print("\nModel Evaluation Results:")
    for model_name, (loss, accuracy) in results.items():
        print(f"\n{model_name}:")
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
    
    # Generate predictions and confusion matrices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Image model predictions
    with torch.no_grad():
        X_image = torch.FloatTensor(X_image_test).to(device)
        y_image_pred = model.image_model(X_image).argmax(1).cpu().numpy()
        
        # Spectral model predictions
        X_spectral = torch.FloatTensor(X_spectral_test).to(device)
        y_spectral_pred = model.spectral_model(X_spectral).argmax(1).cpu().numpy()
        
        # Combined model predictions
        y_combined_pred = model.combined_model(X_image, X_spectral).argmax(1).cpu().numpy()
    
    # Plot confusion matrices
    classes = [f"Class {i}" for i in range(num_classes)]  # Replace with actual class names
    plot_confusion_matrix(y_image_test, y_image_pred, classes, "Image Model")
    plot_confusion_matrix(y_spectral_test, y_spectral_pred, classes, "Spectral Model")
    plot_confusion_matrix(y_spectral_test, y_combined_pred, classes, "Combined Model")
    
    # Save classification reports
    with open('results/classification_reports.txt', 'w') as f:
        f.write("Classification Reports:\n\n")
        f.write("Image Model:\n")
        f.write(classification_report(y_image_test, y_image_pred, target_names=classes))
        f.write("\nSpectral Model:\n")
        f.write(classification_report(y_spectral_test, y_spectral_pred, target_names=classes))
        f.write("\nCombined Model:\n")
        f.write(classification_report(y_spectral_test, y_combined_pred, target_names=classes))

if __name__ == "__main__":
    main() 