import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImageModel(nn.Module):
    def __init__(self, num_classes):
        super(ImageModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 26 * 26, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SpectralModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SpectralModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class CombinedModel(nn.Module):
    def __init__(self, spectral_input_size, num_classes):
        super(CombinedModel, self).__init__()
        self.image_model = ImageModel(num_classes)
        self.spectral_model = SpectralModel(spectral_input_size, num_classes)
        self.fc1 = nn.Linear(num_classes * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, image_input, spectral_input):
        image_features = self.image_model(image_input)
        spectral_features = self.spectral_model(spectral_input)
        combined = torch.cat([image_features, spectral_features], dim=1)
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CancerDetectionModel:
    def __init__(self, spectral_input_size, num_classes):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Initialize models
        self.image_model = ImageModel(num_classes).to(self.device)
        self.spectral_model = SpectralModel(spectral_input_size, num_classes).to(self.device)
        self.combined_model = CombinedModel(spectral_input_size, num_classes).to(self.device)
        
        # Initialize optimizers
        self.image_optimizer = torch.optim.Adam(self.image_model.parameters())
        self.spectral_optimizer = torch.optim.Adam(self.spectral_model.parameters())
        self.combined_optimizer = torch.optim.Adam(self.combined_model.parameters())
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def train_image_model(self, X_train, y_train, epochs=10, batch_size=32):
        self.image_model.train()
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                batch_X = torch.FloatTensor(X_train[i:i+batch_size]).to(self.device)
                batch_y = torch.LongTensor(y_train[i:i+batch_size]).to(self.device)
                
                self.image_optimizer.zero_grad()
                outputs = self.image_model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.image_optimizer.step()
                
                if (i + batch_size) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(X_train)}], Loss: {loss.item():.4f}')
    
    def train_spectral_model(self, X_train, y_train, epochs=10, batch_size=32):
        self.spectral_model.train()
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                batch_X = torch.FloatTensor(X_train[i:i+batch_size]).to(self.device)
                batch_y = torch.LongTensor(y_train[i:i+batch_size]).to(self.device)
                
                self.spectral_optimizer.zero_grad()
                outputs = self.spectral_model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.spectral_optimizer.step()
                
                if (i + batch_size) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(X_train)}], Loss: {loss.item():.4f}')
    
    def train_combined_model(self, X_image_train, X_spectral_train, y_train, epochs=10, batch_size=32):
        self.combined_model.train()
        for epoch in range(epochs):
            for i in range(0, len(X_image_train), batch_size):
                batch_X_image = torch.FloatTensor(X_image_train[i:i+batch_size]).to(self.device)
                batch_X_spectral = torch.FloatTensor(X_spectral_train[i:i+batch_size]).to(self.device)
                batch_y = torch.LongTensor(y_train[i:i+batch_size]).to(self.device)
                
                self.combined_optimizer.zero_grad()
                outputs = self.combined_model(batch_X_image, batch_X_spectral)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.combined_optimizer.step()
                
                if (i + batch_size) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(X_image_train)}], Loss: {loss.item():.4f}')
    
    def evaluate_models(self, X_image_test, X_spectral_test, y_test):
        self.image_model.eval()
        self.spectral_model.eval()
        self.combined_model.eval()
        
        with torch.no_grad():
            # Evaluate image model
            X_image = torch.FloatTensor(X_image_test).to(self.device)
            y = torch.LongTensor(y_test).to(self.device)
            outputs = self.image_model(X_image)
            image_loss = self.criterion(outputs, y)
            image_accuracy = (outputs.argmax(1) == y).float().mean()
            
            # Evaluate spectral model
            X_spectral = torch.FloatTensor(X_spectral_test).to(self.device)
            outputs = self.spectral_model(X_spectral)
            spectral_loss = self.criterion(outputs, y)
            spectral_accuracy = (outputs.argmax(1) == y).float().mean()
            
            # Evaluate combined model
            outputs = self.combined_model(X_image, X_spectral)
            combined_loss = self.criterion(outputs, y)
            combined_accuracy = (outputs.argmax(1) == y).float().mean()
        
        return {
            'image_model': (image_loss.item(), image_accuracy.item()),
            'spectral_model': (spectral_loss.item(), spectral_accuracy.item()),
            'combined_model': (combined_loss.item(), combined_accuracy.item())
        }

if __name__ == "__main__":
    # Example usage
    spectral_input_size = 100  # Replace with actual spectral input size
    num_classes = 3  # Replace with actual number of cancer types
    model = CancerDetectionModel(spectral_input_size, num_classes)
    print("Models created successfully!") 