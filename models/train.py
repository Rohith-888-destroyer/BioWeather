import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Ground-Truth / Proxy Target Limitation Notice:
# Ground-truth outbreak case feeds (e.g. WHO/CDC daily case numbers) are not available
# as a live, open API. Therefore, this model is trained on epidemiological climate-suitability
# proxy targets derived from vector thermal response curves (Mordecai et al. 2016, 2019).
# The output represents a Climate Suitability Risk Index (0.0 to 1.0), NOT clinical case counts.

class BioWeatherRiskModel(nn.Module):
    """PyTorch Deep Learning model for vector-borne transmission risk forecasting."""
    def __init__(self, input_dim=5, hidden_dim=32):
        super(BioWeatherRiskModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, 16)
        self.out = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.out(x))

def normalize_features(temp, min_temp, max_temp, humidity, vc_proxy):
    """Normalize raw input metrics to standard [0, 1] neural network scale."""
    norm_temp = np.clip(temp / 40.0, 0.0, 1.0)
    norm_min_temp = np.clip(min_temp / 40.0, 0.0, 1.0)
    norm_max_temp = np.clip(max_temp / 40.0, 0.0, 1.0)
    norm_hum = np.clip(humidity / 100.0, 0.0, 1.0)
    norm_vc = np.clip(vc_proxy, 0.0, 1.0)
    return np.column_stack([norm_temp, norm_min_temp, norm_max_temp, norm_hum, norm_vc])

def generate_synthetic_training_data(n_samples=2500):
    """Generate synthetic climate samples with epidemiological suitability ground truth."""
    np.random.seed(42)
    temp = np.random.uniform(10.0, 40.0, n_samples)
    min_temp = temp - np.random.uniform(1.0, 4.0, n_samples)
    max_temp = temp + np.random.uniform(1.0, 4.0, n_samples)
    humidity = np.random.uniform(20.0, 100.0, n_samples)
    
    # Target suitability calculation based on Mordecai vector thermal response
    vc_proxy = np.maximum(0.0, 1.0 - ((temp - 27.5) / 12.5) ** 2) * (humidity / 100.0)
    target = np.clip(vc_proxy + np.random.normal(0, 0.02, n_samples), 0.0, 1.0)

    features = normalize_features(temp, min_temp, max_temp, humidity, vc_proxy)
    return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32).unsqueeze(1)

def train_model(output_path="models/bioweather_model.pt", epochs=200):
    """Train PyTorch risk model and save weights."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    X_train, y_train = generate_synthetic_training_data()

    model = BioWeatherRiskModel(input_dim=5, hidden_dim=32)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("Training PyTorch BioWeather risk model...")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 40 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.5f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": 5,
        "feature_names": ["mean_temp_14d", "min_temp_14d", "max_temp_14d", "mean_humidity_14d", "vectorial_capacity_proxy"]
    }, output_path)
    print(f"Model saved successfully to {output_path}")

if __name__ == "__main__":
    train_model()
