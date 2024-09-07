import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
import re
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
import librosa
import numpy as np

# CUDA diagnostics
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA current device: {torch.cuda.current_device()}")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def bpm_loss(outputs, targets):
    # Calculate absolute difference
    diff = torch.abs(outputs - targets)
    
    # Calculate difference if we double or halve the prediction
    diff_double = torch.abs(outputs * 2 - targets)
    diff_half = torch.abs(outputs / 2 - targets)
    
    # Take the minimum of the three differences
    min_diff = torch.min(torch.min(diff, diff_double), diff_half)
    
    return torch.mean(min_diff)

# Replace the existing criterion with this custom loss
criterion = bpm_loss

class BPMPredictor(nn.Module):
    def __init__(self, input_size):
        super(BPMPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout(self.relu(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        return x

class AudioDataset(Dataset):
    def __init__(self, folder_path, device):
        self.folder_path = folder_path
        self.device = device
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.flac') or f.endswith('.wav')]
        print(f"Found {len(self.file_list)} files in {folder_path}")
        if len(self.file_list) == 0:
            print("No files found. Check the folder path and file extensions.")
        
        for file in self.file_list:
            bpm = self.extract_bpm_from_filename(file)
            print(f"File: {file}, Extracted BPM: {bpm}")
    
    def __len__(self):
        return len(self.file_list)
    

    def extract_bpm_from_filename(self, filename):
        match = re.search(r'(?i)(minor|major)-(\d+)', filename)
        if match:
            return int(match.group(2))
        print(f"Failed to extract BPM from filename: {filename}")
        return 0  # Default BPM if not found

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.file_list[idx])
        bpm = self.extract_bpm_from_filename(self.file_list[idx])
        features = self.extract_audio_features(file_path)
        return features.to(self.device), torch.tensor([bpm], dtype=torch.float32).to(self.device)

    def extract_audio_features(self, file_path):
        y, sr = librosa.load(file_path, sr=None)
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Compute statistics
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.mean(mel, axis=1),
            np.mean(chroma, axis=1),
        ])
        
        return torch.FloatTensor(features)
    
def extract_audio_features(self, file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    waveform = waveform.to(self.device)  # Move to GPU

    # If stereo, convert to mono by averaging channels
    if waveform.shape[0] == 2:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # MFCC
    mfcc_transform = T.MFCC(sample_rate=sample_rate, n_mfcc=13, melkwargs={'n_mels': 64}).to(self.device)
    mfcc = mfcc_transform(waveform)

    # Mel Spectrogram
    mel_spectrogram = T.MelSpectrogram(sample_rate=sample_rate, n_mels=64).to(self.device)
    mel = mel_spectrogram(waveform)

    # Spectral Centroid
    spectral_centroid = T.SpectralCentroid(sample_rate=sample_rate).to(self.device)
    centroid = spectral_centroid(waveform)

    # Onset Detection
    onset_env = torchaudio.functional.detect_pitch_frequency(waveform, sample_rate)
    onset = onset_env.unsqueeze(0)  # Add a channel dimension

    features = torch.cat([
        torch.mean(mfcc, dim=2),
        torch.mean(mel, dim=2),
        torch.mean(centroid, dim=1).unsqueeze(0),  # Ensure it's 2D
        torch.mean(onset, dim=2)
    ], dim=1)

    return features.squeeze(0)  # Remove batch dimension

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, patience=10):
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for features, bpms in train_loader:
            features, bpms = features.to(device), bpms.to(device)
            
            optimizer.zero_grad()
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(features)
                loss = criterion(outputs, bpms)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, bpms in val_loader:
                features, bpms = features.to(device), bpms.to(device)
                outputs = model(features)
                loss = criterion(outputs, bpms)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

    return model

def evaluate_model(model, eval_loader, device):
    model.to(device)
    model.eval()
    
    total_error = 0.0
    with torch.no_grad():
        for features, bpms in eval_loader:
            features, bpms = features.to(device), bpms.to(device)
            outputs = model(features)
            
            # Calculate error considering doubling and halving
            error = torch.abs(outputs - bpms)
            error_double = torch.abs(outputs * 2 - bpms)
            error_half = torch.abs(outputs / 2 - bpms)
            
            min_error = torch.min(torch.min(error, error_double), error_half)
            total_error += min_error.mean().item()

    avg_error = total_error / len(eval_loader)
    print(f'Average BPM Error: {avg_error:.2f}')
    return avg_error

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set the default tensor type to CUDA if available
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(script_dir, 'train')
    eval_dir = os.path.join(script_dir, 'eval')

    train_dataset = AudioDataset(train_dir, device)
    eval_dataset = AudioDataset(eval_dir, device)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    # Use a CUDA-compatible generator for random_split
    generator = torch.Generator(device=device)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, 
        [train_size, val_size],
        generator=generator
    )

    # Use a CUDA-compatible sampler
    train_sampler = torch.utils.data.RandomSampler(train_dataset, generator=generator)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=32, sampler=val_sampler, num_workers=0, pin_memory=False)
    eval_loader = DataLoader(eval_dataset, batch_size=32, num_workers=0, pin_memory=False)

    # Get the input size from the dataset directly
    input_size = train_dataset[0][0].shape[0]  # Assuming the first element is representative
    model = BPMPredictor(input_size).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)


    num_epochs = 100
    max_retrain_attempts = 5
    accuracy_threshold = 5.0  # BPM error threshold

    for attempt in range(max_retrain_attempts):
        print(f"Training attempt {attempt + 1}")
        model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)
        avg_error = evaluate_model(model, eval_loader, device)
        
        if avg_error <= accuracy_threshold:
            print(f"Achieved desired accuracy after {attempt + 1} attempts.")
            break
        else:
            print(f"Accuracy not sufficient. Retraining (attempt {attempt + 2})...")

    torch.save(model.state_dict(), 'final_bpm_predictor_model.pth')
    print("Model training completed and saved.")