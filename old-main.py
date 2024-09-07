import streamlit as st
import os
import librosa
import numpy as np
import pywt
from scipy import signal
import wave
import array
import math
import tempfile
import boto3
import uuid
import torch
from torch import nn
import torchaudio
import torchaudio.transforms as T

from dotenv import load_dotenv
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
REGION_NAME = os.getenv('REGION_NAME')
BUCKET_NAME = os.getenv('BUCKET_NAME')
s3 = boto3.client('s3')


def upload_to_s3(file_content, file_name):
    bucket_name = BUCKET_NAME  
    folder_name = str(uuid.uuid4())  # Generate a random folder name
    s3_path = f"{folder_name}/{file_name}"
    
    try:
        s3.put_object(Bucket=bucket_name, Key=s3_path, Body=file_content)
        return f"s3://{bucket_name}/{s3_path}"
    except Exception as e:
        st.error(f"Error uploading audio file: {e}")
        return None

def download_from_s3(s3_uri):
    s3 = boto3.client('s3')
    bucket_name = s3_uri.split('/')[2]
    s3_path = '/'.join(s3_uri.split('/')[3:])
    
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            s3.download_fileobj(bucket_name, s3_path, tmp_file)
            return tmp_file.name
    except Exception as e:
        st.error(f"Error downloading from S3: {e}")
        return None

def peak_detect(data):
    max_val = np.amax(abs(data))
    peak_ndx = np.where(data == max_val)
    if len(peak_ndx[0]) == 0:  # if nothing found then the max must be negative
        peak_ndx = np.where(data == -max_val)
    return peak_ndx[0]

def wavelet_bpm_detector(audio_file_path):
    try:
        samps, fs = read_wav(audio_file_path)
        data = samps
        
        cA = []
        cD = []
        correl = []
        cD_sum = []
        levels = 4
        max_decimation = 2 ** (levels - 1)
        min_ndx = math.floor(60.0 / 220 * (fs / max_decimation))
        max_ndx = math.floor(60.0 / 40 * (fs / max_decimation))

        for loop in range(0, levels):
            cD = []
            if loop == 0:
                [cA, cD] = pywt.dwt(data, "db4")
                cD_minlen = len(cD) // max_decimation + 1
                cD_sum = np.zeros(math.floor(cD_minlen))
            else:
                [cA, cD] = pywt.dwt(cA, "db4")

            cD = signal.lfilter([0.01], [1 - 0.99], cD)
            cD = abs(cD[:: (2 ** (levels - loop - 1))])
            cD = cD - np.mean(cD)
            cD_sum = cD[0 : math.floor(cD_minlen)] + cD_sum

        if [b for b in cA if b != 0.0] == []:
            return None

        cA = signal.lfilter([0.01], [1 - 0.99], cA)
        cA = abs(cA)
        cA = cA - np.mean(cA)
        cD_sum = cA[0 : math.floor(cD_minlen)] + cD_sum

        correl = np.correlate(cD_sum, cD_sum, "full")

        midpoint = len(correl) // 2
        correl_midpoint_tmp = correl[midpoint:]
        peak_ndx = peak_detect(correl_midpoint_tmp[min_ndx:max_ndx])
        if len(peak_ndx) > 1:
            return None

        peak_ndx_adjusted = peak_ndx[0] + min_ndx
        bpm = 60.0 / peak_ndx_adjusted * (fs / max_decimation)
        return int(round(bpm))
    except Exception as e:
        print(f"Error in wavelet BPM detection: {e}")
        return None

def read_wav(filename):
    try:
        y, sr = librosa.load(filename, sr=None)
        return y, sr
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return None, None

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

def extract_audio_features(file_path, device):
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
    
    return torch.FloatTensor(features).to(device)

def detect_bpm(audio_file):
    try:
        # Existing methods
        y, sr = librosa.load(audio_file)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        tempo_from_tempogram = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0]
        
        # New wavelet-based method
        wavelet_bpm = wavelet_bpm_detector(audio_file)
        
        # Load the trained model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_size = 13 + 64 + 12  # 13 (MFCC) + 64 (Mel) + 12 (Chroma)
        model = BPMPredictor(input_size)
        model_path = os.path.join('model-train', 'best_model.pth')
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        
        # Extract features and predict BPM
        features = extract_audio_features(audio_file, device)
        with torch.no_grad():
            model_bpm = model(features.unsqueeze(0)).item()
        
        # Combine all methods
        all_tempos = [tempo, tempo_from_tempogram, model_bpm]
        if wavelet_bpm is not None:
            all_tempos.append(wavelet_bpm)
        
        # Check for half or double tempo
        potential_tempos = []
        for t in all_tempos:
            if isinstance(t, np.ndarray):
                t = t.item()
            potential_tempos.extend([t/2, t, t*2])
        potential_tempos = [t for t in potential_tempos if 60 <= t <= 200]
        
        if not potential_tempos:
            return int(round(np.mean(all_tempos)))
        
        # Choose the tempo that best matches the onset strength
        best_tempo = max(potential_tempos, key=lambda t: onset_strength_at_tempo(onset_env, sr, t))
        
        return int(round(best_tempo))
    except Exception as e:
        st.error(f"Error detecting BPM: {str(e)}")
        return None

def verify_and_adjust_bpm(audio, detected_bpm):
    duration_ms = len(audio)
    beat_duration_ms = (60 / detected_bpm) * 1000
    expected_beats = duration_ms / beat_duration_ms
    rounded_beats = round(expected_beats)
    
    if abs(expected_beats - rounded_beats) < 0.1:  # 10% tolerance
        return detected_bpm
    
    # If not matching, try adjusting BPM
    adjusted_bpm = (detected_bpm * rounded_beats) / expected_beats
    
    # Check if doubling or halving the BPM would be more accurate
    if abs(adjusted_bpm - detected_bpm * 2) < abs(adjusted_bpm - detected_bpm):
        return int(round(detected_bpm * 2))
    elif abs(adjusted_bpm - detected_bpm / 2) < abs(adjusted_bpm - detected_bpm):
        return int(round(detected_bpm / 2))
    
    return int(round(adjusted_bpm))

def onset_strength_at_tempo(onset_env, sr, tempo):
    # Helper function to calculate onset strength at a given tempo
    tempo_period = 60.0 / tempo
    hop_length = 512  # This should match the hop_length used in librosa.onset.onset_strength
    beats = librosa.util.fix_frames(np.arange(0, len(onset_env), tempo_period * sr / hop_length))
    return np.mean(onset_env[beats])

# Streamlit app
def main():
    st.title("Audio BPM Detector")
    st.write("Upload an audio file (MP3, FLAC, or WAV) to detect its BPM.")

    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "flac", "wav"])

    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(file_details)

        # Check file size
        if uploaded_file.size > 100 * 1024 * 1024:  # 100 MB
            st.error("File size exceeds the maximum limit of 100 MB.")
        else:
            # Upload to S3
            s3_uri = upload_to_s3(uploaded_file.getvalue(), uploaded_file.name)
            
            if s3_uri:
                st.info(f"File uploaded to cloud")
                st.info(f"Running detection on file...")
                
                # Download from S3
                tmp_file_path = download_from_s3(s3_uri)
                
                if tmp_file_path:
                    try:
                        # Detect BPM
                        detected_bpm = detect_bpm(tmp_file_path)

                        if detected_bpm:
                            st.success(f"Detected BPM: {detected_bpm}")

                        else:
                            st.warning("Unable to detect BPM. Please try a different audio file.")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                    finally:
                        # Clean up the temporary file
                        os.unlink(tmp_file_path)
                else:
                    st.error("Failed to download file from S3.")
            else:
                st.error("Failed to upload file to S3.")

if __name__ == "__main__":
    main()
