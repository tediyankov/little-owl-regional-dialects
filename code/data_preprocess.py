## libraries 
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# config
INPUT_DIR = './data/Athene_noctua'
OUTPUT_DIR = './spectrograms'
SAMPLE_RATE = 22050 
N_FFT = 2048 # window size for FFT
HOP_LENGTH = 512 # stride
N_MELS = 128 # num of Mel bands (height of the image)
DURATION = 5 # duration in seconds to process (clip or pad)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_spectrogram(file_path, output_path):
    try:
        # loading audio
        y, sr = librosa.load(file_path, sr = SAMPLE_RATE, duration = DURATION)

        # padding audio if its shorter than desired duration
        target_length = int(DURATION * SAMPLE_RATE)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode = 'constant')

        # generating Mel spectrogram
        S = librosa.feature.melspectrogram (y = y, sr = sr, n_fft = N_FFT, hop_length = HOP_LENGTH, n_mels = N_MELS)
        
        # converting to log scale (dB)
        S_dB = librosa.power_to_db(S, ref = np.max)

        # plotting and saving the spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr = sr, hop_length = HOP_LENGTH, x_axis = 'time', y_axis = 'mel')
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Processed: {os.path.basename(file_path)}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    ensure_dir(OUTPUT_DIR)
    
    # walking through the data directory
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.mp3', '.wav'))]
    
    print(f"Found {len(files)} audio files. Starting conversion...")

    for filename in files:
        input_path = os.path.join(INPUT_DIR, filename)
        
        # creating output filename
        file_stem = Path(filename).stem
        output_path = os.path.join(OUTPUT_DIR, f"{file_stem}.png")
        
        # skipping if already exists (optional)
        if not os.path.exists(output_path):
            save_spectrogram(input_path, output_path)
        else:
            print(f"Skipping {filename}, already exists.")

if __name__ == "__main__":
    main()