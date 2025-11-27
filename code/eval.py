## libraries
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from code.model_train import LittleOwlDataset, CountryClassifier, MIN_RECORDINGS_PER_COUNTRY

## config
SPECTROGRAM_DIR = './spectrograms'
METADATA_FILE = './data/metadata.csv'
MODEL_PATH = './little_owl_cnn.pth'
CLASSES_PATH = './classes.npy'
BATCH_SIZE = 16
DEVICE = torch.device('mps' if torch.cuda.is_available() else 'cpu')

def main():
    # load metadata and classes
    df = pd.read_csv(METADATA_FILE)
    classes = np.load(CLASSES_PATH, allow_pickle = True)
    print(f"Classes found: {classes}")
    # filtering data exactly like training 
    country_counts = df['cnt'].value_counts()
    valid_countries = country_counts[country_counts >= MIN_RECORDINGS_PER_COUNTRY].index
    df = df[df['cnt'].isin(valid_countries)].copy()
    # encoding labels (countries)
    le = LabelEncoder()
    le.fit(classes)
    df['country_encoded'] = le.transform(df['cnt'])
    # prepping data loader
    data_transform = transforms.Compose([
        transforms.Resize((128, 424)),
        transforms.ToTensor(),
    ])
    dataset = LittleOwlDataset(df, SPECTROGRAM_DIR, transform = data_transform)
    loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = False)
    # loading model
    model = CountryClassifier(num_classes = len(classes)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location = DEVICE))
    model.eval()
    print("Model loaded successfully.")
    # running preds
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    # generating confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', 
                xticklabels = classes, yticklabels = classes)
    plt.xlabel('predicted country')
    plt.ylabel('actual country')
    plt.title('Little Owl Dialect Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")
    # printing report
    print("\nclassification report:")
    print(classification_report(all_labels, all_preds, target_names = classes))

if __name__ == "__main__":
    main()