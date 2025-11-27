## libraries
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

## config
SPECTROGRAM_DIR = './spectrograms'
METADATA_FILE = './data/metadata.csv'
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 10
MIN_RECORDINGS_PER_COUNTRY = 3 
DEVICE = torch.device('mps' if torch.cuda.is_available() else 'cpu')
# check device
print(f"Using device: {DEVICE}")

## custom dataset class
class LittleOwlDataset (Dataset):
    def __init__(self, dataframe, img_dir, transform = None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        original_filename = row['file-name']
        img_name = os.path.splitext(original_filename)[0] + '.png'
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            # if file missing fallback
            return self.__getitem__((idx + 1) % len(self))
        label = row['country_encoded']
        if self.transform:
            image = self.transform(image)
        return image, label

## model 
class CountryClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CountryClassifier, self).__init__()
        # block 1: conv -> relu -> pool (input size halved)
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2) 
        # block 2: conv -> relu -> pool (input size halved again)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2) 
        # block 3: conv -> relu -> pool (input size halved AGAIN)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) 
        # classification head (where we flatten the features)
        self.flatten = nn.Flatten()
        # approximating the flattened size after 3 poolings for 128 Mel bands and ~5s duration
        self.fc1 = nn.Linear(128 * 16 * 53, 512) 
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # pass through Block 1
        x = self.pool1(self.relu1(self.conv1(x)))
        # pass through Block 2
        x = self.pool2(self.relu2(self.conv2(x)))
        # pass through Block 3
        x = self.pool3(self.relu3(self.conv3(x)))
        # flatten and classify
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x
    
## execution 
def main(): 
    # prep data 
    print("Loading metadata...")
    try:
        df = pd.read_csv(METADATA_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find {METADATA_FILE}. Make sure xcapi saved it.")
        return
    # only keep countries with enough recordings
    country_counts = df['cnt'].value_counts()
    valid_countries = country_counts[country_counts >= MIN_RECORDINGS_PER_COUNTRY].index
    df = df[df['cnt'].isin(valid_countries)].copy()
    print(f"Training on {len(df)} recordings from {len(valid_countries)} countries: {list(valid_countries)}")
    # encode labels (ie countries)
    le = LabelEncoder()
    df['country_encoded'] = le.fit_transform(df['cnt'])
    # train-test split
    train_df, val_df = train_test_split(df, test_size = 0.2, stratify = df['country_encoded'], random_state = 13)
    # transforming data
    data_transform = transforms.Compose([
        transforms.Resize((128, 424)), 
        transforms.ToTensor(),
    ])
    train_dataset = LittleOwlDataset(train_df, SPECTROGRAM_DIR, transform = data_transform)
    val_dataset = LittleOwlDataset(val_df, SPECTROGRAM_DIR, transform = data_transform)
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False)
    # initialise model
    model = CountryClassifier(num_classes = len(valid_countries)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    # training loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            # zero gradients
            optimizer.zero_grad()
            # fwd pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # backwd pass
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Val Accuracy: {100 * correct / total:.2f}%")
    print("Training complete.")
    # saving the learned weights
    save_path = './little_owl_cnn.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    # saving also the label encoder classes so we know that 0=France, 1=Spain, etc.
    import numpy as np
    np.save('./classes.npy', le.classes_)
    print("Class labels saved to ./classes.npy")

if __name__ == "__main__":
    main()