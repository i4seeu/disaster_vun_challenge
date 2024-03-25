# Importing all the necessary libraries
# Import Fastai libraries
from fastai.vision.all import *

# Import PyTorch
import torch

# Import other necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

#lets set the reproducibility of the code

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Set seed for reproducibility
seed = 42
set_seed(seed)

# Step 1: Read and parse train.csv
train_df = pd.read_csv('../data/Train.csv')
train_df = train_df.dropna(subset=['bbox', 'category_id'])

# Step 2: Preprocess the data
def parse_bbox(bbox_str):
    # Assuming bbox_str is in the format '[x, y, width, height]'
    bbox_str = str(bbox_str)
    bbox = eval(bbox_str)
    return tuple(map(float, bbox))

train_df['bbox'] = train_df['bbox'].apply(parse_bbox)

class ObjectDetectionDataLoader:
    def __init__(self, df, path, image_col, bbox_col, label_col):
        self.df = df
        self.path = path
        self.image_col = image_col
        self.bbox_col = bbox_col
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx]['image_id']
        image_path = self.path / f"{image_id}.tif"
        image = PILImage.create(image_path)
        bbox = eval(self.df.iloc[idx][self.bbox_col])  # Assuming bbox is a string representation of a list
        label = self.df.iloc[idx][self.label_col]
        return image, (bbox, label)

image_path = Path('../data/train/Images')

# Define column names for images, bounding boxes, and labels
image_col = 'image_id'
bbox_col = 'bbox'
label_col = 'category_id'

# Create dataset object
dataset = ObjectDetectionDataLoader(train_df, image_path, image_col, bbox_col, label_col)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)