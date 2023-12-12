from utils.dataset import CustomDataset
from utils.utils import *
from torch.utils.data import Dataset, DataLoader
import os

batch_size = 16
num_workers = os.cpu_count()

train_dir = "dataset/training_set/training_set"
train_transform = create_train_transform()
train_dataset = CustomDataset(train_dir, train_transform)
train_loader = DataLoader(train_dataset, shuffle = True, num_workers = num_workers, batch_size = batch_size, drop_last = True)

test_dir = 'dataset/test_set/test_set'
test_transform = create_test_transform()
test_dataset = CustomDataset(root_dir = test_dir, transform = test_transform)
test_loader = DataLoader(test_dataset, num_workers = num_workers, batch_size = batch_size)

import random

idx = random.randint(0, 100)

img, label = test_dataset.__getitem__(idx)

# img