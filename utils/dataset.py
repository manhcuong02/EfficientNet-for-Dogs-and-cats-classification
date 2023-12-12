from PIL import Image
from torchvision import transforms as T
from torch. utils.data import Dataset, DataLoader
import os

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.labels = os.listdir(self.root_dir)
        
        self.path_list = []
        self.label_list = []
        
        for label in self.labels:
            for filename in os.listdir(os.path.join(self.root_dir, label)): 
                file_path = os.path.join(self.root_dir, label, filename)
                if os.path.isfile(file_path) and (filename.lower().endswith('.jpg') or filename.lower().endswith('.png')):
                    self.path_list.append(file_path)
                    
                    self.label_list.append(int(label == "dogs"))
    def __len__(self):
        return len(self.label_list)
    
    def __getitem__(self, idx): 
        file_path = self.path_list[idx]
        label = self.label_list[idx]

        img = Image.open(file_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
            return img, label

        return img, label
        