from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt

def create_train_transform():
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p = 0.5),
            T.ToTensor(),
            T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ]
    )
    
    return transform

def create_test_transform():
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ]
    )
    return transform

def split_dataloader(train_data, validation_split):
    # Chia DataLoader thành phần train và test
    train_ratio = 1 - validation_split  # Tỷ lệ phần train (80%)
    train_size = int(train_ratio * len(train_data.dataset))  # Số lượng mẫu dùng cho train

    indices = list(range(len(train_data.dataset)))  # Danh sách các chỉ số của dataset
    train_indices = indices[:train_size]  # Chỉ số của mẫu dùng cho train
    val_indices = indices[train_size:]  # Chỉ số của mẫu dùng cho test

    # lấy dữ liệu từ dataloader
    dataset = train_data.dataset
    batch_size = train_data.batch_size
    num_workers = train_data.num_workers
    
    # Tạo ra các SubsetRandomSampler để chọn một phần dữ liệu cho train và test
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Tạo DataLoader mới từ SubsetRandomSampler
    train_data = DataLoader(dataset, batch_size = batch_size, sampler = train_sampler, num_workers = num_workers, drop_last = True)
    val_data = DataLoader(dataset, batch_size = batch_size, sampler = val_sampler, num_workers = num_workers, drop_last = True)
    
    return train_data, val_data

def visual_results(history, save_path):
    plt.figure(figsize = (10,5))
    plt.subplot(1,2,1)
    plt.plot(range(1, len(history['train_acc']) + 1), history['train_acc'], c = 'r', label = 'train_acc', marker = '.')
    plt.plot(range(1, len(history['val_acc']) + 1), history['val_acc'], c = 'orange', label = 'val_acc', marker = '.')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy plot')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], c = 'r', label = 'train_loss', marker = '.')
    plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], c = 'orange', label = 'val_loss', marker = '.')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss plot')
    plt.legend()
    
    plt.savefig(save_path)