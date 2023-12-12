from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from utils.utils import *
from eval import evaluate
from torch.optim import Adam 
from torch import nn  
from utils.dataset import CustomDataset
import os
from models import EfficientNet

def train(model, train_data: DataLoader, loss_fn, optimizer, epochs, weights = None, save_last_weights_path = None,
          save_best_weights_path = None, steps_per_epoch = None,
          device = 'cpu', validation_data = None, validation_split = None, scheduler = None):
    
    # đặt validation_data and validation_split không đồng thời khác None
    assert not(validation_data is not None and validation_split is not None)
    
    # nếu truyền vào model trọng số có sẵn, thì nó sẽ lưu trọng số lại 
    if weights:
        model.load_state_dict(torch.load(weights))
        print('Weights loaded successfully from path:', weights)
        print('====================================================')
    
    # set device
    if (device == 'gpu' or device == 'cuda') and torch.cuda.is_available():
        device = torch.device('cuda')
    elif isinstance(device, torch.device): 
        device = device
    else: 
        device = torch.device('cpu')
        
    # chia dữ liệu thành 2 tập train và val    
    if validation_data is not None:
        val_data = validation_data
    elif validation_split is not None: 
        train_data, val_data = split_dataloader(train_data, validation_split)
    else: 
        val_data = None
        
    # save best model
    if save_best_weights_path: 
        if val_data is None:
            train_data, val_data = split_dataloader(train_data, 0.2)
        best_loss, _ = evaluate(model, val_data, device = device, loss_fn = loss_fn)  
        
    # đặt số lần update weights trong 1 epoch
    if steps_per_epoch is None: 
        steps_per_epoch = len(train_data)

    num_steps = len(train_data)
    iterator = iter(train_data)
    count_steps = 1    
    
    ## history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_loss': []
    }
        
    # add model to device
    model = model.to(device)
    
    ############################### Train and Val ##########################################
    for epoch in range(1, epochs + 1):
        # tính tổng giá trị hàm mất mát cho mỗi epoch
        running_loss = 0.
        train_correct = 0
        train_total = steps_per_epoch*train_data.batch_size
        
        # đặt model ở chế độ huấn luyện 
        model.train()
        
        for step in tqdm(range(steps_per_epoch), desc = f'epoch: {epoch}/{epochs}: ', ncols = 100): 
            img_batch, label_batch = next(iterator)
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            
            # Xóa các gradient
            optimizer.zero_grad()
            
            # tính toán đầu ra
            output_batch = model(img_batch)
            
            # tính loss
            loss = loss_fn(output_batch, label_batch)
            
            # lan truyền ngược
            loss.backward()
            
            # cập nhật trọng số cho mạng
            optimizer.step()
            
            # dự đoán đầu ra với softmax
            _, predicted_labels = torch.max(output_batch.data, dim = 1)

            # so sánh nhãn dự đoán với nhãn thật (ground-truth label)

            train_correct += (label_batch == predicted_labels).sum().item()
                
            # Cập nhật tổng hàm mất mát
            running_loss += loss.item()
                
            if count_steps == num_steps:
                count_steps = 0
                iterator = iter(train_data)
            count_steps += 1
            
        train_loss = running_loss / steps_per_epoch
        train_accuracy = train_correct/train_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_accuracy)
        
        if val_data is not None: 
            val_loss, val_acc = evaluate(model, val_data, device = device, loss_fn = loss_fn)
            print(f'epoch: {epoch}, train_accuracy: {train_accuracy: .2f}, loss: {train_loss: .3f}, val_accuracy: {val_acc: .2f}, val_loss: {val_loss:.3f}')

            if save_best_weights_path:
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), save_best_weights_path)
                    print(f'Saved successfully best weights to:', save_best_weights_path)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
        else:
            print(f'epoch: {epoch}, train_accuracy: {train_accuracy: .2f}, loss: {train_loss: .3f}')
    if save_last_weights_path:  
        torch.save(model.state_dict(), save_last_weights_path)
        print(f'Saved successfully last weights to:', save_last_weights_path)
    return model, history


if __name__ == '__main__':
    batch_size = 64
    num_workers = os.cpu_count()
    
    train_dir = "dataset/training_set/training_set"
    train_transform = create_train_transform()
    train_dataset = CustomDataset(train_dir, train_transform)
    train_loader = DataLoader(train_dataset, shuffle = True, num_workers = num_workers, batch_size = batch_size, drop_last = True)
    
    test_dir = 'dataset/test_set/test_set'
    test_transform = create_test_transform()
    test_dataset = CustomDataset(test_dir, test_transform)
    test_loader = DataLoader(test_dataset, num_workers = num_workers, batch_size = batch_size, drop_last = True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 2
    epochs = 100
    model = EfficientNet(num_classes = num_classes, in_channels = 3)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr = 0.00125)
    
    weights_path = 'weights/efficient_weights.pt'
    
    model, history = train(model, train_loader, loss_fn, optimizer,
                        epochs = epochs, save_last_weights_path = weights_path, steps_per_epoch = 100, device = device, validation_data = test_loader)
     
    visual_results(history, save_path = "history.png")