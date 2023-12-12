import torch

def evaluate(model, val_data, loss_fn, device = 'cpu'):
    # set device
    if (device == 'gpu' or device == 'cuda') and torch.cuda.is_available():
        device = torch.device('cuda')
    elif isinstance(device, torch.device): 
        device = device
    else: 
        device = torch.device('cpu')
    
    model = model.to(device)

    with torch.no_grad():
        model.eval()
        val_correct = 0
        val_total = len(val_data)*val_data.batch_size
        running_loss = 0.
        for data_batch, label_batch in val_data:
            data_batch, label_batch = data_batch.to(device), label_batch.to(device)

            # tính toán đầu ra cho bộ valid
            output_batch = model(data_batch)

            loss = loss_fn(output_batch, label_batch)
            running_loss += loss.item()

            # dự đoán đầu ra với softmax
            _, predicted_labels = torch.max(output_batch.data, dim = 1)

            # so sánh nhãn dự đoán với nhãn thật (ground-truth label)
            val_correct += (label_batch == predicted_labels).sum().item()
        val_loss = running_loss/len(val_data)
        val_acc = val_correct/val_total
        return val_loss, val_acc
