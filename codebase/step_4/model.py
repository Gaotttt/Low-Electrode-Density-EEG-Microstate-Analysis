import os
import mne
import pandas as pd
import numpy as np
from mne.io import read_raw_eeglab
from pycrostates.cluster import ModKMeans
from pycrostates.io import ChData
import logging
import torch
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from pycrostates.preprocessing import extract_gfp_peaks
from fastapi import FastAPI, Request
import uvicorn
import pickle
from scipy.interpolate import interp1d

global_channels = None

def process_eeg_files(data_folder, window_size, overlap, labels, channels_to_drop):
    global global_channels
    files = [f for f in os.listdir(data_folder) if f.endswith('.set')]
    all_samples = []
    all_labels = []
    
    if files:
        first_file = os.path.join(data_folder, files[0])
        raw = mne.io.read_raw_eeglab(first_file, preload=True)
        raw.pick("eeg")
        existing_unwanted = [ch for ch in channels_to_drop if ch in raw.ch_names]

        if existing_unwanted:
            raw.drop_channels(existing_unwanted)
            print(f"Dropped channels: {existing_unwanted}")

        global_channels = raw.ch_names
        print(f"Found channels: {global_channels}")
    
    for idx, file in enumerate(files):
        file_path = os.path.join(data_folder, file)
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
        raw.pick("eeg")
        raw.drop_channels(channels_to_drop)
        data = raw.get_data()

        # Get the corresponding label using the filename
        if file in labels:
            label = labels[file]
        else:
            print(f"Warning: No label found for file {file}")
            continue

        n_timepoints = data.shape[1]
        if n_timepoints != len(label):
            print(f'Warning: Timepoints mismatch for {file}: {n_timepoints} != {len(label)}')
            continue

        step_size = int(window_size * (1 - overlap))
        for start in range(0, n_timepoints - window_size + 1, step_size):
            end = start + window_size
            sample = data[:, start:end]
            sample_label = label[start:end]
            all_samples.append(sample)
            all_labels.append(sample_label)
    return np.array(all_samples), np.array(all_labels)


class EEGDataset(Dataset):
    def __init__(self, eeg_data, labels, transform=None):
        self.eeg_data = eeg_data
        self.labels = labels
        self.transform = transform
        
        self.scaler = StandardScaler()
        self.eeg_data = self.scaler.fit_transform(self.eeg_data.reshape(-1, self.eeg_data.shape[-1])).reshape(self.eeg_data.shape)
        
        self.labels = np.where(self.labels == -1, 4, self.labels)
    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        sample = self.eeg_data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class EEGTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=8, num_layers=4):
        super(EEGTransformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_projection(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        x = self.classifier(x)
        return x


class InputGradCAM_new:
    def __init__(self, model, channel_names):
        self.model = model
        self.channel_names = channel_names
        self.gradients = None

    def save_gradients(self, grad):
        self.gradients = grad

    def generate_cam(self, inputs, target_class):
        inputs.requires_grad = True
        outputs = self.model(inputs)
        logits = outputs[:, :, target_class].sum()

        self.model.zero_grad()
        logits.backward()

        gradients = inputs.grad
        cam = gradients[0].mean(dim=1)
        return cam

    def save_cam(self, channel_names, cam, target_class, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        grad_filename = os.path.join(output_dir, f'class_{target_class}_grad.txt')
        cam = cam.cpu().numpy()
        with open(grad_filename, 'w') as f:
            # Iterate over the channels
            for i, channel_name in enumerate(channel_names):
                # Write each line as "channel_name + cam[i]"
                f.write(f'{channel_name} {cam[i]:.6f}\n')  # Adjust formatting if needed

        image_filename = os.path.join(output_dir, f'class_{target_class}.png')

        colors = ['blue' if value >=0 else 'red' for value in cam]

        plt.figure(figsize=(20, 5))
        plt.bar(self.channel_names, cam, color=colors)
        plt.xticks(rotation=45, fontsize=10)
        plt.title(f"Accumulated Grad-CAM for Class {target_class}")
        plt.xlabel('EEG Channels (58)')
        plt.ylabel('Accumulated Gradient Contribution')
        plt.savefig(image_filename)
        plt.close()




def run_app():
    app = FastAPI()

    @app.post("/")
    async def get_answer(request: Request):
        global channels_to_drop, global_channels
        request_dict = await request.json()

        raw_data = request_dict.get("raw_data")
        label_file = request_dict.get("label_file")
        output_dir = request_dict.get("output_dir")
        sampling_rate = request_dict.get("sampling_rate")
        overlap = request_dict.get("overlap")
        batch_size = request_dict.get("batch_size")
        total_epoch = request_dict.get("total_epoch")
        input_channel_num = request_dict.get("input_channel")

        # check raw_data
        if not raw_data:
            print("Warning: No input paths provided. Using the default path.")
            raw_data = "/home/medicine/test_data"

        if not label_file:
            print("Warning: No EEG Microstate labels provided. Using the default lables.")
            label_file = "/home/medicine/test_labels/segmentation_labels.pkl"

        # check output_dir
        if not output_dir:
            print("Warning: No output paths provided. Using the default path.")
            output_dir = "/home/medicine/output"

        if not sampling_rate:
            print("Warning: No sampling_rate provided. Using the default value.")
            sampling_rate = 500

        if not overlap:
            print("Warning: No overlap provided. Using the default value.")
            overlap = 0

        if not batch_size:
            print("Warning: No batch_size provided. Using the default value.")
            batch_size = 64

        if not total_epoch:
            print("Warning: No total_epoch provided. Using the default value.")
            total_epoch = 200

        if not input_channel_num:
            print("Warning: No input_channel_num provided. Using the default value.")
            input_channel_num = 60

        window_size = 4 * sampling_rate

        with open(label_file, 'rb') as f:
            labels = pickle.load(f)

        eeg_samples, eeg_labels = process_eeg_files(raw_data, window_size, overlap, labels, channels_to_drop)

        if global_channels is None:
            return "Error: No channels found in the input data."

        if len(global_channels) != input_channel_num:
            return f"Error: Channel mismatch. Expected {input_channel_num} channels, got {len(global_channels)}."

        dataset = EEGDataset(eeg_samples, eeg_labels)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        test_size = int(1.0 * len(dataset))
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = EEGTransformer(input_dim=input_channel_num, num_classes=5).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch)

        early_stopping_patience = 10
        best_epoch = 0
        writer = SummaryWriter(log_dir=os.path.join(output_dir, 'runs'))

        best_val_loss = float('inf')
        best_model_path = os.path.join(output_dir, 'best_model.pth')

        def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
            nonlocal best_val_loss, best_epoch
            best_model_wts = None
            early_stopping_counter = 0
            model.train()
            print(f"************** Start training ****************")

            os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

            for epoch in range(num_epochs):
                running_loss = 0.0
                for i, (inputs, labels) in enumerate(train_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)
                print(f'Epoch {epoch}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

                # Save the model checkpoint at every epoch
                checkpoint_path = os.path.join(output_dir, 'checkpoints', f'model_epoch_{epoch}.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at: {checkpoint_path}")

                # 验证模型并保存最佳模型
                val_loss = validate_model(model, val_loader, criterion, epoch)

                # 学习率调度
                scheduler.step()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_wts = model.state_dict()
                    best_epoch = epoch
                    early_stopping_counter = 0

                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

                print(f"best epoch: {best_epoch}")
                torch.save(best_model_wts, best_model_path)


        def validate_model(model, val_loader, criterion, epoch):
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            all_preds = []
            all_labels = []
            all_probs = []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)

                    loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 2)
                    total += labels.numel()
                    correct += (predicted == labels).sum().item()

                    all_preds.append(predicted.cpu().numpy().reshape(-1))
                    all_labels.append(labels.cpu().numpy().reshape(-1))

                    probs = torch.softmax(outputs, dim=-1)
                    all_probs.append(probs.cpu().numpy().reshape(-1, probs.shape[-1]))

            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            all_probs = np.concatenate(all_probs)

            accuracy = 100 * correct / total

            precision = precision_score(all_labels, all_preds, average='macro')
            recall = recall_score(all_labels, all_preds, average='macro')
            f1 = f1_score(all_labels, all_preds, average='macro')

            try:
                auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
            except ValueError:
                auc = float('nan')

            writer.add_scalar('Loss/val', val_loss / len(val_loader), epoch)
            writer.add_scalar('Accuracy/val', accuracy, epoch)
            writer.add_scalar('Precision/val', precision, epoch)
            writer.add_scalar('Recall/val', recall, epoch)
            writer.add_scalar('F1/val', f1, epoch)
            writer.add_scalar('AUC/val', auc, epoch)

            print(
                f'Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {accuracy:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}')
            return val_loss / len(val_loader)

        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=total_epoch)
        print("train finish")

        model.load_state_dict(torch.load(best_model_path))
        model.to(device)
        print("Best model loaded for Grad-CAM visualization")

        input_grad_cam = InputGradCAM_new(model, global_channels)
        grad_cam_results = os.path.join(output_dir, 'grad_cam_results')
        os.makedirs(grad_cam_results, exist_ok=True)
        accumulated_grad_cam = torch.zeros(5, input_channel_num).to(device)
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            for sample_idx in range(inputs.shape[0]):
                for target_class in range(5):
                    cam = input_grad_cam.generate_cam(inputs[sample_idx].unsqueeze(0), target_class)
                    accumulated_grad_cam[target_class] += cam

        for target_class in range(5):
            print(f"Saving accumulated Grad-CAM for class {target_class}...")
            input_grad_cam.save_cam(global_channels, accumulated_grad_cam[target_class], target_class, grad_cam_results)


        return {"message": "Successfully.",
                "log_path": os.path.join(output_dir, 'runs'),
                "checkpoint_path": os.path.join(output_dir, 'checkpoints'),
                "results_path": os.path.join(output_dir, 'grad_cam_results')}

    uvicorn.run(app, host='0.0.0.0', port=8080, workers=1)


if __name__ == '__main__':
    # check CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    # Load model
    channels_to_drop = ['TP9', 'TP10', 'HEOG', 'VEOG']

    # Start FastAPI server
    run_app()
