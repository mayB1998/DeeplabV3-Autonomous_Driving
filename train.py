import argparse
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import segmentation_models_pytorch as smp

from utils import meanIoU, plot_training_results, evaluate_model, train_validate_model, get_datasets, get_dataloaders
from model import deeplabv3_plus




def train_deeplabv3_plus(images, labels, output_path):
    images = np.load(os.path.join(images_path, 'inputs.npy'))
    labels = np.load(os.path.join(labels_path, 'labels.npy'))

    # Get datasets and dataloaders
    train_set, val_set, test_set = get_datasets(images, labels)
    sample_image, sample_label = train_set[0]
    print(f"There are {len(train_set)} train images, {len(val_set)} validation images, {len(test_set)} test Images")
    print(f"Input shape = {sample_image.shape}, output label shape = {sample_label.shape}")

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(train_set, val_set, test_set)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model hyperparameters
    N_EPOCHS = 5
    NUM_CLASSES = 3
    MAX_LR = 3e-4
    MODEL_NAME = 'deeplabv3_plus_resnet50_os'

    criterion = smp.losses.DiceLoss('multiclass', classes=[0, 1, 2], log_loss=True, smooth=1.0)

    # Create model, optimizer, lr_scheduler and pass to training function
    model = deeplabv3_plus(in_channels=3, output_stride=8, num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=MAX_LR)
    scheduler = OneCycleLR(optimizer, max_lr=MAX_LR, epochs=N_EPOCHS, steps_per_epoch=len(train_dataloader),
                           pct_start=0.3, div_factor=10, anneal_strategy='cos')

    _ = train_validate_model(model, N_EPOCHS, MODEL_NAME, criterion, optimizer,
                             device, train_dataloader, val_dataloader, meanIoU, 'meanIoU',
                             NUM_CLASSES, lr_scheduler=scheduler, output_path=output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DeeplabV3 Plus for image semantic segmentation.')

    parser.add_argument('--images', type=str, help='Path to the images directory', default='dataset/')
    parser.add_argument('--labels', type=str, help='Path to the labels directory', default='dataset/')
    parser.add_argument('--output_path', type=str, help='Path to save the trained model', default='output/')

    argv = parser.parse_args()

    images_path = argv.images
    labels_path = argv.labels
    output_path = argv.output_path

    train_deeplabv3_plus(images_path, labels_path, output_path)
