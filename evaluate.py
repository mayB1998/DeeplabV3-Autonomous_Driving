import argparse
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from utils import meanIoU, evaluate_model, visualize_predictions, get_datasets, get_dataloaders, predict_video
from model import deeplabv3_plus

from collections import namedtuple

# Each label is a tuple with name, class id and color
Label = namedtuple("Label", ["name", "train_id", "color"])
drivables = [
    Label("direct", 0, (32, 146, 190)),        # red
    Label("alternative", 1, (119, 231, 124)),  # cyan
    Label("background", 2, (0, 0, 0)),        # black
]

train_id_to_color = [c.color for c in drivables if (c.train_id != -1 and c.train_id != 255)]
train_id_to_color = np.array(train_id_to_color)
criterion = smp.losses.DiceLoss('multiclass', classes=[0, 1, 2], log_loss=True, smooth=1.0)

def evaluate_and_visualize(output_path, MODEL_NAME, test_set, test_dataloader, criterion, meanIoU, NUM_CLASSES, device, num_test_samples, get_video=True):
    model = deeplabv3_plus(in_channels=3, output_stride=8, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(f'{output_path}/{MODEL_NAME}.pt', map_location=device))

    _, test_metric = evaluate_model(model, test_dataloader, criterion, meanIoU, NUM_CLASSES, device)
    print(f"\nModel has {test_metric} mean IoU in the test set")

    _, axes = plt.subplots(num_test_samples, 3, figsize=(3 * 6, num_test_samples * 4))
    visualize_predictions(model, test_set, axes, device, numTestSamples=num_test_samples, id_to_color=train_id_to_color)
    
    while get_video:
        print("Running")
        predict_video(model, MODEL_NAME, video_path, output_path, 1242, 375, "cuda", train_id_to_color)
        print("Video Saved in Output folder")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate and visualize predictions using DeeplabV3 Plus.')

    parser.add_argument('--video', type=str, help='Path where the demo video is saved', default='dataset/stuttgart_1024_512_360_640.avi')
    parser.add_argument('--output_path', type=str, help='Path where the trained model is saved', default='output/')
    parser.add_argument('--MODEL_NAME', type=str, help='Name of the trained model', default='deeplabv3_plus_resnet50_os')
    parser.add_argument('--get_video', action='store_true', help='Gets us a video of the segmented area')
    parser.add_argument('--images', type=str, help='Path to the images directory', default='dataset/')
    parser.add_argument('--labels', type=str, help='Path to the labels directory', default='dataset/')

    argv = parser.parse_args()

    images_path = argv.images
    labels_path = argv.labels
    output_path = argv.output_path
    MODEL_NAME = argv.MODEL_NAME
    get_video = argv.get_video
    video_path = argv.video

    # Other parameters (You might need to set these accordingly)
    images = np.load(os.path.join(images_path, 'inputs.npy'))
    labels = np.load(os.path.join(labels_path, 'labels.npy'))
    NUM_CLASSES = 3
    num_test_samples = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get datasets and dataloaders
    test_set, _, _ = get_datasets(images, labels)
    _, _, test_dataloader = get_dataloaders(_, _, test_set)

    evaluate_and_visualize(output_path, MODEL_NAME, test_set, test_dataloader, criterion, meanIoU, NUM_CLASSES, device, num_test_samples, get_video)
