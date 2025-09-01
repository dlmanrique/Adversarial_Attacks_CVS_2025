# This is the main.py file for this inference
print('Importing libraries...')
# Standard library imports
import argparse
import os
import random
import json

# Third-party imports
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import torch.nn.functional as F

# Local imports
from scripts.f_environment import get_config
from scripts.f_dataset import get_datasets, get_dataloaders
from scripts.f_build import build_model
from pgd_adjusted import PGD_BCE
from utils import save_adv_example

warnings.filterwarnings("ignore")


# Load configuration
parser = argparse.ArgumentParser(description="Run SwinCVS with specified config")
parser.add_argument('--config_path', type=str, required=False, default='config/SwinCVS_config.yaml' , help='Path to config YAML file')
parser.add_argument('--ckpt_path',  type=str, required=False, default='weights/Sages_Fold2_bestMAP.pt')
parser.add_argument('--adversarial_split', type=str, default='All', choices=['CVS_only', 'One_only', 'All'])
args = parser.parse_args()

config = get_config(args.config_path)

seed = config.SEED
# Environment Standardisation
random.seed(seed)                      # Set random seed
np.random.seed(seed)                   # Set NumPy seed
torch.manual_seed(seed)                # Set PyTorch seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(seed)           # Set CUDA seed
torch.use_deterministic_algorithms(True) # Force deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # CUDA workspace config


##############################################################################################
##############################################################################################
# DATASET and DATALOADER
train_dataset = get_datasets(config, args)
train_dataloader = get_dataloaders(config, train_dataset)

##############################################################################################
##############################################################################################

# Initialise SwinCVS according to config
model = None
model = build_model(config)
model.head = nn.Linear(in_features=1024, out_features=3, bias=True)
print('Full model initialised successfully!\n')

# Load saved weights for inference
model.load_state_dict(torch.load(args.ckpt_path, weights_only=True))
print(f"Trained SwinCVS weights loaded successfully for INFERENCE - name: {args.ckpt_path}")
model.to('cuda')
torch.cuda.empty_cache()

criterion = nn.BCEWithLogitsLoss()

model.eval()

atk = PGD_BCE(model, eps=6/255, alpha=2/255, steps=8)
n_iters = 100
os.makedirs(f'visualizations/Fold{config.FOLD}', exist_ok=True)
save_path = '/media/lambda001/SSD3/leoshared/Dataset/frames_adv'
os.makedirs(save_path, exist_ok=True)


all_img_info = []

for idx, (samples, targets, img_path) in enumerate(tqdm(train_dataloader)):
    
    #Get preds
    samples, targets = samples.to('cuda'), targets.to('cuda')
    adv_images = atk(samples, targets)

    # Forward normal
    outputs = model(samples)
    # Forward adversario
    adv_outputs = model(adv_images)

    preds = torch.sigmoid(outputs).round()
    adv_preds = torch.sigmoid(adv_outputs).round()

    # Save adversarial example
    formated_img_name = save_adv_example(adv_images, img_path, save_path)
    # Add info for json file
    all_img_info.append({'file_name': formated_img_name, 'ds': targets.cpu().tolist()})


    # Guardar visualización cada n_iters
    if idx % n_iters == 0:
        batch_size = min(4, samples.size(0))  # mostrar hasta 4 imágenes
        fig, axes = plt.subplots(batch_size, 2, figsize=(6, 3*batch_size), constrained_layout=True)
        if batch_size == 1:
            axes = [axes]  # mantener indexable
            
        mse_per_image = F.mse_loss(adv_images, samples, reduction='none')
        mse_per_image = mse_per_image.view(mse_per_image.size(0), -1).mean(dim=1)

        for i in range(batch_size):
            # Imagen original
            axes[i][0].imshow(samples[i].permute(1, 2, 0).detach().cpu().numpy())
            axes[i][0].axis("off")
            axes[i][0].set_title(
                f"GT: {targets[i].cpu().detach().numpy()}\n Ori Pred: {preds[i].cpu().detach().numpy()}", 
                fontsize=9
            )

            # Imagen adversaria
            axes[i][1].imshow(adv_images[i].permute(1, 2, 0).detach().cpu().numpy())
            axes[i][1].axis("off")
            axes[i][1].set_title(
                f"GT: {targets[i].cpu().detach().numpy()}\nAdv Pred: {adv_preds[i].cpu().detach().numpy()}\nMSE: {mse_per_image[i].item():.6f}",
                fontsize=9
            )

        plt.tight_layout()
        plt.savefig(f"visualizations/Fold{config.FOLD}/adv_examples_batch{idx}.png")
        plt.close()

os.makedirs('adversarial_training_files', exist_ok=True)

with open(f"adversarial_training_files/fold{config.FOLD}_train.json", "w") as f:
    json.dump({'images': all_img_info}, f, indent=4)
