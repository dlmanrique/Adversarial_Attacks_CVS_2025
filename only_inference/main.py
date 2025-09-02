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
import wandb
from datetime import datetime
import time
import torch.nn.functional as F
from autoattack import AutoAttack

# Local imports
from scripts.f_environment import get_config
from scripts.f_dataset import get_datasets, get_dataloaders
from scripts.f_build import build_model
from scripts.f_metrics import get_map
from pgd_adjusted import PGD_BCE
from scripts.f_training_utils import build_optimizer, update_params, NativeScalerWithGradNormCount, save_weights
from utils import save_adv_example

warnings.filterwarnings("ignore")

torch.set_num_threads(1)

# Load configuration
parser = argparse.ArgumentParser(description="Run SwinCVS with specified config")
parser.add_argument('--config_path', type=str, required=False, default='config/SwinCVS_config.yaml' , help='Path to config YAML file')
parser.add_argument('--adv_attack', type=str, default='No', choices=['PGD', 'APGD', 'No'])
args = parser.parse_args()

config, experiment_name = get_config(args)
# Wanndb configuration ---------------------------------

# La forma de diferenciar entre solo el SwinV2 y el SWINCVS completo es si MODEL.LSTM = False
# convertir a dict
args_dict = vars(args)           # argumentos CLI
config_dict = vars(config) if not isinstance(config, dict) else config  # YAML config

# unir ambos (args tiene prioridad si hay colisión de claves)
wandb_config = {**config_dict, **args_dict}

# nombre experimento
exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

wandb.init(
    project='SwinCVS_adversarial', 
    entity='endovis_bcv',
    config=wandb_config,
    name=exp_name + '_none'
)

# (Opcional) imprimir para verificar
print("Config final usada:")
print(config)

# Create folder for saving outputs
os.makedirs(os.path.join(config.SAVING_DATASET, experiment_name, exp_name), exist_ok=True)
complete_exp_info_folder = os.path.join(config.SAVING_DATASET, experiment_name, exp_name)
                                        

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
training_dataset, val_dataset, test_dataset = get_datasets(config, args)
train_dataloader, val_dataloader, test_dataloader = get_dataloaders(config, training_dataset, val_dataset, test_dataset)

##############################################################################################
##############################################################################################

# Initialise SwinCVS according to config
model = None
model = build_model(config)
print('Full model initialised successfully!\n')
print(f"Trained SwinCVS weights loaded successfully for INFERENCE - name: {config.BACKBONE.PRETRAINED}")
model.to('cuda')
torch.cuda.empty_cache()

optimizer = build_optimizer(config, model)
loss_scaler = NativeScalerWithGradNormCount()
class_weights = torch.tensor(config.TRAIN.CLASS_WEIGHTS).to('cuda')
criterion = nn.BCEWithLogitsLoss(weight=class_weights).to('cuda')


desired_attack = 'pgd' if args.adv_attack == 'PGD' else 'apgd-ce'
"""adversary = AutoAttack(
    model,
    norm='Linf',
    eps=8/255,
    version='custom',
    attacks_to_run=[desired_attack],   # puedes cambiar a 'pgd' o 'apgd-dlr'
)"""

if desired_attack == 'pgd':
    adversary = PGD_BCE(model, eps=6/255, alpha=2/255, steps=8)
else:
    adversary=None

checkpoint_path = os.path.join(complete_exp_info_folder, 'weights')
os.makedirs(checkpoint_path, exist_ok=True)
results_dict = {}
best_mAP = 0

print("Beginning training...")
for epoch in range(config.TRAIN.EPOCHS):
    print(f"Epoch: {epoch+1:02}/{config.TRAIN.EPOCHS:02}")

    train_loss = 0.0
    val_loss = 0.0

    model.train()
    optimizer.zero_grad()

    print("Training")
    start_time = time.time()
    for idx, (samples, targets) in enumerate(tqdm(train_dataloader)):
        # Get predictions
        # samples.shape -> (batch, 3, 384, 384)
        # targets.shape -> (batch, 3)
        samples, targets = samples.to('cuda'), targets.to('cuda')

        if adversary is not None:
            model.eval()
            adv_images = adversary(samples, targets)
            model.train()
        
            # Forward limpio
            outputs_lstm_original = model(samples)
            loss_clean = criterion(outputs_lstm_original, targets)

            # Forward adversario
            outputs_adv = model(adv_images)
            loss_adv = criterion(outputs_adv, targets)
            
            loss_train = 0.9 * loss_clean + 0.1 * loss_adv

            wandb.log({'Training original Loss': loss_clean.item()})
            wandb.log({'Training adv Loss': loss_adv.item()})

        else:
            outputs_lstm_original = model(samples)
            loss_clean = criterion(outputs_lstm_original, targets)
            loss_train = loss_clean

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss_train, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        
        optimizer.zero_grad()
        train_loss+=loss_train.item()
        wandb.log({'Training Loss': loss_train.item()})
        torch.cuda.synchronize()

    # Validation Epochs
    print("\nValidation")
    model.eval()
    val_probabilities = []
    val_predictions = []
    val_targets = []
    with torch.inference_mode():
        for idx, (samples, targets) in enumerate(tqdm(val_dataloader)):
            # Get predictions
            samples, targets = samples.to('cuda'), targets.to('cuda')
            outputs_lstm = model(samples)
            
            # Get outputs
            val_probability = torch.sigmoid(outputs_lstm)
            val_prediction = torch.round(val_probability)

            # Save outputs
            val_probabilities.append(val_probability.to('cpu'))
            val_predictions.append(val_prediction.to('cpu'))
            val_targets.append(targets.to('cpu'))

            # Loss
            loss_val = criterion(outputs_lstm, targets)
            val_loss += loss_val.item()
            wandb.log({'Val Loss': loss_val.item()})
            torch.cuda.synchronize()

    # Get validation scores
    C1_ap, C2_ap, C3_ap, mAP = get_map(val_targets, val_probabilities)
    print('mAP', round(mAP, 4))
    print('C1 ap', round(C1_ap, 4))
    print('C2 ap', round(C2_ap, 4))
    print('C3 ap', round(C3_ap, 4))

    # Save validation scores
    val_predictions_2save = torch.cat(val_predictions, dim=0).tolist()
    val_probabilities_2save = torch.cat(val_probabilities, dim=0).tolist()
    val_targets_2save = torch.cat(val_targets, dim=0).tolist()

    epoch_results = {'avg_map': round(mAP, 4),
                    'C1_map': round(C1_ap, 4), 'C2_map': round(C2_ap, 4), 'C3_map': round(C3_ap, 4),
                    'preds': val_predictions_2save, 'true': val_targets_2save, 'preds_prob': val_probabilities_2save,
                    'train_loss': train_loss, 'val_loss': val_loss}
    results_dict[f"Epoch {epoch+1}"] = epoch_results

    
    results_file_path = '/'.join(checkpoint_path.split('/')[:-1])
    with open(os.path.join(results_file_path, f'results.json'), 'w') as file:
        json.dump(results_dict, file, indent=4)

    keys_a_borrar = ["preds", "true", 'preds_prob', 'train_loss', 'val_loss']
    for key in keys_a_borrar:
        epoch_results.pop(key, None)

    wandb.log({'Val metrics': epoch_results})

    
    # Estimate remaining time
    end_time = time.time()
    time_of_epoch = int(end_time-start_time)
    print(F"Epoch duration: {time_of_epoch}s")
    
    # Save weights of the best epoch
    if mAP >= best_mAP:
        best_mAP = mAP
        print(f"New best result (Epoch {epoch+1}), saving weights...")
        save_weights(model, checkpoint_path, epoch)
        wandb.log({'Best_Val_mAP': mAP})
        wandb.log({'Best C1': C1_ap})
        wandb.log({'Best C2': C2_ap})
        wandb.log({'Best C3': C3_ap})

    else:
        print('\n')


"""os.makedirs(f'visualizations/Fold{config.FOLD}', exist_ok=True)
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
    json.dump({'images': all_img_info}, f, indent=4)"""
