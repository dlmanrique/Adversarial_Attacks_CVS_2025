from scripts.m_swinv2 import SwinTransformerV2, load_pretrained
import torch
import torch.nn as nn
import os
import random
import numpy as np


seed = 5
# Environment Standardisation
random.seed(seed)                      # Set random seed
np.random.seed(seed)                   # Set NumPy seed
torch.manual_seed(seed)                # Set PyTorch seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(seed)           # Set CUDA seed
torch.use_deterministic_algorithms(True) # Force deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # CUDA workspace config

def build_model(config):

    # Initialise Backbone
    model = SwinTransformerV2(  img_size=384, # important to load correct IMNET pretrained weights
                                patch_size=config.BACKBONE.SWINV2.PATCH_SIZE,
                                in_chans=config.BACKBONE.SWINV2.IN_CHANS,
                                num_classes=config.BACKBONE.NUM_CLASSES,
                                embed_dim=config.BACKBONE.SWINV2.EMBED_DIM,
                                depths=config.BACKBONE.SWINV2.DEPTHS,
                                num_heads=config.BACKBONE.SWINV2.NUM_HEADS,
                                window_size=config.BACKBONE.SWINV2.WINDOW_SIZE,
                                mlp_ratio=config.BACKBONE.SWINV2.MLP_RATIO,
                                qkv_bias=config.BACKBONE.SWINV2.QKV_BIAS,
                                drop_rate=config.BACKBONE.DROP_RATE,
                                drop_path_rate=config.BACKBONE.DROP_PATH_RATE,
                                ape=config.BACKBONE.SWINV2.APE,
                                patch_norm=config.BACKBONE.SWINV2.PATCH_NORM,
                                use_checkpoint=config.BACKBONE.USE_CHECKPOINT,
                                pretrained_window_sizes=config.BACKBONE.SWINV2.PRETRAINED_WINDOW_SIZES)
    

    model.head = nn.Linear(in_features=1024, out_features=3, bias=True)
    model.load_state_dict(torch.load(os.path.join('weights', config.BACKBONE.PRETRAINED), weights_only=True), strict=True)
    # Only runs if the pure SwinV2 model option is selected

    return model
