import h5py
import os
import pickle
import yaml
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel

# from titan.utils import get_eval_metrics, TEMPLATES, bootstrap
from TITAN.titan.utils import get_eval_metrics, TEMPLATES, bootstrap

os.environ["OMP_NUM_THREADS"] = "8"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_id = 0
device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")

# # load model from huggingface
# model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
model = AutoModel.from_pretrained('/home/gjx/can_pretrained-model/TITAN', local_files_only=True, trust_remote_code=True)
# from TITAN.titan import Titan, TitanConfig
# model = Titan(TitanConfig()) # Pre-trained weights are not loaded.
model = model.to(device)

# Single feature classifier

# load example data
# from huggingface_hub import hf_hub_download
# demo_h5_path = hf_hub_download(
#     "MahmoodLab/TITAN",
#     filename="TCGA_demo_features/TCGA-PC-A5DK-01Z-00-DX1.C2D3BC09-411F-46CF-811B-FDBA7C2A295B.h5",
# )
demo_h5_path = '/home/gjx/can_pretrained-model/TITAN/TCGA_demo_features/TCGA-PC-A5DK-01Z-00-DX1.C2D3BC09-411F-46CF-811B-FDBA7C2A295B.h5'
file = h5py.File(demo_h5_path, 'r')
features = torch.from_numpy(file['features'][:])
coords = torch.from_numpy(file['coords'][:])
patch_size_lv0 = file['coords'].attrs['patch_size_level0']

# load configs and prompts for TCGA-OT task
with open('../datasets/config_tcga-ot.yaml', 'r') as file:
    task_config = yaml.load(file, Loader=yaml.FullLoader)
class_prompts = task_config['prompts']
target = task_config['target']
label_dict = task_config['label_dict']

# extract slide embedding
with torch.autocast('cuda', torch.float16), torch.inference_mode():
    features = features.to(device)
    coords = coords.to(device)
    slide_embedding = model.encode_slide_from_patch_features(features, coords, patch_size_lv0)

print('ok')