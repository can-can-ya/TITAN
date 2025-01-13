import os
from tqdm import tqdm
import h5py
import torch
from transformers import AutoModel

# gpu_id = 0
# device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu") # out of memory
device = torch.device("cpu")
model = AutoModel.from_pretrained('/home/gjx/can_pretrained-model/TITAN', local_files_only=True, trust_remote_code=True)
# from TITAN.titan import Titan, TitanConfig
# model = Titan(TitanConfig()) # Pre-trained weights are not loaded.
model = model.to(device)

datasets = ['lung', 'brca', 'rcc', 'esca'] # four datasets
patch_size_lv0 = 1024

for dataset in datasets:
    slide_feats_TITAN_dir = f'/home/gjx/can_dataset/tcga_{dataset}/feats-l0-s1024_CONCH_v_1_5/slide_feats_TITAN'
    os.makedirs(slide_feats_TITAN_dir, exist_ok=True)
    h5_files_dir = f'/home/gjx/can_dataset/tcga_{dataset}/feats-l0-s1024_CONCH_v_1_5/h5_files'
    h5_file_names = os.listdir(h5_files_dir)

    for h5_file_name in tqdm(h5_file_names):
        slide_feature_save_path = os.path.join(slide_feats_TITAN_dir, os.path.splitext(h5_file_name)[0]+'.pt')
        if os.path.exists(slide_feature_save_path):
            continue
        h5_file_path = os.path.join(h5_files_dir, h5_file_name)
        with h5py.File(h5_file_path, 'r') as f:
            features = torch.from_numpy(f['features'][:])
            coords = torch.from_numpy(f['coords'][:])

        # extract slide feature
        with torch.autocast('cuda', torch.float16), torch.inference_mode():
            features = features.unsqueeze(0).to(device)
            coords = coords.unsqueeze(0).to(device)
            slide_feature = model.encode_slide_from_patch_features(features, coords, patch_size_lv0).cpu()
            torch.save(slide_feature, slide_feature_save_path)