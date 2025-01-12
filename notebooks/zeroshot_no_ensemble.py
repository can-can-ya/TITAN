import h5py
import os
import pickle
import yaml
from pathlib import Path
from tqdm import tqdm
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

from TITAN.titan.utils import get_eval_metrics, TEMPLATES, bootstrap
from TITAN.titan.wsi_datasets import WSIEmbeddingDataset

def read_datasplit_npz(path: str):
    data_npz = np.load(path, allow_pickle=True)

    pids_train = [str(s) for s in data_npz['train_patients']]
    if 'val_patients' in data_npz:
        pids_val = [str(s) for s in data_npz['val_patients']]
    else:
        pids_val = None
    if 'test_patients' in data_npz:
        pids_test = [str(s) for s in data_npz['test_patients']]
    else:
        pids_test = None
    return pids_train, pids_val, pids_test

os.environ["OMP_NUM_THREADS"] = "8"
gpu_id = 0
device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")

model = AutoModel.from_pretrained('/home/gjx/can_pretrained-model/TITAN', local_files_only=True, trust_remote_code=True)
# from TITAN.titan import Titan, TitanConfig
# model = Titan(TitanConfig()) # Pre-trained weights are not loaded.
model = model.to(device)

# create prompts for zero-shot classification
idx_to_class = {0: 'LUAD', 1: 'LUSC', 2: 'IDC', 3: 'ILC', 4: 'CCRCC', 5: 'PRCC', 6: 'ESAD', 7: 'ESCC'}
print(idx_to_class)
prompt_file = '/home/gjx/Code/TITAN/TITAN/prompts/eight_subtype_prompts_all_per_class_no_ensemble.json'
with open(prompt_file) as f:
    prompts = json.load(f)['0']
classnames = prompts['classnames']
templates = prompts['templates']
n_classes = len(classnames)
classnames_text = [classnames[idx_to_class[idx]] for idx in range(n_classes)]
for class_idx, classname in enumerate(classnames_text):
    print(f'{class_idx}: {classname}')
with torch.autocast('cuda', torch.float16), torch.inference_mode():
    classifier = model.zero_shot_classifier(classnames_text, templates, device=device)

index_col = 'pathology_id' # column with the slide ids
target_col = 'subtype' # column with the target labels
patient_id = 'patient_id' # column with patient id
datasets = ['lung', 'brca', 'rcc', 'esca'] # four datasets
all_label_map = {
    'lung': {'LUAD': 0, 'LUSC': 1},
    'brca': {'IDC': 0, 'ILC': 1},
    'rcc': {'CCRCC': 0, 'PRCC': 1},
    'esca': {'ESAD': 0, 'ESCC': 1}
} # maps values in target_col to integers
dataset_label_shift = {
    'lung': 0,
    'brca': 2,
    'rcc': 4,
    'esca': 6
}
patch_size_lv0 = 1024

for fold in range(1, 11):
    for dataset_name in datasets:
        label_map = all_label_map[dataset_name]

        df = pd.read_csv(f'/home/gjx/can_dataset/tcga_{dataset_name}/table/TCGA_{dataset_name.upper()}_path_subtype_x40_processed.csv')
        data_source = f'/home/gjx/can_dataset/tcga_{dataset_name}/feats-l0-s1024_CONCH_v_1_5/'
        datasplit_path = f'/home/gjx/can_dataset/tcga_{dataset_name}/datasplit/fold_{fold}.npz'

        pids_train, pids_val, pids_test = read_datasplit_npz(datasplit_path)
        df = df[df[patient_id].isin(pids_test)].reset_index(drop=True)

        dataset = WSIEmbeddingDataset(data_source=data_source,
                                      df=df,
                                      index_col=index_col,
                                      target_col=target_col,
                                      use_h5=True,
                                      label_map=label_map)
        dataloader = DataLoader(dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=4)
        print("num samples: ", len(dataloader.dataset))

        probs = []
        targets = []

        for idx, data in enumerate(tqdm(dataloader)):
            with torch.autocast('cuda', torch.float16), torch.inference_mode():
                image_features = data['img'].to(device)
                coords = data['coords'].to(device)
                slide_embedding = model.encode_slide_from_patch_features(image_features, coords, patch_size_lv0)
                scores = model.zero_shot(slide_embedding, classifier).squeeze(0).cpu()
            probs.append(scores)
            target = data['label'] + dataset_label_shift[dataset_name]
            targets.append(target)
        probs_all = torch.cat(probs, dim=0)
        preds_all = probs_all.argmax(dim=1)
        targets_all = torch.cat(targets, dim=0)

        results = get_eval_metrics(targets_all, preds_all, probs_all, roc_kwargs={'multi_class': 'ovo', 'average': 'macro'})
        print(f"{fold}-{dataset_name}-acc: {results['/acc']:.6f}")