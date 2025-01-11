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

# load configs and prompts for TCGA-OT task
with open('../datasets/config_tcga-ot.yaml', 'r') as file:
    task_config = yaml.load(file, Loader=yaml.FullLoader)
class_prompts = task_config['prompts']
target = task_config['target']
label_dict = task_config['label_dict']

# create prompts for zero-shot classification
sorted_class_prompts = dict(sorted(class_prompts.items(), key=lambda item: label_dict.get(item[0], float('inf'))))
classes = list(sorted_class_prompts.keys())
class_prompts = [sorted_class_prompts[key] for key in sorted_class_prompts.keys()]
with torch.autocast('cuda', torch.float16), torch.inference_mode():
    classifier = model.zero_shot_classifier(class_prompts, TEMPLATES, device=device)  # will take approx 3 mins for 46 classes of TCGA-OncoTree (23 templates)



# # Single feature classifier
#
# # load example data
# # from huggingface_hub import hf_hub_download
# # demo_h5_path = hf_hub_download(
# #     "MahmoodLab/TITAN",
# #     filename="TCGA_demo_features/TCGA-PC-A5DK-01Z-00-DX1.C2D3BC09-411F-46CF-811B-FDBA7C2A295B.h5",
# # )
# demo_h5_path = '/home/gjx/can_pretrained-model/TITAN/TCGA_demo_features/TCGA-PC-A5DK-01Z-00-DX1.C2D3BC09-411F-46CF-811B-FDBA7C2A295B.h5'
# file = h5py.File(demo_h5_path, 'r')
# features = torch.from_numpy(file['features'][:])
# coords = torch.from_numpy(file['coords'][:])
# patch_size_lv0 = file['coords'].attrs['patch_size_level0']
#
# # extract slide embedding
# with torch.autocast('cuda', torch.float16), torch.inference_mode():
#     features = features.to(device)
#     coords = coords.to(device)
#     slide_embedding = model.encode_slide_from_patch_features(features, coords, patch_size_lv0)
#
# with torch.autocast('cuda', torch.float16), torch.inference_mode():
#     scores = model.zero_shot(slide_embedding, classifier)
# print("Predicted class:", classes[scores.argmax()])
# print("Normalized similarity scores:", [f"{c}: {score:.3f}" for c, score in zip(classes, scores[0][0])])



# Evaluate classifier on TCGA-OncoTree
# Reproduce the zeroshot results on the dataset TCGA-OncoTree based on pre-computed slide embeddings. The TITAN embeddings of TCGA-OT are available in our huggingface model hub.

task_csv = pd.read_csv('../datasets/tcga-ot_test.csv')

# load pre-extracted TITAN slide embeddings for TCGA
# import pickle
# from huggingface_hub import hf_hub_download
# slide_feature_path = hf_hub_download(
#     "MahmoodLab/TITAN",
#     filename="TCGA_TITAN_features.pkl",
# )
slide_feature_path = '/home/gjx/can_pretrained-model/TITAN/TCGA_TITAN_features.pkl'
with open(slide_feature_path, 'rb') as file:
  data = pickle.load(file)
slide_embeddings = torch.from_numpy(data['embeddings'][:])
slide_names = np.array(data['filenames'])

# get indices of slide_names that are in the task csv
slide_names_series = pd.Series(slide_names)
indices = slide_names_series[slide_names_series.isin(task_csv['slide_id'])].index
slide_embeddings = slide_embeddings[indices]
slide_names = slide_names[indices]

probs = []
targets = []

for slide_emb, slide_id in tqdm(zip(slide_embeddings, slide_names), total=len(slide_embeddings)):
    with torch.autocast('cuda', torch.float16), torch.inference_mode():
        slide_emb = slide_emb.to(device)
        probs.append(model.zero_shot(slide_emb, classifier).cpu())
    targets.append(label_dict[task_csv[task_csv['slide_id'] == slide_id][target].values[0]])
probs_all = torch.cat(probs, dim=0)
targets_all = torch.tensor(targets)
preds_all = probs_all.argmax(dim=1)

results = get_eval_metrics(targets_all, preds_all, probs_all, roc_kwargs={'multi_class': 'ovo', 'average': 'macro'})
for key, value in results.items():
    print(f"{key.split('/')[-1]: <12}: {value:.3f}")

# outputs = {
#     "targets": targets_all,
#     "preds": preds_all,
#     "probs": probs_all,
# }
# bootstrap_kwargs = {'n': 1000, 'alpha': 0.95}
# results_mean, results_std = bootstrap(results_dict=outputs, **bootstrap_kwargs)
# for keys, values in results_mean.items():
#     print(f"{keys.split('/')[-1]: <12}: {values:.4f} ± {results_std[keys]:.4f}")