# -*- coding: utf-8 -*-
'''
This scripts performs kNN search on inferenced image and text features (on single-GPU) and outputs image-to-text retrieval prediction file for evaluation.
'''

import argparse
import numpy
from tqdm import tqdm
import json

import numpy as np
import torch


top_k=10
output_path=r'/hdd3/hdd/hdd2/cbh/datasets/OpenClip/CLIP4.0/eval_CLIP/predictions_tr.jsonl'
eval_batch_size=8
if __name__ == "__main__":

    print("Begin to load text features...")
    text_ids = []
    text_feats = []
    with open(r'/hdd3/hdd/hdd2/cbh/datasets/OpenClip/CLIP4.0/eval_CLIP/extract_text_feats.jsonl', "r") as fin:
        for line in tqdm(fin):
            obj = json.loads(line.strip())
            text_ids.append(obj['text_id'])
            text_feats.append(obj['feature'])
    text_feats_array = np.array(text_feats, dtype=np.float32)
    print("Finished loading text features.")

    print("Begin to compute top-{} predictions for images...".format(top_k))
    with open(output_path, "w") as fout:
        with open(r'/hdd3/hdd/hdd2/cbh/datasets/OpenClip/CLIP4.0/eval_CLIP/extract_image_feats.jsonl', "r") as fin:
            for line in tqdm(fin):
                obj = json.loads(line.strip())
                image_id = obj['image_id']
                image_feat = obj['feature']
                score_tuples = []
                image_feat_tensor = torch.tensor([image_feat], dtype=torch.float).cuda()  # [1, feature_dim]
                idx = 0
                while idx < len(text_ids):
                    text_feats_tensor = torch.from_numpy(text_feats_array[idx: min(idx + eval_batch_size,
                                                                                   len(text_ids))]).cuda()  # [batch_size, feature_dim]
                    batch_scores = image_feat_tensor @ text_feats_tensor.t()  # [1, batch_size]
                    for text_id, score in zip(text_ids[idx: min(idx + eval_batch_size, len(text_ids))],
                                              batch_scores.squeeze(0).tolist()):
                        score_tuples.append((text_id, score))
                    idx += eval_batch_size
                top_k_predictions = sorted(score_tuples, key=lambda x: x[1], reverse=True)[:top_k]
                fout.write("{}\n".format(
                    json.dumps({"image_id": image_id, "text_ids": [entry[0] for entry in top_k_predictions]})))

    print("Top-{} predictions are saved in {}".format(top_k, output_path))
    print("Done!")