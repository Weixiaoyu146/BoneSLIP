# -*- coding: utf-8 -*-


import argparse
import numpy
from tqdm import tqdm
import json

import numpy as np
import torch

top_k=10
output_path=r'/hdd3/hdd/hdd2/cbh/datasets/OpenClip/CLIP4.0/eval_CLIP/predictions.jsonl'
eval_batch_size=16
if __name__ == "__main__":

    print("Begin to load image features...")
    image_ids = []
    image_feats = []
    with open(r'/hdd3/hdd/hdd2/cbh/datasets/OpenClip/CLIP4.0/eval_CLIP/extract_image_feats.jsonl', "r") as fin:
        for line in tqdm(fin):
            obj = json.loads(line.strip())
            image_ids.append(obj['image_id'])
            image_feats.append(obj['feature'])
    image_feats_array = np.array(image_feats, dtype=np.float32)
    print("Finished loading image features.")

    print("Begin to compute top-{} predictions for texts...".format(top_k))
    with open(output_path, "w") as fout:
        with open(r'/hdd3/hdd/hdd2/cbh/datasets/OpenClip/CLIP4.0/eval_CLIP/extract_text_feats.jsonl', "r") as fin:
            for line in tqdm(fin):
                obj = json.loads(line.strip())
                text_id = obj['text_id']
                text_feat = obj['feature']
                score_tuples = []
                text_feat_tensor = torch.tensor([text_feat], dtype=torch.float).cuda()  # [1, feature_dim]
                idx = 0
                while idx < len(image_ids):
                    img_feats_tensor = torch.from_numpy(image_feats_array[idx: min(idx + eval_batch_size,
                                                                                   len(image_ids))]).cuda()  # [batch_size, feature_dim]
                    batch_scores = text_feat_tensor @ img_feats_tensor.t()  # [1, batch_size]
                    for image_id, score in zip(image_ids[idx: min(idx + eval_batch_size, len(image_ids))],
                                               batch_scores.squeeze(0).tolist()):
                        score_tuples.append((image_id, score))
                    idx += eval_batch_size
                top_k_predictions = sorted(score_tuples, key=lambda x: x[1], reverse=True)[:top_k]
                fout.write("{}\n".format(
                    json.dumps({"text_id": text_id, "image_ids": [entry[0] for entry in top_k_predictions]})))

    print("Top-{} predictions are saved in {}".format(top_k, output_path))
    print("Done!")