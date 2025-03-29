# -*- coding: utf-8 -*-
'''
This script performs zero-shot evaluation on ImageNet-1K. (with single-GPU)
'''

import os
import argparse
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from cn_clip.clip.model import convert_weights, CLIP
from cn_clip.clip import tokenize
from cn_clip.training.main import convert_models_to_fp32
from cn_clip.clip.utils import image_transform
from cn_clip.eval.data import get_zeroshot_dataset, _preprocess_text
from cn_clip.eval.cvinw_zeroshot_templates import (
    openai_templates,
    flower_templates,
    food_templates,
    aircraft_templates,
    eurosat_templates,
    country211_templates,
)
import math
# import latexify


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vision-model",
        choices=["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14", "RN50"],
        default="ViT-B-16",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--text-model",
        choices=["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese", "RBT3-chinese"],
        default="RoBERTa-wwm-ext-base-chinese",
        help="Name of the text backbone to use.",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )
    parser.add_argument(
        "--label-file",
        type=str,
        help="file for labels",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        required=True,
        help="Path to the test set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        help="Specified dataset.",
    )
    parser.add_argument(
        "--index",
        type=str,
        default="",
        help="Specify image paths.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="Specified dataset.",
    )
    # parser.add_argument(
    #     "--imagenet-val",
    #     type=str,
    #     required=True,
    #     help="Path to imagenet val set for conducting zero shot evaluation.",
    # )
    parser.add_argument(
        "--img-batch-size", type=int, default=64, help="Image batch size."
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=52,
        help="The maximum length of input text (include [CLS] & [SEP] tokens)."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for ImageNet dataloader."
    )
    args = parser.parse_args()

    return args


def zero_shot_classifier(model, classnames, templates, args):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [_preprocess_text(template(classname)) for template in templates]  # format with class
            texts = tokenize(texts, context_length=args.context_length).to(args.gpu)  # tokenize
            class_embeddings = model(None, texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.gpu)
    return zeroshot_weights

def perf_measure(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    truth = target.view(1, -1).expand_as(pred)
    positive = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    negative = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(pred.size(1)):
        if pred[0][i] in positive and truth[0][i] == pred[0][i]:
           TP += 1
        if pred[0][i] in positive and truth[0][i] != pred[0][i]:
           FP += 1
        if pred[0][i] in negative and truth[0][i] == pred[0][i]:
           TN += 1
        if pred[0][i] in negative and truth[0][i] != pred[0][i]:
           FN += 1

    return TP, FP, TN, FN

def calculate(output, target, true_labels, pred_labels, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    truth = target.view(1, -1).expand_as(pred)
    pred = pred.squeeze()
    truth = truth.squeeze()
    true_labels = true_labels + truth.cpu().tolist()
    pred_labels = pred_labels + pred.cpu().tolist()
    return true_labels, pred_labels

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


# @latexify.with_latex
def run(model, classifier, dataloader, args):
    total_logits = []
    total_targets = []
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        top1, a_top1, r_top1, p_top1, F1_top1, top5, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        a_2, r_2, p_2 = 0.0, 0.0, 0.0
        TP1, FP1, TN1, FN1 = 0, 0, 0, 0
        for images, target in tqdm(dataloader):
            images = images.to(args.gpu)
            target = target.to(args.gpu)
            total_targets.append(target)

            # predict
            image_features = model(images, None)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_features @ classifier).softmax(dim=-1)
            total_logits.append(logits)

            # measure accuracy
            true_labels, pred_labels = calculate(logits, target, true_labels, pred_labels, topk=(1, 1))
            TP, FP, TN, FN = perf_measure(logits, target, topk=(1, 1))
            TP1 += TP
            FP1 += FP
            TN1 += TN
            FN1 += FN
            acc1, acc5 = accuracy(logits, target, topk=(1, 1))
            accuracy1 = TP + TN
            accuracy2 = TP + TN + FP + FN

            recall1 = TP + TN
            recall2 = TP + TN + FN

            precision1 = TP + TN
            precision2 = TP + TN + FP

            top1 += acc1

            a_top1 += accuracy1
            a_2 += accuracy2

            r_top1 += recall1
            r_2 += recall2

            p_top1 += precision1
            p_2 += precision2
            n += images.size(0)
            # top1 = top1 / n
            # a_top1 = (a_top1 / a_2) if a_2 != 0 else 0
            # r_top1 = (r_top1 / r_2) if r_2 != 0 else 0
            # p_top1 = (p_top1 / p_2) if p_2 != 0 else 0
            # F1_top1 = (2 * p_top1 * r_top1) / (p_top1 + r_top1) if (p_top1 + r_top1) != 0 else 0

    outputs = torch.cat(total_logits, dim=0)
    targets = torch.cat(total_targets, dim=0)

    if getattr(args, "index", ""):
        print("Use index to rearrange the logits...")
        with open(args.index, "r", encoding="utf-8") as f:
            index = json.load(f)
            print(index)
        outputs = outputs[index]
        targets = targets[index]
        print(targets)

    print("TP: ", TP1)
    print("FP: ", FP1)
    print("TN: ", TN1)
    print("FN: ", FN1)
    y_true = np.array(true_labels)
    y_pred = np.array(pred_labels)
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # 计算精确率
    macro_precision = precision_score(y_true, y_pred, average='macro')
    print("macro_precision:", macro_precision)

    micro_precision = precision_score(y_true, y_pred, average='micro')
    print("micro_precision:", micro_precision)

    # 计算召回率
    macro_recall = recall_score(y_true, y_pred, average='macro')
    print("macro_recall:", macro_recall)

    micro_recall = recall_score(y_true, y_pred, average='micro')
    print("micro_recall:", micro_recall)

    # 计算F1分数
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print("macro_f1-score:", macro_f1)

    micro_f1 = f1_score(y_true, y_pred, average='micro')
    print("micro_f1-score:", micro_f1)

    labels = []
    for i in range(1, 27):
        labels.append('class {}'.format(i))

    x_labels = ['$ankleL_a$', '$ankleL_n$', '$ankleR_a$', '$ankleR_n$', '$chestL_a$', '$chestL_n$', '$chestR_a$', '$chestR_n$', '$elbowL_a$', '$elbowL_n$', '$elbowR_a$', '$elbowR_n$', '$head_a$', '$head_n$', '$kneeL_a$', '$kneeL_n$', '$kneeR_a$', '$kneeR_n$', '$pelvis_a$', '$pelvis_n$', '$shoulderL_a$', '$shoulderL_n$', '$shoulderR_a$', '$shoulderR_n$', '$vertebra_a$', '$vertebra_n$']
    y_labels = ['$ankleL_a$', '$ankleL_n$', '$ankleR_a$', '$ankleR_n$', '$chestL_a$', '$chestL_n$', '$chestR_a$', '$chestR_n$', '$elbowL_a$', '$elbowL_n$', '$elbowR_a$', '$elbowR_n$', '$head_a$', '$head_n$', '$kneeL_a$', '$kneeL_n$', '$kneeR_a$', '$kneeR_n$', '$pelvis_a$', '$pelvis_n$', '$shoulderL_a$', '$shoulderL_n$', '$shoulderR_a$', '$shoulderR_n$', '$vertebra_a$', '$vertebra_n$']

    # 绘制热力图
    plt.figure(figsize=(14.5, 11.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=x_labels, yticklabels=y_labels)
    plt.title('Confusion Matrix', fontsize=20)
    plt.xlabel('Predicted Labels', fontsize=20)
    plt.ylabel('True Labels', fontsize=20)
    plt.show()

    top1 = top1 / n
    a_top1 = (a_top1 / a_2) if a_2 != 0 else 0
    r_top1 = (r_top1 / r_2) if r_2 != 0 else 0
    p_top1 = (p_top1 / p_2) if p_2 != 0 else 0
    F1_top1 = (2 * p_top1 * r_top1) / (p_top1 + r_top1) if (p_top1 + r_top1) != 0 else 0
    return top1, a_top1, r_top1, p_top1, F1_top1, outputs


if __name__ == "__main__":
    args = parse_args()

    # Log params.
    print("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"  {name}: {val}")

    args.gpu = 0
    torch.cuda.set_device(args.gpu)

    # Initialize the model.
    vision_model_config_file = Path(
        __file__).parent.parent / f"clip/model_configs/{args.vision_model.replace('/', '-')}.json"
    print('Loading vision model config from', vision_model_config_file)
    assert os.path.exists(vision_model_config_file)

    text_model_config_file = Path(
        __file__).parent.parent / f"clip/model_configs/{args.text_model.replace('/', '-')}.json"
    print('Loading text model config from', text_model_config_file)
    assert os.path.exists(text_model_config_file)

    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        if isinstance(model_info['vision_layers'], str):
            model_info['vision_layers'] = eval(model_info['vision_layers'])
        for k, v in json.load(ft).items():
            model_info[k] = v

    model = CLIP(**model_info)
    convert_weights(model)

    # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
    if args.precision == "amp" or args.precision == "fp32":
        convert_models_to_fp32(model)
    model.cuda(args.gpu)
    if args.precision == "fp16":
        convert_weights(model)

    # Get eval data.
    print("Preparing zeroshot dataset.")
    data = {}
    print(f"{model_info['image_resolution']}")
    data[args.dataset] = get_zeroshot_dataset(
        args, image_transform(model_info["image_resolution"])
    )

    # Resume from a checkpoint.
    print("Begin to load model checkpoint from {}.".format(args.resume))
    assert os.path.exists(args.resume), "The checkpoint file {} not exists!".format(args.resume)
    # Map model to be loaded to specified single gpu.
    loc = "cuda:{}".format(args.gpu)
    checkpoint = torch.load(args.resume, map_location='cpu')
    start_epoch = checkpoint["epoch"]
    sd = checkpoint["state_dict"]
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
    model.load_state_dict(sd)
    print(
        f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']} @ {checkpoint['step']} steps)"
    )

    # Compute ensembled class embeddings
    print('Building zero-shot classifier')

    model.eval()

    f = open(args.label_file, "r", encoding="utf8")
    classnames = [line.strip() for line in f.readlines()]

    template_dict = {
        "fgvc-aircraft-2013b-variants102": aircraft_templates,
        "food-101": food_templates,
        "oxford-flower-102": flower_templates,
        "eurosat_clip": eurosat_templates,
        "resisc45_clip": eurosat_templates,
        "country211": country211_templates,
        "openai": openai_templates,
    }
    if args.dataset in template_dict.keys():
        templates = template_dict[args.dataset]
    else:
        templates = template_dict['openai']

    # Make inference and evaluation
    print('Using classifier')
    classifier = zero_shot_classifier(model, classnames, templates, args)
    results = {}
    top1, a_top1, r_top1, p_top1, F1_top1, logits = run(model, classifier, data[args.dataset].dataloader, args)


    def json_prec_dump(data, prec=6):
        return json.dumps(
            json.loads(json.dumps(data), parse_float=lambda x: round(float(x), prec))
        )


    print(logits.size())
    output_dict = {
        "model_name": "CN-CLIP-" + args.vision_model,
        "dataset_name": args.dataset,
        "num_trainable_params": 0,
        "num_params": sum(x.numel() for x in model.parameters()),
        "num_visual_params": sum(x.numel() for x in model.visual.parameters()),
        "num_backbone_params": sum(x.numel() for x in model.parameters()),
        "n_shot": 0,
        "rnd_seeds": [123],
        "predictions": [logits.cpu().data.numpy().tolist()],
    }
    json_string = json_prec_dump(output_dict)
    with open(os.path.join(args.save_dir, f"{args.dataset}.json"), "w", encoding="utf-8") as w:
        w.write(json_string)

    results["zeroshot-top1"] = top1

    results["Accuracy"] = a_top1

    results["Recall"] = r_top1

    results["Precision"] = p_top1

    results["F1-Score"] = F1_top1

    print('Result:')
    print(", ".join(["{}: {}".format(k, v) for k, v in results.items()]))
    print('Finished.')
