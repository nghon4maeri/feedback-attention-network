"""Evaluate FANet with iterative test-time refinement.

Usage:
    python scripts/evaluate.py [--config configs/kvasir_sessile.yaml]
                               [--checkpoint checkpoints/checkpoint.pth]
                               [--num-iter 10]
"""
import argparse
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from fanet.config import load_config
from fanet.data import load_data
from fanet.metrics import calculate_metrics
from fanet.models import FANet
from fanet.utils import create_dir, seeding, init_mask, rle_encode, rle_decode


def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/kvasir_sessile.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-iter", type=int, default=None)
    parser.add_argument("--save-masks", action="store_true",
                        help="save per-iteration mask montages into results/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    size = tuple(cfg["dataset"]["image_size"])
    num_iter = args.num_iter or cfg["eval"]["num_iter"]
    checkpoint_path = args.checkpoint or cfg["paths"]["checkpoint"]
    results_dir = cfg["paths"]["results"]

    seeding(cfg["train"]["seed"])
    create_dir(results_dir)

    (train_x, train_y), (test_x, test_y) = load_data(cfg["dataset"]["root"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FANet()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    prev_masks = init_mask(test_x, size)
    save_data = []
    file = open(f"{results_dir}/test_results.csv", "w")
    file.write("Iteration,Jaccard,F1,Recall,Precision,Specificity,Accuracy,F2,Mean Time,Mean FPS\n")

    for iter in range(num_iter):
        metrics_score = [0.0] * 7
        tmp_masks = []
        time_taken = []

        for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
            image = cv2.imread(x, cv2.IMREAD_COLOR)
            image = cv2.resize(image, size)
            img_x = image
            image = np.transpose(image, (2, 0, 1))
            image = image / 255.0
            image = np.expand_dims(image, axis=0)
            image = image.astype(np.float32)
            image = torch.from_numpy(image).to(device)

            mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, size)
            mask = np.expand_dims(mask, axis=0)
            mask = mask / 255.0
            mask = np.expand_dims(mask, axis=0)
            mask = mask.astype(np.float32)
            mask = torch.from_numpy(mask).to(device)

            pmask = str(" ".join(str(d) for d in prev_masks[i]))
            pmask = rle_decode(pmask, size)
            pmask = np.expand_dims(pmask, axis=0)
            pmask = np.expand_dims(pmask, axis=0)
            pmask = pmask.astype(np.float32)
            if iter == 0:
                pmask = np.transpose(pmask, (0, 1, 3, 2))
            pmask = torch.from_numpy(pmask).to(device)

            with torch.no_grad():
                start_time = time.time()
                pred_y = torch.sigmoid(model([image, pmask]))
                time_taken.append(time.time() - start_time)

                score = calculate_metrics(mask, pred_y)
                metrics_score = [a + b for a, b in zip(metrics_score, score)]
                pred_y = pred_y[0][0].cpu().numpy()
                pred_y = pred_y > 0.5
                pred_y = np.transpose(pred_y, (1, 0))
                pred_y = np.array(pred_y, dtype=np.uint8)
                pred_y = rle_encode(pred_y)
                prev_masks[i] = pred_y
                tmp_masks.append(pred_y)

        metrics = [m / len(test_x) for m in metrics_score]
        mean_time_taken = np.mean(time_taken)
        mean_fps = 1 / mean_time_taken

        print(f"Iteration {iter+1}: Jaccard: {metrics[0]:1.4f} - F1: {metrics[1]:1.4f} - "
              f"Recall: {metrics[2]:1.4f} - Precision: {metrics[3]:1.4f} - "
              f"Specificity: {metrics[4]:1.4f} - Acc: {metrics[5]:1.4f} - F2: {metrics[6]:1.4f} "
              f"- Mean Time: {mean_time_taken:1.3f}s - Mean FPS: {mean_fps:1.3f}")

        file.write(f"{iter+1}," + ",".join(f"{m:1.4f}" for m in metrics) +
                   f",{mean_time_taken:1.3f},{mean_fps:1.3f}\n")
        save_data.append(tmp_masks)

    file.close()

    if args.save_masks:
        for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
            image = cv2.imread(x, cv2.IMREAD_COLOR)
            image = cv2.resize(image, size)
            mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, size)
            mask = mask_parse(mask)

            name = y.split("/")[-1].split(".")[0]
            sep_line = np.ones((size[0], 10, 3)) * 128
            tmp = [image, sep_line, mask]

            for data in save_data:
                tmp.append(sep_line)
                d = rle_decode(str(" ".join(str(z) for z in data[i])), size)
                d = d * 255
                d = mask_parse(d)
                tmp.append(d)

            cat_images = np.concatenate(tmp, axis=1)
            cv2.imwrite(f"{results_dir}/{name}.png", cat_images)


if __name__ == "__main__":
    main()
