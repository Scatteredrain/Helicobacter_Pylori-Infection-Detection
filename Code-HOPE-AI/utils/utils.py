import os
import numpy as np
import sys
import cv2
import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.nn.functional as F
from datetime import datetime
import json
import pandas as pd

from PIL import Image
from torchvision import transforms

def inference_preprocess(folder_path, testsize=256):

    transform = transforms.Compose([
        transforms.Resize((testsize, testsize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    images = []
    image_names = []

    for filename in sorted(os.listdir(folder_path)): 
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in image_extensions:
            try:

                img = Image.open(file_path).convert('RGB')  
                img_tensor = transform(img)
                images.append(img_tensor)
                image_names.append(filename)
            except Exception as e:
                print(f"error processing {filename} : {e}")

    return images, image_names

def inference_one_bag(model, model_center_loss, model_center_loss_img, bags, names, topk = 7, save_img_name=None):

  max_instances, _, _, _ = bags.size()
  bag_softmax_out = []
  bag_img_feature = []

  bag_len = bags.size()[1]
  for instance in range(0, bags.size()[1], 40):
    split_images = bags[instance:instance+40, :, :, :]
    split_images = split_images.cuda()
    
    img_features, _, split_softmax_out = model(split_images.unsqueeze(0), is_train=False)
    bag_softmax_out.append(split_softmax_out.squeeze(0))
    bag_img_feature.append(img_features.squeeze(0))
    split_images = split_images.cpu()

  bag_softmax_out = torch.cat(bag_softmax_out, dim=0)
  probs_class_1 = bag_softmax_out[:, 1]
  topk_values, topk_indices = torch.topk(probs_class_1, min(probs_class_1.size(0), topk))
  topk_indices = topk_indices.cpu()
  topk_imgs = bags[topk_indices, :, :, :] #[b, topk, 3, h, w]

  img_feature, x_fc, bag_feature, lstm_out = model(topk_imgs.cuda().unsqueeze(0), is_train=True)

  topk_imgs = topk_imgs.squeeze(0).cpu()

  # -- 1. LSTM Output
  patient_pred_lstm = torch.tensor([1]) if F.softmax(lstm_out, dim=1)[-1][-1].item() > 0.5 else torch.tensor([0])
  # -- 2. topk img Output mean
  patient_pred_img_mean = torch.tensor([1]) if topk_values.mean() > 0.5 else torch.tensor([0])
  topk_img_pred = (topk_values > 0.5).float().cpu()
 # -- 3. Patient CenterLoss Output
  center_loss_output = model_center_loss.get_assignment(bag_feature)
  center_loss_probs = center_loss_output.detach()[:, 1].clone()
  patient_pred_pat_center = torch.tensor([1]) if center_loss_probs[0] > 0.5 else torch.tensor([0])

  # -- 4. topk img CenterLoss Output
  img_feature = torch.cat(bag_img_feature, dim=0)
  center_loss_img_output = model_center_loss_img.get_assignment(img_feature.squeeze(0))
  # center_loss_img_output.size():torch.Size([7,2])
  center_loss_img_probs = center_loss_img_output.detach()[:, 1].clone().cpu()
  # center_loss_img_output.size():torch.Size([7])
  topk_values_img_center, _ = torch.topk(center_loss_img_probs, min(center_loss_img_probs.size(0), topk))
  topk_img_pred_img_center = (topk_values_img_center > 0.5).float()

  patient_pred_img_center_mean = torch.tensor([1]) if topk_values_img_center.mean() > 0.5 else torch.tensor([0])

  # -- Concat all Outputs
  preds_patient = [patient_pred_lstm, patient_pred_img_mean, patient_pred_pat_center, patient_pred_img_center_mean]
  preds_topkImgs = [[F.softmax(lstm_out, dim=1)[-1][-1].cpu().item()], [topk_values.mean().cpu().item()], 
      [center_loss_probs[0].cpu().item()], [topk_values_img_center.mean().cpu().item()]]
  
  if save_img_name is not None:
    debout = save_result(
        bags, names, center_loss_img_probs.cpu().numpy(), probs_class_1.cpu().numpy(),
        topk_indices.cpu(), F.softmax(lstm_out, dim=1)[-1][-1].cpu().item(),
        center_loss_probs[0].cpu().item(), topk_values_img_center.mean().item())
    cv2.imwrite(save_img_name, debout)

  return preds_patient, preds_topkImgs, bag_len

def save_result(out, topk_names, top_labels, top_values, topk_indices, lstm_out, center_pred=None, center_pred_img=None, images_per_row=8, size=(200, 200)):
    rows = []
    h, w = size
    front_scale = int(size[0] / 200.)

    sorted_indices = np.argsort(-top_values)
    out = [out[i] for i in sorted_indices]
    topk_names = [topk_names[i] for i in sorted_indices]
    top_labels = [top_labels[i] for i in sorted_indices]
    top_values = [top_values[i] for i in sorted_indices]

    for i in range(0, len(out), images_per_row):
        row_images = []
        for j in range(images_per_row):
            if i + j >= len(out):
                log = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                tmp_name = topk_names[i + j].split("_")[-1]
                deb = out[i + j]
                # text = f"{tmp_name[-20:]}, {format(top_labels[i+j], '.10f')[:5]}, {format(top_values[i+j], '.10f')[:5]}"
                text = f"img_name: {tmp_name[-20:]}"
                text2 = f"risk_prob: {format(top_values[i+j], '.10f')[:5]}"

                org = (5, 10)  
                org2 = (5, 22)  
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4 * front_scale
                color = (0, 255, 0)  # 绿色
                thickness = 1

                log = deb.clone().cpu().detach().numpy().squeeze()
                log = log.transpose(1,2,0)
                log = (log - log.min()) / (log.max() - log.min() + 1e-8)
                log *= 255
                log = log.astype(np.uint8)
                log = cv2.cvtColor(log, cv2.COLOR_BGR2RGB)
                log = cv2.resize(log, size)
                log = cv2.putText(log, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
                log = cv2.putText(log, text2, org2, font, font_scale, color, thickness, cv2.LINE_AA)

                if topk_names[i + j][0] == "zeros":
                    row_images.append(log)
                    continue
                patient_name = topk_names[i + j][0].split("_")[0]
                    
            row_images.append(log)
        rows.append(np.hstack(row_images))

    tiled_image = np.vstack(rows)
 
    title_bar_height = 50
    topk_mean = np.array(top_values[:topk_indices.size(0)]).mean()
    
    title = f"Bag pred: {format(lstm_out, '.10f')[:5]}, Img mean: {format(topk_mean, '.10f')[:5]}"
    if center_pred is not None:
        title += f", Bag center: {format(center_pred, '.10f')[:5]}"
    if center_pred_img is not None:
        title += f", Img center mean: {format(center_pred_img, '.10f')[:5]}"

    title_bar = np.zeros((title_bar_height, tiled_image.shape[1], 3), dtype=np.uint8)
    title_bar = cv2.putText(title_bar, title, (10, title_bar_height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)

    final_image = np.vstack([title_bar, tiled_image])
    
    return final_image