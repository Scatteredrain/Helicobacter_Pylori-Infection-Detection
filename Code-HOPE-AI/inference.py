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
from lib.pvtv2_lstm import LSTMModel
from lib.admil import CenterLoss

from utils.utils import inference_preprocess, inference_one_bag

if __name__ == '__main__':

  # model load
  weight_path = '/mnt/data/yizhenyu/data/HP识别/workspace/MIL-HP_YZY_BACKUP/weights'
  
  model = LSTMModel()
  model_center_loss = CenterLoss(2, 256).cuda()
  model_center_loss_img = CenterLoss(2, 512).cuda()

  model.load_state_dict(torch.load(weight_path+'/model.pth',map_location="cpu"),strict=False)
  model_center_loss.load_state_dict(torch.load(weight_path+'/center_loss.pth',map_location="cpu"),strict=False)
  model_center_loss_img.load_state_dict(torch.load(weight_path+'/center_loss_img.pth',map_location="cpu"),strict=False)

  model.cuda().eval()
  model_center_loss.cuda().eval()
  model_center_loss_img.cuda().eval()

  # data load
  data_path = '/mnt/data/yizhenyu/data/HP识别/workspace/Code_HP-Infection/cases/Positive_2' 
  images, names = inference_preprocess(data_path, testsize=352)
  bag_tensor = torch.stack(images).cuda()  # shape: [N, 3, 256, 256]

  # save result path
  save_img_name = data_path.replace('cases', 'results') + '_res.png'
  os.makedirs(os.path.dirname(save_img_name), exist_ok=True)

  with torch.no_grad():
    preds_patient, probs_patient, bag_len = inference_one_bag(model, model_center_loss, model_center_loss_img, bag_tensor, names, topk = 7, save_img_name=save_img_name)
    # print(preds_patient)
    pred_vote = torch.stack(preds_patient).float().mean() > 0.5

  print('sample path:', data_path)
  print('predict result:', 'HP - positive' if pred_vote else 'HP - negative')
  print('Visualization saved to:', save_img_name)