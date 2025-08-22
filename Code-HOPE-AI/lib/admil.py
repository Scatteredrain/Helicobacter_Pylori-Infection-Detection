import torch.nn as nn
import math
import torch
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss

class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=2, feat_dim=512):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim  = feat_dim

        center_init = torch.zeros(self.num_classes, self.feat_dim).cuda()

        nn.init.xavier_uniform_(center_init)
        self.centers = nn.Parameter(center_init)

        # self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())

    def forward(self, x, labels, class_weight=None):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long() # should be long()
        classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask   = labels.eq(classes.expand(batch_size, self.num_classes))
        if class_weight is not None:
            class_weight = class_weight.cuda()
            dist = distmat * mask.float() * class_weight[labels.squeeze()]
        else:
            dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

    def get_assignment(self, batch):
        alpha = 1.0
        norm_squared = torch.sum((batch.unsqueeze(1) - self.centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / alpha))
        power = float(alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def target_distribution(self, batch):
        """
        Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        weight = (batch ** 2) / torch.sum(batch, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

class Bag_Classifier_Attention_Head(nn.Module):
    def __init__(self, num_classes, init=False, withoutAtten=False, input_feat_dim=512):
        super(Bag_Classifier_Attention_Head, self).__init__()
        self.withoutAtten=withoutAtten
        self.classifier = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(input_feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))
        self.L = 1024
        self.D = 512
        self.K = 1

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        # self.headcount = len(num_classes)
        self.return_features = False
        self.top_layer = nn.Linear(1024, num_classes)
        if init:
            self._initialize_weights()

    def forward(self, x, returnBeforeSoftMaxA=False, softmax=False):
        x = self.classifier(x)

        # Attention module
        A_ = self.attention(x)  # NxK
        A_ = torch.transpose(A_, 1, 0)  # KxN
        if softmax:
            A = torch.softmax(A_, dim=1)  # softmax over N
        else:
            A = torch.sigmoid(A_)  # softmax over N
        

        x = torch.mm(A, x)  # KxL

        x = self.top_layer(x)
        if returnBeforeSoftMaxA:
            return x, torch.zeros_like(x), A, A_.squeeze(0)
        return x, A

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class Bag_Classifier_DSMIL_Head(nn.Module):
    def __init__(self, num_classes, init=True, input_feat_dim=512):
        super(Bag_Classifier_DSMIL_Head, self).__init__()
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(input_feat_dim, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(inplace=True))

        self.fc_dsmil = nn.Sequential(nn.Linear(512, 2))
        self.q_dsmil = nn.Linear(512, 512)
        self.v_dsmil = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(512, 512)
        )
        self.fcc_dsmil = nn.Conv1d(2, 2, kernel_size=512)

        self.return_features = False
        if init:
            self._initialize_weights()

    def forward(self, x):
        img_feature = x.view(x.shape[0], -1)

        device = img_feature.device
        instance_pred = self.fc_dsmil(img_feature)
        V = self.v_dsmil(img_feature)
        Q = self.q_dsmil(img_feature).view(img_feature.shape[0], -1)
        _, m_indices = torch.sort(instance_pred, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(img_feature, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K
        q_max = self.q_dsmil(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C,
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
        bag_feature = B.view(1, -1)
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc_dsmil(B) # 1 x C x 1
        C = C.view(1, -1)
        return instance_pred, C, F.normalize(img_feature, dim=-1), F.normalize(bag_feature, dim=-1)

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

import numpy as np
from nystrom_attention import NystromAttention

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, num_classes):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = num_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)


    def forward(self, h):

        # h = kwargs['data'].float() #[B, n, 1024]
        h = h.unsqueeze(0)
        h = self._fc1(h) #[B, n, 512]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        
        return logits, Y_hat, Y_prob