import time
import math
import torch 
import torch.nn as nn
from torch import optim
import torchvision
from a5_helper import *
import matplotlib.pyplot as plt


def hello_single_stage_detector():
    print("Hello from single_stage_detector.py!")


def GenerateAnchor(anc, grid):
  """
  Anchor generator.

  Inputs:
  - anc: Tensor of shape (A, 2) giving the shapes of anchor boxes to consider at
    each point in the grid. anc[a] = (w, h) gives the width and height of the
    a'th anchor shape.
  - grid: Tensor of shape (B, H', W', 2) giving the (x, y) coordinates of the
    center of each feature from the backbone feature map. This is the tensor
    returned from GenerateGrid.
  
  Outputs:
  - anchors: Tensor of shape (B, A, H', W', 4) giving the positions of all
    anchor boxes for the entire image. anchors[b, a, h, w] is an anchor box
    centered at grid[b, h, w], whose shape is given by anc[a]; we parameterize
    boxes as anchors[b, a, h, w] = (x_tl, y_tl, x_br, y_br), where (x_tl, y_tl)
    and (x_br, y_br) give the xy coordinates of the top-left and bottom-right
    corners of the box.
  """
  anchors = None
  ##############################################################################
  # TODO: Given a set of anchor shapes and a grid cell on the activation map,  #
  # generate all the anchor coordinates for each image. Support batch input.   #
  ##############################################################################
  # Replace "pass" statement with your code
  anchors = []

  Hp, Wp = grid.shape[1], grid.shape[2]
  A = anc.shape[0]

  for curr_grid in grid:
    curr_grid_anc = torch.zeros((A, Hp, Wp, 4))

    for line, curr_gridl in enumerate(curr_grid):
      for col, curr_gridc in enumerate(curr_gridl):
        x, y = curr_gridc[0].item(), curr_gridc[1].item()
        for idx_anc, curr_anc in enumerate(anc):
          anc_h, anc_w = curr_anc[0], curr_anc[1]
          x_tl, y_tl = (x - anc_h / 2), (y - anc_w / 2)
          x_br, y_br = (x + anc_h / 2), (y + anc_w / 2)

          curr_grid_anc[idx_anc, line, col] = torch.tensor((x_tl, y_tl, x_br, y_br))
    anchors.append(curr_grid_anc)

  anchors = torch.stack(anchors).cuda()
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return anchors


def GenerateProposal(anchors, offsets, method='YOLO'):
  """
  Proposal generator.

  Inputs:
  - anchors: Anchor boxes, of shape (B, A, H', W', 4). Anchors are represented
    by the coordinates of their top-left and bottom-right corners.
  - offsets: Transformations of shape (B, A, H', W', 4) that will be used to
    convert anchor boxes into region proposals. The transformation
    offsets[b, a, h, w] = (tx, ty, tw, th) will be applied to the anchor
    anchors[b, a, h, w]. For YOLO, assume that tx and ty are in the range
    (-0.5, 0.5).
  - method: Which transformation formula to use, either 'YOLO' or 'FasterRCNN'
  
  Outputs:
  - proposals: Region proposals of shape (B, A, H', W', 4), represented by the
    coordinates of their top-left and bottom-right corners. Applying the
    transform offsets[b, a, h, w] to the anchor [b, a, h, w] should give the
    proposal proposals[b, a, h, w].
  
  """
  assert(method in ['YOLO', 'FasterRCNN'])
  proposals = None
  ##############################################################################
  # TODO: Given anchor coordinates and the proposed offset for each anchor,    #
  # compute the proposal coordinates using the transformation formulas above.  #
  ##############################################################################
  # Replace "pass" statement with your code
  center_anchors = torch.zeros_like(anchors)
  center_anchors[..., 0] = (anchors[..., 0] + anchors[..., 2]) / 2
  center_anchors[..., 1] = (anchors[..., 1] + anchors[..., 3]) / 2
  center_anchors[..., 2] = anchors[..., 2] - anchors[..., 0]
  center_anchors[..., 3] = anchors[..., 3] - anchors[..., 1]

  if method == 'YOLO':
    center_anchors[..., 0] += offsets[..., 0]
    center_anchors[..., 1] += offsets[..., 1]
  else:
    center_anchors[..., 0] += (offsets[..., 0] * center_anchors[..., 2]).detach()
    center_anchors[..., 1] += (offsets[..., 1] * center_anchors[..., 3]).detach()

  center_anchors[..., 2] *= torch.exp(offsets[..., 2])
  center_anchors[..., 3] *= torch.exp(offsets[..., 3])

  proposals = torch.zeros_like(center_anchors)
  proposals[..., 0] = center_anchors[..., 0] - center_anchors[..., 2] / 2 
  proposals[..., 1] = center_anchors[..., 1] - center_anchors[..., 3] / 2
  proposals[..., 2] = center_anchors[..., 0] + center_anchors[..., 2] / 2
  proposals[..., 3] = center_anchors[..., 1] + center_anchors[..., 3] / 2
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return proposals


def IoU(proposals, bboxes):
  """
  Compute intersection over union between sets of bounding boxes.

  Inputs:
  - proposals: Proposals of shape (B, A, H', W', 4)
  - bboxes: Ground-truth boxes from the DataLoader of shape (B, N, 5).
    Each ground-truth box is represented as tuple (x_lr, y_lr, x_rb, y_rb, class).
    If image i has fewer than N boxes, then bboxes[i] will be padded with extra
    rows of -1.
  
  Outputs:
  - iou_mat: IoU matrix of shape (B, A*H'*W', N) where iou_mat[b, i, n] gives
    the IoU between one element of proposals[b] and bboxes[b, n].

  For this implementation you DO NOT need to filter invalid proposals or boxes;
  in particular you don't need any special handling for bboxxes that are padded
  with -1.
  """
  iou_mat = None
  ##############################################################################
  # TODO: Compute the Intersection over Union (IoU) on proposals and GT boxes. #
  # No need to filter invalid proposals/bboxes (i.e., allow region area <= 0). #
  # However, you need to make sure to compute the IoU correctly (it should be  #
  # 0 in those cases.                                                          # 
  # You need to ensure your implementation is efficient (no for loops).        #
  # HINT:                                                                      #
  # IoU = Area of Intersection / Area of Union, where                          #
  # Area of Union = Area of Proposal + Area of BBox - Area of Intersection     #
  # and the Area of Intersection can be computed using the top-left corner and #
  # bottom-right corner of proposal and bbox. Think about their relationships. #
  ##############################################################################
  # Replace "pass" statement with your code
  proposals = torch.flatten(proposals, start_dim=1, end_dim=-2)
  proposals_area = (proposals[..., 2] - proposals[..., 0]) * (proposals[..., 3] - proposals[..., 1])

  bboxes = torch.unsqueeze(bboxes, dim=1)
  bboxes_area = (bboxes[..., 2] - bboxes[..., 0]) * (bboxes[..., 3] - bboxes[..., 1])

  red_union = (proposals_area.T + bboxes_area.T).T

  proposals = torch.unsqueeze(proposals, dim=2)

  maxs = torch.maximum(bboxes[..., :4], proposals)
  mins = torch.minimum(bboxes[..., :4], proposals)

  inter_heights = mins[..., 2] - maxs[..., 0]
  inter_widths = mins[..., 3] - maxs[..., 1]
  inter_heights[inter_heights < 0] = 0
  inter_widths[inter_widths < 0] = 0
  inter_area = inter_heights * inter_widths

  # Compute the IoU, which equals to: Area of Intersection / Area of Union
  iou_mat = inter_area / (red_union - inter_area)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return iou_mat


class PredictionNetwork(nn.Module):
  def __init__(self, in_dim, hidden_dim=128, num_anchors=9, num_classes=20, drop_ratio=0.3):
    super().__init__()

    assert(num_classes != 0 and num_anchors != 0)
    self.num_classes = num_classes
    self.num_anchors = num_anchors

    ##############################################################################
    # TODO: Set up a network that will predict outputs for all anchors. This     #
    # network should have a 1x1 convolution with hidden_dim filters, followed    #
    # by a Dropout layer with p=drop_ratio, a Leaky ReLU nonlinearity, and       #
    # finally another 1x1 convolution layer to predict all outputs. You can      #
    # use an nn.Sequential for this network, and store it in a member variable.  #
    # HINT: The output should be of shape (B, 5*A+C, 7, 7), where                #
    # A=self.num_anchors and C=self.num_classes.                                 #
    ##############################################################################
    # Make sure to name your prediction network pred_layer.
    self.pred_layer = None
    # Replace "pass" statement with your code
    self.pred_layer = torch.nn.Sequential(
      torch.nn.Conv2d(in_dim, hidden_dim, 1, stride=1),
      torch.nn.Dropout(drop_ratio),
      torch.nn.LeakyReLU(),
      torch.nn.Conv2d(hidden_dim, int(5 * num_anchors + num_classes), 1, stride=1)
    )
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

  def _extract_anchor_data(self, anchor_data, anchor_idx):
    """
    Inputs:
    - anchor_data: Tensor of shape (B, A, D, H, W) giving a vector of length
      D for each of A anchors at each point in an H x W grid.
    - anchor_idx: int64 Tensor of shape (M,) giving anchor indices to extract

    Returns:
    - extracted_anchors: Tensor of shape (M, D) giving anchor data for each
      of the anchors specified by anchor_idx.
    """
    B, A, D, H, W = anchor_data.shape
    anchor_data = anchor_data.permute(0, 1, 3, 4, 2).contiguous().view(-1, D)
    extracted_anchors = anchor_data[anchor_idx]
    return extracted_anchors
  
  def _extract_class_scores(self, all_scores, anchor_idx):
    """
    Inputs:
    - all_scores: Tensor of shape (B, C, H, W) giving classification scores for
      C classes at each point in an H x W grid.
    - anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors at
      which to extract classification scores

    Returns:
    - extracted_scores: Tensor of shape (M, C) giving the classification scores
      for each of the anchors specified by anchor_idx.
    """
    B, C, H, W = all_scores.shape
    A = self.num_anchors
    all_scores = all_scores.contiguous().permute(0, 2, 3, 1).contiguous()
    all_scores = all_scores.view(B, 1, H, W, C).expand(B, A, H, W, C)
    all_scores = all_scores.reshape(B * A * H * W, C)
    extracted_scores = all_scores[anchor_idx]
    return extracted_scores

  def forward(self, features, pos_anchor_idx=None, neg_anchor_idx=None):
    """
    Run the forward pass of the network to predict outputs given features
    from the backbone network.

    Inputs:
    - features: Tensor of shape (B, in_dim, 7, 7) giving image features computed
      by the backbone network.
    - pos_anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors
      marked as positive. These are only given during training; at test-time
      this should be None.
    - neg_anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors
      marked as negative. These are only given at training; at test-time this
      should be None.
    
    The outputs from this method are different during training and inference.
    
    During training, pos_anchor_idx and neg_anchor_idx are given and identify
    which anchors should be positive and negative, and this forward pass needs
    to extract only the predictions for the positive and negative anchors.

    During inference, only features are provided and this method needs to return
    predictions for all anchors.

    Outputs (During training):
    - conf_scores: Tensor of shape (2*M, 1) giving the predicted classification
      scores for positive anchors and negative anchors (in that order).
    - offsets: Tensor of shape (M, 4) giving predicted transformation for
      positive anchors.
    - class_scores: Tensor of shape (M, C) giving classification scores for
      positive anchors.

    Outputs (During inference):
    - conf_scores: Tensor of shape (B, A, H, W) giving predicted classification
      scores for all anchors.
    - offsets: Tensor of shape (B, A, 4, H, W) giving predicted transformations
      all all anchors.
    - class_scores: Tensor of shape (B, C, H, W) giving classification scores for
      each spatial position.
    """
    conf_scores, offsets, class_scores = None, None, None
    ############################################################################
    # TODO: Use backbone features to predict conf_scores, offsets, and         #
    # class_scores. Make sure conf_scores is between 0 and 1 by squashing the  #
    # network output with a sigmoid. Also make sure the first two elements t^x #
    # and t^y of offsets are between -0.5 and 0.5 by squashing with a sigmoid  #
    # and subtracting 0.5.                                                     #
    #                                                                          #
    # During training you need to extract the outputs for only the positive    #
    # and negative anchors as specified above.                                 #
    #                                                                          #
    # HINT: You can use the provided helper methods self._extract_anchor_data  #
    # and self._extract_class_scores to extract information for positive and   #
    # negative anchors specified by pos_anchor_idx and neg_anchor_idx.         #
    ############################################################################
    # Replace "pass" statement with your code
    B, _, H, W = features.shape
    A = self.num_anchors

    out_anchors = self.pred_layer(features)

    all_conf_scores = out_anchors[:, 0:5*A:5, ...]
    all_conf_scores = torch.sigmoid(all_conf_scores)

    all_offsets = out_anchors[:, :5*A, ...]
    all_offsets = all_offsets.reshape((B, A, 5, H, W))
    all_offsets = all_offsets[:, :, 1:, ...]
    all_offsets[:, :, :2, ...] = torch.sigmoid(all_offsets[:, :, :2, ...]) - 0.5

    all_class_scores = out_anchors[:, 5*A:, ...]

    if pos_anchor_idx is not None:  # Training mode.
      all_conf_scores = all_conf_scores.reshape((B, A, 1, H, W))

      pos_anc_conf = self._extract_anchor_data(all_conf_scores, pos_anchor_idx)
      neg_anc_conf = self._extract_anchor_data(all_conf_scores, neg_anchor_idx)
      conf_scores = torch.cat((pos_anc_conf, neg_anc_conf))

      offsets = self._extract_anchor_data(all_offsets, pos_anchor_idx)
      class_scores = self._extract_class_scores(all_class_scores, pos_anchor_idx)

    else:  # Inference mode.
      conf_scores = all_conf_scores
      offsets = all_offsets
      class_scores = all_class_scores
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return conf_scores, offsets, class_scores


class SingleStageDetector(nn.Module):
  def __init__(self):
    super().__init__()

    self.anchor_list = torch.tensor([[1., 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3]]) # READ ONLY
    self.feat_extractor = FeatureExtractor()
    self.num_classes = 20
    self.pred_network = PredictionNetwork(1280, num_anchors=self.anchor_list.shape[0], \
                                          num_classes=self.num_classes)
  def forward(self, images, bboxes):
    """
    Training-time forward pass for the single-stage detector.

    Inputs:
    - images: Input images, of shape (B, 3, 224, 224)
    - bboxes: GT bounding boxes of shape (B, N, 5) (padded)

    Outputs:
    - total_loss: Torch scalar giving the total loss for the batch.
    """
    # weights to multiple to each loss term
    w_conf = 1 # for conf_scores
    w_reg = 1 # for offsets
    w_cls = 1 # for class_prob

    total_loss = None
    ##############################################################################
    # TODO: Implement the forward pass of SingleStageDetector.                   #
    # A few key steps are outlined as follows:                                   #
    # i) Image feature extraction,                                               #
    # ii) Grid and anchor generation,                                            #
    # iii) Compute IoU between anchors and GT boxes and then determine activated/#
    #      negative anchors, and GT_conf_scores, GT_offsets, GT_class,           #
    # iv) Compute conf_scores, offsets, class_prob through the prediction network#
    # v) Compute the total_loss which is formulated as:                          #
    #    total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss,  #
    #    where conf_loss is determined by ConfScoreRegression, w_reg by          #
    #    BboxRegression, and w_cls by ObjectClassification.                      #
    # HINT: Set `neg_thresh=0.2` in ReferenceOnActivatedAnchors in this notebook #
    #       (A5-1) for a better performance than with the default value.         #
    ##############################################################################
    # Replace "pass" statement with your code
    batch_size = images.shape[0]
    features = self.feat_extractor(images)

    grid = GenerateGrid(batch_size)
    anchors = GenerateAnchor(self.anchor_list, grid)

    iou_mat = IoU(anchors, bboxes)
    activated_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class, _, _ = \
        ReferenceOnActivatedAnchors(anchors, bboxes, grid, iou_mat, pos_thresh=0.7, 
                                    neg_thresh=0.2, method='YOLO')

    conf_scores, offsets, class_scores = self.pred_network(features, activated_anc_ind, negative_anc_ind)
    conf_loss = ConfScoreRegression(conf_scores, GT_conf_scores)
    reg_loss = BboxRegression(offsets, GT_offsets)

    _, A, H, W, _ = anchors.shape
    anc_per_img = A * H * W
    cls_loss = ObjectClassification(class_scores, GT_class, batch_size, anc_per_img, activated_anc_ind)

    total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return total_loss
  
  def inference(self, images, thresh=0.5, nms_thresh=0.7):
    """"
    Inference-time forward pass for the single stage detector.

    Inputs:
    - images: Input images
    - thresh: Threshold value on confidence scores
    - nms_thresh: Threshold value on NMS

    Outputs:
    - final_propsals: Keeped proposals after confidence score thresholding and NMS,
                      a list of B (*x4) tensors
    - final_conf_scores: Corresponding confidence scores, a list of B (*x1) tensors
    - final_class: Corresponding class predictions, a list of B  (*x1) tensors
    """
    final_proposals, final_conf_scores, final_class = [], [], []
    ##############################################################################
    # TODO: Predicting the final proposal coordinates `final_proposals`,         #
    # confidence scores `final_conf_scores`, and the class index `final_class`.  #
    # The overall steps are similar to the forward pass but now you do not need  #
    # to decide the activated nor negative anchors.                              #
    # HINT: Thresholding the conf_scores based on the threshold value `thresh`.  #
    # Then, apply NMS (torchvision.ops.nms) to the filtered proposals given the  #
    # threshold `nms_thresh`.                                                    #
    # The class index is determined by the class with the maximal probability.   #
    # Note that `final_propsals`, `final_conf_scores`, and `final_class` are all #
    # lists of B 2-D tensors (you may need to unsqueeze dim=1 for the last two). #
    ##############################################################################
    # Replace "pass" statement with your code
    batch_size = images.shape[0]

    # Image feature extraction (using the backbone CNN network).
    features = self.feat_extractor(images)

    # Grid and anchor generation.
    grid = GenerateGrid(batch_size)
    anchors = GenerateAnchor(self.anchor_list, grid)

    B, A, Hp, Wp, _ = anchors.shape

    conf_scores, offsets, class_scores = self.pred_network(features)
    conf_scores = torch.flatten(conf_scores, start_dim=1)

    offsets = torch.transpose(offsets, 2, 4)

    class_indices = torch.max(class_scores, dim=1)[1]
    class_indices = class_indices.unsqueeze(1)
    class_indices = torch.broadcast_to(class_indices, (B, A, Hp, Wp))
    class_indices = torch.flatten(class_indices, start_dim=1)

    proposals = GenerateProposal(anchors, offsets, method='YOLO')
    proposals = torch.flatten(proposals, start_dim=1, end_dim=-2)

    final_proposals, final_conf_scores, final_class = [], [], []

    for idx in range(batch_size):
      cr_proposal = proposals[idx]       
      cr_conf_scores = conf_scores[idx]  
      cr_classes = class_indices[idx]   

      del_idx_mask = ~ (cr_conf_scores < thresh)

      cr_conf_scores = cr_conf_scores[del_idx_mask]
      K = cr_conf_scores.shape[0]
      cr_classes = cr_classes[del_idx_mask].unsqueeze(1)

      del_idx_mask = del_idx_mask.unsqueeze(1)
      del_idx_mask = torch.broadcast_to(del_idx_mask, (A*Hp*Wp, 4))

      cr_proposal = cr_proposal[del_idx_mask].reshape(K, 4)
      icr_proposal = torchvision.ops.nms(cr_proposal, cr_conf_scores, nms_thresh)

      final_proposals.append(cr_proposal[icr_proposal])
      final_conf_scores.append(cr_conf_scores[icr_proposal].unsqueeze(1))
      final_class.append(cr_classes[icr_proposal])
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return final_proposals, final_conf_scores, final_class


def nms(boxes, scores, iou_threshold=0.5, topk=None):
  """
  Non-maximum suppression removes overlapping bounding boxes.

  Inputs:
  - boxes: top-left and bottom-right coordinate values of the bounding boxes
    to perform NMS on, of shape Nx4
  - scores: scores for each one of the boxes, of shape N
  - iou_threshold: discards all overlapping boxes with IoU > iou_threshold; float
  - topk: If this is not None, then return only the topk highest-scoring boxes.
    Otherwise if this is None, then return all boxes that pass NMS.

  Outputs:
  - keep: torch.long tensor with the indices of the elements that have been
    kept by NMS, sorted in decreasing order of scores; of shape [num_kept_boxes]
  """

  if (not boxes.numel()) or (not scores.numel()):
    return torch.zeros(0, dtype=torch.long)

  keep = None
  #############################################################################
  # TODO: Implement non-maximum suppression which iterates the following:     #
  #       1. Select the highest-scoring box among the remaining ones,         #
  #          which has not been chosen in this step before                    #
  #       2. Eliminate boxes with IoU > threshold                             #
  #       3. If any boxes remain, GOTO 1                                      #
  #       Your implementation should not depend on a specific device type;    #
  #       you can use the device of the input if necessary.                   #
  # HINT: You can refer to the torchvision library code:                      #
  #   github.com/pytorch/vision/blob/master/torchvision/csrc/cpu/nms_cpu.cpp  #
  #############################################################################
  # Replace "pass" statement with your code
  keep = []

  indices = torch.sort(scores, descending=True)[1]
  boxes = boxes[indices]

  while len(boxes) != 0:
    keep.append(indices[0])
    hbox = boxes[0]
    boxes = boxes[1:]
    indices = indices[1:]

    hbox = hbox.reshape(1, 1, 4)
    rboxes = boxes.reshape(1, -1, 1, 1, 4)

    iou_mat = IoU(rboxes, hbox).squeeze()

    del_idx_mask = ~ (iou_mat > iou_threshold)

    indices = indices[del_idx_mask]
    boxes = boxes[del_idx_mask]

  if topk is not None:
    keep = keep[:topk]

  keep = torch.tensor(keep)
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  return keep


def ConfScoreRegression(conf_scores, GT_conf_scores):
  """
  Use sum-squared error as in YOLO

  Inputs:
  - conf_scores: Predicted confidence scores
  - GT_conf_scores: GT confidence scores
  
  Outputs:
  - conf_score_loss
  """
  # the target conf_scores for negative samples are zeros
  GT_conf_scores = torch.cat((torch.ones_like(GT_conf_scores), \
                              torch.zeros_like(GT_conf_scores)), dim=0).view(-1, 1)
  conf_score_loss = torch.sum((conf_scores - GT_conf_scores)**2) * 1. / GT_conf_scores.shape[0]
  return conf_score_loss


def BboxRegression(offsets, GT_offsets):
  """"
  Use sum-squared error as in YOLO
  For both xy and wh

  Inputs:
  - offsets: Predicted box offsets
  - GT_offsets: GT box offsets
  
  Outputs:
  - bbox_reg_loss
  """
  bbox_reg_loss = torch.sum((offsets - GT_offsets)**2) * 1. / GT_offsets.shape[0]
  return bbox_reg_loss

