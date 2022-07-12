# YOLO v3
## Before training
Use KMeans to predefine 9 different anchor dimenions $(p_w, p_h)$, namely 3 anchor dimensions each scale.

## Training
### Architecture: DarkNet53
![DarkNet53](Images/DarkNet53.jpg)

### What to do with these feature maps
1. Use larger predefined anchor dimensions for smaller feature maps
2. Each position in every feature map contains 255 channels, which come from 3 * (80 + 1 + 4)
   - 3: 3 anchor dimension for each scale
   - 80: 80 classes in that dataset
   - 1: objctness score. Should be 1 if it's the best fit for one ground truth, 0 if it doesn't fit any groud truth better than an IOU threshold
   - 4: $t_x, t_y, t_w, t_h$
  ![t](Images/YOLO%20v23%20t.png)

### Loss
![Loss](Images/YOLO%20v123%20Loss.png)
- Coorfination loss: only for positive bounding box predictions (best fits for ground truths)
- Objectness score loss: 
  - for positive predictions: should be as close to 1 as possible
  - for negative predictions (don't fit any ground truth better than an IOU threshold): should be as close to 0 as possible
- Classification loss: [BCE loss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) instead of cross entropy loss only for positive predictions
- Predictions that are neither positive nor negative are ignored in loss calculation

# After training
- Ignore all predictions that has an objectness score smaller than a threshold
- NMS

# SSD (Single Shot MultiBox Detector)
## Model
![Framework](Images/SSD%20Framework.png)

1. Evaluate a small set (e.g. 4) of anchor boxes of different **hand-picked** aspect ratios at each location in several feature maps with different scales.
2. Bounding box and anchor box have one-to-one correspondence. For each anchor box, SSD predicts shape offsets and **confidences for all object categories $(c_1,...,c_p)$**
3. During Training: These anchor boxes are matched with ground truths (i.e. positive), **all other anchor boxes are negative**. An anchor box is matched with one ground truth if
   - It has the best IOU with this ground truth or
   - It has an IOU higher than a threshold(0.5) with this ground truth

## Training Objectives
$$L(x,c,l,g)=\frac{1}{N}\left(L_{conf}(x,c)+\alpha L_{loc}(x,l,g)\right)$$
- $x_{ij}^p = 1$ if i-th anchor box matches the j-th ground truth box of category p.
- $c$: output of softmax layer
- $l$: predicted box parameters
- $g$: ground truth box parameters

### Confidence loss
$$L_{conf}(x,c)=-\sum_{i\in Pos}^Nlog(c_i^p)-\sum_{i\in Neg}^N\log(c_i^0)$$
- $p$: classification ground truth
- $c_i^p$: predicted probability of anchor $i$ belongs to class $p$
- $p=0$ means there is no object in this anchor. **Background is also a category in SSD**

### Localization loss
$$L_{loc}(x,l,g)=\sum_{i\in Pos}^N\sum_{m\in\{cx,cy,w,h\}}x_{ij}^k smoothL1(l_i^m-\hat g_j^m)$$
details see [Faster RCNN](#Train-RPN)

### Hard Negative Mining
Similar as [Training RPN](#details), there would be a significant imbalance between the positive and negative training examples if we use all negative examples. SSD picks negative examples with highest confidence loss (the network cannot believe it's negative, i.e. hard negative), so that the ratio between the negatives and positives is at most 3:1.

### Data argumentation


# YOLO v4
