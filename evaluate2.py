import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff,iou_,multiclass_iou_


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    iou_score = 0
    accuracy = 0
    precision = 0
    recall = 0

    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.num_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            mask_pred = net(image)

            if net.num_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                iou_score += iou_(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.num_classes).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
                iou_score += multiclass_iou_(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
            # Compute IOU

            # intersection = (mask_pred * mask_true).sum()
            # union = mask_pred.sum() + mask_true.sum() - intersection
            # iou = intersection / union
            # iou_score += iou

            # Compute Accuracy
            correct_pixels = torch.sum((mask_pred.argmax(dim=1) == mask_true.argmax(dim=1)).float())
            total_pixels = torch.numel(mask_true)
            accuracy += correct_pixels / total_pixels

            # Compute Precision
            true_positive = torch.sum((mask_pred.argmax(dim=1) == 1) & (mask_true.argmax(dim=1) == 1)).float()
            predicted_positive = torch.sum(mask_pred.argmax(dim=1) == 1).float()
            precision += true_positive / predicted_positive

            # Compute Recall
            actual_positive = torch.sum(mask_true.argmax(dim=1) == 1).float()
            recall += true_positive / actual_positive

    net.train()

    if num_val_batches == 0:
        return dice_score, iou_score, accuracy, precision,

    return dice_score / num_val_batches, iou_score / num_val_batches, accuracy / num_val_batches, precision / num_val_batches, recall / num_val_batches
