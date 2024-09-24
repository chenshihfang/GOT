import torch.nn as nn
import torch
from torch.nn import functional as F


class batch_act_ce_loss(nn.Module):

    def __init__(self):
        super().__init__()
        print("batch_act_ce_loss")

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        Returns:
            Loss tensor
        """
        hw = inputs.shape[1]

        pos = F.binary_cross_entropy_with_logits(
            inputs, torch.ones_like(inputs), reduction="none"
        )
        neg = F.binary_cross_entropy_with_logits(
            inputs, torch.zeros_like(inputs), reduction="none"
        )

        # print("pos.shape", pos.shape)
        # print("targets.shape", targets.shape)
        # print("neg.shape", neg.shape)

        pos = pos.flatten(1)
        targets = targets.flatten(1)
        neg = neg.flatten(1)

        # print("pos.shape", pos.shape)
        # print("targets.shape", targets.shape)
        # print("neg.shape", neg.shape)

        loss = torch.einsum("nc,mc->nm", pos, targets) 
        + torch.einsum("nc,mc->nm", neg, (1 - targets)
        )

        return loss / hw

    def forward(self, prediction, label):

        return self.loss(prediction, label)

class batch_dice_loss(nn.Module):

    def __init__(self):
        super().__init__()
        print("batch_dice_loss")

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        # inputs = inputs.sigmoid()
        # print("inputs.shape", inputs.shape)
        # print("targets.shape", targets.shape)
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        # print("inputs.shape 2", inputs.shape)
        # print("targets.shape 2", targets.shape)
        numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
        denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss

    def forward(self, prediction, label):

        return self.loss(prediction, label)


class LBHinge(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """
    def __init__(self, error_metric=nn.MSELoss(), threshold=None, clip=None):
        super().__init__()
        self.error_metric = error_metric
        self.threshold = threshold if threshold is not None else -100
        self.clip = clip

    def forward(self, prediction, label, target_bb=None):
        negative_mask = (label < self.threshold).float()
        positive_mask = (1.0 - negative_mask)

        prediction = negative_mask * F.relu(prediction) + positive_mask * prediction

        loss = self.error_metric(prediction, positive_mask * label)

        if self.clip is not None:
            loss = torch.min(loss, torch.tensor([self.clip], device=loss.device))
        return loss


class LBHingev2(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=None, threshold=None, return_per_sequence=False):
        super().__init__()

        if error_metric is None:
            if return_per_sequence:
                reduction = 'none'
            else:
                reduction = 'mean'
            error_metric = nn.MSELoss(reduction=reduction)

        self.error_metric = error_metric
        self.threshold = threshold if threshold is not None else -100

        self.return_per_sequence = return_per_sequence

    def forward(self, prediction, label, target_bb=None, valid_samples=None):
        assert prediction.dim() == 4 and label.dim() == 4

        negative_mask = (label < self.threshold).float()
        positive_mask = (1.0 - negative_mask)

        prediction = negative_mask * F.relu(prediction) + positive_mask * prediction

        # Mask invalid samples
        if valid_samples is not None:
            valid_samples = valid_samples.float()
            prediction = prediction * valid_samples
            label = label * valid_samples

            loss = self.error_metric(prediction, positive_mask * label)

            if self.return_per_sequence:
                loss = loss.mean((-2, -1))
            else:
                loss = loss * valid_samples.numel() / (valid_samples.sum() + 1e-12)
        else:
            loss = self.error_metric(prediction, positive_mask * label)

            if self.return_per_sequence:
                loss = loss.mean((-2, -1))

        return loss


class IsTargetCellLoss(nn.Module):
    def __init__(self, return_per_sequence=False, use_with_logits=True):
        super(IsTargetCellLoss, self).__init__()
        self.return_per_sequence = return_per_sequence
        self.use_with_logits = use_with_logits

    def forward(self, prediction, label, target_bb=None, valid_samples=None):
        score_shape = label.shape[-2:]

        prediction = prediction.view(-1, score_shape[0], score_shape[1])
        label = label.view(-1, score_shape[0], score_shape[1])

        if valid_samples is not None:
            valid_samples = valid_samples.float().view(-1)

            if self.use_with_logits:
                prediction_accuracy_persample = F.binary_cross_entropy_with_logits(prediction, label, reduction='none')
            else:
                prediction_accuracy_persample = F.binary_cross_entropy(prediction, label, reduction='none')

            prediction_accuracy = prediction_accuracy_persample.mean((-2, -1))
            prediction_accuracy = prediction_accuracy * valid_samples

            if not self.return_per_sequence:
                num_valid_samples = valid_samples.sum()
                if num_valid_samples > 0:
                    prediction_accuracy = prediction_accuracy.sum() / num_valid_samples
                else:
                    prediction_accuracy = 0.0 * prediction_accuracy.sum()
        else:
            if self.use_with_logits:
                prediction_accuracy = F.binary_cross_entropy_with_logits(prediction, label, reduction='none')
            else:
                prediction_accuracy = F.binary_cross_entropy(prediction, label, reduction='none')

            if self.return_per_sequence:
                prediction_accuracy = prediction_accuracy.mean((-2, -1))
            else:
                prediction_accuracy = prediction_accuracy.mean()

        return prediction_accuracy


class TrackingClassificationAccuracy(nn.Module):
    """ Estimates tracking accuracy by computing whether the peak of the predicted score map matches with the target
        location.
    """
    def __init__(self, threshold, neg_threshold=None):
        super(TrackingClassificationAccuracy, self).__init__()
        self.threshold = threshold

        if neg_threshold is None:
            neg_threshold = threshold
        self.neg_threshold = neg_threshold

    def forward(self, prediction, label, valid_samples=None):
        prediction_reshaped = prediction.view(-1, prediction.shape[-2] * prediction.shape[-1])
        label_reshaped = label.view(-1, label.shape[-2] * label.shape[-1])

        prediction_max_val, argmax_id = prediction_reshaped.max(dim=1)
        label_max_val, _ = label_reshaped.max(dim=1)

        label_val_at_peak = label_reshaped[torch.arange(len(argmax_id)), argmax_id]
        label_val_at_peak = torch.max(label_val_at_peak, torch.zeros_like(label_val_at_peak))

        prediction_correct = ((label_val_at_peak >= self.threshold) & (label_max_val > 0.25)) | ((label_val_at_peak < self.neg_threshold) & (label_max_val < 0.25))

        if valid_samples is not None:
            valid_samples = valid_samples.float().view(-1)

            num_valid_samples = valid_samples.sum()
            if num_valid_samples > 0:
                prediction_accuracy = (valid_samples * prediction_correct.float()).sum() / num_valid_samples
            else:
                prediction_accuracy = 1.0
        else:
            prediction_accuracy = prediction_correct.float().mean()

        return prediction_accuracy, prediction_correct.float()