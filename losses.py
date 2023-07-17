from typing import Any, List, Tuple
from torch import Tensor
import torch
from torch.nn import functional as F

EPS = 1e-5


def loss_bce(pred: Tensor, labels: Tensor, only_pos: bool = False) -> Tensor:
    assert torch.isnan((pred + EPS).log()).any() == False, f"pred: {pred}"
    assert torch.isinf((pred + EPS).log()).any() == False

    num_classes = labels.size(1)
    if not only_pos:
        return (
            -(
                (pred + EPS).log() * labels
                + (-pred + 1.0 + EPS).log() * (-labels + 1.0 + EPS)
            ).sum(dim=1)
            / num_classes
        )
    else:
        assert (
            labels.dtype == torch.int or labels.dtype == torch.long
        )
        return (
            -((pred + EPS).log() * labels).masked_fill(labels != 1, 0.0).sum(dim=1)
            / num_classes
        )


def loss_epr(
    pred: Tensor, strong_labels: Tensor, expected_num_pos_labels: float
) -> Tensor:
    batch_size = pred.size(0)
    label_size = pred.size(1)
    
    first_part = loss_bce(pred, strong_labels, only_pos=True).sum(0) / batch_size
    second_part = ((pred.sum(1).mean(0) - expected_num_pos_labels) / label_size) ** 2

    assert first_part.isnan().any() == False
    assert second_part.isnan().any() == False
    assert first_part.isinf().any() == False
    assert second_part.isinf().any() == False
    return first_part + second_part


def loss(
    pred_targets: Tensor,
    esti_targets: Tensor,
    observed_targets: Tensor,
    expected_num_pos_labels: float,
) -> Tuple[Tensor, int]:
    # make sure all labels are positive
    assert torch.min(observed_targets) >= 0

    batch_size = pred_targets.size(0)

    # freeze estimated targets
    esti_targets_detach = esti_targets.detach()
    loss_1_1 = loss_bce(pred_targets, esti_targets_detach).sum(
        dim=0
    ) / batch_size + loss_epr(pred_targets, observed_targets, expected_num_pos_labels)
    loss_1 = loss_1_1
    # freeze predicted targets
    pred_targets_detach = pred_targets.detach()
    loss_2 = loss_bce(esti_targets, pred_targets_detach).sum(
        dim=0
    ) / batch_size + loss_epr(esti_targets, observed_targets, expected_num_pos_labels)

    assert loss_1.isnan().any() == False
    assert loss_1.isinf().any() == False
    assert loss_2.isnan().any() == False
    assert loss_2.isinf().any() == False

    return (loss_1 + loss_2) / 2, batch_size


def loss_contrastive_infonce(
    feats1: Tensor,
    feats2: Tensor,
    targets: Tensor,
    correlations: Tensor,
    pq: Tensor,
    tau: float = 1,
    hard_correlation=False,
):
    """
    targets: [1, 0, 9...]
    """
    batch_size = feats1.size(0)
    num_labels = pq.size(0)
    feat_num = pq.size(1)
    feat_dim = pq.size(2)

    # batch_size x feat_dim
    feats1 = F.normalize(feats1, dim=1)
    # batch_size x feat_dim
    feats2 = F.normalize(feats2, dim=1)
    # num_labels x feat_num x feat_dim
    pq = F.normalize(pq, dim=2)

    # first part: positive pair
    # -> batch_size
    pos_sim = (feats1 * feats2).sum(dim=1)

    # second part: negative pairs
    if hard_correlation:
        correlations[correlations < 1] = 0
        assert (
            correlations[targets] == 1
        ).sum() == batch_size, f"{(correlations[targets]==1).sum()}!=batch_size"

    # batch_size x num_labels -> batch_size x num_labels x 1 -> batch_size x num_labels x feat_num -> batch_size x (num_labels x feat_num)
    possibilities = (
        correlations[targets].unsqueeze(-1).repeat(1, 1, feat_num).view(batch_size, -1)
    )
    # num_labels x feat_num x feat_dim -> (num_labels x feat_num) x feat_dim -> feat_dim x (num_labels x feat_num)
    pq = pq.view(num_labels * feat_num, feat_dim).T

    # batch_size x feat_dim * feat_dim x (num_labels x feat_num) -> batch_size x (num_labels x feat_num)
    neg_sim = torch.mm(feats1, pq)

    sim = torch.zeros((batch_size, num_labels * feat_num + 1), device=feats1.device)
    sim[:, 0] = pos_sim
    sim[:, 1:] = neg_sim
    sim += 1
    weights = torch.zeros_like(sim)
    weights[:, 0] = 1
    weights[:, 1:] = 1 - possibilities.pow(2)
    sim = sim * weights.detach()
    sim /= tau

    t = F.cross_entropy(
        sim, torch.zeros(batch_size, dtype=torch.long, device=feats1.device)
    )
    return t
