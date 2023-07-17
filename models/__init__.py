from typing import Any, List, Optional, Union
from torch.functional import Tensor
import torchvision
import numpy as np
from torch import nn
import torch
from .pos_embed import build_position_encoding
from .transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
)
from torch.nn import functional as F


class ImageBackbone(nn.Module):
    def __init__(self, freeze: bool) -> None:
        super().__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: Tensor):
        x = self.backbone.conv1(x)  # 224
        x = self.backbone.bn1(x)  # 224
        x = self.backbone.relu(x)  # 224
        x = self.backbone.maxpool(x)  # 112

        x = self.backbone.layer1(x)  # 112
        x = self.backbone.layer2(x)  # 56
        x28 = self.backbone.layer3(x)  # 28
        x14 = self.backbone.layer4(x28)  # 14

        return x14, x28


class ImageMid(nn.Module):
    def __init__(self, in_feat_dim: int, feat_dim: int) -> None:
        super().__init__()

        self.feat_dim = feat_dim

        # create transformer
        encoder_layer = TransformerEncoderLayer(
            feat_dim, 8, activation=F.leaky_relu, batch_first=True # type: ignore
        )
        encoder_norm = nn.LayerNorm(feat_dim)
        self.encoder = TransformerEncoder(encoder_layer, 2, encoder_norm)

        self.feat_conv = nn.Conv2d(in_feat_dim, feat_dim, (1, 1))
        self.pos_encoder = build_position_encoding(feat_dim, "learned")

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear_parts = nn.ModuleList(
            [self.feat_conv, self.pos_encoder, self.avgpool]
        )
        self.transformer_parts = self.encoder

    def forward(self, x: Tensor, image_masks: Optional[Tensor] = None) -> Tensor:
        batch_size = x.size(0)
        patch_size = x.size(2)

        x = self.feat_conv(x)
        x = x + self.pos_encoder(x)
        x = x.permute(0, 2, 3, 1).flatten(1, 2)  # batch x (? x ?) x feat_dim
        if image_masks != None:
            image_masks = image_masks.flatten(1, 2)

        feat: Tensor = self.encoder(
            x, src_key_padding_mask=image_masks
        )  # batch x (? x ?) x feat_dim

        # batch x feat_dim x (? x ?) -> batch x feat_dim x ? x ?
        feat = feat.permute(0, 2, 1).view(
            batch_size, self.feat_dim, patch_size, patch_size
        )
        feat = self.avgpool(feat)
        feat = feat.flatten(1)

        return feat


class FeatProjector(nn.Module):
    def __init__(self, feat_dim: int) -> None:
        super().__init__()

        self.projector1 = nn.Linear(feat_dim, feat_dim)
        self.relu = nn.LeakyReLU()
        self.projector2 = nn.Linear(feat_dim, feat_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.projector1(x)
        x = self.relu(x)
        x = self.projector2(x)
        return x


class ImageClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.classifier = nn.Linear(feat_dim, num_classes)

    def _gen_full_connected_adj(self, correlation: Tensor) -> Tensor:
        return correlation

    def forward(self, feat: Tensor):
        # batch x class
        logits = self.classifier(feat)
        res = torch.sigmoid(logits)
        return res


def inverse_sigmoid(x):
    EPS = 1e-5
    # clip
    x = min(x, 1 - EPS)
    x = max(x, EPS)
    return np.log(x / (1 - x))


class LabelEstimator(nn.Module):
    correlation_cache: Tensor

    def __init__(
        self,
        num_classes: int,
        observed_label_matrix: Tensor,
        raw_matrix: Union[Tensor, None] = None,
    ):
        super().__init__()
        self.num_classes = num_classes

        num_data = observed_label_matrix.size(0)
        observed_matrix_np = np.array(observed_label_matrix)

        params_matrix = torch.zeros(num_data, num_classes)
        torch.nn.init.uniform_(
            params_matrix, inverse_sigmoid(0.2), inverse_sigmoid(0.8)
        )
        if raw_matrix:
            params_matrix = raw_matrix

        idx_pos = torch.from_numpy((observed_matrix_np == 1).astype(bool))
        params_matrix[idx_pos] = inverse_sigmoid(0.995)
        idx_neg = torch.from_numpy((observed_matrix_np == -1).astype(bool))
        params_matrix[idx_neg] = inverse_sigmoid(0.005)

        self.logits = nn.parameter.Parameter(params_matrix.cuda())

    @torch.no_grad()
    def calculate_correlation(self) -> Tensor:
        x = torch.sigmoid(self.logits.detach())
        res = torch.zeros((self.num_classes, self.num_classes), device=x.device)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                res[i, j] = (x[:, i] * x[:, j]).mean() / x[:, i].mean()
                if i == j:
                    res[i, j] = 1.0
        self.correlation_cache = res
        return res

    @torch.no_grad()
    def get_correlation_matrix(self) -> Tensor:
        return self.correlation_cache

    def forward(self, indices):
        x = self.logits[indices, :]
        x = torch.sigmoid(x)
        return x


class Scob(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        observed_matrix: Tensor,
        raw_matrix: Optional[Tensor] = None,
    ):
        super().__init__()

        assert feat_dim // 2 * 2 == feat_dim

        self.fb = ImageBackbone(True)
        self.fm1_14 = ImageMid(2048, feat_dim // 2)
        self.fm1_28 = ImageMid(1024, feat_dim // 2)

        self.fm2_14 = ImageMid(2048, feat_dim // 2)
        self.fm2_28 = ImageMid(1024, feat_dim // 2)

        self.projector = FeatProjector(feat_dim)
        self.fh = ImageClassifier(num_classes=num_classes, feat_dim=feat_dim)

        self.g = LabelEstimator(num_classes, observed_matrix, raw_matrix)

    def forward(
        self,
        x: Tensor,
        idx: Tensor,
        image_masks14: Optional[Tensor] = None,
        image_masks28: Optional[Tensor] = None,
    ):
        """
        image_masks: If a BoolTensor is provided, the positions with the value of ``True``
        will be ignored while the position with the value of ``False`` will be unchanged.
        """
        low_feats14, low_feats28 = self.fb(x)
        high_feats14 = self.fm1_14(low_feats14, image_masks14)
        high_feats28 = self.fm1_28(low_feats28, image_masks28)
        high_feats = torch.cat([high_feats14, high_feats28], dim=1)
        logits = self.fh(high_feats)

        projected_feats = self.projector(high_feats)
        estimated_labels = self.g(idx)
        return logits, projected_feats, estimated_labels

    @torch.no_grad()
    def forward_k(
        self,
        x: Tensor,
        image_masks14: Optional[Tensor] = None,
        image_masks28: Optional[Tensor] = None,
    ) -> Tensor:
        x14, x28 = self.fb(x)
        x14 = self.fm2_14(x14, image_masks14)
        x28 = self.fm2_28(x28, image_masks28)

        return torch.cat([x14, x28], dim=1)

    @torch.no_grad()
    def move_k(self, alpha: float):
        for fast, slow in zip(self.fm1_14.parameters(), self.fm2_14.parameters()):
            slow = alpha * slow + (1 - alpha) * fast
        for fast, slow in zip(self.fm1_28.parameters(), self.fm2_28.parameters()):
            slow = alpha * slow + (1 - alpha) * fast
