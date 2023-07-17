import heapq
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchvision import transforms
import chinopie
from chinopie.modelhelper import HyperparameterManager, ModelStaff, PhaseHelper
from cam import (
    GradCamWrapper,
    crop_corner,
    generate_binary_mask,
    resize_mask,
)
from losses import (
    loss,
    loss_contrastive_infonce,
)
from models import Scob
from probes.average_precision_meter import AveragePrecisionMeter
from data.coco import COCO2014SinglePositive
from data.voc import Preprocess, VOC07Dataset, VOC12Dataset
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch
from typing import Any, Dict, List, Optional
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from chinopie import ModuleRecipe, TrainBootstrap
from chinopie.filehelper import InstanceFileHelper


def collate_fn(batch):
    indices = torch.tensor([b["index"] for b in batch])
    name = [b["name"] for b in batch]
    images = torch.cat([b["image"].unsqueeze(0) for b in batch])
    target = torch.cat([b["target"].unsqueeze(0) for b in batch])

    return {
        "indices": indices,
        "name": name,
        "image": images,
        "target": target,
    }


class IPT:
    def __init__(
        self, num_classes: int, max_feat_queue: int, feat_dim: int, device: str
    ) -> None:
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.max_feat_queue = max_feat_queue
        self.instance_queues = [[] for _ in range(num_classes)]
        self.instance_feats = torch.zeros(
            (num_classes, max_feat_queue, feat_dim), device=device, dtype=torch.float
        )
        self.instance_queue_idset = set()

    def update(
        self,
        label: int,
        confidence: float,
        image_id: int,
        mask: Tensor,
        feat: Tensor,
    ):
        packed = (
            (confidence, image_id),
            {
                "slot_id": -1,
                "image_id": image_id,
                "mask": mask,
                "feat": feat,
            },
        )
        if len(self.instance_queues[label]) >= self.max_feat_queue:
            if (
                self.instance_queues[label][0][0][0] < confidence
                and image_id not in self.instance_queue_idset
            ):
                old = heapq.heappushpop(self.instance_queues[label], packed)
                assert old[1]["slot_id"] != -1
                packed[1]["slot_id"] = old[1]["slot_id"]
                self.instance_queue_idset.remove(old[1]["image_id"])
                self.instance_queue_idset.add(image_id)
                self.instance_feats[label, packed[1]["slot_id"]] = feat
        else:
            heapq.heappush(self.instance_queues[label], packed)
            packed[1]["slot_id"] = len(self.instance_queues[label]) - 1
            self.instance_queue_idset.add(image_id)
            self.instance_feats[label, packed[1]["slot_id"]] = feat

    def get_most_confident_feats(self, topk: int):
        res = torch.zeros((self.num_classes, topk, self.feat_dim), dtype=torch.float)
        for label in range(self.num_classes):
            kth = heapq.nlargest(topk, self.instance_queues[label])
            for k, v in enumerate(kth):
                res[label, k] = v[1]["feat"]
        return res
    
    def clear(self):
        self.instance_queues = [[] for _ in range(self.num_classes)]
        self.instance_queue_idset.clear()
        self.instance_feats.fill_(0)


train_preprocess = Preprocess(
    [
        transforms.RandomHorizontalFlip(2),
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ],
    [0.5, 1, 1, 1],
    114514,
)
val_preprocess = Preprocess(
    [
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ],
    [1, 1, 1],
    114514,
)

hard_augment = Preprocess(
    [
        transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.255],
        ),
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
        transforms.RandomGrayscale(),
        transforms.GaussianBlur((3, 3), (1.0, 2.0)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ],
    [1, 0.3, 0.2, 0.2, 1],
)


def get_datasets(name: str, helper: InstanceFileHelper):
    if name == "voc12":
        return VOC12Dataset(
            helper.get_dataset_slot(name),
            train_preprocess,
            phase="train",
        ), VOC12Dataset(
            helper.get_dataset_slot(name),
            val_preprocess,
            phase="val",
        )
    elif name == "voc07":
        return VOC07Dataset(
            helper.get_dataset_slot(name),
            train_preprocess,
            phase="train",
        ), VOC07Dataset(
            helper.get_dataset_slot(name),
            val_preprocess,
            phase="val",
        )
    elif name == "coco2014":
        return COCO2014SinglePositive(
            helper.get_dataset_slot(name),
            train_preprocess,
            phase="train",
        ), COCO2014SinglePositive(
            helper.get_dataset_slot(name),
            val_preprocess,
            phase="val",
        )
    else:
        raise RuntimeError("unknown dataset")


class BasicRecipe(ModuleRecipe):
    model: Scob

    def __init__(self, dataset: str):
        super().__init__(clamp_grad=None)
        self.dataset = dataset
        if self.dataset in ["voc07", "voc12"]:
            self.expected_positive_labels = 1.5
        elif self.dataset in['coco2014']:
            self.expected_positive_labels = 3.9

    def prepare(
        self,
        hp: HyperparameterManager,
        staff: ModelStaff,
        inherited_states: Dict[str, Any],
    ):
        self.trainset, self.valset = get_datasets(self.dataset, staff.file)
        self.trainset.drop_to_single(114514)
        self.valset.debug_retain_all_labels()

        self.observed_labels = self.trainset.get_all_single_labels()
        self.num_labels = self.observed_labels.size(1)

        batch_size = hp.suggest_int("batch_size", 2, 128, log=True)
        trainloader = DataLoader(
            self.trainset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=True,
            num_workers=8,
        )
        valloader = DataLoader(
            self.valset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=8,
        )
        staff.register_dataset(self.trainset, trainloader, self.valset, valloader)

        model = Scob(self.num_labels, 1024, self.observed_labels)
        model.move_k(0)
        chinopie.freeze_model(model.fb)
        staff.reg_model(model)

        self.alpha = hp.suggest_float("alpha", 0.9, 0.99999, log=True)
        self.cl_temperature = hp.suggest_float("cl_temperature", 1e-2, 1e2, log=True)
        self.lambda_cl = hp.suggest_float("lambda_cl", 1e-2, 1e2, log=True)

    def set_optimizers(
        self,
        model: Scob,
        hp: HyperparameterManager,
        staff: ModelStaff,
    ) -> Optimizer:
        lr_linear = hp.suggest_float("lr_linear", 1e-4, 1e-1, log=True)
        lr_tf = hp.suggest_float("lr_tf", 1e-4, 1e-1, log=True)
        lr_pj = hp.suggest_float("lr_pj", 1e-4, 1e-1, log=True)
        lr_es = hp.suggest_float("lr_es", 1e-4, 1e-1, log=True)
        return torch.optim.Adam(
            [
                {
                    "tag": "SELFNET",
                    "params": torch.nn.ModuleList(
                        [model.fm1_14.linear_parts, model.fm1_28.linear_parts]
                    ).parameters(),
                    "lr": lr_linear,
                },
                {
                    "tag": "SELFNET",
                    "params": torch.nn.ModuleList(
                        [model.fm1_14.transformer_parts, model.fm1_28.transformer_parts]
                    ).parameters(),
                    "lr": lr_tf,
                },
                {
                    "tag": "SELFNET",
                    "params": model.projector.parameters(),
                    "lr": lr_pj,
                },
                {
                    "tag": "SELFNET",
                    "params": model.fh.parameters(),
                    "lr": lr_linear,
                },
                {
                    "tag": "SELFNET",
                    "params": model.g.parameters(),
                    "lr": lr_es,
                },
            ],
        )
    
    def set_scheduler(self, optimizer: Optimizer, hp: HyperparameterManager, staff: ModelStaff) -> LRScheduler | None:
        return torch.optim.lr_scheduler.StepLR(optimizer,step_size=16,gamma=0.1,verbose=True)

    def before_epoch(self):
        self.instance_manager = IPT(self.num_labels, 2, 1024, self._staff.dev)
        self.model.g.calculate_correlation()

        self.scores = {
            "train": AveragePrecisionMeter(False),
            "val": AveragePrecisionMeter(False),
        }

    def run_train_iter(self, data, p: PhaseHelper):
        self.model.move_k(self.alpha)

        dev_data: Any = chinopie.any_to(data, self.dev)
        output = self.forward(dev_data)
        loss = self.cal_loss(dev_data, output)
        p.update_loss(loss.detach().cpu())

        self.optimizer.zero_grad()
        loss.backward()
        if self._clamp_grad is not None:
            torch.nn.utils.clip_grad.clip_grad_norm_(
                self.model.parameters(), max_norm=self._clamp_grad
            )
        self.optimizer.step()
        self.update_probe(data, chinopie.any_to(output, "cpu"), p)
        self.after_iter(dev_data, output, "train")

        image = dev_data["image"]
        indices = dev_data["indices"]
        logits = output["logits"]
        # generate patch masks
        batch_size = image.size(0)
        with GradCamWrapper(self.model, True if self.dev == "cuda" else False) as gc:
            self.model.eval()
            self.optimizer.zero_grad()

            real_confidences, real_labels = dev_data["target"].max(dim=1)
            predicted_confidences = logits[range(batch_size), real_labels]

            images: Tensor = dev_data["image"]
            images.requires_grad_(True)
            masks: Tensor = torch.tensor(
                gc(
                    images,
                    [ClassifierOutputTarget(x) for x in real_labels.tolist()],
                ),
                device=self.dev,
            )
            patch_masks14 = crop_corner(resize_mask(masks, 14))
            patch_masks28 = crop_corner(resize_mask(masks, 28))
        # begin of contrastive learning
        binary_patch_masks14 = generate_binary_mask(patch_masks14)
        binary_patch_masks28 = generate_binary_mask(patch_masks28)
        self.model.train()
        momentum_feats = self.model.forward_k(
            image,
            image_masks14=~binary_patch_masks14,
            image_masks28=~binary_patch_masks28,
        )
        self.optimizer.zero_grad()
        cl_logits, cl_feats, cl_esti_labels = self.model(
            hard_augment.forward_batch(image),
            indices,
            image_masks14=~binary_patch_masks14,
            image_masks28=~binary_patch_masks28,
        )
        if binary_patch_masks14.any() and binary_patch_masks28.any():
            cl_loss_viewandneg = loss_contrastive_infonce(
                cl_feats,
                momentum_feats.detach(),
                real_labels.detach(),
                correlations=self.model.g.get_correlation_matrix().detach(),
                pq=self.instance_manager.get_most_confident_feats(1).to(self.dev),
                tau=self.cl_temperature,
                hard_correlation=True,
            )
            cl_loss = cl_loss_viewandneg * self.lambda_cl

            if p.validate_loss(cl_loss, False):
                cl_loss.backward()
                self.optimizer.step()
        # end of contrastive learning

        for i in range(batch_size):
            id = data["indices"][i].item()
            label = real_labels[i]
            # update most confident masks
            if momentum_feats[i].isnan().any() == False:
                self.instance_manager.update(
                    label,
                    predicted_confidences[i].item(),
                    id,
                    patch_masks28[i].cpu().detach(),
                    momentum_feats[i].detach(),
                )

    def forward_val(self, data) -> Any:
        image = data["image"]
        indices = torch.zeros_like(data["indices"])
        logits, feats, estimated_labels = self.model(image, indices)

        return {"logits": logits, "feats": feats, "estimated_labels": estimated_labels}

    def forward(self, data) -> Any:
        image = data["image"]
        indices = data["indices"]
        logits, feats, estimated_labels = self.model(image, indices)

        return {"logits": logits, "feats": feats, "estimated_labels": estimated_labels}

    def cal_loss(self, data, output) -> Tensor:
        _loss, n = loss(
            pred_targets=output["logits"],
            esti_targets=output["estimated_labels"],
            observed_targets=data["target"],
            expected_num_pos_labels=self.expected_positive_labels,
        )
        return _loss

    def after_iter(self, data, output, phase: str):
        dataset = self.trainset if phase == "train" else self.valset
        self.scores[phase].add(
            output["logits"].data,
            dataset.get_real_labels_by_id(data["indices"]),
            data["name"],
        )

    def report_score(self, phase: str) -> float:
        return self.scores[phase].value().mean().item()  # type: ignore


if __name__ == "__main__":
    dataset = chinopie.get_env("dataset")
    tb = TrainBootstrap(
        "deps", 20, load_checkpoint=True, save_checkpoint=True, comment="a", dev="cuda"
    )

    tb.hp.reg_int("batch_size", 8)
    tb.hp.reg_float("cl_temperature", 1.0)
    tb.hp.reg_float("alpha", 0.999)
    tb.hp.reg_float("lambda_cl", 0.1)
    tb.hp.reg_float("lr_linear", 1e-3)
    tb.hp.reg_float("lr_tf", 4e-5)
    tb.hp.reg_float("lr_pj", 1e-2)
    tb.hp.reg_float("lr_es", 1e-2)
    tb.optimize(BasicRecipe(dataset), direction="maximize", inf_score=0, n_trials=1)

    tb.release()
