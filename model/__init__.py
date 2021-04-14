import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import tqdm
import matplotlib.pyplot as plt
import os


class Meter:
    def __init__(self):
        self.n_sum = 0.
        self.n_counts = 0.

    def add(self, n_sum, n_counts):
        self.n_sum += n_sum
        self.n_counts += n_counts

    def get_avg_score(self):
        return self.n_sum / self.n_counts


class SemanticSegmentationNet(torch.nn.Module):
    def __init__(self, model_base=None, optimizer=None, optimizer_args=None, scheduler=None, scheduler_args=None, criterion=None):
        super().__init__()
        self.model_base = model_base
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model_base = nn.DataParallel(self.model_base)

        self.model_base.to(self.device)

        self.criterion = criterion if criterion is not None else nn.BCELoss()

        if optimizer is None:
            self.opt = torch.optim.Adam(
                self.model_base.parameters(), lr=0.0001, betas=(0.99, 0.999), weight_decay=0.0005)
        else:
            self.opt = optimizer(self.model_base.parameters(), **optimizer_args)

        if scheduler is None:
            self.scheduler = ReduceLROnPlateau(self.opt, mode='min', factor=0.5, patience=5, verbose=True,
                                               threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=5e-7,
                                               eps=1e-08)
        else:
            self.scheduler = scheduler(self.opt, **scheduler_args)

    def train_on_loader(self, train_loader):
        self.train()
        n_batches = len(train_loader)
        train_meter = Meter()

        pbar = tqdm.tqdm(total=n_batches)
        for batch in train_loader:
            score_dict = self.train_on_batch(batch)
            train_meter.add(score_dict['train_loss'], 1)

            pbar.set_description("Training Loss: %.4f" % train_meter.get_avg_score())
            pbar.update(1)

        self.scheduler.step(train_meter.get_avg_score())
        pbar.close()

        return {'train_loss': train_meter.get_avg_score()}

    @torch.no_grad()
    def val_on_loader(self, val_loader, savedir_images=None, n_images=3):
        self.eval()

        n_batches = len(val_loader)
        val_meter = Meter()
        pbar = tqdm.tqdm(total=n_batches)
        for i, batch in enumerate(tqdm.tqdm(val_loader)):
            score_dict = self.val_on_batch(batch)
            val_meter.add(score_dict['valloss'], batch['images'].shape[0])

            # pbar.update(1)

            if savedir_images and i < n_images:
                os.makedirs(savedir_images, exist_ok=True)
                self.vis_on_batch(batch, savedir_image=os.path.join(
                    savedir_images, "%d.jpg" % i))

                pbar.set_description("Validating. MAE: %.4f" % val_meter.get_avg_score())
                pbar.update(1)

        pbar.close()
        val_mae = val_meter.get_avg_score()
        val_dict = {'val_mae': val_mae, 'val_score': val_mae}
        return val_dict

    @torch.no_grad()
    def test_on_loader(self, test_loader):
        self.eval()

        n_batches = len(test_loader)
        test_meter = Meter()
        pbar = tqdm.tqdm(total=n_batches)
        for i, batch in enumerate(tqdm.tqdm(test_loader)):
            score_dict = self.test_on_batch(batch)
            test_meter.add(score_dict['testloss'], 1)

            pbar.set_description("Testing. iou: %.4f" % test_meter.get_avg_score())
            pbar.update(1)

        pbar.close()
        test_iou = test_meter.get_avg_score()
        test_dict = {'test_iou': test_iou, 'test_score': -test_iou}
        return test_dict

    def train_on_batch(self, batch, **extras):
        self.opt.zero_grad()
        self.model_base.train()

        images = batch["img"].float().to(self.device)
        masks = batch["mask"][:, None].float().to(self.device)
        logits = self.model_base.forward(images)

        loss = self.criterion(logits.sigmoid(), masks)

        loss.backward()

        self.opt.step()

        return {"train_loss": loss.item()}

    def get_state_dict(self):
        model_to_get = self.model_base.module if hasattr(self.model_base, 'module') else self.model_base
        state_dict = {"model": model_to_get.state_dict(),
                      "opt": self.opt.state_dict()}

        return state_dict

    def load_state_dict(self, state_dict):
        self.model_base.load_state_dict(state_dict["model"], map_location=self.device)
        self.opt.load_state_dict(state_dict["opt"])

    def val_on_batch(self, batch):
        self.eval()
        images = batch["img"].to(self.device)
        mask = batch["mask"].to(self.device)
        logits = self.model_base.forward(images)
        prob = logits.sigmoid()
        val_loss = self.iou_pytorch(prob, mask)

        return {'valloss': val_loss.item()}

    def test_on_batch(self, batch):
        self.eval()
        images = batch["img"].float().to(self.device)
        masks = batch["mask"][:, None].float().to(self.device)
        logits = self.model_base.forward(images)
        prob = logits.sigmoid()
        test_loss = self.iou_pytorch(prob, masks)

        return {"testloss": test_loss.item()}

    def vis_on_batch(self, batch, savedir_image):
        self.eval()
        images = batch["img"].to(self.device)
        mask = batch["mask"].to(self.device)
        logits = self.model_base.forward(images)
        prob = logits.sigmoid()
        seg = torch.argmax(prob, dim=1)
        #         import pdb
        #         pdb.set_trace()
        fig, axes = plt.subplots(1, 3, figsize=(30, 10))
        axes[0].imshow(images[0].detach().cpu().numpy().transpose(1, 2, 0))
        axes[1].imshow(mask[0].detach().cpu().numpy(), vmax=7, vmin=0)
        axes[2].imshow(seg[0].detach().cpu().numpy(), vmax=7, vmin=0)
        for ax in axes:
            ax.axis('off')
        fig.savefig(savedir_image)
        plt.close()

    def iou_pytorch(self, outputs: torch.Tensor, labels: torch.Tensor):
        smooth = 1e-6
        outputs = outputs[:,0]>0.5
        labels = labels.squeeze(1).round() if labels.dtype is not torch.bool else labels
        iou = 0.0
        outputs_cls = outputs.bool()
        labels_cls = labels.bool()
        intersection = (outputs_cls & labels_cls).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
        union = (outputs_cls | labels_cls).float().sum((1, 2))  # Will be zzero if both are 0

        iou += (intersection + smooth) / (union + smooth)  # We smooth our devision to avoid 0/0

        return torch.mean(iou)
