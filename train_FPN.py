from dataset import FileLoader
import glob, os
from torch.utils.data import DataLoader
import torch
import segmentation_models_pytorch as smp
from model import SemanticSegmentationNet

def worker_init_fn(worker_id):
    # ! to make the seed chain reproducible, must use the torch random, not numpy
    # the torch rng from main thread will regenerate a base seed, which is then
    # copied into the dataloader each time it created (i.e start of each epoch)
    # then dataloader with this seed will spawn worker, now we reseed the worker
    worker_info = torch.utils.data.get_worker_info()
    # to make it more random, simply switch torch.randint to np.randint
    worker_seed = torch.randint(0, 2 ** 32, (1,))[0].cpu().item() + worker_id
    # print('Loader Worker %d Uses RNG Seed: %d' % (worker_id, worker_seed))
    # retrieve the dataset copied into this worker process
    # then set the random seed for each augmentation
    worker_info.dataset.setup_augmentor(worker_id, worker_seed)
    return

batch_size = {"train": 16, "valid": 16}

tr_file_list = glob.glob("../hover_net/dataset/training_data/consep/consep/train/540x540_164x164/*.npy")
ts_file_list = glob.glob("../hover_net/dataset/training_data/consep/consep/valid/540x540_164x164/*.npy")
tr_file_list.sort()
ts_file_list.sort()


train_data = FileLoader(tr_file_list,
                        input_shape=(256, 256),
                        mask_shape=(256, 256),
                        mode="train",
                        )

test_data = FileLoader(ts_file_list,
                       input_shape=(256, 256),
                       mask_shape=(256, 256),
                       mode="valid")

tr_loader = DataLoader(train_data,
                       num_workers=1,
                       batch_size=batch_size["train"],
                       shuffle=True,
                       drop_last=True,
                       worker_init_fn=worker_init_fn,
                       )

ts_loader = DataLoader(test_data,
                       num_workers=1,
                       batch_size=batch_size["valid"],
                       shuffle=False,
                       drop_last=False,
                       worker_init_fn=worker_init_fn,
                       )

model_base = smp.FPN('resnet50',
                     in_channels=3,
                     classes=1,
                     encoder_weights="imagenet",
                     decoder_merge_policy='cat',
                     )

model = SemanticSegmentationNet(model_base=model_base)

best_score = 0.0
for epoch in range(1000):
    model.train_on_loader(tr_loader)
    test_dict = model.test_on_loader(ts_loader)

    if test_dict['test_iou'] >= best_score:
        model_to_save = model.model_base.module if hasattr(model.model_base, "module") else model.model_base
        torch.save(model_to_save.state_dict(), "FPN.pth")



