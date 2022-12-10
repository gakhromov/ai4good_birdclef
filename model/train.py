from config import config, wandb_key, cnn_conf
from helpers import DotDict, fetch_scheduler, varying_threshold_metrics
from models import get_model
from dataset import get_dataset
from tqdm import tqdm
import torch
import gc
from torch.optim import AdamW, Adam, SGD
import wandb
import argparse
from contextlib import nullcontext
import numpy as np
from torchvision.ops import sigmoid_focal_loss
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
DEBUG = False

def do_epoch(train, model, data_loader, optimizer, scheduler, scaler, conf, epoch):
    device = conf.device
    if conf.use_secondary:
        loss_focal = sigmoid_focal_loss
        loss_bce = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05, reduction='none')

    if train:
        model.train()
        prefix = "Train"
        context = nullcontext()
    else:
        model.eval()
        prefix = "Valid"
        context = torch.no_grad()

    predictions = torch.tensor([], device='cpu')
    targets = torch.tensor([], device='cpu')
    running_loss = 0
    iters = len(data_loader)
    loop = tqdm(data_loader, position=0)
    with context:
        for i, (data, labels) in enumerate(loop):
            if train: model.zero_grad()
            mels = data['mels'].to(device=device, non_blocking=True, dtype=torch.float)
            labels = labels.to(device=device, non_blocking=True)
            score = data['score'].to(device=device)

            with torch.cuda.amp.autocast():
                if conf.use_slices:
                    # swap so we have B 1 H W
                    mels = torch.swapaxes(mels, 0, 1)
                    # create slices
                    slices = mels.split(8)
                    # concatenate the slices in the output vector
                    outputs = torch.tensor([], device=device)

                    for i, s in enumerate(slices):
                        if conf.ast:
                            s = s.squeeze(1) # squeeze out the channel dimension for ast model
                        s = s.to(device=device, non_blocking=True)
                        pred = model(s)
                        outputs = torch.cat((outputs, pred), dim=0)

                    # combine the outputs together
                    outputs = torch.mean(outputs, dim=0, keepdim=True)
                    del slices
                else:
                    if not conf.ast:
                        mels = torch.unsqueeze(mels, 1)
                    outputs = model(mels)
                # scale to logspace for KL
                if config['mixup'] and conf.use_secondary:
                    # do bce loss plus focal loss
                    loss = loss_bce(outputs, labels) + loss_focal(outputs, labels, alpha=0.25, gamma=2, reduction='mean')
                else:
                    loss_scaled =  loss_fn(outputs, labels) * score / 5
                    loss = torch.mean(loss_scaled, dim=0)

            # calculate sigmoid for multilabel
            if conf.use_secondary:
                preds = torch.sigmoid(outputs)
            else:
                _, preds = torch.max(outputs, 1)

            # plot the last result in a good way TODO change to last
            if i == 0:
                #log_imgs = [wandb.Image(img) for img in mels.detach().cpu().numpy()]
                log_labels = [label for label in labels.detach().cpu().numpy()]
                log_preds = [pred for pred in preds.detach().cpu().numpy()]
                log_preds_thresh = [np.array(pred > 0.2, dtype=np.int8) for pred in preds.detach().cpu().numpy()]
                print(f"{prefix} Labels: ", log_labels[0])
                print(f"{prefix} Preds: ", log_preds[0])
                print(f"{prefix} Preds threshold: ", log_preds_thresh[0])

            # do optimizer and scheduler steps
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step(epoch + i / iters)
                del outputs, mels
                #torch.cuda.empty_cache()

            running_loss += loss.item()

            # append for metrics in validation
            predictions = torch.cat((predictions, preds.detach().cpu()), dim=0)
            targets = torch.cat((targets, labels.detach().cpu()), dim=0)
            loop.set_description(f"{prefix} Epoch [{epoch + 1}/{conf.epochs}")
            loop.set_postfix(loss=loss.item())
            if DEBUG:
                break
    
    # calculate metrics for varying thresholds
    running_loss /= len(data_loader)
    metrics = varying_threshold_metrics(predictions, targets)
    results = pd.DataFrame(data=metrics, columns=["threshold", "precision", "recall", "f1 score"])
    wandb.Table(data=results)
    wandb.log({"Results": results}, step=epoch)
    wandb.log({f"{prefix} loss": running_loss}, step=epoch)

    best_f1 = results['f1 score'].max()
    return best_f1


def run(data, fold, args):
    train_loader, valid_loader = data
    cfg = DotDict(config)

    if cfg.ast:
        model = get_model("ast").to(cfg.device)
    else:
        model = get_model("cnn").to(cfg.device)

    wandb.watch(model)

    optimizer = Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = fetch_scheduler(optimizer, "CosineAnnealingWarmRestarts")
    scaler = torch.cuda.amp.GradScaler()

    best_valid_f1 = 0

    for epoch in range(cfg.epochs):
        do_epoch(True, model, train_loader, optimizer, scheduler, scaler, cfg, epoch)
        val_f1 = do_epoch(False, model, valid_loader, optimizer, scheduler, scaler, cfg, epoch)

        if val_f1 > best_valid_f1:
            print(f"Validation F1 Improved - {best_valid_f1} ---> {val_f1}")
            torch.save(model.state_dict(), f'./model_{fold}_{wandb.run.name}.bin')
            print(f"Saved model checkpoint at ./model_{fold}_{wandb.run.name}.bin")
            best_valid_f1 = val_f1

    return best_valid_f1


def main():
    parser = argparse.ArgumentParser(description="Run the training pipeline")
    parser.add_argument("--data_path", type=str, default="../datasets/numpy_mel", help="Location of the metadata csv")
    parser.add_argument("--data_folder", type=str, default="../datasets/numpy_mel/data",
                        help="Location of the individual bird folders")
    parser.add_argument("--pretrain", type=bool, default=False)

    args = parser.parse_args()

    # enable benchmark mode for more power
    torch.backends.cudnn.benchmark = True

    # setup wandb
    args.use_secondary = config['use_secondary']
    args.mixup = config["mixup"]

    wandb.login(key=wandb_key)
    wandb.init(
        project="ai4good",
        entity="lessgoo",
        config={**config, **cnn_conf}
    )

    # get dataset
    dataset = get_dataset(args)

    # train n_folds models
    for fold in range(config['n_folds']):
        # only do the first fold
        if not fold == 0:
            continue
        torch.cuda.empty_cache()
        print("=" * 30)
        print("Training Fold - ", fold)
        print("=" * 30)
        best_valid_f1 = run(dataset[fold], fold, args)
        print(f'Best F1 Score: {best_valid_f1:.5f}')

        gc.collect()
        torch.cuda.empty_cache()
        


if __name__ == "__main__":
    main()
