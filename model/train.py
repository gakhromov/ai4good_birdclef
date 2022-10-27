from config import config
from helpers import dotdict
from models import loss_ce, get_model, loss_bcefocal
from dataset import get_dataset
from tqdm import tqdm
import torch
import gc
from torch.optim import Adam
from sklearn.metrics import precision_recall_fscore_support
import wandb
import argparse
import numpy as np

def train(model, data_loader, optimizer, scheduler, conf, epoch):
    device = conf.device
    loss_fn = loss_bcefocal if conf.use_secondary else loss_ce
    model.to(device)
    model.train()
    running_loss = 0

    prec= 0
    rec = 0
    f1 = 0
    loop = tqdm(data_loader, position=0)

    for i, (mels, labels) in enumerate(loop):
        model.zero_grad()
        mels = mels.to(device)
        labels = labels.to(device)

        outputs = model(mels)

        # calculate sigmoid for multilabel
        if conf.use_secondary:
            preds = torch.sigmoid(outputs)
        else:
            _, preds = torch.max(outputs, 1)

        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()
        

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()

        if conf.use_secondary:
            p,r, f, _ = precision_recall_fscore_support(
                average='micro',
                y_pred = preds.view(-1).detach().cpu().numpy() > 0.5,
                y_true=labels.view(-1).detach().cpu().numpy(),
                zero_division=0)
        else:
            p,r, f, _ = precision_recall_fscore_support(
                average='macro',
                y_pred = preds.view(-1).detach().cpu().numpy(),
                y_true=labels.view(-1).detach().cpu().numpy(),
                zero_division=0)
        prec += p
        rec += r
        f1 += f

        loop.set_description(f"Train Epoch [{epoch + 1}/{conf.epochs}")
        loop.set_postfix(loss=loss.item(), f1=f, precision=p, recall=r)

    lendl = len(data_loader)
    running_loss /= lendl
    f1 /= lendl
    prec /= lendl
    rec /= lendl

    return running_loss, prec, rec, f1


def valid(model, data_loader, conf, epoch):
    model.eval()
    loss_fn = loss_bcefocal if conf.use_secondary else loss_ce
    device = conf.device
    running_loss = 0
    f1 = 0
    prec = 0
    rec = 0

    loop = tqdm(data_loader, position=0)
    for mels, labels in loop:
        mels = mels.to(device)
        labels = labels.to(device)

        outputs = model(mels)
        # calculate sigmoid for multilabel
        if conf.use_secondary:
            preds = torch.sigmoid(outputs)
        else:
            _, preds = torch.max(outputs, 1)

        loss = loss_fn(outputs, labels)

        running_loss += loss.item()

        if conf.use_secondary:
            p,r, f, _ = precision_recall_fscore_support(
                average='micro',
                y_pred = preds.view(-1).detach().cpu().numpy() > 0.5,
                y_true=labels.view(-1).detach().cpu().numpy(),
                zero_division=0)
        else:
            p,r, f, _ = precision_recall_fscore_support(
                average='macro',
                y_pred = preds.view(-1).detach().cpu().numpy(),
                y_true=labels.view(-1).detach().cpu().numpy(),
                zero_division=0)
        
        prec += p
        rec += r
        f1 += f
        loop.set_description(f"Valid Epoch [{epoch + 1}/{conf.epochs}")
        loop.set_postfix(loss=loss.item(), f1=f, precision=p, recall=r)

    lendl = len(data_loader)
    running_loss /= lendl
    f1 /= lendl
    prec /= lendl
    rec /= lendl

    return running_loss, prec, rec, f1


def run(data, fold, args):
    train_loader, valid_loader = data

    model = get_model("cnn")
    wandb.watch(model)

    cfg = dotdict(config)

    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=10)

    best_valid_f1 = 0

    for epoch in range(cfg.epochs):
        train_loss, train_prec, train_rec, train_f1 = train(model, train_loader, optimizer, scheduler, cfg, epoch)
        valid_loss, valid_prec, valid_rec, valid_f1 = valid(model, valid_loader, cfg, epoch)

        wandb.log({
            "train_loss": train_loss,
            "train_f1": train_f1,
            "train_recall": train_rec,
            "train_precision": train_prec,
            "valid_loss": valid_loss,
            "valid_precision": valid_prec,
            "valid_recall": valid_rec,
            "valid_f1": valid_f1
            })

        if valid_f1 > best_valid_f1:
            print(f"Validation F1 Improved - {best_valid_f1} ---> {valid_f1}")
            torch.save(model.state_dict(), f'./model_{fold}.bin')
            print(f"Saved model checkpoint at ./model_{fold}.bin")
            best_valid_f1 = valid_f1

    return best_valid_f1


def main():
    parser = argparse.ArgumentParser(description="Run the training pipeline")
    parser.add_argument("--data_path", type=str, default="../datasets/numpy_mel", help="Location of the metadata csv")
    parser.add_argument("--data_folder", type=str, default="../datasets/numpy_mel/data", help="Location of the individual bird folders")
    parser.add_argument("--use_secondary", type=bool, default=False, help="Use the secondary label")
    parser.add_argument("--dtype", type=str, default="mel")
    args = parser.parse_args()
    print("Running training with following args: \n", args)
    # enable benchmark mode for more power
    torch.backends.cudnn.benchmark = True

    # setup wandb

    print(config)
    wandb.init(
        project="birdclef",
        entity="matvogel",
        config=config)

    # get dataset
    dataset = get_dataset(args)

    # train n_folds models
    for fold in range(config['n_folds']):
        print("=" * 30)
        print("Training Fold - ", fold)
        print("=" * 30)
        best_valid_f1 = run(dataset[fold], fold, args)
        print(f'Best F1 Score: {best_valid_f1:.5f}')

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
