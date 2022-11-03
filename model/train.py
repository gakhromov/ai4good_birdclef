from config import config, wandb_key
from helpers import DotDict, fetch_scheduler
from models import loss_ce, get_model, loss_bcefocal, loss_bce
from dataset import get_dataset
from tqdm import tqdm
import torch
import gc
from torch.optim import Adam, AdamW
from sklearn.metrics import precision_recall_fscore_support
import wandb
import argparse
from contextlib import nullcontext


def do_epoch(train, model, data_loader, optimizer, scheduler, scaler, conf, epoch):
    device = conf.device
    loss_fn = loss_bcefocal if conf.use_secondary else loss_ce

    if train:
        model.train()
        prefix = "Train"
        context = nullcontext()
    else:
        model.eval()
        prefix = "Valid"
        context = torch.no_grad()

    predictions = torch.tensor([], device=device)
    targets = torch.tensor([], device=device)
    running_loss = 0;

    loop = tqdm(data_loader, position=0)
    with context:
        for i, (mels, labels) in enumerate(loop):
            if train: model.zero_grad()

            mels = mels.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(mels)
                loss = loss_fn(outputs, labels)

            # calculate sigmoid for multilabel
            if conf.use_secondary:
                preds = torch.sigmoid(outputs)
            else:
                _, preds = torch.max(outputs, 1)

            # do optimizer and scheduler steps
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

            running_loss += loss.item()

            if conf.use_secondary:
                predictions = torch.cat((predictions, preds.view(-1) > 0.5), dim=0)
                targets = torch.cat((targets, labels.view(-1)), dim=0)
            else:
                predictions = torch.cat((predictions, preds.view(-1)), dim=0)
                targets = torch.cat((targets, labels.view(-1)), dim=0)

            loop.set_description(f"{prefix} Epoch [{epoch + 1}/{conf.epochs}")
            loop.set_postfix(loss=loss.item())

        # test last output
        #test_pred = torch.tensor(torch.sigmoid(model(mels)[0]) > 0.5).type(torch.int8)
        #print(test_pred, labels[0], sep='\n\n')

    # calculate metrics
    if conf.use_secondary:
        pre, rec, f, _ = precision_recall_fscore_support(
            average='macro',
            y_pred=predictions.detach().cpu().numpy(),
            y_true=targets.detach().cpu().numpy(),
            zero_division=0)
    else:
        pre, rec, f, _ = precision_recall_fscore_support(
            average='macro',
            y_pred=predictions.detach().cpu().numpy(),
            y_true=targets.detach().cpu().numpy(),
            zero_division=0)

    running_loss /= len(data_loader)

    return running_loss, pre, rec, f


def run(data, fold, args):
    train_loader, valid_loader = data

    model = get_model("cnn").to(config['device'])
    wandb.watch(model)

    cfg = DotDict(config)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = fetch_scheduler(optimizer, "OneCycle", spe=len(train_loader), epochs=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_valid_f1 = 0

    for epoch in range(cfg.epochs):
        loss, prec, rec, f1 = do_epoch(True, model, train_loader, optimizer, scheduler, scaler, cfg, epoch)
        val_loss, val_prec, val_rec, val_f1 = do_epoch(False, model, valid_loader, optimizer, scheduler, scaler, cfg,
                                                       epoch)

        wandb.log({
            "train_loss": loss,
            "train_f1": f1,
            "train_recall": rec,
            "train_precision": prec,
            "valid_loss": val_loss,
            "valid_precision": val_prec,
            "valid_recall": val_rec,
            "valid_f1": val_f1
        })

        if val_f1 > best_valid_f1:
            print(f"Validation F1 Improved - {best_valid_f1} ---> {val_f1}")
            torch.save(model.state_dict(), f'./model_{fold}.bin')
            print(f"Saved model checkpoint at ./model_{fold}.bin")
            best_valid_f1 = val_f1

    return best_valid_f1


def main():
    parser = argparse.ArgumentParser(description="Run the training pipeline")
    parser.add_argument("--data_path", type=str, default="../datasets/numpy_mel", help="Location of the metadata csv")
    parser.add_argument("--data_folder", type=str, default="../datasets/numpy_mel/data",
                        help="Location of the individual bird folders")
    parser.add_argument("--dtype", type=str, default="mel")
    args = parser.parse_args()
    print("Running training with following args: \n", args)

    # enable benchmark mode for more power
    torch.backends.cudnn.benchmark = True

    # setup wandb
    args.use_secondary = config['use_secondary']
    print(config)
    wandb.login(key=wandb_key)
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
