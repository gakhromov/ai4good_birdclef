from config import config, wandb_key, cnn_conf
from helpers import DotDict, fetch_scheduler, varying_threshold_metrics, varying_threshold_metrics_sklearn, calculate_metrics_sklearn
from models import get_model
from dataset import get_dataset, get_dataset_pretrain
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
from dataset import onehot_primary
from sklearn.metrics import precision_recall_fscore_support

warnings.filterwarnings("ignore")
DEBUG = False
SKLEARN_METRICS = False

def do_epoch(train, model, data_loader, optimizer, scheduler, scaler, conf, epoch):
    device = conf.device
    loss_bce = torch.nn.BCEWithLogitsLoss()
    loss_ce = torch.nn.CrossEntropyLoss(label_smoothing=1e-3, reduction='none')

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
    running_loss = 0
    iters = len(data_loader)
    loop = tqdm(data_loader, position=0)

    with context:
        for i, (data, labels) in enumerate(loop):
            if train: model.zero_grad()
            mels = data['mels'].to(device=device, non_blocking=True, dtype=torch.float)

            # extract the labels in case of mixup
            if conf.mixup:
                #labels = torch.stack(labels)
                label1 = labels[0].to(device=device, non_blocking=True)
                label2 = labels[1].to(device=device, non_blocking=True)
                r = data['rval'].to(device=device, non_blocking=True)
            else:
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
                if conf.use_secondary:
                    # do bce loss plus focal loss
                    loss = loss_bce(outputs, labels)
                elif conf.mixup:
                    loss = r * loss_ce(outputs, label1) + (1-r) * loss_ce(outputs, label2)
                    loss = torch.mean(loss, dim=0)
                else:
                    loss_scaled =  loss_ce(outputs, labels) * score / 5
                    loss = torch.mean(loss_scaled, dim=0)

            # calculate sigmoid for multilabel
            if conf.use_secondary:
                preds = torch.sigmoid(outputs)
            else:
                _, preds = torch.max(outputs, -1)

            # plot the last result in a good way TODO change to last
            if i == 0 and DEBUG:
                try:
                    #log_imgs = [wandb.Image(img) for img in mels.detach().cpu().numpy()]
                    log_labels = [label for label in labels.detach().cpu().numpy()]
                    log_preds = [pred for pred in preds.detach().cpu().numpy()]
                    log_preds_thresh = [np.array(pred > 0.2, dtype=np.int8) for pred in preds.detach().cpu().numpy()]
                    #log_imgs = [wandb.Image(img) for img in mels.detach().cpu().numpy()]
                    log_labels = [label for label in labels.detach().cpu().numpy()]
                    log_preds = [pred for pred in preds.detach().cpu().numpy()]
                    log_preds_thresh = [np.array(pred > 0.2, dtype=np.int8) for pred in preds.detach().cpu().numpy()]
                    print(f"{prefix} Labels: ", log_labels[0])
                    print(f"{prefix} Preds: ", log_preds[0])
                    print(f"{prefix} Preds threshold: ", log_preds_thresh[0])
                    # find the activations where there should be a positive prediction
                    print("Activations at TP: ", log_preds[0][log_labels[0] == 1])
                except:
                    print("Couldn't print DEBUG infos")

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
            predictions = torch.cat((predictions, preds.detach()), dim=0)
            if conf.mixup:
                labels = torch.where(r>(1-r), label1, label2)
                targets = torch.cat((targets, labels.detach()), dim=0)
            else:
                targets = torch.cat((targets, labels.detach()), dim=0)
            loop.set_description(f"{prefix} Epoch [{epoch}]")
            loop.set_postfix(loss=loss.item())

            if DEBUG:
                break
    
    # calculate metrics for varying thresholds
    if SKLEARN_METRICS or conf.pretrain or conf.mixup:
        p, r, f1, _ = precision_recall_fscore_support(
            y_pred= predictions.cpu().view(-1).numpy(),
            y_true= targets.cpu().view(-1).numpy(),
            zero_division=0,
            average='macro'
        )
        wandb.log({
            f"{prefix} Precision": p,
            f"{prefix} Recall": r,
            f"{prefix} F1": f1,
            }, step=epoch)
    # VARYING THRESHOLD WHICH ONLY MAKES SENSE AT INFERENCE
    else:
        metrics = varying_threshold_metrics(predictions, targets)
        results = pd.DataFrame(data=metrics, columns=["threshold", "precision", "recall", "f1 score"])
        wandb.log({f"{prefix} Results": wandb.Table(dataframe=results)}, step=epoch)
        f1 = results['f1 score'].max()
        wandb.log({f"Best {prefix} f1": f1}, step=epoch)

    # LOG TO WANDB
    running_loss /= len(data_loader)
    wandb.log({f"{prefix} loss": running_loss}, step=epoch)
    return f1, running_loss


def run(data, fold, args):
    train_loader, valid_loader = data
    cfg = DotDict(config)
    cfg.pretrain = args.pretrain

    if cfg.ast:
        model = get_model("ast").to(cfg.device)
    else:
        model = get_model("cnn").to(cfg.device)
    
    if args.load_weights:
        name = 'ast' if cfg.ast else 'cnn'
        name += f'_{fold}.bin'
        model.load_state_dict(torch.load(f"pretrain/{name}"))
        print("Loaded weights!")

    
    wandb.watch(model)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = fetch_scheduler(optimizer, "OneCycle", spe=len(train_loader), epochs=cfg.epochs)
    #scheduler = fetch_scheduler(optimizer, "CosineAnnealingLR")
    #scheduler = None
    scaler = torch.cuda.amp.GradScaler()

    best_valid_f1 = 0
    best_valid_loss = np.inf

    prefix = './pretrain/' if cfg.pretrain else './'

    for epoch in range(cfg.epochs):
        do_epoch(True, model, train_loader, optimizer, scheduler, scaler, cfg, epoch)
        _, val_loss = do_epoch(False, model, valid_loader, optimizer, scheduler, scaler, cfg, epoch)

        if val_loss < best_valid_loss:
            print(f"Validation Loss Improved - {best_valid_loss} ---> {val_loss}")
            torch.save(model.state_dict(),  f'{prefix}model_{fold}_{wandb.run.name}.bin')
            print(f"Saved model checkpoint at {prefix}model_{fold}_{wandb.run.name}.bin")
            best_valid_loss = val_loss

    return best_valid_loss


def main():
    parser = argparse.ArgumentParser(description="Run the training pipeline")
    parser.add_argument("--data_path",   type=str, default="../datasets/numpy_mel", help="Location of the metadata csv")
    parser.add_argument("--data_folder", type=str, default="../datasets/numpy_mel/data", help="Location of the individual bird folders")
    parser.add_argument("--pretrain", type=bool, default=False)
    parser.add_argument("--load_weights", type=bool, default=True)
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
    dataset = get_dataset_pretrain(args) if args.pretrain == True else get_dataset(args)

    if args.load_weights:
        assert config['n_folds'] == 3
    # train n_folds models
    for fold in range(config['n_folds']):
        # only do the first fold
        if not fold == 0:
            continue
        torch.cuda.empty_cache()
        print("=" * 30)
        print("Training Fold - ", fold)
        print("=" * 30)
        best_val = run(dataset[fold], fold, args)
        print(f'Best Loss: {best_val:.5f}')

        gc.collect()
        torch.cuda.empty_cache()
        


if __name__ == "__main__":
    main()
