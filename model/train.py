from config import config, wandb_key
from helpers import DotDict, fetch_scheduler
from models import loss_ce, get_model, loss_bcefocal, loss_bce
from dataset import get_dataset
from tqdm import tqdm
import time
import torch
import gc
from torch.optim import Adam, AdamW
from sklearn.metrics import precision_recall_fscore_support
from torch.cuda.amp import autocast
import wandb
import argparse
import numpy as np
from contextlib import nullcontext


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




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
        '''for i, (mels, labels) in enumerate(loop):
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
                    scheduler.step()'''
                    
        ### Train AST ###
        
        batch_time = AverageMeter()
        per_sample_time = AverageMeter()
        data_time = AverageMeter()
        per_sample_data_time = AverageMeter()
        loss_meter = AverageMeter()
        per_sample_dnn_time = AverageMeter()
        global_step, epoch = 0, 0
        start_time = time.time()
        warmup = False
                    
        for i, (mels, lbls) in enumerate(loop):
        
          
          audio_input = mels['mels']
          labels = lbls
          B = audio_input.size(0)
          
          audio_input = audio_input.to(device=device, non_blocking=True)
          labels = labels.to(device=device, non_blocking=True)


          if global_step <= 1000 and global_step % 50 == 0 and warmup == True:
            warm_lr = (global_step / 1000) * (1e-3)
            for param_group in optimizer.param_groups:
              param_group['lr'] = warm_lr
            print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

          with torch.cuda.amp.autocast():
            outputs = model(audio_input)
            if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
              loss = loss_fn(outputs, labels)#torch.argmax(labels.long(), axis=1))
            else:
              loss = loss_fn(outputs, labels)

            if train:
              optimizer.zero_grad()
              scaler.scale(loss).backward()
              scaler.step(optimizer)
              scaler.update()
              if scheduler is not None:
                scheduler.step()

            loss_meter.update(loss.item(), B)

            print_step = global_step % 20 == 0
            early_print_step = epoch == 0 and global_step % (20/10) == 0
            print_step = print_step or early_print_step
            
            # calculate sigmoid for multilabel
            if conf.use_secondary:
                preds = torch.sigmoid(outputs)
            else:
                _, preds = torch.max(outputs, 1)

            if print_step and global_step != 0:
              print('Epoch: [{0}][{1}/{2}]\t'
                  'Train Loss {loss_meter.avg:.4f}\t'.format(
                  epoch, i, len(loop), loss_meter=loss_meter), flush=True)
              if np.isnan(loss_meter.avg):
                print("training diverged...")
                return
        
        ##########

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

    model = get_model("ast").to(config['device'])
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
        project="ai4good",
        entity="lessgoo",
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
