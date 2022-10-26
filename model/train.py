from config import config, CFG
from models import loss_ce, get_model, loss_bcefocal
from dataset import get_dataset
from tqdm import tqdm
import torch
import gc
from torch.optim import Adam
from sklearn.metrics import f1_score
import wandb
import argparse

def train(model, data_loader, optimizer, scheduler, loss_fn, device, epoch):
    model.to(device)
    model.train()

    running_loss = 0
    running_f1 = 0
    pred = []
    label = []
    loop = tqdm(data_loader, position=0)

    for i, (mels, labels) in enumerate(loop):
        mels = mels.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(mels)
        preds = torch.sigmoid(outputs) > 0.5

        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()
        

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()

        pred.extend(preds.view(-1).cpu().detach().numpy())
        label.extend(labels.view(-1).cpu().detach().numpy())
        running_f1 += f1_score(label, pred, average='macro')

        loop.set_description(f"Train Epoch [{epoch + 1}/{config['epochs']}]")
        loop.set_postfix(loss=loss.item())

    return running_loss / len(data_loader), running_f1 / len(data_loader)


def valid(model, data_loader, loss_fn, device, epoch):
    model.eval()

    running_loss = 0
    running_f1 = 0
    pred = []
    label = []

    loop = tqdm(data_loader, position=0)
    for mels, labels in loop:
        mels = mels.to(device)
        labels = labels.to(device)

        outputs = model(mels)
        preds = torch.sigmoid(outputs) > 0.5
        loss = loss_fn(outputs, labels)

        running_loss += loss.item()

        pred.extend(preds.view(-1).cpu().detach().numpy())
        label.extend(labels.view(-1).cpu().detach().numpy())

        running_f1 += f1_score(label, pred, average='macro')

        loop.set_description(f"Valid Epoch [{epoch + 1}/{config['epochs']}]")
        loop.set_postfix(loss=loss.item())

    valid_f1 = running_f1 / len(data_loader)

    return running_loss / len(data_loader), valid_f1


def run(data, fold, args):
    train_loader, valid_loader = data

    model = get_model("cnn")
    wandb.watch(model)

    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=10)

    loss = loss_bcefocal if args.use_secondary else loss_ce

    best_valid_f1 = 0

    for epoch in range(config['epochs']):
        train_loss, train_f1 = train(model, train_loader, optimizer, scheduler, loss, config['device'], epoch)
        valid_loss, valid_f1 = valid(model, valid_loader,loss, config['device'], epoch)

        wandb.log({"train_loss": train_loss, "train_f1": train_f1, "valid_loss": valid_loss, "valid_f1": valid_f1})

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
