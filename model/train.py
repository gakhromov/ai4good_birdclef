from config import config, CFG
from models import loss_fn, get_model
from dataset import get_dataset
from tqdm import tqdm
import torch
import gc
from torch.optim import Adam
from sklearn.metrics import f1_score
import wandb
from adamp import AdamP
import argparse

def train(model, data_loader, optimizer, scheduler, device, epoch):
    model.to(device)
    model.train()

    running_loss = 0
    loop = tqdm(data_loader, position=0)

    for i, (mels, labels) in enumerate(loop):
        mels = mels.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(mels)
        _, preds = torch.max(outputs, 1)

        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()
        

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()

    return running_loss / len(data_loader)


def valid(model, data_loader, device, epoch):
    model.eval()

    running_loss = 0
    pred = []
    label = []

    loop = tqdm(data_loader, position=0)
    for mels, labels in loop:
        mels = mels.to(device)
        labels = labels.to(device)

        outputs = model(mels)
        _, preds = torch.max(outputs, 1)

        loss = loss_fn(outputs, labels)

        running_loss += loss.item()

        pred.extend(preds.view(-1).cpu().detach().numpy())
        label.extend(labels.view(-1).cpu().detach().numpy())

        loop.set_description(f"Epoch [{epoch + 1}/{config['epochs']}]")
        loop.set_postfix(loss=loss.item())

    valid_f1 = f1_score(label, pred, average='macro')

    return running_loss / len(data_loader), valid_f1


def run(data, fold):
    train_loader, valid_loader = data

    model = get_model("cnn")
    wandb.watch(model)

    #optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    optimizer = AdamP(model.parameters(), lr=CFG.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=10)

    best_valid_f1 = 0

    for epoch in range(config['epochs']):
        train_loss = train(model, train_loader, optimizer, scheduler, config['device'], epoch)
        valid_loss, valid_f1 = valid(model, valid_loader, config['device'], epoch)

        wandb.log({"train_loss": train_loss, "valid_loss": valid_loss, "valid_f1": valid_f1})

        if valid_f1 > best_valid_f1:
            print(f"Validation F1 Improved - {best_valid_f1} ---> {valid_f1}")
            torch.save(model.state_dict(), f'./model_{fold}.bin')
            print(f"Saved model checkpoint at ./model_{fold}.bin")
            best_valid_f1 = valid_f1

    return best_valid_f1


def main():
    parser = argparse.ArgumentParser(description="Run the training pipeline")
    parser.add_argument("--data_path", type=str, default="../datasets/birdclef22")
    parser.add_argument("--data_folder", type=str, default="../datasets/birdclef22/train_audio")
    parser.add_argument("--dtype", type=str, default="ogg")
    args = parser.parse_args()

    # enable benchmark mode for more power
    torch.backends.cudnn.benchmark = True

    # setup wandb

    print(config)
    wandb.init(
        project="birdclef",
        entity="matvogel",
        config=config)

    # get dataset
    dataset = get_dataset(path=args.data_path, data_folder=args.data_folder, data_type=args.dtype)

    # train n_folds models
    for fold in range(config['n_folds']):
        print("=" * 30)
        print("Training Fold - ", fold)
        print("=" * 30)
        best_valid_f1 = run(dataset[fold], fold)
        print(f'Best F1 Score: {best_valid_f1:.5f}')

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
