import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataloader_img import HYDRoSWOT
from torch.optim import lr_scheduler
import time
import os

from model import ZaRA, LauRA
from utils import loss_decay_plot

def main():

    PATH = f"./model_weights/cnns/model_test"

    try:
        os.makedirs(PATH)
    except FileExistsError:
        pass

    cudnn.enabled = True
    cudnn.benchmark = True

    dataset = HYDRoSWOT("./IMAGES", split="train", transform=True)

    val_chunk = int(len(dataset) * 0.15)
    train_chunk = len(dataset) - val_chunk

    train_set, val_set = torch.utils.data.random_split(dataset, [train_chunk, val_chunk],
                                                       generator=torch.Generator().manual_seed(7))
    datasets = {"train": train_set, "val": val_set}

    dataloaders = {x: DataLoader(datasets[x], batch_size=64, shuffle=True, num_workers=2,
                                 pin_memory=True, drop_last=False) for x in ["train", "val"]}

    model = LauRA()
    model = model.cuda()

    learning_rate = 1.75e-3
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.009)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    # 3) Training loop
    num_epochs = 5
    loss_record = {"train": [], "val": []}
    since = time.time()

    for epoch in range(num_epochs):

        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            for img, X, y in dataloaders[phase]:

                y = y.reshape(-1, 1)

                img = img.cuda()
                X = X.cuda()
                y = y.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    if model.name == "ZaRA":
                        y_predicted = model((img, X))
                        loss = criterion(y_predicted, y)

                    else:
                        aux, outputs = model((img, X))
                        loss = criterion(outputs, y) + 0.4 * criterion(aux, y)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * X.size(0)

            if phase == "train":
                scheduler.step()
                lr = optimizer.param_groups[0]['lr']

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            if (epoch + 1) % 1 == 0:
                print(f'epoch: {epoch + 1}, phase: {phase}, loss: {epoch_loss:.4f}, lr: {lr:.2E}')
                loss_record[phase].append(epoch_loss)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    state = {
        "epoch": num_epochs,
        "learning_rate": learning_rate,
        "model_state": model.state_dict(),
        "train_loss": loss_record["train"],
        "val_loss": loss_record["val"]
        }
    torch.save(state, f"{PATH}/model.pth")

    loss_decay_plot(num_epochs, loss_record["train"], loss_record["val"], PATH)

if __name__ == "__main__":
    main()