import torch
import torch.nn as nn
from torchvision import models

class MLP(nn.Module):
    def __init__(self, input_ftrs=26):
        super(MLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_ftrs, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=64, out_features=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class ZaRA(nn.Module):
    def __init__(self, scalar_ftrs=26, model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)):
        super(ZaRA, self).__init__()
        self.name = "ZaRA"

        model.conv1 = nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(model.fc.in_features, 64)

        self.cnn_model = model

        self.scalars_warmup = nn.Sequential(nn.Linear(in_features=scalar_ftrs, out_features=64),
                                                nn.ReLU())
        self.encoder = nn.Sequential(
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(in_features=256, out_features=1)
        )

    def forward(self, x):

        out = self.cnn_model(x[0])
        ftrs = self.scalars_warmup(x[1])
        out = torch.cat((ftrs, out), axis=1)
        out = self.encoder(out)

        return out

class LauRA(nn.Module):
    def __init__(self, scalar_ftrs=26, model=models.vgg16(weights=models.VGG16_Weights.DEFAULT)):
        super(LauRA, self).__init__()
        self.name = "LauRA"

        model.features[0] = nn.Conv2d(5, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)

        self.pooling = model.avgpool
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(model.classifier[0],
                                nn.ReLU(),
                                nn.Linear(in_features=4096, out_features=1024),
                                nn.ReLU(),
                                nn.Linear(in_features=1024, out_features=64),
                                nn.ReLU())

        self.dsn = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1)
        )
        self.scalars_warmup = nn.Sequential(nn.Linear(in_features=scalar_ftrs, out_features=64),
                                                nn.ReLU())
        self.encoder = nn.Sequential(
            nn.Linear(in_features=128, out_features=192),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(in_features=192, out_features=192),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(in_features=192, out_features=192),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(in_features=192, out_features=1)
        )

    def forward(self, x):

        out = self.features(x[0])
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out)

        aux = self.dsn(out)
        ftrs = self.scalars_warmup(x[1])
        out = torch.cat((ftrs, out), axis=1)
        out = self.encoder(out)
        return aux, out

if __name__ == "__main__":

    # model = ZaRA()
    model = LauRA()

    X = (torch.randn(2, 5, 500, 500), torch.randn(2, 26))

    # ZaRA:
    # print(model(X).shape)

    # LauRA:
    aux, pred = model(X)
    print(aux.shape)
    print(pred.shape)
