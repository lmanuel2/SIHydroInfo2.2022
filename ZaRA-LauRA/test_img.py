import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataloader_img import HYDRoSWOT
from sklearn.metrics import r2_score
import numpy as np
import scipy

from model import ZaRA, LauRA
from utils import loss_decay_plot, vs_q_plot, pred_vs_gt_plot

def main():

    PATH = f"./model_weights/cnns/model_test"

    cudnn.enabled = True
    cudnn.benchmark = True

    dataset = HYDRoSWOT("./IMAGES", split="test", transform=True)

    testloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                num_workers=1, pin_memory=True, drop_last=False)

    model = LauRA()
    model = model.cuda()

    loss_record = {"train": [], "val": []}

    checkpoint = torch.load(f"{PATH}/model.pth")
    model.load_state_dict(checkpoint['model_state'])

    num_epochs = checkpoint['epoch']
    loss_record["train"] = checkpoint['train_loss']
    loss_record["val"] = checkpoint['val_loss']

    loss_decay_plot(num_epochs, loss_record["train"], loss_record["val"], PATH)

    pred_lst = []
    gt_lst = []
    covs_lst = []

    model.eval()
    with torch.no_grad():
        for img_test, X_test, y_test in testloader:

            y_test = y_test.reshape(-1, 1)

            img_test = img_test.cuda()
            X_test = X_test.cuda()

            if model.name == "ZaRA":
                pred = model((img_test, X_test))
            else:
                _, pred = model((img_test, X_test))

            covariates = X_test.detach().cpu().numpy()
            pred_width = pred.detach().cpu().numpy()
            obs_width = y_test.numpy()

            pred_lst.append(pred_width)
            gt_lst.append(obs_width)
            covs_lst.append(covariates)

    pred_lst = np.array(pred_lst).reshape(-1, 1)
    gt_lst = np.array(gt_lst).reshape(-1, 1)
    covs_lst = np.squeeze(np.array(covs_lst), axis=1)


    covs = dataset.X_scaler.inverse_transform(covs_lst[:, :16])
    log_q = covs[:, 0]
    log_wdth = dataset.y_scaler.inverse_transform(gt_lst)
    log_pred = dataset.y_scaler.inverse_transform(pred_lst)

    q = 10 ** log_q
    wdth = 10 ** log_wdth
    pred = 10 ** log_pred

    metric_dict = dict()
    metric_dict["NSE_log"] = r2_score(log_wdth, log_pred)
    _, _, metric_dict["r_value_log"], _, _ = scipy.stats.linregress(log_wdth.reshape(-1), log_pred.reshape(-1))
    metric_dict["pbias_log"] = 100 * np.sum((log_pred - log_wdth)) / np.sum(log_wdth)

    print(f'NSE log: {metric_dict["NSE_log"]:.4f}, Rsquared_log: {metric_dict["r_value_log"]:.4f}, pbias_log: {metric_dict["pbias_log"]}')

    vs_q_plot(q, wdth, pred, PATH)

    pred_vs_gt_plot(pred, wdth, metric_dict, PATH)

if __name__ == "__main__":
    main()