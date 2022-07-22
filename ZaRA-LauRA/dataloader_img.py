import os
import numpy as np
import tifffile
import pandas as pd
import torch
from torch.utils import data
from PIL import Image
from torchvision.transforms import Compose, ToTensor
from sklearn.preprocessing import StandardScaler


class HYDRoSWOT(data.Dataset):

    def __init__(self, root, split, transform=None):
        super(HYDRoSWOT, self).__init__()
        self.dem_base = os.path.join(root, "DEM")
        self.naipr_base = os.path.join(root, "NAIPR")
        self.naipg_base = os.path.join(root, "NAIPG")
        self.naipb_base = os.path.join(root, "NAIPB")
        self.naipn_base = os.path.join(root, "NAIPN")
        self.items_list = self.get_images_list()

        df = pd.read_csv("data/stream_wdth_all_data_trimmed_v3_MEDIAN_train_PUB.csv", low_memory=False, index_col=0)
        # df = df.astype({'site_no': 'Int64'})

        stat_list = ['stream_wdth_va', 'q_va', 'TotDASqKM', 'D50[mm]', 'AI1', 'EVI_JAS_2012', 'EVI_JFM_2012',
                     'MINELEVSMO', 'SLOPE', 'Percent_Clay1km_0_100cm', 'Percent_Silt_1km100cmfinal',
                     'percent_sand1km_0_100cm', 'ACC_NDAMS2010', 'NLCD_developed_16', 'NLCD_forest_16',
                     'NLCD_agriculture_16', 'CAT_POPDENS10', 'StreamOrde', 'site_no']

        dft = df[stat_list]

        not_in_imgs = [6091700, 2061000, 2208300, 2231458, 3404500, 4165710, 5133500, 6178500, 7374525,
                       4010500, 4176540, 4264331, 5113360, 5129515, 5137500, 5496000, 6149500, 6164000,
                       6906200, 9522000, 10254970, 10302025, 11253115, 11253130, 12113346, 12212390, 12306500,
                       12355000, 12399500, 13132535]
                       # 21536096, 21623976, 21720724, 21720816, 22907084, 23177484, 23358684, 35825880, 23505264]



        dft = dft[dft['TotDASqKM'] != 0]
        dft = dft[dft['q_va'] > 0]
        dft = dft[dft['stream_wdth_va'] < 22000.00]
        # dft = dft[(dft['q_va'] > 0) & (dft['q_va'] < dft['q_va'].quantile(0.95))]
        dft.dropna(axis=0, inplace=True)

        dft["q_va"] = np.log10(dft["q_va"])
        dft["stream_wdth_va"] = np.log10(dft["stream_wdth_va"])
        dft["TotDASqKM"] = np.log10(dft["TotDASqKM"])
        dft["D50[mm]"] = np.log10(dft["D50[mm]"])

        dft = dft.astype({'StreamOrde': 'Int64'})
        dummies = pd.get_dummies(dft['StreamOrde'], prefix='StreamOrde')

        drop_list = ['stream_wdth_va', 'StreamOrde', 'site_no']

        final_list = [var for var in stat_list if var not in drop_list[1:]] + list(dummies.columns) + ['site_no']

        X_scaler = StandardScaler()
        X = X_scaler.fit_transform(dft.drop(drop_list, axis=1))
        X = np.concatenate([X, dummies.to_numpy(), dft['site_no'].to_numpy().reshape(-1, 1)], axis=1)
        self.X_scaler = X_scaler

        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(dft["stream_wdth_va"].to_numpy().reshape(-1, 1))
        self.y_scaler = y_scaler

        if split == "train":
            data = np.concatenate((y, X), axis=1)
            data = pd.DataFrame(data, columns=final_list).astype({'site_no': 'Int64'})
            data = data[~data.site_no.isin(not_in_imgs)].reset_index(drop=True)
            self.data = data

        if split == "test":
            df_test = pd.read_csv("data/stream_wdth_all_data_trimmed_v3_MEDIAN_test_PUB.csv", low_memory=False, index_col=0)
            df_test = df_test[stat_list]

            df_test = df_test[df_test['TotDASqKM'] != 0]
            df_test = df_test[df_test['q_va'] > 0]
            df_test.dropna(axis=0, inplace=True)

            df_test["q_va"] = np.log10(df_test["q_va"])
            df_test["stream_wdth_va"] = np.log10(df_test["stream_wdth_va"])
            df_test["TotDASqKM"] = np.log10(df_test["TotDASqKM"])
            df_test["D50[mm]"] = np.log10(df_test["D50[mm]"])

            df_test = df_test.astype({'StreamOrde': 'Int64'})
            dummies = pd.get_dummies(df_test['StreamOrde'], prefix='StreamOrde')

            X = X_scaler.transform(df_test.drop(drop_list, axis=1))
            X = np.concatenate([X, dummies.to_numpy(), df_test['site_no'].to_numpy().reshape(-1, 1)], axis=1)
            y = y_scaler.transform(np.array(df_test["stream_wdth_va"]).reshape(-1, 1)).astype('float32')

            data = np.concatenate((y, X), axis=1)
            data = pd.DataFrame(data, columns=final_list).astype({'site_no': 'Int64'})
            site2 = data["site_no"].values.tolist()
            data = data[~data.site_no.isin(not_in_imgs)].reset_index(drop=True)
            self.data = data


        self.transform = transform
        image_transforms_list = [ToTensor()]
        self.image_transforms = Compose(image_transforms_list)


    def get_images_list(self):
        items_list = dict()
        for root, dirs, files in os.walk(self.dem_base, topdown=True):

            for name in files:
                if name.endswith(".tif"):
                    dem_file = os.path.join(root, name)
                    naipr_file = os.path.join(self.naipr_base, name)
                    naipg_file = os.path.join(self.naipg_base, name)
                    naipb_file = os.path.join(self.naipb_base, name)
                    naipn_file = os.path.join(self.naipn_base, name)
                    items_list.update({
                        int(name.split(".")[0]): {
                        "dem": dem_file,
                        "naipr": naipr_file,
                        "naipg": naipg_file,
                        "naipb": naipb_file,
                        "naipn": naipn_file
                        }})
        return items_list

    def __getitem__(self, index):
        ftrs = self.data.iloc[index, :-1].to_numpy(np.float32)
        name = self.data.loc[index, "site_no"]

        dem_path = self.items_list[name]["dem"]
        naipr_path = self.items_list[name]["naipr"]
        naipg_path = self.items_list[name]["naipg"]
        naipb_path = self.items_list[name]["naipb"]
        naipn_path = self.items_list[name]["naipn"]

        dem = Image.fromarray(tifffile.imread(dem_path)[10:50, 10:50]).resize((400, 400), resample=Image.NONE)
        naipr = Image.fromarray(tifffile.imread(naipr_path)[100:500, 100:500])
        naipg = Image.fromarray(tifffile.imread(naipg_path)[100:500, 100:500])
        naipb = Image.fromarray(tifffile.imread(naipb_path)[100:500, 100:500])
        naipn = Image.fromarray(tifffile.imread(naipn_path)[100:500, 100:500])

        dem = np.expand_dims(np.array(dem), axis=2)
        naipr = np.expand_dims(np.array(naipr), axis=2)
        naipg = np.expand_dims(np.array(naipg), axis=2)
        naipb = np.expand_dims(np.array(naipb), axis=2)
        naipn = np.expand_dims(np.array(naipn), axis=2)


        naip = np.concatenate([dem, naipr, naipg, naipb, naipn], axis=2)

        naip = (naip - np.min(naip, axis=(0, 1), keepdims=True)) / (np.max(naip, axis=(0, 1), keepdims=True) - np.min(naip, axis=(0, 1), keepdims=True))

        if self.transform:
            naip = self.image_transforms(naip)

        return naip, torch.from_numpy(np.asarray(ftrs[1:])), torch.from_numpy(np.asarray(ftrs[0]))

    def __len__(self):
        return len(self.data)


def main():
    dataset = HYDRoSWOT("./IMAGES", "train", transform=True)
    print(len(dataset))
    dataiter = iter(dataset)
    img, X, y = next(dataiter)
    print(img.shape, X, y)


if __name__ == "__main__":
    main()
