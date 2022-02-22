# %%
from glob import glob
from torch.utils.data import Dataset
import os
import h5py
import numpy as np


class ModelNet40(Dataset):
    """
    dataset Class for ModelNet40
    Params
    ------
    data_dir: the dataset directory
    num_points: how many point should we use in the training process
    partition: train or test, which means the dataset is for training or test
    """

    def __init__(self, data_dir: str, num_points: int, partition="train") -> None:
        super().__init__()
        all_data = []
        all_label = []
        # select all data for training and load them into memory
        for h5_name in glob(os.path.join(data_dir, 'modelnet40_ply_hdf5_2048', '*%s*.h5' % partition)):
            f = h5py.File(h5_name, "r+")
            data = f["data"][:].astype("float32")
            label = f["label"][:].astype("int64")
            f.close()
            # then append the data and label to the list
            all_data.append(data)
            all_label.append(label)

        # concatenate all data from different h5 files
        self.data = np.concatenate(all_data, axis=0)
        self.label = np.concatenate(all_label, axis=0)

        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, index):
        point_cloud = self.data[index][:self.num_points]
        label = self.label[index]
        if self.partition == "train":
            xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
            xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
            point_cloud = np.add(np.multiply(
                point_cloud, xyz1), xyz2).astype("float32")
            np.random.shuffle(point_cloud)

        return point_cloud, label

    def __len__(self):
        return self.data.shape[0]


# # %% test for the dataloader
# from torch.utils.data import DataLoader
# data_path = "C:/Users/ssk/Desktop/GNN/CodeFromGithub/dgcnn.pytorch/data/"
# dataset = ModelNet40(data_path, 2048, "train")
# train_loader = DataLoader(dataset,batch_size=32,shuffle=True,drop_last=True)

# for data,label in train_loader:
#     print(data.shape,label.shape)
#     break
# # %%
