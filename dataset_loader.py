from torch.utils.data import DataLoader, Dataset, random_split
import torch
from utils import matrix_to_rotation_6d
import numpy as np


class ActionDatasets(Dataset):
    def __init__(self, filepath):
        super(ActionDatasets, self).__init__()
        self.data = np.load(filepath, allow_pickle=True)

        self.poses = self.data['poses']
        if "s_foot" in self.data:
            self.s_foot = self.data['s_foot']
            self.root_velocity = self.data['trans']
        else:
            self.s_foot = []
            self.root_velocity = []

        self.ori = self.data['ori']
        self.acc = self.data['acc']
        self.point = self.data['point']

    def __getitem__(self, idx):
        if len(self.root_velocity) != 0:
            all_root_velocity = self.root_velocity[idx]
            # 根节点位移
            root_velocity = torch.FloatTensor(all_root_velocity)

            all_s_foot = self.s_foot[idx]
            # 脚触地概率
            s_foot = torch.FloatTensor(all_s_foot)

        else:
            root_velocity =[]
            s_foot = []

        all_ori = self.ori[idx]
        all_pose = self.poses[idx]

        all_pose_6d = []
        for pose in all_pose:
            all_pose_6d.append(matrix_to_rotation_6d(torch.FloatTensor(pose)))

        
        # 所有节点加速度
        all_acc = self.acc[idx]
        acc = torch.FloatTensor(all_acc)
        #所有关节位置
        all_point = self.point[idx]
        # 叶节点方向
        ori = torch.FloatTensor(all_ori)
        root_inv = ori[:, -1].inverse()
        # normalize

        all_acc[:, :-1] = torch.einsum("ijk,imk->imj", root_inv, acc[:, :-1] - acc[:, -1].reshape(-1, 1, 3))
        all_acc[:, -1] = torch.einsum("ijk,ik->ij", root_inv, acc[:, -1])
        
        ori[:, :-1] = torch.einsum("ijk,itkm->itjm", root_inv, ori[:, :-1])
        ori = ori.reshape(-1, 6, 9)
        # 所有节点的6d旋转
        all_pose_6d = torch.stack(all_pose_6d)
        pose_6d = all_pose_6d.reshape(-1, 15 * 6)
        # 根节点方向
        root_ori = ori[:, -1, :]

        # 结合的x0输入
        x0 = torch.cat([acc, ori], 2).reshape(-1, 6 * 12)
        # 相对于胯的各节点位置
        root_pos = all_point[:, 0, :].reshape(-1, 1, 3)
        leaf_pos = all_point[:, 1:, :] - root_pos
        points = torch.FloatTensor(leaf_pos)
        # 叶节点位置
        p_leaf_gt = torch.cat((points[:, 18 - 1, :], points[:, 19 - 1, :], points[:, 4 - 1, :], points[:, 5 - 1, :],
                               points[:, 15 - 1, :]), 1).reshape(-1, 5 * 3)
        # 除胯的节点位置
        p_all_gt = points.reshape(-1, 23 * 3)

        return x0, p_leaf_gt, p_all_gt, pose_6d, s_foot, root_velocity, root_ori

    def __len__(self):
        return len(self.ori)


# 数据集处理函数
def padding_fn(batch_data):
    batch_size = len(batch_data)

    sequence_lengths = np.array([s[1].shape[0] for s in batch_data], dtype=np.int32)

    max_len = max(sequence_lengths)
    sorted_indices = np.argsort(sequence_lengths)[::-1]

    batch_x0 = np.zeros((batch_size, max_len, 72))
    batch_p_leaf_gt = np.zeros((batch_size, max_len, 15))
    batch_p_all_gt = np.zeros((batch_size, max_len, 69))
    batch_pose_6d = np.zeros((batch_size, max_len, 90))
    batch_s_foot = np.zeros((batch_size, max_len, 2))
    batch_root_velocity = np.zeros((batch_size, max_len, 3))
    batch_root_ori = np.zeros((batch_size, max_len, 9))
    batch_mask = np.zeros((batch_size, max_len))

    for idx, item in enumerate(batch_data):
        x0, p_leaf_gt, p_all_gt, pose_6d, s_foot, root_velocity, root_ori = item
        seq_lenghth = x0.shape[0]
        if seq_lenghth != max_len:
            pad = np.zeros((max_len - seq_lenghth, 72))
            z1 = np.concatenate((x0, pad), axis=0)
            batch_x0[idx] = z1

            pad = np.zeros((max_len - seq_lenghth, 15))
            z2 = np.concatenate((p_leaf_gt, pad), axis=0)
            batch_p_leaf_gt[idx] = z2

            pad = np.zeros((max_len - seq_lenghth, 69))
            z3 = np.concatenate((p_all_gt, pad), axis=0)
            batch_p_all_gt[idx] = z3

            pad = np.zeros((max_len - seq_lenghth, 90))
            z4 = np.concatenate((pose_6d, pad), axis=0)
            batch_pose_6d[idx] = z4

            if not s_foot is None:
                pad = np.zeros((max_len - seq_lenghth, 2))
                z5 = np.concatenate((s_foot, pad), axis=0)
                batch_s_foot[idx] = z5

                pad = np.zeros((max_len - seq_lenghth, 3))
                z6 = np.concatenate((root_velocity, pad), axis=0)
                batch_root_velocity[idx] = z6

            pad = np.zeros((max_len - seq_lenghth, 9))
            z7 = np.concatenate((root_ori, pad), axis=0)
            batch_root_ori[idx] = z7

        else:
            batch_x0[idx] = x0
            batch_p_leaf_gt[idx] = p_leaf_gt
            batch_p_all_gt[idx] = p_all_gt
            batch_pose_6d[idx] = pose_6d
            if not s_foot is None:
                batch_s_foot[idx] = s_foot
                batch_root_velocity[idx] = root_velocity
            batch_root_ori[idx] = root_ori

        batch_mask[idx] = np.concatenate((np.ones((seq_lenghth)), np.zeros((max_len - seq_lenghth))))
    return sequence_lengths[sorted_indices], \
           torch.FloatTensor(batch_x0[sorted_indices]), \
           torch.FloatTensor(batch_p_leaf_gt[sorted_indices]), \
           torch.FloatTensor(batch_p_all_gt[sorted_indices]), \
           torch.FloatTensor(batch_pose_6d[sorted_indices]), \
           torch.LongTensor(batch_s_foot[sorted_indices]), \
           torch.FloatTensor(batch_root_velocity[sorted_indices]), \
           torch.FloatTensor(batch_root_ori[sorted_indices]), \
           torch.FloatTensor(batch_mask[sorted_indices])


train_datasets = None

def create_data_loader(batch_size):
    global train_datasets
    if train_datasets is None:
        train_datasets = ActionDatasets("dataset/merge_data_all_no_std.npz")
    test_datasets = ActionDatasets("dataset/test_no_std_300.npz")
    split_rate = 0.99  # 训练集占整个数据集的比例
    train_len = int(split_rate * len(train_datasets))
    valid_len = len(train_datasets) - train_len

    train_sets, valid_sets = random_split(train_datasets, [train_len, valid_len])

    train_loader = DataLoader(train_sets, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True,
                               num_workers=4)
    test_loader = DataLoader(test_datasets, batch_size=1, shuffle=True, pin_memory=True,
                              num_workers=4)
    valid_loader = DataLoader(valid_sets, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True,
                            num_workers=4)

    print(f"训练集大小{len(train_sets)}， 验证集大小{len(valid_sets)}， 测试集大小{len(test_datasets)}")
    return train_loader, test_loader, valid_loader


if __name__ == "__main__":
    train_loader, test_loader, valid_loader = create_data_loader(2)
    data = next(iter(train_loader))
    print(data[0].shape)
    print(data[1].shape)
    print(data[2].shape)
    print(data[3].shape)
    print(data[4].shape)
    print(data[5].shape)
    print(data[6].shape)

