import torch
import os
import pickle
from config import paths

def process_dipimu():
    imu_mask = [7, 8, 11, 12, 0, 2]
    # test_split = ['s_09', 's_10']
    test_split = ['s_01']
    accs, oris, poses, trans = [], [], [], []

    for subject_name in test_split:
        #'D:/OLD/Xu/test_transpose/source/DIPIMUandOthers/DIP_IMU_and_Others/raw_dipimu_dir\\s_01'  #04.pkl
        for motion_name in os.listdir(os.path.join(paths.raw_dipimu_dir, subject_name)):
            path = os.path.join(paths.raw_dipimu_dir, subject_name, motion_name)
            data = pickle.load(open(path, 'rb'), encoding='latin1')
            acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
            ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
            pose = torch.from_numpy(data['gt']).float()

            # fill nan with nearest neighbors
            for _ in range(4):
                acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

            acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]
            if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                accs.append(acc.clone())
                oris.append(ori.clone())
                poses.append(pose.clone())
                trans.append(torch.zeros(pose.shape[0], 3))  # dip-imu does not contain translations
            else:
                print('DIP-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))

    os.makedirs(paths.dipimu_dir, exist_ok=True)
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'tran': trans}, os.path.join(paths.dipimu_dir, 'test.pt'))
    print('Preprocessed DIP-IMU dataset is saved at', paths.dipimu_dir)

if __name__ == '__main__':
    process_dipimu()
