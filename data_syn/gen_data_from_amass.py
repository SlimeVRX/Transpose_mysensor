'''
Generate synthesis IMU data from AMASS
'''
import os
import sys
sys.path.append(os.getcwd())
from glob import glob
from utils import axis_angle_to_matrix, matrix_to_axis_angle
import torch
from tqdm import tqdm, trange
from human_body_prior.body_model.body_model import BodyModel
import numpy as np
import pickle as pkl


comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'data_syn/smplh/models/%s/model.npz'
male_fname = MODEL_PATH % 'male'
female_fname = MODEL_PATH % 'female'
num_betas = 16
TARGET_FPS = 60

# 排除手和脚的关节点，胯部关节点使用采集到的
SMPL_IDS = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]

model_male = BodyModel(bm_fname=male_fname,
                       num_betas=num_betas).to(comp_device)
model_female = BodyModel(bm_fname=female_fname,
                         num_betas=num_betas).to(comp_device)
# IMU 穿戴位置
# 头 198
# 胯 3026
# 右手 5425
# 左手 1970
# 左脚 1372
# 右脚 4845
VERTEX_IDS = [1970, 5425, 1372, 4845, 198, 3026]


def get_ori_acc(A_global_list, vertexs, frame_rate, n):
    orientation = []  # 旋转矩阵
    acceleration = []  # 加速度
    for j in range(A_global_list.shape[0]):
        ori_tmp = []
        for i in [18, 19, 4, 5, 15, 0]:
            a_global = A_global_list[j]
            ori_ = a_global[i, :3, :3]
            ori_tmp.append(ori_)

        orientation.append(np.array(ori_tmp))
    ori = np.array(orientation)

    time_interval = 1.0 / frame_rate
    total_number = len(A_global_list)
    for idx in range(n, total_number - n):
        vertex_0 = vertexs[idx - n]  # 6 * 3
        vertex_1 = vertexs[idx]
        vertex_2 = vertexs[idx + n]
        # 1 加速度合成
        accel_tmp = (vertex_2 + vertex_0 - 2 * vertex_1) / \
            (n * n * time_interval * time_interval)
        acceleration.append(accel_tmp)

    ori = ori[n:-n]
    acc = np.array(acceleration)
    return ori, acc


def compute_imu_data(gender, poses, trans):
    if gender == 'male':
        model = model_male
    else:
        model = model_female
    time_length = len(poses)

    poses = np.array(poses)
    batch_size = 512
    As, Vs, Js = [], [], []
    for i in range(time_length // batch_size):
        body_parms = {
            'root_orient': torch.Tensor(poses[i * batch_size:(i + 1) * batch_size, :3]).to(comp_device),
            'pose_body': torch.Tensor(poses[i * batch_size:(i + 1) * batch_size, 3:66]).to(comp_device),
            'pose_hand': torch.Tensor(poses[i * batch_size:(i + 1) * batch_size, 66:]).to(comp_device),
            'trans': torch.Tensor(trans[i * batch_size:(i + 1) * batch_size]).to(comp_device),
        }

        body_pose_beta = model(
            **{k: v for k, v in body_parms.items() if k in ['pose_body', 'root_orient', 'pose_hand', 'trans']})
        A = body_pose_beta.A[:, :24].cpu().numpy()
        V = body_pose_beta.v[:, VERTEX_IDS, :].cpu().numpy()
        J = body_pose_beta.Jtr[:, :24].cpu().numpy()
        As.append(A)
        Vs.append(V)
        Js.append(J)
    last_count = time_length % batch_size
    if last_count != 0:
        body_parms = {
            'root_orient': torch.Tensor(poses[-last_count:, :3]).to(comp_device),
            'pose_body': torch.Tensor(poses[-last_count:, 3:66]).to(comp_device),
            'pose_hand': torch.Tensor(poses[-last_count:, 66:]).to(comp_device),
            'trans': torch.Tensor(trans[-last_count:]).to(comp_device),
        }

        body_pose_beta = model(
            **{k: v for k, v in body_parms.items() if k in ['pose_body', 'root_orient', 'pose_hand', 'trans']})
        A = body_pose_beta.A[:, :24].cpu().numpy()
        V = body_pose_beta.v[:, VERTEX_IDS, :].cpu().numpy()
        J = body_pose_beta.Jtr[:, :24].cpu().numpy()

        As.append(A)
        Vs.append(V)
        Js.append(J)
    As = np.concatenate(As, axis=0)
    Vs = np.concatenate(Vs, axis=0)
    Js = np.concatenate(Js, axis=0)
    return As, Vs, Js


def findNearest(t, t_list):
    list_tmp = np.array(t_list) - t
    list_tmp = np.abs(list_tmp)
    index = np.argsort(list_tmp)[:2]
    return index


def interpolation_integer(poses_ori, fps):
    poses = []
    n_tmp = int(fps / TARGET_FPS)
    poses_ori = poses_ori[::n_tmp]

    for t in poses_ori:
        poses.append(t)

    return poses


def interpolation(poses_ori, fps):
    poses = []
    total_time = len(poses_ori) / fps
    times_ori = np.arange(0, total_time, 1.0 / fps)
    times = np.arange(0, total_time, 1.0 / TARGET_FPS)

    for t in times:
        index = findNearest(t, times_ori)
        if len(index) != 2:
            continue
        if index[0] == len(poses_ori):
            break
        a = poses_ori[index[0]]
        t_a = times_ori[index[0]]
        if index[1] == len(poses_ori):
            break
        b = poses_ori[index[1]]
        t_b = times_ori[index[1]]

        if t_a == t:
            tmp_pose = a
        elif t_b == t:
            tmp_pose = b
        else:
            tmp_pose = a + (b - a) * ((t_b - t) / (t_b - t_a))
        poses.append(tmp_pose)

    return poses


def generate_data1(npz_dir, res_path):
    if os.path.exists(res_path):
        return
    data_in = np.load(npz_dir, allow_pickle=True)

    if len(data_in['poses']) == 1:
        return

    data_out = {}
    data_out['gender'] = data_in['gender']
    data_out['betas'] = data_in['betas']

    fps_ori = data_in['mocap_framerate']
    n = 4  # 加速度合成间隔帧数

    if (fps_ori % TARGET_FPS) == 0:
        data_out['poses'] = interpolation_integer(data_in['poses'], fps_ori)
        trans = interpolation_integer(data_in['trans'], fps_ori)
    else:
        data_out['poses'] = interpolation(data_in['poses'], fps_ori)
        trans = interpolation(data_in['trans'], fps_ori)

    data_out['betas'] = np.array(data_out['betas'])
    trans = np.array(trans)

    A, V, J = compute_imu_data(data_out['gender'], data_out['poses'], trans)

    data_out_dict = {"A": A, "V": V, "J": J,
                     "poses": np.array(data_out['poses']), "trans": trans}
    with open(res_path, 'wb') as fout:
        pkl.dump(data_out_dict, fout)


def generate_data2(npz_dir, res_path):
    if not os.path.exists(res_path):
        return
    #if os.path.exists(res_path[:-8] + "_syn.pkl"):
    #    return

    data_out = {}
    n = 4  # 加速度合成间隔帧数

    with open(res_path, 'rb') as fout:
        data_out_dict = pkl.load(fout)
    poses = data_out_dict['poses']
    poses = poses.reshape(-1, 52, 3)[:, SMPL_IDS]
    trans = data_out_dict['trans']
    pre_trans = []
    for idx in range(1, len(trans)):
        pre_trans.append(trans[idx] - trans[idx - 1])
    trans = np.array(pre_trans)[n:-n+1]

    A_global_arr, VERTEXS, joint_location = data_out_dict[
        'A'], data_out_dict['V'], data_out_dict['J']
    mu = 0.008  # 单位米，移动小于mu则为支撑脚
    s_foot = []
    for idx in range(n, len(joint_location)-n):
        S_foot = [0, 0]
        if np.linalg.norm((joint_location[idx][10] - joint_location[idx - 1][10])) < mu:
            S_foot[0] = 1
        if np.linalg.norm((joint_location[idx][11] - joint_location[idx - 1][11])) < mu:
            S_foot[1] = 1

        s_foot.append(S_foot)
    s_foot = np.array(s_foot)  # 脚触底数据
    position = joint_location[n:-n]  # 24个关节点在空间上的坐标 (x, y, z)
    # orientation传感器方向（自身坐标系）, acceleration传感器加速度(自身坐标系)
    orientation, acceleration = get_ori_acc(
        A_global_arr, VERTEXS, TARGET_FPS, n)
    data_out['ori'] = orientation
    data_out['acc'] = acceleration
    data_out['point'] = position
    data_out['s_foot'] = s_foot
    data_out['trans'] = trans
    poses = torch.FloatTensor(poses)
    data_out['poses'] = axis_angle_to_matrix(poses).numpy()[n:-n]
    # for k, v in data_out.items():
    #     print(k, v.shape)
    with open(res_path[:-8] + "_syn.pkl", 'wb') as fout:
        pkl.dump(data_out, fout)


def main(pkl_path, res_data_path):
    #print(res_data_path)
    generate_data2(pkl_path, res_data_path)


if __name__ == '__main__':

    files = glob("../acc2pos/dataset/**/*.npz", recursive=True)

    for file_ in tqdm(files):
        name_arr = file_.split("/")
        filename = name_arr[-1].split(".")[0]

        if filename == 'shape':
            continue
        new_file = "/".join(name_arr[:-1]) + "/" + filename + "_avj.pkl"

        pkl_path = file_
        res_data_path = new_file
        main(pkl_path, res_data_path)
