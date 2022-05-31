r"""
    Config for paths, joint set, and normalizing scales.
"""


# datasets (directory names) in AMASS
# e.g., for ACCAD, the path should be `paths.raw_amass_dir/ACCAD/ACCAD/s001/*.npz`
# amass_data = ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh', 'Transitions_mocap', 'SSM_synced', 'CMU',
#               'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BMLmovi', 'EKUT', 'TCD_handMocap', 'ACCAD',
#               'BioMotionLab_NTroje', 'BMLhandball', 'MPI_Limits', 'DFaust67']
amass_data = ['ACCAD']


class paths:
    # raw_amass_dir = 'data/dataset_raw/AMASS'      # raw AMASS dataset path (raw_amass_dir/ACCAD/ACCAD/s001/*.npz)
    raw_amass_dir = '../acc2pos/AMASS'

    amass_dir = 'data/datassmpl_file_et_work/AMASS'         # output path for the synthetic AMASS dataset

    smpl_file = '../acc2pos/models/SMPL_male.pkl'              # official SMPL model path



class joint_set:
    leaf = [7, 8, 12, 20, 21]
    full = list(range(1, 24))
    reduced = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]

    lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]

    n_leaf = len(leaf)
    n_full = len(full)
    n_reduced = len(reduced)
    n_ignored = len(ignored)


acc_scale = 30
vel_scale = 3
