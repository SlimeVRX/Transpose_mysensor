from glob import glob
from random import shuffle

import numpy as np
from tqdm import tqdm
import pickle
import os


def progress(files):
    global data, item
    shuffle(files)
    # files = files[:int(len(files) /10)]
    for idx, filepath in tqdm(enumerate(files), total=len(files)):
        #print(filepath)
        data = pickle.load(open(filepath, 'rb'))


        if len(data['ori']) == 0:
            continue

        for key in ['poses', 'ori', 'acc', 'point']:
            item = data[key]
            item = np.array(item)
            #print(key, item.shape)
            item[np.isnan(item)] = 0
            if key == 'ori':
                if item.shape[1] != 6:
                    item = item[:, [18, 19, 4, 5, 15, 0], :, :]
                else:
                    item = item[:, [2, 3, 4, 5, 0, 1], :, :]
            if key == 'acc':
                if item.shape[1] != 6:
                    item = item[:, [18, 19, 4, 5, 15, 0], :]
                else:
                    item = item[:, [2, 3, 4, 5, 0, 1], :]
            data_len = len(data['ori'])
            split_len = data_len // seq_len
            s_len = data_len % seq_len
            for ii in range(seq_len):

                item_ = item[ii * seq_len:(ii + 1) * seq_len]

                if len(item_) == 0:
                    break
                if key not in all_data:
                    all_data[key] = [item_, ]
                else:
                    # print(len(item))
                    all_data[key].append(item_)
            if s_len > 0:
                item_ = item[(ii + 1) * seq_len:]
                if len(item_) != 0:
                    if key not in all_data:
                        all_data[key] = [item_, ]
                    else:
                        all_data[key].append(item_)
        #print()


if __name__ == '__main__':
    filedir = "../acc2pos/dataset/*"
    filedir2 = "../acc2pos/DIP_IMU_and_Others/DIP_IMU/*/*_syn.pkl"
    action_type = glob(filedir)
    print(action_type)
    seq_len = 256
    all_data = {}
    files = []
    for type_ in action_type:
        print(type_)
        actions = type_ + "/*/*_syn.pkl"
        files.extend(list(glob(actions)))
    files.extend(list(glob(filedir2)))
    print(files)

    progress(files)
    #   #  if (idx+1)%1000==0:
    np.savez(f"../acc2pos/dataset/merge_data_notrans.npz", **all_data)
    
    # data = pickle.load(open("/Users/oswin/Downloads/amass/merge_data.pkl", 'rb'))
    #
    # print(type(data['acc']))
    # print(type(data['ori']))
