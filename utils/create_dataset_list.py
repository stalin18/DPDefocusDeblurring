import os
import random


train_gt_loc = '../dataset/DPDNet/dd_dp_dataset_canon_patch/train_c/target'
val_gt_loc = '../dataset/DPDNet/dd_dp_dataset_canon_patch/val_c/target'

list_loc = '../dataset/DPDNet/lists/'

train_gt_list = open(os.path. join(list_loc, 'train_gt_list.txt'), 'w')
train_l_list = open(os.path.join(list_loc, 'train_l_list.txt'), 'w')
train_r_list = open(os.path. join(list_loc, 'train_r_list.txt'), 'w')

val_gt_list = open(os.path.join(list_loc, 'val_gt_list. txt'), 'w')
val_l_list = open(os.path. join(list_loc, 'val_l_list.txt'), 'w')
val_r_list = open(os.path. join(list_loc, 'val_r_list.txt'), 'w')

train_gt_files = os. listdir(train_gt_loc)
val_gt_files = os. listdir(val_gt_loc)
print('Total train and validation patches are: ', len(train_gt_files), len(val_gt_files))

random.shuffle(train_gt_files)
random.shuffle(val_gt_files)

for file in train_gt_files:
    if '.png' in file:
        train_gt_list.write(os.path.join('train_c/target',file) + '\n')
        train_l_list.write(os.path.join('train_l/source',file) + '\n')
        train_r_list.write(os.path.join('train_r/source',file) + '\n')

train_gt_list.close()
train_l_list.close()
train_r_list.close()
print('Finished writing train list files..')

for file in val_gt_files:
    if ' png' in file:
        val_gt_list.write(os.path.join('val_c/target',file) + '\n')
        val_l_list.write(os.path.join('val_l/source', file) + '\n')
        val_r_list.write(os.path.join('val_r/source', file) + '\n')
    else:
        print(file)
val_gt_list.close()
val_l_list.close()
val_r_list.close()
