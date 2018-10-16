import os
import shutil
from glob import glob
import ntpath

root_dir = '/home/siiva/RUDY/SIIVA/GoalCam/annotations/NBA_CBA_CUBA/'

dst = 'ALL'
src = '190918/002'
src_name = '190918_002'
src_files = glob(os.path.join(root_dir, src, 'training/image_2', '*.jpg'))
src_labels = glob(os.path.join(root_dir, src, 'training/label_2', '*.txt'))

for file_ in src_files:
    shutil.copy(file_, os.path.join(root_dir, dst, 'training/image_2', src_name + '_' + ntpath.basename(file_)))

for label in src_labels:
    shutil.copy(label, os.path.join(root_dir, dst, 'training/label_2', src_name + '_' + ntpath.basename(label)))
