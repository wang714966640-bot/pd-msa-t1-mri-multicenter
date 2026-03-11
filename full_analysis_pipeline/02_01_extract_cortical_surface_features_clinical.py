import os
import pandas as pd
import nibabel as nib
import numpy as np
from brainspace.datasets import load_conte69, load_mask
from enigmatoolbox.utils.parcellation import surface_to_parcel

# 读取demo文件
file_path = '/media/neurox/T7_Shield/PD_analyse/demo_yanzheng_clinical2.xlsx'
demo = pd.read_excel(file_path)

# 获取被试数量
n_subjects = len(demo['sub'])
print(f"被试数量: {n_subjects}")

# 加载mask
mask_conte = load_mask(join=True)

# 数据路径
datapath = '/media/neurox/T7_Shield/PD/download_yanzheng_clinical2'

# 初始化数据数组: (400个ROI, n_subjects个被试, 2个测量值: area和thickness)
tank = np.zeros((400, n_subjects, 2))
# 遍历每个被试
for id in range(n_subjects):
    subject_id = demo['sub'][id]
    print(f"处理被试 {id+1}/{n_subjects}: {subject_id}")
    
    try:
        # 处理 area 数据
        meas = 'area'
        tmp = []
        for _, h in enumerate(['lh', 'rh']):
            filename = f'anat/conte69-32k_desc-{h}_{meas}_10mm.mgh'
            full_path = os.path.join(datapath, subject_id, filename)
            
            if not os.path.exists(full_path):
                print(f"警告: 文件不存在 {full_path}")
                continue
                
            data = nib.load(full_path).dataobj[:, 0, 0]
            tmp = np.append(tmp, np.asarray(data))
        
        if len(tmp) > 0:
            tmp = tmp * mask_conte
            sch400 = surface_to_parcel(tmp, 'schaefer_400_conte69')
            tank[:, id, 0] = sch400[1:401]  # 修正: 使用索引0存储area
        
        # 处理 thickness 数据
        meas = 'thickness'
        tmp = []
        for _, h in enumerate(['lh', 'rh']):
            filename = f'anat/conte69-32k_desc-{h}_{meas}_10mm.mgh'
            full_path = os.path.join(datapath, subject_id, filename)
            
            if not os.path.exists(full_path):
                print(f"警告: 文件不存在 {full_path}")
                continue
                
            data = nib.load(full_path).dataobj[:, 0, 0]
            tmp = np.append(tmp, np.asarray(data))
        
        if len(tmp) > 0:
            tmp = tmp * mask_conte
            sch400 = surface_to_parcel(tmp, 'schaefer_400_conte69')
            tank[:, id, 1] = sch400[1:401]  # 修正: 使用索引1存储thickness
            
    except Exception as e:
        print(f"处理被试 {subject_id} 时出错: {e}")
        continue

# 保存数据
output_path = '/media/neurox/T7_Shield/PD_analyse/surface_yanzheng_clinical2.npy'
np.save(output_path, tank)

print(f"数据保存完成: {output_path}")
print(f"数据形状: {tank.shape}")
print(f"数据说明: (400个ROI, {n_subjects}个被试, 2个测量值)")
print(f"测量值: [0]=area, [1]=thickness")