import os
import pandas as pd
import nibabel as nib
import numpy as np
from numpy import loadtxt
from brainspace.datasets import load_conte69, load_mask
from enigmatoolbox.utils.parcellation import surface_to_parcel
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '/media/neurox/T7_Shield/PD_analyse/demo_xuanwuyanzheng_1.xls'
demo = pd.read_excel(file_path)

mask_conte = load_mask(join=True)
surf_lh, surf_rh = load_conte69()
surf_conte = load_conte69(join=True)

datapath = '/media/neurox/T7_Shield/PD/download_yanzheng_xuanwu'
tank = np.zeros((400, 305, 5))
for id in range(len(demo['sub'])):
    print(demo['sub'][id])

    # volume
    meas = 'volume'
    tmp = []
    for _, h in enumerate(['lh', 'rh']):
        filename = f'anat/conte69-32k_desc-{h}_{meas}_10mm.mgh'
        full_path = os.path.join(datapath, demo['sub'][id], filename)
        data = nib.load(full_path).dataobj[:, 0, 0]
        tmp = np.append(tmp, np.asarray(data))
    tmp = tmp*mask_conte
    sch400 = surface_to_parcel(tmp, 'schaefer_400_conte69')
    tank[:, id, 0] = sch400[1:401]

    # area
    meas = 'area'
    tmp = []
    for _, h in enumerate(['lh', 'rh']):
        filename = f'anat/conte69-32k_desc-{h}_{meas}_10mm.mgh'
        full_path = os.path.join(datapath, demo['sub'][id], filename)
        data = nib.load(full_path).dataobj[:, 0, 0]
        tmp = np.append(tmp, np.asarray(data))
    tmp = tmp*mask_conte
    sch400 = surface_to_parcel(tmp, 'schaefer_400_conte69')
    tank[:, id, 1] = sch400[1:401]

    # thickness
    meas = 'thickness'
    tmp = []
    for _, h in enumerate(['lh', 'rh']):
        filename = f'anat/conte69-32k_desc-{h}_{meas}_10mm.mgh'
        full_path = os.path.join(datapath, demo['sub'][id], filename)
        data = nib.load(full_path).dataobj[:, 0, 0]
        tmp = np.append(tmp, np.asarray(data))
    tmp = tmp*mask_conte
    sch400 = surface_to_parcel(tmp, 'schaefer_400_conte69')
    tank[:, id, 2] = sch400[1:401]

    # avg_curv
    meas = 'avg_curv'
    tmp = []
    for _, h in enumerate(['lh', 'rh']):
        filename = f'anat/conte69-32k_desc-{h}_{meas}_10mm.mgh'
        full_path = os.path.join(datapath, demo['sub'][id], filename)
        data = nib.load(full_path).dataobj[:, 0, 0]
        tmp = np.append(tmp, np.asarray(data))
    tmp = tmp*mask_conte
    sch400 = surface_to_parcel(tmp, 'schaefer_400_conte69')
    tank[:, id, 3] = sch400[1:401]

    # gau_curv
    meas = 'gau_curv'
    tmp = []
    for _, h in enumerate(['lh', 'rh']):
        filename = f'anat/conte69-32k_desc-{h}_{meas}_10mm.mgh'
        full_path = os.path.join(datapath, demo['sub'][id], filename)
        data = nib.load(full_path).dataobj[:, 0, 0]
        tmp = np.append(tmp, np.asarray(data))
    tmp = tmp*mask_conte
    sch400 = surface_to_parcel(tmp, 'schaefer_400_conte69')
    tank[:, id, 4] = sch400[1:401]

np.save('/media/neurox/T7_Shield/PD_analyse/surface_xuanwuyanzheng_1.npy', tank)

# ==============================================================================
# 补充部分: 将3D数据重塑为2D表格并保存为Excel文件
# ==============================================================================


#%% visual check
# calculate the mean of second dimension of tank
tank_mean = np.mean(tank, axis=1)

# plot tank_mean with labels
plt.figure()
plt.plot(tank_mean[:, 0], label='volume')
plt.plot(tank_mean[:, 1], label='area')
plt.plot(tank_mean[:, 2], label='thickness')
plt.plot(tank_mean[:, 3], label='avg_curv')
plt.plot(tank_mean[:, 4], label='gau_curv')
plt.legend()
plt.show()

# 确保目录存在
output_dir = 'E:\\PD_analyse'
os.makedirs(output_dir, exist_ok=True)

# 设置全局样式为暖色调,创建一个新的 figure
sns.set_theme(style="whitegrid", palette="flare")
plt.figure(figsize=(8, 10))  # 调整为更合适的尺寸

# 定义颜色列表
colors = ['#e41a1c', '#ff7f00', '#a6d75f', '#abd9e9', '#d79b00']

# 遍历每个度量并绘制 KDE 图
for i in range(5):
    plt.subplot(5, 1, i+1)
    # 使用 seaborn 绘制 KDE 图
    sns.kdeplot(data=tank_mean[:, i], bw_adjust=.25, fill=True, color=colors[i], linewidth=2)

    # 设置横纵坐标标题
    if i == 0:
        plt.title('Volume', fontsize=12)
    elif i == 1:
        plt.title('Area', fontsize=12)
    elif i == 2:
        plt.title('Thickness', fontsize=12)
    elif i == 3:
        plt.title('Avg Curv', fontsize=12)
    elif i == 4:
        plt.title('Gau Curv', fontsize=12)
    plt.xlabel('Value', fontsize=10)
    plt.ylabel('Density', fontsize=10)
    # 去除网格线，只保留子图内的横纵坐标线
    plt.grid(False)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', labelsize=8)
    # 调整子图间距
    plt.tight_layout()
    # 保存每个子图
    plt.savefig(os.path.join(output_dir, f'subfigure_{i+1}.png'), dpi=300, bbox_inches='tight')

# 显示图形
plt.show()

# check if there are zeros or nans in tank
np.sum(tank == 0)
np.sum(np.isnan(tank))

#%% load the BOLD and calc the FC
# Initialize the 'tank' as a list to hold arrays of varying second dimension sizes
tank = []

# Initialize the 'corrtank' array
corrtank = np.zeros((418, 400, 400))

# Loop over each subject
for id, subject in enumerate(demo['sub']):
    print(subject)

    # Load bold time series, variable time points each hemisphere
    tmp = [None] * 2
    for num, h in enumerate(['lh', 'rh']):
        tmp[num] = np.asarray(nib.load(datapath + demo['sub'][id] + '/bold/' + 'conte69_{}_10mm.mgh'.
                                       format(h)).get_fdata()[:, 0, 0, :], dtype=np.float32)
    timeseries = np.vstack(tmp)
    ts = np.transpose(timeseries)
    tmp = ts * mask_conte
    sch400 = surface_to_parcel(tmp, 'schaefer_400_conte69')
    print(sch400.shape)

    # The second dimension of sch400 is now variable
    tank.append(sch400[:, 1:401])

    # Calculate correlation matrix for the current subject
    # Ensure that the number of time points is sufficient for correlation calculation
    if tank[-1].shape[1] >= 2:  # At least two time points are needed for correlation
        corr = np.corrcoef(tank[-1], rowvar=False)
        corrtank[id, :, :] = corr
    else:
        print(f"Insufficient time points for correlation calculation for subject {subject}")

    print(f"Processed subject {id + 1}/{len(demo['sub'])}")

# save out
np.save(r'E:\PD_analyse\sch400-correlation-matrix_valid_data_RBD.npy', corrtank)

# %% load SC
# loop the subjects and load the features
sc = np.zeros((418, 400, 400))
for id in range(len(demo['sub'])):
    print(demo['sub'][id])
    df = pd.read_csv(datapath + demo['sub'][id] + '/'+ f'dti/nos_connectome.csv', header=None)
    print(df.shape)
    sc[id, :, :] = df

np.save('E:\PD_analyse\schaefer400-SC_RBD.npy', sc)
#%% plot the heatmap of sc using seaborn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# mean the sc
sc_mean = np.mean(sc, axis=0)

plt.figure()
# set the color range of the heatmap
sns.heatmap(sc_mean, vmin=0, vmax=1, cmap='Reds')
plt.show()

#%% load the fc
fc = np.load(r'E:\PD_analyse\sch400-correlation-matrix_valid_data.npy')
fc_mean = np.mean(fc, axis=0)
plt.figure()
# set the color range of the heatmap
sns.heatmap(fc_mean, vmin=-0, vmax=1, cmap='Reds')
plt.show()
