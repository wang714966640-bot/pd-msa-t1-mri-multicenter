import os
import numpy as np
import pandas as pd
import nibabel as nib
from brainspace.plotting import plot_hemispheres
from brainspace.datasets import load_conte69, load_parcellation, load_mask
from enigmatoolbox.utils.parcellation import surface_to_parcel, parcel_to_surface
from brainstat.stats.SLM import SLM
from brainstat.datasets import fetch_mask, fetch_template_surface
from brainstat.stats.terms import FixedEffect, MixedEffect
from brainspace.utils.parcellation import map_to_labels
import seaborn as sns
import matplotlib.pyplot as plt

# 设置路径（使用原始字符串避免转义问题）
data_dir = r"E:\PD_analyse"
print(f'DataDir: "{data_dir}"')

# 创建输出目录
output_dir = os.path.join(data_dir, "ALL_AGE_spinperm")
os.makedirs(output_dir, exist_ok=True)

# 加载人口统计学数据
demo_path = os.path.join(data_dir, "demo_all.xlsx")
demo = pd.read_excel(demo_path, sheet_name=0)
demo['age'] = demo['age'] - demo['age'].mean()  # 年龄去中心化

# 保存处理后的数据
output_demo_path = os.path.join(data_dir, "processed_demo_all.xlsx")
demo.to_excel(output_demo_path, index=False)
print(f"处理后的数据已保存到 {output_demo_path}")

# 加载脑模板和掩膜
mask_conte = load_mask(join=True)  # 确认参数名是否正确
surf_lh, surf_rh = load_conte69()
surf_conte = load_conte69(join=True)

# 遍历每个被试的MRI数据
base_path = r"E:\PD\download"
tank = []
for i, subject_id in enumerate(demo['sub']):
    print(f"Processing subject: {subject_id}")
    subject_path = os.path.join(base_path, subject_id, 'anat')
    ct = []

    # 处理左右半球
    for hemisphere in ['lh', 'rh']:
        file_name = f"conte69-32k_desc-{hemisphere}_area_10mm.mgh"
        file_path = os.path.join(subject_path, file_name)

        # 加载并提取数据
        data = nib.load(file_path).get_fdata().flatten()
        ct.append(data)

    # 合并半球数据并应用掩膜
    ct_combined = np.concatenate(ct) * mask_conte

    # 转换为Schaefer 400分区
    CT_schaefer_400 = surface_to_parcel(ct_combined, 'schaefer_400_conte69')
    tank.append(CT_schaefer_400)

# 保存结果
output_file = os.path.join(data_dir, "CT_schaefer_400_results.csv")
np.savetxt(output_file, tank, delimiter=',')
np.save(os.path.join(data_dir, "ct_parcel.npy"), np.array(tank))

#%%load numpy array
ct = np.load(os.path.join(data_dir, "ct_parcel.npy"))
# subcor_data = [282, 14]

# # as.factor
# demo['gender'] = demo['gender'].astype('category')

# construct the model
term_age = FixedEffect(demo.age)
term_sex = FixedEffect(demo.gender)
group = FixedEffect(demo.group)
model = term_age + term_sex + group + term_age*group

# fit slm, set contrasts
contrast_sex = (demo.gender == "M").astype(int) - (demo.gender == "F").astype(int)
contrast_age = demo.age
contrast_group = demo.group
contrast_site = demo.site

# read annotations of networks
import pandas as pd
annot = pd.read_csv("E:/R_done/analysis/surfaces/parcellations/lut/lut_schaefer-400_mics.csv")
annot = annot.drop('roi', axis=1)
annot.loc[0:200, 'mics'] = annot.loc[0:200, 'mics'] - 1000
annot.loc[201:401, 'mics'] = annot.loc[201:401, 'mics'] - 1900
annot = annot.drop([0, 201])
annot = annot.reset_index(drop=True)
print(annot.head())

# Extract labels from the annotation file
labels = annot['label'].values[:401]

# colormap
colormap1 = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r',
             'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2']
colormap2 = ['Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r',
             'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r']
colormap3 = ['PuBu','PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r',
             'Purples', 'Purples_r', 'RdBu','RdBu_r', 'RdGy']
colormap4 = ['RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r','Set1', 'Set1_r', 'Set2',
             'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r']
colormap5 = ['YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r',
             'autumn','autumn_r', 'binary', 'binary_r', 'bone']
colormap6 = ['bwr_r', 'cividis', 'cividis_r','cool', 'cool_r', 'coolwarm',
             'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r']
colormap7 = ['gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r','gist_rainbow',
             'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg',
             'gist_yarg_r', 'gnuplot', 'gnuplot2']
colormap8 = ['hot_r', 'hsv', 'hsv_r', 'inferno', 'jet_r',
             'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r',
             'pink', 'pink_r']
colormap9 = ['prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring',
             'spring_r', 'summer', 'summer_r','tab10', 'tab10_r', 'tab20']
colormap10 = ['tab20c_r', 'terrain', 'terrain_r', 'turbo','turbo_r', 'twilight', 'twilight_r',
              'twilight_shifted', 'twilight_shifted_r']
colormap11 = ['GnBu_r', 'Greens', 'PiYG', 'PiYG_r', 'RdPu_r', 'RdYlBu', 'YlGn','YlGnBu',
              'gist_gray', 'gist_gray_r', 'gray', 'gray_r', 'hot']
colormap12 = ['plasma', 'plasma_r', 'tab20b_r', 'tab20c', 'winter','winter_r',
              'brg_r', 'bwr','gist_earth', 'gist_earth_r','gnuplot2_r', 'gnuplot_r', 'tab20b']
colormap13 = ['tab20c', 'Dark2_r', 'GnBu', 'Pastel2', 'Pastel2_r', 'RdGy_r',
              'RdPu', 'bone_r', 'brg', 'inferno_r', 'jet', 'viridis', 'viridis_r', 'tab20_r']
colormap = ['tab20c']

## loop the comparisons
from os import path
import shutil
for cmap in colormap6:

    print(cmap)
    #是否有意义（生成的空文件夹）
    if path.exists('surf_plot/' + cmap):
        shutil.rmtree('surf_plot/' + cmap)

    os.mkdir('surf_plot/' + cmap)

#%% 定义数据类型和比较类型
data_types = ['area']
comparisons = ['HC-PD','HC-MSA', 'PD-MSA','HC-RBD','RBD-MSA', 'RBD-PD']

# 遍历数据类型和比较类型
for data_type in data_types:
    for comparison in comparisons:

        # data_type = ['area']
        # comparison = 'HC-PD'
        # 加载相应的数据
        print(f'Loading data for {data_type} and {comparison}')
        # remove first row of data
        data = ct[:,1:]

        # 定义比较
        comp1, comp2 = comparison.split('-')
        print(comp1, comp2)
        comparison_array = (demo.group == comp1).astype(int) - (demo.group == comp2).astype(int)

        # 进行统计分析
        slm_comparison = SLM(model, comparison_array, correction="fdr")
        slm_comparison.fit(data)

        # 构建对比：年龄之间的差异
        contrast_age = demo.age
        slm_age = SLM(model, contrast_age, correction=["fdr"])
        slm_age.fit(data)

        # 获取 FDR 校正后的结果
        parcel_age = slm_age.Q
        labeling = load_parcellation('schaefer', scale=400, join=True)
        vert_age = map_to_labels(parcel_age, labeling, mask=labeling != 0, fill=np.nan)

        # 绘制 FDR 校正后的结果图
        name_age = os.path.join(output_dir, f'{comparison}-{data_type}-age-compare.png')
        plot_hemispheres(surf_lh, surf_rh, array_name=vert_age, size=(900, 250), color_bar='right',
                         screenshot=True, embed_nb=False, interactive=False, share='both',
                         filename=name_age, color_range=(0, 0.05), label_text=[f'{comparison} P'],
                         transparent_bg=False, scale=2, nan_color=(0.5, 0.5, 0.5, 1),
                         zoom=1, cmap='Reds_r')

        # 获取 t 结果
        parcel_t_age = slm_age.t
        print(np.mean(parcel_t_age))
        vert_t_age = map_to_labels(parcel_t_age, labeling, mask=labeling != 0, fill=np.nan)

        # 绘制 t结果图
        name_t_age = os.path.join(output_dir, f'{comparison}-{data_type}_age.png')
        plot_hemispheres(surf_lh, surf_rh, array_name=vert_t_age, size=(900, 250), color_bar='right',
                         screenshot=True, embed_nb=False, interactive=False, share='both',
                         filename=name_t_age, color_range=(-3, 3), label_text=['beta'],
                         transparent_bg=False, scale=2, nan_color=(0.5, 0.5, 0.5, 1),
                         zoom=1, cmap='RdBu_r')

        # 保存统计结果
        stats_age = pd.DataFrame(index=range(400), columns=['label', 't-value', 'FDR'])
        stats_age['label'] = labels
        stats_age['t-value'] = np.reshape(slm_age.t, -1)
        stats_age['FDR'] = slm_age.Q
        stats_age.to_csv(os.path.join(output_dir, f'stats_{comparison}-{data_type}_age.csv'), index=False)

        # 构建对比：组别之间的差异
        contrast_group = (demo.group == comp1).astype(int) - (demo.group == comp2).astype(int)
        slm_group = SLM(model, contrast_group, correction=["fdr"])
        slm_group.fit(data)

        # 获取 FDR 校正后的结果
        parcel_group = slm_group.Q
        vert_group = map_to_labels(parcel_group, labeling, mask=labeling != 0, fill=np.nan)

        # 绘制 FDR 校正后的结果图
        name_group = os.path.join(output_dir, f'{comparison}-{data_type}-group-compare.png')
        plot_hemispheres(surf_lh, surf_rh, array_name=vert_group, size=(900, 250), color_bar='right',
                         screenshot=True, embed_nb=False, interactive=False, share='both',
                         filename=name_group, color_range=(0, 0.05), label_text=[f'{comparison} P'],
                         transparent_bg=False, scale=2, nan_color=(0.5, 0.5, 0.5, 1),
                         zoom=1, cmap='Reds_r')

        # 获取 t 结果
        parcel_t_group = slm_group.t
        print(np.mean(parcel_t_group))
        vert_t_group = map_to_labels(parcel_t_group, labeling, mask=labeling != 0, fill=np.nan)

        # 绘制 t结果图
        name_t_group = os.path.join(output_dir, f'{comparison}-{data_type}_group.png')
        plot_hemispheres(surf_lh, surf_rh, array_name=vert_t_group, size=(900, 250), color_bar='right',
                         screenshot=True, embed_nb=False, interactive=False, share='both',
                         filename=name_t_group, color_range=(-3, 3), label_text=['beta'],
                         transparent_bg=False, scale=2, nan_color=(0.5, 0.5, 0.5, 1),
                         zoom=1, cmap='RdBu_r')

        # 保存统计结果
        stats_group = pd.DataFrame(index=range(400), columns=['label', 't-value', 'FDR'])
        stats_group['label'] = labels
        stats_group['t-value'] = np.reshape(slm_group.t, -1)
        stats_group['FDR'] = slm_group.Q
        stats_group.to_csv(os.path.join(output_dir, f'stats_{comparison}-{data_type}_group.csv'), index=False)

        # 构建对比：年龄与组别交互作用
        contrast_age_group = (demo.age * (demo.group == comp1)) - (demo.age * (demo.group == comp2))
        slm_age_group = SLM(model, contrast_age_group, correction=["fdr"])
        slm_age_group.fit(data)

        # 获取 FDR 校正后的结果
        parcel_age_group = slm_age_group.Q
        vert_age_group = map_to_labels(parcel_age_group, labeling, mask=labeling != 0, fill=np.nan)

        # 绘制 FDR 校正后的结果图
        name_age_group = os.path.join(output_dir, f'{comparison}-{data_type}-age-group-compare.png')
        plot_hemispheres(surf_lh, surf_rh, array_name=vert_age_group, size=(900, 250), color_bar='right',
                         screenshot=True, embed_nb=False, interactive=False, share='both',
                         filename=name_age_group, color_range=(0, 0.05), label_text=[f'{comparison} P'],
                         transparent_bg=False, scale=2, nan_color=(0.5, 0.5, 0.5, 1),
                         zoom=1, cmap='Reds_r')

        # 获取 t 结果
        parcel_t_age_group = slm_age_group.t
        print(np.mean(parcel_t_age_group))
        vert_t_age_group = map_to_labels(parcel_t_age_group, labeling, mask=labeling != 0, fill=np.nan)

        # 绘制 t结果图
        name_t_age_group = os.path.join(output_dir, f'{comparison}-{data_type}_age_group.png')
        plot_hemispheres(surf_lh, surf_rh, array_name=vert_t_age_group, size=(900, 250), color_bar='right',
                         screenshot=True, embed_nb=False, interactive=False, share='both',
                         filename=name_t_age_group, color_range=(-3, 3), label_text=['beta'],
                         transparent_bg=False, scale=2, nan_color=(0.5, 0.5, 0.5, 1),
                         zoom=1, cmap='RdBu_r')

        # 保存统计结果
        stats_age_group = pd.DataFrame(index=range(400), columns=['label', 't-value', 'FDR'])
        stats_age_group['label'] = labels
        stats_age_group['t-value'] = np.reshape(slm_age_group.t, -1)
        stats_age_group['FDR'] = slm_age_group.Q
        stats_age_group.to_csv(os.path.join(output_dir, f'stats_{comparison}-{data_type}_age_group.csv'), index=False)

        # hc - pd
        hc_pd = (demo.group == "HC").astype(int) - (demo.group == "PD").astype(int)
        slm_hcpd = SLM(model, hc_pd, correction="fdr")
        slm_hcpd.fit(data)
        p_parcel = np.copy(slm_hcpd.Q)
        p_vertex = parcel_to_surface(p_parcel, 'schaefer_400_conte69')
        name = 'surf_plot/%s/hc-pd-fdrP_parcel.png' % cmap
        plot_hemispheres(surf_lh, surf_rh, array_name=p_vertex, size=(900, 250),color_bar='bottom',
                         screenshot=True,embed_nb=False,interactive=False,share='both',
                         filename=name,color_range=(0,0.05),label_text=['HC-PD P'],
                         transparent_bg=False,scale=2,nan_color=(0, 0, 0, 1),
                         zoom=1, cmap=cmap)

        # hc-pd statistics
        stats = ({'label':[],
                  't':[],
                  'FDR':[]})
        stats['label'] = labels
        stats['FDR'] = list(p_parcel[:400])
        stats['t'] = list(np.transpose(np.copy(slm_hcpd.t))[:400])
        stats = pd.DataFrame(stats)
        stats.to_csv('stats_HC-PD.csv', index=False)

        # hc - msa
        hc_msa = (demo.group == "HC").astype(int) - (demo.group == "MSA").astype(int)
        slm_hcmsa = SLM(model, hc_msa, correction="fdr")
        slm_hcmsa.fit(data)
        p_parcel = np.copy(slm_hcmsa.Q)
        p_vertex = parcel_to_surface(p_parcel, 'schaefer_400_conte69')
        name = 'surf_plot/%s/hc-msa-fdrP_parcel.png' % cmap
        plot_hemispheres(surf_lh, surf_rh, array_name=p_vertex, size=(900, 250), color_bar='bottom',
                         screenshot=True, embed_nb=False, interactive=False, share='both',
                         filename=name, color_range=(0, 0.05), label_text=['HC-MSA P'],
                         transparent_bg=False, scale=2, nan_color=(0, 0, 0, 1),
                         zoom=1, cmap=cmap)

        # hc-msa statistics
        stats = ({'label':[],
                  't':[],
                  'FDR':[]})
        stats['label'] = labels
        stats['FDR'] = list(p_parcel[:400])
        stats['t'] = list(np.transpose(np.copy(slm_hcmsa.t))[:400])
        stats = pd.DataFrame(stats)
        stats.to_csv('stats_HC-MSA.csv', index=False)

        # pd - msa
        pd_msa = (demo.group == "PD").astype(int) - (demo.group == "MSA").astype(int)
        slm_pdmsa = SLM(model, pd_msa, correction="fdr")
        slm_pdmsa.fit(data)
        p_parcel = np.copy(slm_pdmsa.Q)
        p_vertex = parcel_to_surface(p_parcel, 'schaefer_400_conte69')
        name = 'surf_plot/%s/pd-msa-fdrP_parcel.png' % cmap
        plot_hemispheres(surf_lh, surf_rh, array_name=p_vertex, size=(900, 250), color_bar='bottom',
                         screenshot=True, embed_nb=False, interactive=False, share='both',
                         filename=name, color_range=(0, 0.05), label_text=['PD-MSA P'],
                         transparent_bg=False, scale=2, nan_color=(0, 0, 0, 1),
                         zoom=1, cmap=cmap)

        # pd-msa statistics
        stats = ({'label':[],
                  't':[],
                  'FDR':[]})
        stats['label'] = labels
        stats['FDR'] = list(p_parcel[:400])
        stats['t'] = list(np.transpose(np.copy(slm_pdmsa.t))[:400])
        stats = pd.DataFrame(stats)
        stats.to_csv('stats_PD-MSA.csv', index=False)
        # hc - rbd
        hc_rbd_contrast = (demo.group == "HC").astype(int) - (demo.group == "RBD").astype(int)
        slm_hcrbd = SLM(model, hc_rbd_contrast, correction="fdr")
        slm_hcrbd.fit(data)
        p_parcel = np.copy(slm_hcrbd.Q) # p_parcel变量在每个块内被重新赋值和使用
        p_vertex = parcel_to_surface(p_parcel, 'schaefer_400_conte69')
        name = os.path.join('surf_plot', cmap, 'hc-rbd-fdrP_parcel.png')
        plot_hemispheres(surf_lh, surf_rh, array_name=p_vertex, size=(900, 250), color_bar='bottom',
                         screenshot=True, embed_nb=False, interactive=False, share='both',
                         filename=name, color_range=(0, 0.05), label_text=['HC-RBD P'],
                         transparent_bg=False, scale=2, nan_color=(0, 0, 0, 1),
                         zoom=1, cmap=cmap)

        # hc-rbd statistics
        stats = ({'label':[], 't':[], 'FDR':[]}) # stats字典在每个块内被重新初始化和使用
        stats['label'] = labels
        stats['FDR'] = list(p_parcel[:400])
        stats['t'] = list(np.transpose(np.copy(slm_hcrbd.t))[:400])
        current_stats_df = pd.DataFrame(stats)
        current_stats_df.to_csv(os.path.join(output_dir, 'stats_HC-RBD.csv'), index=False)

        # rbd - msa
        rbd_msa_contrast = (demo.group == "RBD").astype(int) - (demo.group == "MSA").astype(int)
        slm_rbdmsa = SLM(model, rbd_msa_contrast, correction="fdr")
        slm_rbdmsa.fit(data)
        p_parcel = np.copy(slm_rbdmsa.Q)
        p_vertex = parcel_to_surface(p_parcel, 'schaefer_400_conte69')
        name = os.path.join('surf_plot', cmap, 'rbd-msa-fdrP_parcel.png')
        plot_hemispheres(surf_lh, surf_rh, array_name=p_vertex, size=(900, 250), color_bar='bottom',
                         screenshot=True, embed_nb=False, interactive=False, share='both',
                         filename=name, color_range=(0, 0.05), label_text=['RBD-MSA P'],
                         transparent_bg=False, scale=2, nan_color=(0, 0, 0, 1),
                         zoom=1, cmap=cmap)

        # rbd-msa statistics
        stats = ({'label':[], 't':[], 'FDR':[]})
        stats['label'] = labels
        stats['FDR'] = list(p_parcel[:400])
        stats['t'] = list(np.transpose(np.copy(slm_rbdmsa.t))[:400])
        current_stats_df = pd.DataFrame(stats)
        current_stats_df.to_csv(os.path.join(output_dir, 'stats_RBD-MSA.csv'), index=False)

        # rbd - pd
        rbd_pd_contrast = (demo.group == "RBD").astype(int) - (demo.group == "PD").astype(int)
        slm_rbdpd = SLM(model, rbd_pd_contrast, correction="fdr")
        slm_rbdpd.fit(data)
        p_parcel = np.copy(slm_rbdpd.Q)
        p_vertex = parcel_to_surface(p_parcel, 'schaefer_400_conte69')
        name = os.path.join('surf_plot', cmap, 'rbd-pd-fdrP_parcel.png')
        plot_hemispheres(surf_lh, surf_rh, array_name=p_vertex, size=(900, 250), color_bar='bottom',
                         screenshot=True, embed_nb=False, interactive=False, share='both',
                         filename=name, color_range=(0, 0.05), label_text=['RBD-PD P'],
                         transparent_bg=False, scale=2, nan_color=(0, 0, 0, 1),
                         zoom=1, cmap=cmap)

        # rbd-pd statistics
        stats = ({'label':[], 't':[], 'FDR':[]})
        stats['label'] = labels
        stats['FDR'] = list(p_parcel[:400])
        stats['t'] = list(np.transpose(np.copy(slm_rbdpd.t))[:400])
        current_stats_df = pd.DataFrame(stats)
        current_stats_df.to_csv(os.path.join(output_dir, 'stats_RBD-PD.csv'), index=False)