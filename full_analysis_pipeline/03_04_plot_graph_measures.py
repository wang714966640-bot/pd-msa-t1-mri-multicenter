#%%绘制用于模型训练的综合神经影像指标：FC/SC的拓扑属性全局直方图和脑图
import pandas as pd
import numpy as np
from brainspace.datasets import load_conte69, load_mask, load_parcellation
from brainspace.plotting import plot_hemispheres
from brainspace.utils.parcellation import map_to_labels
import matplotlib.pyplot as plt

#%% load demo
demo = pd.read_excel('E:\PD_analyse\demo1126.xlsx', sheet_name=0)

# # Find the row indices where age > 240 and group is 'BP'
# idx = demo[(demo['age'] > 240) & (demo['group'] == 'BP')].index
#
# # Remove the rows in demo using the indices
# demo = demo.drop(idx)

#%% local measures（脑图）
clustering_fc = pd.read_csv('E:/PD_analyse/FC_SC_MatPlots/clustering_fc.csv', header=None)
clustering_sc = pd.read_csv('E:/PD_analyse/FC_SC_MatPlots/clustering_sc.csv', header=None)
local_efficiency_fc = pd.read_csv('E:/PD_analyse/FC_SC_MatPlots/local_efficiency_fc.csv', header=None)
local_efficiency_sc = pd.read_csv('E:/PD_analyse/FC_SC_MatPlots/local_efficiency_sc.csv', header=None)
degree_fc = pd.read_csv('E:/PD_analyse/FC_SC_MatPlots/degree_fc.csv', header=None)
degree_sc = pd.read_csv('E:/PD_analyse/FC_SC_MatPlots/degree_sc.csv', header=None)
betweenness_fc = pd.read_csv('E:/PD_analyse/FC_SC_MatPlots/betweenness_fc.csv', header=None)
betweenness_sc = pd.read_csv('E:/PD_analyse/FC_SC_MatPlots/betweenness_sc.csv', header=None)

# # drop the idx of local measures
# clustering_fc = clustering_fc.drop(idx)
# clustering_sc = clustering_sc.drop(idx)
# local_efficiency_fc = local_efficiency_fc.drop(idx)
# local_efficiency_sc = local_efficiency_sc.drop(idx)
# degree_fc = degree_fc.drop(idx)
# degree_sc = degree_sc.drop(idx)
# betweenness_fc = betweenness_fc.drop(idx)
# betweenness_sc = betweenness_sc.drop(idx)

# calculate the mean of each measure
clustering_fc_mean = clustering_fc.mean(axis=0)
clustering_sc_mean = clustering_sc.mean(axis=0)
local_efficiency_fc_mean = local_efficiency_fc.mean(axis=0)
local_efficiency_sc_mean = local_efficiency_sc.mean(axis=0)
degree_fc_mean = degree_fc.mean(axis=0)
degree_sc_mean = degree_sc.mean(axis=0)
betweenness_fc_mean = betweenness_fc.mean(axis=0)
betweenness_sc_mean = betweenness_sc.mean(axis=0)

# print the min and max of each measure
print('clustering_fc_mean: ', clustering_fc_mean.min(), clustering_fc_mean.max())
print('clustering_sc_mean: ', clustering_sc_mean.min(), clustering_sc_mean.max())
print('local_efficiency_fc_mean: ', local_efficiency_fc_mean.min(), local_efficiency_fc_mean.max())
print('local_efficiency_sc_mean: ', local_efficiency_sc_mean.min(), local_efficiency_sc_mean.max())
print('degree_fc_mean: ', degree_fc_mean.min(), degree_fc_mean.max())
print('degree_sc_mean: ', degree_sc_mean.min(), degree_sc_mean.max())
print('betweenness_fc_mean: ', betweenness_fc_mean.min(), betweenness_fc_mean.max())
print('betweenness_sc_mean: ', betweenness_sc_mean.min(), betweenness_sc_mean.max())

#%% plot the measures
# load mask, labels, surface
labeling = load_parcellation('schaefer', scale=400, join=True)
mask = labeling != 0
surf_lh, surf_rh = load_conte69()

# plot the measures
measures = [map_to_labels(clustering_fc_mean, labeling, mask=mask, fill=np.nan),
            map_to_labels(clustering_sc_mean, labeling, mask=mask, fill=np.nan),
            map_to_labels(local_efficiency_fc_mean, labeling, mask=mask, fill=np.nan),
            map_to_labels(local_efficiency_sc_mean, labeling, mask=mask, fill=np.nan),
            map_to_labels(degree_fc_mean, labeling, mask=mask, fill=np.nan),
            map_to_labels(degree_sc_mean, labeling, mask=mask, fill=np.nan),
            map_to_labels(betweenness_fc_mean, labeling, mask=mask, fill=np.nan),
            map_to_labels(betweenness_sc_mean, labeling, mask=mask, fill=np.nan)]
labels = ['Clustering FC', 'Clustering SC', 'Efficiency FC', 'Efficiency SC',
            'Degree FC', 'Degree SC', 'Betweenness FC', 'Betweenness SC']
color_range = [(0.2, 0.45), (0, 0.7), (0, 0.15), (0.5, 1), (20, 80), (12300, 80000),
               (46, 800), (100, 2000)]
filename = r'E:/PD_analyse/local_measures.png'
plot_hemispheres(surf_lh, surf_rh, array_name=measures, size=(1000, 1600), color_range=color_range,
                 cmap='PiYG_r', color_bar=True, filename=filename, transparent_bg=False,
                 screenshot=True, embed_nb=False, interactive=False, nan_color=(0.5, 0.5, 0.5, 1),
                 label_text=labels, zoom=1.2)


#%% global measures（直方图）
characteristic_path_length_fc = pd.read_csv('E:/PD_analyse/FC_SC_MatPlots/characteristic_path_length_fc.csv', header=None)
characteristic_path_length_sc = pd.read_csv('E:/PD_analyse/FC_SC_MatPlots/characteristic_path_length_sc.csv', header=None)
global_efficiency_fc = pd.read_csv('E:/PD_analyse/FC_SC_MatPlots/global_efficiency_fc.csv', header=None)
global_efficiency_sc = pd.read_csv('E:/PD_analyse/FC_SC_MatPlots/global_efficiency_sc.csv', header=None)
modularity_fc = pd.read_csv('E:/PD_analyse/FC_SC_MatPlots/modularity_fc.csv', header=None)
modularity_sc = pd.read_csv('E:/PD_analyse/FC_SC_MatPlots/modularity_sc.csv', header=None)

# remove the idx in global measures
# characteristic_path_length_fc = characteristic_path_length_fc.drop(idx)
# characteristic_path_length_sc = characteristic_path_length_sc.drop(idx)
# global_efficiency_fc = global_efficiency_fc.drop(idx)
# global_efficiency_sc = global_efficiency_sc.drop(idx)
# modularity_fc = modularity_fc.drop(idx)
# modularity_sc = modularity_sc.drop(idx)
# small_worldness_fc = small_worldness_fc.drop(idx)
# small_worldness_sc = small_worldness_sc.drop(idx)

idx1 = demo['group'] == 'HC'
idx2 = demo['group'] == 'PD'
idx3 = demo['group'] == 'MSA'


# %% plot the distribution of global measures in different groups
fig, axes = plt.subplots(1, 6, figsize=(30, 5))  # Changed the subplot layout to 1 row and 6 columns

# First subplot
axes[0].hist(characteristic_path_length_fc[idx1].values, bins=20, color='g', alpha=0.5, label='HC')
axes[0].hist(characteristic_path_length_fc[idx2].values, bins=20, color='b', alpha=0.5, label='PD')
axes[0].hist(characteristic_path_length_fc[idx3].values, bins=20, color='r', alpha=0.5, label='MSA')
axes[0].set_title('Characteristic Path Length FC', fontdict={'fontsize': 18,'fontweight': 'bold'})
axes[0].legend(fontsize=18, frameon=False, bbox_to_anchor=(0.63, 1))
axes[0].locator_params(axis='both', nbins=5)  # Set the number of ticks
axes[0].tick_params(axis='both', labelsize=18)  # Enlarge the tick label size

# Second subplot
axes[1].hist(global_efficiency_fc[idx1].values, bins=20, color='g', alpha=0.5, label='HC')
axes[1].hist(global_efficiency_fc[idx2].values, bins=20, color='b', alpha=0.5, label='PD')
axes[1].hist(global_efficiency_fc[idx3].values, bins=20, color='r', alpha=0.5, label='MSA')
axes[1].set_title('Global Efficiency FC', fontdict={'fontsize': 18,'fontweight': 'bold'})
axes[1].locator_params(axis='both', nbins=5)  # Set the number of ticks
axes[1].tick_params(axis='both', labelsize=18)  # Enlarge the tick label size

# Third subplot
axes[2].hist(modularity_fc[idx1].values, bins=20, color='g', alpha=0.5, label='HC')
axes[2].hist(modularity_fc[idx2].values, bins=20, color='b', alpha=0.5, label='PD')
axes[2].hist(modularity_fc[idx3].values, bins=20, color='r', alpha=0.5, label='MSA')
axes[2].set_title('Modularity FC', fontdict={'fontsize': 18,'fontweight': 'bold'})
axes[2].locator_params(axis='both', nbins=5)  # Set the number of ticks
axes[2].tick_params(axis='both', labelsize=18)  # Enlarge the tick label size

# Fourth subplot
axes[3].hist(characteristic_path_length_sc[idx1].values, bins=20, color='g', alpha=0.5, label='HC')
axes[3].hist(characteristic_path_length_sc[idx2].values, bins=20, color='b', alpha=0.5, label='PD')
axes[3].hist(characteristic_path_length_sc[idx3].values, bins=20, color='r', alpha=0.5, label='MSA')
axes[3].set_title('Characteristic Path Length SC', fontdict={'fontsize': 18,'fontweight': 'bold'})
axes[3].locator_params(axis='both', nbins=5)  # Set the number of ticks
axes[3].tick_params(axis='both', labelsize=18)  # Enlarge the tick label size

# Fifth subplot
axes[4].hist(global_efficiency_sc[idx1].values, bins=20, color='g', alpha=0.5, label='HC')
axes[4].hist(global_efficiency_sc[idx2].values, bins=20, color='b', alpha=0.5, label='PD')
axes[4].hist(global_efficiency_sc[idx3].values, bins=20, color='r', alpha=0.5, label='MSA')
axes[4].set_title('Global Efficiency SC', fontdict={'fontsize': 18,'fontweight': 'bold'})
axes[4].locator_params(axis='both', nbins=5)  # Set the number of ticks
axes[4].tick_params(axis='both', labelsize=18)  # Enlarge the tick label size

# Sixth subplot
axes[5].hist(modularity_sc[idx1].values, bins=20, color='g', alpha=0.5, label='HC')
axes[5].hist(modularity_sc[idx2].values, bins=20, color='b', alpha=0.5, label='PD')
axes[5].hist(modularity_sc[idx3].values, bins=20, color='r', alpha=0.5, label='MSA')
axes[5].set_title('Modularity SC', fontdict={'fontsize': 18,'fontweight': 'bold'})
axes[5].locator_params(axis='both', nbins=5)  # Set the number of ticks
axes[5].tick_params(axis='both', labelsize=18)  # Enlarge the tick label size

plt.tight_layout()  # Adjusts the spacing between subplots
plt.show()
plt.close()

# save figure
fig.savefig(r'E:/PD_analyse/global_measures.png', dpi=300)
