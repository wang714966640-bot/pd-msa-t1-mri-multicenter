# %%生成每个队列的SC FC、总体平均SC FC和每个脑区各个结构指标的平均值
import numpy as np
import seaborn as sns
import bct
from matplotlib.colors import ListedColormap
import pandas as pd
import re
from brainspace.datasets import load_conte69, load_mask, load_parcellation
from brainspace.plotting import plot_hemispheres
from brainspace.utils.parcellation import map_to_labels
import matplotlib.pyplot as plt

# %% load the functional connectivity matrix and set the diagonal to zero
fc = np.load('E:/PD_analyse/sch400-correlation-matrix_valid_data.npy')
# Loop through the first dimension and set the diagonal of each 400x400 matrix to zero
for i in range(fc.shape[0]):
    np.fill_diagonal(fc[i], 0)
# remove the negative values of fc
fc[fc < 0] = 0

# load the structural connectivity matrix
sc = np.load('E:/PD_analyse/schaefer400-SC.npy')
for i in range(sc.shape[0]):
    sc[i] = bct.threshold_proportional(sc[i, :, :], 0.2)

# create network labels
# Load the roi labels
annot = pd.read_csv("E:/R_done/analysis/surfaces/parcellations/lut/lut_schaefer-400_mics.csv")
annot = annot.drop([0, 201])  # Remove rows 1 and 202

# your labels go here
labels = annot['label']
# reindex the labels
labels.index = range(400)

# Extract network labels
network_labels = [re.search('(?<=7Networks_.._)\w+?(?=_)', label).group(0) for label in labels]

# Define a dictionary to map network labels to numbers
network_dict = {"Vis": 1, "SomMot": 2, 'DorsAttn': 3, 'SalVentAttn': 4,
                'Limbic': 5, 'Cont': 6, 'Default': 7}

# Convert network labels to numbers
network_numbers = [network_dict[label] for label in network_labels]

print(network_numbers)


# %% your data goes here
def plot_heatmap(matrix, title, filename, vmax=None):
    # Define a list of colors for the networks
    network_colors = sns.color_palette("tab10", 7)

    # Create a color map from the network labels
    cmap = ListedColormap(network_colors)

    # Set strip size
    strip_size = 6

    # Create a figure and a grid of subplots
    fig, ax = plt.subplots(figsize=(10, 10))

    # Generate the heatmap
    sns.heatmap(matrix, vmin=0, vmax=vmax, ax=ax, cbar=True, square=True,
                xticklabels=False, yticklabels=False, cmap='Reds', cbar_kws={'shrink': 0.5})

    # Add a line in the middle of x and y-axis
    ax.axhline(matrix.shape[0] / 2, color='black', linestyle='-', lw=2)
    ax.axvline(matrix.shape[1] / 2, color='black', linestyle='-', lw=2)

    # Create a Pandas Series for the network labels
    network_series = pd.Series(network_numbers)

    # Find the indices where the network labels change
    boundaries = np.where(network_series.diff())[0]

    # Generate color strips for networks
    network_labels = network_numbers
    for i, boundary in enumerate(boundaries):
        print(i, boundary)
        network_label = network_labels[boundary]
        start = 0 if i == 0 else boundaries[i - 1]
        end = boundary

        # print start, end, network_label
        print(start, end, network_label)

        ax.add_patch(plt.Rectangle((start, -strip_size), end - start, strip_size,
                                   facecolor=network_colors[network_label - 2], edgecolor='k', lw=0.5))
        ax.add_patch(plt.Rectangle((-strip_size, start), strip_size, end - start,
                                   facecolor=network_colors[network_label - 2], edgecolor='k', lw=0.5))

        # Add patch as box along the diagonal
        ax.add_patch(plt.Rectangle((start, start), end - start, end - start,
                                   facecolor='none', edgecolor='blue', lw=2))

    # Add last strip
    last_network_label = network_labels[-1]
    last_start = boundaries[-1]
    last_end = len(network_labels)
    ax.add_patch(plt.Rectangle((last_start, -strip_size), last_end - last_start, strip_size,
                               facecolor=network_colors[last_network_label - 1], edgecolor='k', lw=0.5))
    ax.add_patch(plt.Rectangle((-strip_size, last_start), strip_size, last_end - last_start,
                               facecolor=network_colors[last_network_label - 1], edgecolor='k', lw=0.5))

    # Add the final box along the diagonal
    ax.add_patch(plt.Rectangle((last_start, last_start), last_end - last_start, last_end - last_start,
                               facecolor='none', edgecolor='blue', lw=2))

    # Adjust the plot limits to accommodate the strips
    ax.set_xlim([-strip_size, len(network_labels)])
    ax.set_ylim([len(network_labels), -strip_size])

    #Add network labels along the y-axis
    unique_labels = ["Vis", "SomMot", 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default',
                     "Vis", "SomMot", 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
    plt.yticks(boundaries, unique_labels)
    plt.xlabel('400 Regions', fontsize=16, labelpad=20)
    plt.title(title, fontsize=20,fontweight='bold')
    plt.show()

    # save the plot
    fig = ax.get_figure()
    fig.savefig(filename, dpi=300)


#  plot the fc matrix
plot_heatmap(np.mean(fc, axis=0), 'Functional Connectivity Matrix', 'FC_mean.png', vmax=1)

# plot the sc matrix
plot_heatmap(np.mean(sc, axis=0), 'Structural Connectivity Matrix', 'SC_mean.png', vmax=1)

# load demo
demo = pd.read_excel('E:/PD_analyse/demo1126.xlsx', sheet_name=0)

# # Find the row indices where age > 240 and group is 'BP' 没看懂
# idx_bd = demo[(demo['age'] > 240) & (demo['group'] == 'BP')].index
#
# # Remove the rows in demo using the indices
# demo = demo.drop(idx_bd)
#
# # %% drop the idx in fc and sc
# fc = np.delete(fc, idx_bd, axis=0)
# sc = np.delete(sc, idx_bd, axis=0)

idx = demo['group'] == 'HC'
tmp = fc[idx]
tmp1 = sc[idx]
plot_heatmap(np.mean(tmp, axis=0), 'Functional Connectivity Matrix of HC', 'FC_HC.png', vmax=1)
plot_heatmap(np.mean(tmp1, axis=0), 'Structural Connectivity Matrix of HC', 'SC_HC.png', vmax=1)

idx = demo['group'] == 'PD'
tmp = fc[idx]
tmp1 = sc[idx]
plot_heatmap(np.mean(tmp, axis=0), 'Functional Connectivity Matrix of PD', 'FC_PD.png', vmax=1)
plot_heatmap(np.mean(tmp1, axis=0), 'Structural Connectivity Matrix of PD', 'SC_PD.png', vmax=1)

idx = demo['group'] == 'MSA'
tmp = fc[idx]
tmp1 = sc[idx]
plot_heatmap(np.mean(tmp, axis=0), 'Functional Connectivity Matrix of MSA', 'FC_MSA.png', vmax=1)
plot_heatmap(np.mean(tmp1, axis=0), 'Structural Connectivity Matrix of MSA', 'SC_MSA.png', vmax=1)

# # %% load the structural data, remove subjects that missing SC and FC
struc_data = np.load('E:/PD_analyse/surface_metrics.npy')
# sub = [40, 284, 286]  # remove subjects that missing SC and FC
# struc_data = np.delete(struc_data, sub, axis=1)
#
# # drop the idx in struc_data
# struc_data = np.delete(struc_data, idx_bd, axis=1)
#
GC = struc_data[:, :, 4]

# print the min and max of each measure
print(
    np.min(np.min(struc_data, axis=1)[:, 0]),
    np.min(np.min(struc_data, axis=1)[:, 1]),
    np.min(np.min(struc_data, axis=1)[:, 2]),
    np.min(np.min(struc_data, axis=1)[:, 3]),
    np.min(np.min(struc_data, axis=1)[:, 4])
)

print(
    np.max(np.max(struc_data, axis=1)[:, 0]),
    np.max(np.max(struc_data, axis=1)[:, 1]),
    np.max(np.max(struc_data, axis=1)[:, 2]),
    np.max(np.max(struc_data, axis=1)[:, 3]),
    np.max(np.max(struc_data, axis=1)[:, 4])
)

# load mask, labels, surface
labeling = load_parcellation('schaefer', scale=400, join=True)
mask = labeling != 0
surf_lh, surf_rh = load_conte69()
# plot the measures
measures = [map_to_labels(np.mean(struc_data, axis=1)[:, 0], labeling, mask=mask, fill=np.nan),
            map_to_labels(np.mean(struc_data, axis=1)[:, 1], labeling, mask=mask, fill=np.nan),
            map_to_labels(np.mean(struc_data, axis=1)[:, 2], labeling, mask=mask, fill=np.nan),
            map_to_labels(np.mean(struc_data, axis=1)[:, 3], labeling, mask=mask, fill=np.nan),
            map_to_labels(np.mean(struc_data, axis=1)[:, 4], labeling, mask=mask, fill=np.nan),]

# structure labels
struc_labels = ['Volume', 'Area', 'Thickness', 'Avg Curv', 'Gau Curv']
color_range = [(1, 3), (0.6, 0.85), (1.5, 3.5), (-0.15, 0.1), (-0.001, 0.002)]
filename = r'E:\PD_analyse\metrics_MSN_measures.png'
plot_hemispheres(surf_lh, surf_rh, array_name=measures, size=(1000, 1400), color_range=color_range,
                 cmap='PiYG_r', color_bar=True, filename=filename, transparent_bg=False,
                 screenshot=True, embed_nb=False, interactive=False, nan_color=(0.5, 0.5, 0.5, 1),
                 label_text=struc_labels, zoom=1.2)

# 显示图形
plt.show()