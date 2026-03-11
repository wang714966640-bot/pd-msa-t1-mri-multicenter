#循环了每个被试的FC
# %% import packages
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import bct
from multiprocessing import Pool

# %% load data and preprocess
# load the functional connectivity matrix and set the diagonal to zero
fc = np.load('E:/PD_analyse/sch400-correlation-matrix_valid_data_RBD.npy')
# Loop through the first dimension and set the diagonal of each 400x400 matrix to zero
for i in range(fc.shape[0]):
    np.fill_diagonal(fc[i], 0)
# remove the negative values of fc
fc[fc < 0] = 0

for i in range(fc.shape[0]):
     fc[i] = bct.threshold_proportional(fc[i, :, :], 0.2)

# plot the mean fc matrix, set the range to [0, 1]
ax = sns.heatmap(np.mean(fc, axis=0), vmin=0, vmax=1, square=True, annot=False,
                 xticklabels=False, yticklabels=False, cmap='Reds')
plt.show()
plt.close()


# %% check the symmetry of the matrix
def is_symmetrical(matrix: np.ndarray) -> bool:
    # Calculate the transpose of the matrix
    transpose_matrix = matrix.T

    # Check if the matrix is symmetrical
    return np.array_equal(matrix, transpose_matrix)


#  make the matrix symmetrical
def make_symmetrical_upper(matrix: np.ndarray) -> np.ndarray:
    # Create a copy of the matrix to avoid modifying the original matrix
    symmetrical_matrix = matrix.copy()

    # Loop through the matrix elements
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            # Replace the lower triangle values with the upper triangle values
            symmetrical_matrix[j, i] = symmetrical_matrix[i, j]

    return symmetrical_matrix


#  degree, clustering coefficient, betweenness centrality, closeness centrality, eigenvector centrality
def brain_graph(matrix):
    # check if the matrix is symmetrical
    if not is_symmetrical(matrix):
        # print a warning if the matrix is not symmetrical
        print("Warning: The matrix is not symmetrical, making it symmetrical...")
        # make the matrix symmetrical
        matrix = make_symmetrical_upper(matrix)

    # Clustering Coefficient
    Clustering_coef = bct.clustering_coef_wu(matrix)

    # Local Efficiency
    local_efficiency = bct.efficiency_wei(matrix, local=True)

    # Degree Centrality
    degree_centrality = matrix.sum(axis=0)

    # Betweenness Centrality
    betweenness = bct.betweenness_wei(bct.weight_conversion(matrix, 'lengths'))

    # Characteristic Path Length
    charpath = bct.charpath(bct.distance_wei(bct.weight_conversion(matrix, 'lengths'))[0])[1]

    # Global Efficiency
    global_efficiency = bct.efficiency_wei(matrix)

    # Modularity
    partition, modularity = bct.modularity_louvain_und(matrix)

    # return a list of the network metrics
    return {'Clustering Coefficient': Clustering_coef,
            'Local Efficiency': local_efficiency,
            'Degree Centrality': degree_centrality,
            'Betweenness Centrality': betweenness,
            'Characteristic Path Length': charpath,
            'Global Efficiency': global_efficiency,
            'Modularity': modularity}


# %% calculate the network metrics for each subject of fc and sc
# initialize the variables
clustering_fc = np.zeros((fc.shape[0], fc.shape[1]))
local_efficiency_fc = np.zeros((fc.shape[0], fc.shape[1]))
degree_fc = np.zeros((fc.shape[0], fc.shape[1]))
betweenness_fc = np.zeros((fc.shape[0], fc.shape[1]))
characteristic_path_length_fc = np.zeros((fc.shape[0], 1))
global_efficiency_fc = np.zeros((fc.shape[0], 1))
modularity_fc = np.zeros((fc.shape[0], 1))

# loop the 365 subjects of fc
for i in range(fc.shape[0]):
    # print the ith node like this: 'Processing node i'
    print('Processing node - FC', i)

    # plot fc matrix and save it, without showing it
    sns.heatmap(fc[i], vmin=0, vmax=1, square=True, annot=False,
                xticklabels=False, yticklabels=False, cmap='Reds')
    #save the figure with the ith node in the folder FC_SC_MatPlots
    plt.savefig('E:/PD_analyse/FC_SC_MatPlots/FC_' + str(i) + '.png')
    plt.close()

    # calculate the network metrics for the ith subject of fc
    network_metrics = brain_graph(fc[i])

    # save the network metrics for the ith subject of fc
    clustering_fc[i] = network_metrics['Clustering Coefficient']
    local_efficiency_fc[i] = network_metrics['Local Efficiency']
    degree_fc[i] = network_metrics['Degree Centrality']
    betweenness_fc[i] = network_metrics['Betweenness Centrality']
    characteristic_path_length_fc[i] = network_metrics['Characteristic Path Length']
    global_efficiency_fc[i] = network_metrics['Global Efficiency']
    modularity_fc[i] = network_metrics['Modularity']

# save the measures of fc to csv
np.savetxt('E:/PD_analyse/FC_SC_MatPlots/clustering_fc.csv', clustering_fc, delimiter=',')
np.savetxt('E:/PD_analyse/FC_SC_MatPlots/local_efficiency_fc.csv', local_efficiency_fc, delimiter=',')
np.savetxt('E:/PD_analyse/FC_SC_MatPlots/degree_fc.csv', degree_fc, delimiter=',')
np.savetxt('E:/PD_analyse/FC_SC_MatPlots/betweenness_fc.csv', betweenness_fc, delimiter=',')
np.savetxt('E:/PD_analyse/FC_SC_MatPlots/characteristic_path_length_fc.csv', characteristic_path_length_fc, delimiter=',')
np.savetxt('E:/PD_analyse/FC_SC_MatPlots/global_efficiency_fc.csv', global_efficiency_fc, delimiter=',')
np.savetxt('E:/PD_analyse/FC_SC_MatPlots/modularity_fc.csv', modularity_fc, delimiter=',')
print('FC measures saved')

#SC_graph_weighted循环了每个被试的SC
#%% import packages
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import bct
import networkx as nx
import community as community_louvain
import igraph as ig
from multiprocessing import Pool

# %% load data and preprocess
# load the structural connectivity matrix
sc = np.load('E:/PD_analyse/schaefer400-SC_RBD.npy')
for i in range(sc.shape[0]):
    sc[i] = bct.threshold_proportional(sc[i, :, :], 0.2)

# plot the sc_th matrix, set the range to [0, 1]
ax = sns.heatmap(np.mean(sc, axis=0), vmin=0, vmax=1, square=True, annot=False,
                 xticklabels=False, yticklabels=False, cmap='Reds')
plt.show()
plt.close()


# %% check the symmetry of the matrix
def is_symmetrical(matrix: np.ndarray) -> bool:
    # Calculate the transpose of the matrix
    transpose_matrix = matrix.T

    # Check if the matrix is symmetrical
    return np.array_equal(matrix, transpose_matrix)


#  make the matrix symmetrical
def make_symmetrical_upper(matrix: np.ndarray) -> np.ndarray:
    # Create a copy of the matrix to avoid modifying the original matrix
    symmetrical_matrix = matrix.copy()

    # Loop through the matrix elements
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            # Replace the lower triangle values with the upper triangle values
            symmetrical_matrix[j, i] = symmetrical_matrix[i, j]

    return symmetrical_matrix


# %% degree, clustering coefficient, betweenness centrality, closeness centrality, eigenvector centrality
def brain_graph(matrix):
    # check if the matrix is symmetrical
    if not is_symmetrical(matrix):
        # print a warning if the matrix is not symmetrical
        print("Warning: The matrix is not symmetrical, making it symmetrical...")
        # make the matrix symmetrical
        matrix = make_symmetrical_upper(matrix)

    # Clustering Coefficient
    Clustering_coef = bct.clustering_coef_wu(matrix)

    # Local Efficiency
    local_efficiency = bct.efficiency_wei(matrix, local=True)

    # Degree Centrality
    degree_centrality = matrix.sum(axis=0)

    # Betweenness Centrality
    betweenness = bct.betweenness_wei(bct.weight_conversion(matrix, 'lengths'))

    # Characteristic Path Length
    charpath = bct.charpath(bct.distance_wei(bct.weight_conversion(matrix, 'lengths'))[0])[1]

    # Global Efficiency
    global_efficiency = bct.efficiency_wei(matrix)

    # Modularity
    partition, modularity = bct.modularity_louvain_und(matrix)

    # return a list of the network metrics
    return {'Clustering Coefficient': Clustering_coef,
            'Local Efficiency': local_efficiency,
            'Degree Centrality': degree_centrality,
            'Betweenness Centrality': betweenness,
            'Characteristic Path Length': charpath,
            'Global Efficiency': global_efficiency,
            'Modularity': modularity}


# %% calculate the network metrics for each subject of sc
# initialize the variables
clustering_sc = np.zeros((sc.shape[0], sc.shape[1]))
local_efficiency_sc = np.zeros((sc.shape[0], sc.shape[1]))
degree_sc = np.zeros((sc.shape[0], sc.shape[1]))
betweenness_sc = np.zeros((sc.shape[0], sc.shape[1]))
characteristic_path_length_sc = np.zeros((sc.shape[0], 1))
global_efficiency_sc = np.zeros((sc.shape[0], 1))
modularity_sc = np.zeros((sc.shape[0], 1))

# calculate the mean of sc along the first dimension
sc_mean = np.mean(sc, axis=0)

# loop the 392 subjects of sc
for i in range(sc.shape[0]):
    # print the ith node like this: 'Processing node i'
    print('Processing node - SC', i)

    # plot fc matrix and save it, without showing it
    sns.heatmap(sc[i], vmin=0, vmax=1, square=True, annot=False,
                xticklabels=False, yticklabels=False, cmap='Reds')

    # save the figure with the ith node in the folder FC_SC_MatPlots
    plt.savefig('E:/PD_analyse/FC_SC_MatPlots/SC_' + str(i) + '.png')
    plt.close()

    # find the rows that are zeros in sc[i]
    zero_rows = np.where(~sc[i].any(axis=0))[0]
    print('Zero rows in SC', i, ':', zero_rows)

    # replace the rows that are zeros in sc[i] with the mean of sc
    sc[i][zero_rows, :] = sc_mean[zero_rows, :]

    # calculate the network metrics for the ith subject of sc
    network_metrics = brain_graph(sc[i])

    # save the network metrics for the ith subject of sc
    clustering_sc[i] = network_metrics['Clustering Coefficient']
    local_efficiency_sc[i] = network_metrics['Local Efficiency']
    degree_sc[i] = network_metrics['Degree Centrality']
    betweenness_sc[i] = network_metrics['Betweenness Centrality']
    characteristic_path_length_sc[i] = network_metrics['Characteristic Path Length']
    global_efficiency_sc[i] = network_metrics['Global Efficiency']
    modularity_sc[i] = network_metrics['Modularity']

# save the measures of sc to csv
np.savetxt('E:/PD_analyse/FC_SC_MatPlots/clustering_sc.csv', clustering_sc, delimiter=',')
np.savetxt('E:/PD_analyse/FC_SC_MatPlots/local_efficiency_sc.csv', local_efficiency_sc, delimiter=',')
np.savetxt('E:/PD_analyse/FC_SC_MatPlots/degree_sc.csv', degree_sc, delimiter=',')
np.savetxt('E:/PD_analyse/FC_SC_MatPlots/betweenness_sc.csv', betweenness_sc, delimiter=',')
np.savetxt('E:/PD_analyse/FC_SC_MatPlots/characteristic_path_length_sc.csv', characteristic_path_length_sc, delimiter=',')
np.savetxt('E:/PD_analyse/FC_SC_MatPlots/global_efficiency_sc.csv', global_efficiency_sc, delimiter=',')
np.savetxt('E:/PD_analyse/FC_SC_MatPlots/modularity_sc.csv', modularity_sc, delimiter=',')
print('SC measures saved')

# %% calculate the network metrics for each subject of sc in each hemisphere
# initialize the variables
clustering_sc_left = np.zeros((sc.shape[0], sc.shape[1] // 2))
local_efficiency_sc_left = np.zeros((sc.shape[0], sc.shape[1] // 2))
degree_sc_left = np.zeros((sc.shape[0], sc.shape[1] // 2))
betweenness_sc_left = np.zeros((sc.shape[0], sc.shape[1] // 2))
characteristic_path_length_sc_left = np.zeros((sc.shape[0], 1))
global_efficiency_sc_left = np.zeros((sc.shape[0], 1))
modularity_sc_left = np.zeros((sc.shape[0], 1))
clustering_sc_right = np.zeros((sc.shape[0], sc.shape[1] // 2))
local_efficiency_sc_right = np.zeros((sc.shape[0], sc.shape[1] // 2))
degree_sc_right = np.zeros((sc.shape[0], sc.shape[1] // 2))
betweenness_sc_right = np.zeros((sc.shape[0], sc.shape[1] // 2))
characteristic_path_length_sc_right = np.zeros((sc.shape[0], 1))
global_efficiency_sc_right = np.zeros((sc.shape[0], 1))
modularity_sc_right = np.zeros((sc.shape[0], 1))

# calculate the mean of sc along the first dimension
sc_mean = np.mean(sc, axis=0)

# loop the 392 subjects of sc
for i in range(sc.shape[0]):
    # print the ith node like this: 'Processing node i'
    print('Processing node - SC', i)

    # plot fc matrix and save it, without showing it
    sns.heatmap(sc[i], vmin=0, vmax=100, square=True, annot=False,
                xticklabels=False, yticklabels=False, cmap='Reds')

    # save the figure with the ith node in the folder FC_SC_MatPlots
    plt.savefig('FC_SC_MatPlots/SC_' + str(i) + '.png')
    plt.close()

    # find the rows that are zeros in sc[i]
    zero_rows = np.where(~sc[i].any(axis=0))[0]
    print('Zero rows in SC', i, ':', zero_rows)

    # replace the rows that are zeros in sc[i] with the mean of sc
    sc[i][zero_rows, :] = sc_mean[zero_rows, :]

    # calculate the network metrics for the ith subject of sc
    network_metrics_left = brain_graph(sc[i][0:200, 0:200])
    network_metrics_right = brain_graph(sc[i][200:400, 200:400])

    # save the network metrics for the ith subject of sc
    clustering_sc_left[i] = network_metrics_left['Clustering Coefficient']
    local_efficiency_sc_left[i] = network_metrics_left['Local Efficiency']
    degree_sc_left[i] = network_metrics_left['Degree Centrality']
    betweenness_sc_left[i] = network_metrics_left['Betweenness Centrality']
    characteristic_path_length_sc_left[i] = network_metrics_left['Characteristic Path Length']
    global_efficiency_sc_left[i] = network_metrics_left['Global Efficiency']
    modularity_sc_left[i] = network_metrics_left['Modularity']
    clustering_sc_right[i] = network_metrics_right['Clustering Coefficient']
    local_efficiency_sc_right[i] = network_metrics_right['Local Efficiency']
    degree_sc_right[i] = network_metrics_right['Degree Centrality']
    betweenness_sc_right[i] = network_metrics_right['Betweenness Centrality']
    characteristic_path_length_sc_right[i] = network_metrics_right['Characteristic Path Length']
    global_efficiency_sc_right[i] = network_metrics_right['Global Efficiency']
    modularity_sc_right[i] = network_metrics_right['Modularity']

# combine the left and right hemisphere metrics
clustering_sc = np.concatenate((clustering_sc_left, clustering_sc_right), axis=1)
local_efficiency_sc = np.concatenate((local_efficiency_sc_left, local_efficiency_sc_right), axis=1)
degree_sc = np.concatenate((degree_sc_left, degree_sc_right), axis=1)
betweenness_sc = np.concatenate((betweenness_sc_left, betweenness_sc_right), axis=1)
characteristic_path_length_sc = (characteristic_path_length_sc_left + characteristic_path_length_sc_right) / 2
global_efficiency_sc = (global_efficiency_sc_left + global_efficiency_sc_right) / 2
modularity_sc = (modularity_sc_left + modularity_sc_right) / 2

# save the measures of sc to csv
np.savetxt('E:/PD_analyse/FC_SC_MatPlots/clustering_sc_hemisphere.csv', clustering_sc, delimiter=',')
np.savetxt('E:/PD_analyse/FC_SC_MatPlots/local_efficiency_sc_hemisphere.csv', local_efficiency_sc, delimiter=',')
np.savetxt('E:/PD_analyse/FC_SC_MatPlots/degree_sc_hemisphere.csv', degree_sc, delimiter=',')
np.savetxt('E:/PD_analyse/FC_SC_MatPlots/betweenness_sc_hemisphere.csv', betweenness_sc, delimiter=',')
np.savetxt('E:/PD_analyse/FC_SC_MatPlots/characteristic_path_length_sc_hemisphere.csv', characteristic_path_length_sc, delimiter=',')
np.savetxt('E:/PD_analyse/FC_SC_MatPlots/global_efficiency_sc_hemisphere.csv', global_efficiency_sc, delimiter=',')
np.savetxt('E:/PD_analyse/FC_SC_MatPlots/modularity_sc_hemisphere.csv', modularity_sc, delimiter=',')
print('SC measures saved')
