import os
import numpy as np
import pickle
from statsmodels.stats.multitest import fdrcorrection
import scipy.stats as stats
import scipy.io as sio
import matplotlib.pyplot as plt
from nilearn.image import new_img_like
import pandas as pd
import nibabel as nib
import csv
from nilearn import plotting
# from rsatoolbox.inference import eval_fixed
# from rsatoolbox.model import ModelFixed
from glob import glob
import nilearn.image as nlimg
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs, evaluate_models_searchlight

from scipy.stats import spearmanr
from tqdm import tqdm
# from rsatoolbox.data.dataset import Dataset
# from rsatoolbox.rdm.calc import calc_rdm
# from rsatoolbox.rdm import RDMs
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed
import seaborn as sns


stim_file = pd.read_csv('N:/Experimental_Data/yujunchen/projects/IAPS_fMRI_RSA/fmri_conditions_trial_ordered.csv')
stim_Nt = stim_file[0:100]
stim_Pl = stim_file[100:200].reset_index()
stim_Up = stim_file[200:300].reset_index()

stim_Nt_id = pd.DataFrame(stim_Nt[['stim_order']]).T
stim_Nt_id.shape
stim_Pl_id = pd.DataFrame(stim_Pl[['stim_order']]).T
stim_Pl_id.shape
stim_Up_id = pd.DataFrame(stim_Up[['stim_order']]).T
stim_Up_id.shape
print("Neutral    Type: {} Shape/Length: {}".format(type(stim_Nt_id), stim_Nt_id.shape))
print("Pleasant   Type: {} Shape/Length: {}".format(type(stim_Pl_id), stim_Pl_id.shape))
print("Unpleasant Type: {} Shape/Length: {}".format(type(stim_Up_id), stim_Up_id.shape))

allsub_avg = []

subjects = 20
for i in range(subjects):
    sub_avg = []
    #Pleasant
    mat_name_Pl = 'N:\\Experimental_Data\\Ke Bo\\Project_IAPS\\fMRI\\code\\fMRI\\FMRIPreprocess_All\\Beta_Leaveoneout/Pl{}.mat'.format(i+1)
    mat_file_Pl = sio.loadmat(mat_name_Pl)
    data_Pl = mat_file_Pl['Pl'] #(153594, 100)
    data_Pl_reshape = data_Pl.reshape(46, 63, 53, 100)
    data_Pl_reorder = np.moveaxis(data_Pl_reshape, [0,1,2,3], [2,1,0,3])
    data_Pl_reorder_flat = np.reshape(data_Pl_reorder, (153594,100))
    data_Pl_df = pd.DataFrame(data_Pl_reorder_flat) #(153594, 100)
    Pl_df = pd.concat([stim_Pl_id, data_Pl_df]) #(153595,100)
    Pl_df = Pl_df.T
    Pl_df_gr = Pl_df.groupby('stim_order').mean() #(20, 153594)
    Pl_df_gr = Pl_df_gr.reset_index(drop=True) #(20, 153594) 

    #Neutral
    mat_name_Nt = 'N:\\Experimental_Data\\Ke Bo\\Project_IAPS\\fMRI\\code\\fMRI\\FMRIPreprocess_All\\Beta_Leaveoneout/Nt{}.mat'.format(i+1)
    mat_file_Nt = sio.loadmat(mat_name_Nt)
    data_Nt = mat_file_Nt['Nt'] #(153594, 100)
    data_Nt_reshape = data_Nt.reshape(46, 63, 53, 100)
    data_Nt_reorder = np.moveaxis(data_Nt_reshape, [0,1,2,3], [2,1,0,3])
    data_Nt_reorder_flat = np.reshape(data_Nt_reorder, (153594,100))
    data_Nt_df = pd.DataFrame(data_Nt_reorder_flat) #(153594, 100)
    Nt_df = pd.concat([stim_Nt_id,data_Nt_df]) #(153595,100)
    Nt_df = Nt_df.T
    Nt_df_gr = Nt_df.groupby('stim_order').mean() #(20, 153594)
    Nt_df_gr = Nt_df_gr.reset_index(drop=True) #(20, 153594) 


    #Unpleasant
    mat_name_Up = 'N:\\Experimental_Data\\Ke Bo\\Project_IAPS\\fMRI\\code\\fMRI\\FMRIPreprocess_All\\Beta_Leaveoneout/Up{}.mat'.format(i+1)
    mat_file_Up = sio.loadmat(mat_name_Up)
    data_Up = mat_file_Up['Up'] #(153594, 100)
    data_Up_reshape = data_Up.reshape(46, 63, 53, 100)
    data_Up_reorder = np.moveaxis(data_Up_reshape, [0,1,2,3], [2,1,0,3])
    data_Up_reorder_flat = np.reshape(data_Up_reorder, (153594,100))
    data_Up_df = pd.DataFrame(data_Up_reorder_flat) #(153594, 100)
    Up_df = pd.concat([stim_Up_id, data_Up_df]) #(153595,100)
    Up_df = Up_df.T
    Up_df_gr = Up_df.groupby('stim_order').mean() #(20, 153594)
    Up_df_gr = Up_df_gr.reset_index(drop=True) #(20, 153594) 

    sub_df = pd.concat([Pl_df_gr, Nt_df_gr, Up_df_gr])
    allsub_avg.append(sub_df)
    

# Save with full path
save_path = "N:/Experimental_Data/Aanya/Searchlight/allsub_avg.npy"
np.save(save_path, allsub_avg)

# Load from the same path
allsub_avg = np.load(save_path, allow_pickle=True)

allsub_avg.shape

# Create a "mask" of ones with the same dimensions as your data (46, 63, 53)
mask = np.ones((46, 63, 53), dtype=bool)  # Using the dimensions from your data reshape
centers, neighbors = get_volume_searchlight(mask, radius=5, threshold=0.5)

def svm_decode_searchlight(data, labels, neighbors):
    """
    Perform SVM decoding for each searchlight sphere
    data: array of shape (n_samples, n_features)
    labels: array of categorical labels
    neighbors: list of neighbor indices for each searchlight center
    """
    # Initialize classifier pipeline with scaling
    clf = make_pipeline(StandardScaler(), LinearSVC(max_iter=5000))
    
    # Initialize results array
    accuracies = np.zeros(len(neighbors))
    
    # Run decoding for each searchlight sphere
    for i, neighbor in tqdm(enumerate(neighbors), total=len(neighbors), desc='Processing spheres'):
        sphere_data = data[:, neighbor]
        # Get cross-validated accuracy scores
        scores = cross_val_score(clf, sphere_data, labels, cv=5)
        accuracies[i] = np.mean(scores)
    
    return accuracies

# Initialize matrix to store all subjects' results (20 subjects × N voxels)
all_subjects_results = []

# Process subjects in parallel
def process_subject(i):
    print(f"Processing subject {i+1}/20")
    subj_data = allsub_avg[i]
    subj_data_2d = np.nan_to_num(subj_data)
    
    # Create labels (0=Pleasant, 1=Neutral, 2=Unpleasant)
    labels = np.repeat([0, 1, 2], 20)
    
    # Run searchlight SVM decoding
    return svm_decode_searchlight(subj_data_2d, labels, neighbors)

# Run parallel processing with progress bar
all_subjects_results = Parallel(n_jobs=-1)(
    delayed(process_subject)(i) for i in tqdm(range(20), desc='Processing subjects')
)

# Convert to numpy array (20 × N_voxels)
all_subjects_results = np.array(all_subjects_results)
print("Results shape:", all_subjects_results.shape)

# Save the numpy results array
save_path = "N:/Experimental_Data/Aanya/Searchlight/searchlight_svm_results.npy"
np.save(save_path, all_subjects_results)

# If you want to visualize the results in 3D brain space:
x, y, z = mask.shape
mean_results = np.mean(all_subjects_results, axis=0)  # Average across subjects
brain_map = np.zeros([x*y*z])
brain_map[list(centers)] = mean_results
brain_map = brain_map.reshape([x, y, z])

# Get the number of spheres/voxels from your data
n_subjects, n_spheres = all_subjects_results.shape
print(f"Data dimensions: {n_subjects} subjects × {n_spheres} spheres")

# Create heatmap with dimensions matching your data
plt.figure(figsize=(n_spheres/100, 20))  # Dividing n_spheres by 100 to make it manageable
sns.heatmap(all_subjects_results, 
            cmap='viridis',
            xticklabels=False,
            yticklabels=range(1, 21),
            cbar_kws={'label': 'Decoding Accuracy'})
plt.title('SVM Decoding Accuracies Across Subjects and Searchlight Spheres')
plt.xlabel(f'Searchlight Spheres (n={n_spheres})')
plt.ylabel('Subjects')

# Save the heatmap visualization
plt.savefig('N:/Experimental_Data/Aanya/Searchlight/decoding_heatmap.png', 
            dpi=300, 
            bbox_inches='tight')
plt.close()

print(f"Results saved to:")
print(f"1. Numpy array: {save_path}")
print(f"2. Heatmap: N:/Experimental_Data/Aanya/Searchlight/decoding_heatmap.png")
