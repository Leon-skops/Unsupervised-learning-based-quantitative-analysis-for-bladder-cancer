import os
import numpy as np
import nibabel as nib
from sklearn.cluster import KMeans

# Merge the valid voxel values of masks in multilayer medical images into an array
def combine_voxel_values(img, mask):
    img_data = img.get_fdata()
    mask_data = mask.get_fdata()
    combined_data = []

    for i in range(img_data.shape[-1]):
        ori_layer = img_data[..., i]
        mask_layer = mask_data[..., i]

        if np.any(mask_layer == 1):
            combined_data.extend(ori_layer[mask_layer == 1])

    return np.array(combined_data).reshape((-1, 1))

# Clustering
def kmeans_consistent_cluster(img, mask, n_clusters, random_state):
    combined_data = combine_voxel_values(img, mask)
    combined_data[np.isnan(combined_data)] = 0

    # Normalize data
    combined_data = (combined_data - combined_data.min()) / (combined_data.max() - combined_data.min())

    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', init='k-means++', random_state=random_state)

    labels = kmeans.predict(combined_data)

    clusters_img = np.zeros(img.shape)
    mask_data = mask.get_fdata()
    label_idx = 0

    for i in range(img.shape[-1]):
        mask_layer = mask_data[..., i]

        if np.any(mask_layer == 1):
            non_zero_voxels = np.sum(mask_layer == 1)
            labels_layer = labels[label_idx: label_idx + non_zero_voxels]
            clusters_img[..., i][mask_layer == 1] = labels_layer + 1
            label_idx += non_zero_voxels

    return clusters_img

# Batch generation of subregional data
ori_path = r'F:\Bladder_cancer_project'
save_path = r'F:\Bladder_cancer_project\Subregion'

for pt_path in os.listdir(ori_path):
    sub_pt_path = os.path.join(ori_path, pt_path)
    out_path = os.path.join(save_path, pt_path)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for file in os.listdir(sub_pt_path):
        if file.endswith('V_original_resampled_image.nii.gz'):
            ori_file_path = os.path.join(sub_pt_path, file)
            print(ori_file_path)
            ori_file_name = file
            img = nib.load(ori_file_path)

        if file.endswith('V_mask_resampled_image.nii.gz'):
            msk_file_path = os.path.join(sub_pt_path, file)
            msk_file_name = file
            print(msk_file_path)
            mask = nib.load(msk_file_path)

            try:
                subregional = kmeans_consistent_cluster(img, mask, n_clusters=3, random_state=42)
                habitat_mask = nib.Nifti1Image(subregional, img.affine)
                nib.save(habitat_mask, out_path + f'/{ori_file_name}_cluster.nii.gz')

            except ValueError as e:
                print(f"Skipping {ori_file_name} due to error: {e}")
