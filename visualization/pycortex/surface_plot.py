import cortex

# First, download/import fsaverage into pycortex
# You need freesurfer's fsaverage files
cortex.freesurfer.import_subj('fsaverage', freesurfer_subject_dir='/path/to/freesurfer/subjects')

# Then plot your MNI volume
import nibabel as nib
img = nib.load('tmap_beh60_p05_v2.nii.gz')
vol = cortex.Volume('fsaverage', 'identity', data=img.get_fdata())
cortex.quickflat.make_figure(vol)