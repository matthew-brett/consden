""" Script to run SPM processing on ds114 task2

Should be run from directory containing subject directories ('sub001', etc)
"""

from os.path import split as psplit, join as pjoin, abspath, exists

import nibabel as nib

# Import our own configuration for nipype
import nipype_settings

import nipype.interfaces.spm as spm

# Analysis parameters
TR = 2.5
ref_slice = 1  # 1-based indexing
n_dummies = 4
# Realign write_which
# First value: 0 => don't reslice; 1 => all but first; 2 => all
# Second value: 0 => don't write mean; 1 => write mean
write_which = [2, 1]
# Normalize write parameters
bounding_box = [[-78., -112., -46.], [78., 76., 86.]]
voxel_sizes = [2, 2, 2]

def ascending_interleaved(num_slices):
    """ Return acquisition ordering given number of slices

    Note 1-based indexing for MATLAB.

    Return type must be a list for nipype to use it in the SPM interface
    without error.
    """
    odd = range(1, num_slices + 1, 2)
    even = range(2, num_slices + 1, 2)
    return list(odd) + list(even)

order_func = ascending_interleaved

def prefix_path(prefix, path):
    dirname, fname = psplit(path)
    return pjoin(dirname, prefix + fname)


def degz(path):
    return path[:-3] if path.endswith('.gz') else path


def process_anat(anat_fname):
    # Ungz anatomical
    if anat_fname.endswith('.gz'):
        anat_img = nib.load(anat_fname)
        anat_fname = degz(anat_fname)
        nib.save(anat_img, anat_fname)
    return anat_fname


def process_func(func_fname):
    # Drop dummy volumes
    img = nib.load(func_fname);
    dropped_img = nib.Nifti1Image(img.get_data()[..., n_dummies:],
                                img.affine,
                                img.header)
    fixed_fname = prefix_path('f', degz(func_fname))
    nib.save(dropped_img, fixed_fname)

    # Slice time correction
    num_slices = img.shape[2]
    time_for_one_slice = TR / num_slices
    TA = TR - time_for_one_slice
    st = spm.SliceTiming()
    st.inputs.in_files = fixed_fname
    st.inputs.num_slices = num_slices
    st.inputs.time_repetition = TR
    st.inputs.time_acquisition = TA
    st.inputs.slice_order = order_func(num_slices)
    st.inputs.ref_slice = ref_slice
    st.run()
    return prefix_path('a', fixed_fname)


def process_subject(func_fnames, anat_fname):
    anat_fname = process_anat(anat_fname)
    fixed_stimed = [process_func(fname) for fname in func_fnames]

    # Realign
    realign = spm.Realign()
    realign.inputs.in_files = fixed_stimed
    # Write resliced files, do write mean image
    realign.inputs.write_which = write_which
    realign.run()

    # Coregistration
    coreg = spm.Coregister()
    # Coregister structural to mean image from realignment
    coreg.inputs.target = prefix_path('mean', fixed_stimed[0])
    coreg.inputs.source = anat_fname
    coreg.run()

    # Normalization / resampling with normalization + realign params
    seg_norm = spm.Normalize12()
    seg_norm.inputs.image_to_align = anat_fname
    seg_norm.inputs.apply_to_files = fixed_stimed
    seg_norm.inputs.write_bounding_box = bounding_box
    seg_norm.inputs.write_voxel_sizes = voxel_sizes
    seg_norm.run()


def get_scans(subj_dir, runs):
    subj_dir = abspath(subj_dir)
    func_fnames = []
    for run in runs:
        func_fnames.append(pjoin(subj_dir,
                                 'BOLD',
                                 'task002_run{:03d}'.format(run),
                                 'bold.nii.gz'))
    anat_fname = pjoin(subj_dir,
                       'anatomy',
                       'highres001.nii.gz')
    return func_fnames, anat_fname


def pre_process(subjects, runs):
    for subject in subjects:
        func_fnames, anat_fname = get_scans(subject, runs)
        for fname in func_fnames:
            assert exists(fname)
        assert exists(anat_fname)
        process_subject(func_fnames, anat_fname)


SUBJECTS = [
    'sub001',
    'sub002',
    'sub003',
    'sub004',
    'sub005',
    'sub006',
    'sub007',
    'sub008',
    'sub009',
    'sub010']


def main():
    pre_process(SUBJECTS, runs=(1, 2))


if __name__ == '__main__':
    main()
