import numpy as np
import SimpleITK as sitk


def get_median_spacing(img_pths):
    spacings = np.array([sitk.ReadImage(pth).GetSpacing() for pth in img_pths])
    return np.median(spacings, axis=0), spacings


def get_median_size(img_pths):
    sizes = np.array([sitk.ReadImage(pth).GetSize() for pth in img_pths])
    return np.median(sizes, axis=0), sizes


def get_target_spacing(img_pths, anisotropy_check=True, anisotropy_percentile=10):
    median_spacing, spacings = get_median_spacing(img_pths)
    if len(median_spacing) == 2:
        small_spacing_axis, large_spacing_axis = np.argmin(median_spacing), np.argmax(median_spacing)
        if anisotropy_check and median_spacing[large_spacing_axis] > 3. * median_spacing[small_spacing_axis]:
            target_spacing = median_spacing.copy()
            target_spacing[large_spacing_axis] = np.percentile(spacings, anisotropy_percentile, axis=0)[large_spacing_axis]
        else:
            target_spacing = median_spacing

    elif len(median_spacing) == 3:
        print(f'dim: {len(median_spacing)}, median: {median_spacing}')
        small_spacing_axis, large_spacing_axis = np.argmin(median_spacing), np.argmax(median_spacing)
        normal_spacing_axis = list(set([0, 1, 2]) - set([small_spacing_axis, large_spacing_axis]))[0]
        print(small_spacing_axis, normal_spacing_axis, large_spacing_axis)
        if anisotropy_check and median_spacing[small_spacing_axis] * 3. < median_spacing[large_spacing_axis]:
            target_spacing = median_spacing.copy()
            if median_spacing[large_spacing_axis]/median_spacing[normal_spacing_axis] >\
                median_spacing[normal_spacing_axis]/median_spacing[small_spacing_axis]:
                target_spacing[large_spacing_axis] = np.percentile(spacings, anisotropy_percentile,
                                                                   axis=0)[large_spacing_axis]
            else:
                target_spacing[small_spacing_axis] = np.percentile(spacings, 100 - anisotropy_percentile,
                                                                   axis=0)[small_spacing_axis]
        else:
            target_spacing = median_spacing

    else:
        target_spacing = median_spacing

    return target_spacing, median_spacing


if __name__ == '__main__':
    import glob
    pths = glob.glob(r'F:\datasets\ovarian_cancer\batch3_08-12\CT_whole'+'/*Image*')
    target_spacing, median_spacing = get_target_spacing(pths)
    print(f'target: {target_spacing}, median: {median_spacing} !')


