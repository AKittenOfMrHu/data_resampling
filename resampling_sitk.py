import numpy as np
import SimpleITK as sitk


InterpolationOrder = {'NearestNeighbor': sitk.sitkNearestNeighbor,
                      'Linear': sitk.sitkLinear,
                      'BSpline': sitk.sitkBSpline}


def resampling_for_img2d(data,
                         new_spacing=(1, 1),
                         order='BSpline',
                         copy_meta_data=False):
    '''
    Resample data with type sitk.
    :param data: data with type sitk
    :param new_spacing:
    :param order:
    :param in_out_planes_separately:
    :param out_plane_order:
    :return: data with type sitk
    '''
    assert order in InterpolationOrder.keys(), 'The \'order\' should be one of ( NearestNeighbor, Linear, BSpline ).'

    euler2d = sitk.Euler2DTransform()
    old_size = data.GetSize()
    old_spacing = data.GetSpacing()
    origin = data.GetOrigin()
    direction = data.GetDirection()

    new_size = [int((old_size[i] - 1) * old_spacing[i] / new_spacing[i]) + 1 for i in range(2)]
    new_data = sitk.Resample(data, new_size, euler2d,
                             InterpolationOrder[order],
                             origin, new_spacing, direction)

    if copy_meta_data:
        meta_keys = data.GetMetaDataKeys()
        for meta_key in meta_keys:
            meta_val = data.GetMetaData(meta_key)
            new_data.SetMetaData(key=meta_key, val=meta_val)

    return new_data


def resampling_for_lbl2d(data,
                         new_spacing=(1, 1),
                         order='Linear',
                         to_one_hot=True,
                         copy_meta_data=False):
    '''
    Resample data with type sitk.
    :param data: data with type sitk
    :param new_spacing:
    :param order:
    :param in_out_planes_separately:
    :param out_plane_order:
    :param to_one_hot:
    :return: data with type sitk
    '''
    assert order in InterpolationOrder.keys(), 'The \'order\' should be one of ( NearestNeighbor, Linear, BSpline ).'

    euler2d = sitk.Euler2DTransform()
    old_size = data.GetSize()
    old_spacing = data.GetSpacing()
    origin = data.GetOrigin()
    direction = data.GetDirection()

    if to_one_hot:
        arr = sitk.GetArrayFromImage(data)
        dtype = arr.dtype
        max_val = arr.max()
    else:
        arr = None
        dtype = None
        max_val = 1

    data_one_hot = []
    if max_val > 1:
        arr_one_hot = np.zeros([max_val + 1, *arr.shape], dtype=dtype)
        aranges = [np.arange(0, s) for s in arr.shape]
        indices = np.meshgrid(*aranges, indexing='ij')
        arr_one_hot[arr, indices[0], indices[1], indices[2]] = 1

        for ci in range(max_val + 1):
            temp_data = sitk.GetImageFromArray(arr_one_hot[ci])
            temp_data.SetDirection(direction)
            temp_data.SetOrigin(origin)
            temp_data.SetSpacing(old_spacing)
            data_one_hot.append(temp_data)
    else:
        data_one_hot.append(data)

    new_size = [int((old_size[i] - 1) * old_spacing[i] / new_spacing[i]) + 1 for i in range(2)]
    new_datas = []
    for temp_data in data_one_hot:
        new_data = sitk.Resample(temp_data, new_size, euler2d,
                                 InterpolationOrder[order],
                                 origin, new_spacing, direction)
        new_datas.append(new_data)

    if max_val > 1:
        arrs = []
        for temp_data in new_datas:
            arrs.append(sitk.GetArrayFromImage(temp_data))
        new_arr = np.argmax(np.array(arrs), axis=0).astype(dtype)
        new_data = sitk.GetImageFromArray(new_arr)
        new_data.SetOrigin(origin)
        new_data.SetDirection(direction)
        new_data.SetSpacing(new_spacing)
    else:
        new_data = new_datas[0]

    if copy_meta_data:
        meta_keys = data.GetMetaDataKeys()
        for meta_key in meta_keys:
            meta_val = data.GetMetaData(meta_key)
            new_data.SetMetaData(key=meta_key, val=meta_val)

    return new_data


def resampling_for_img3d(data,
                         new_spacing=(1, 1, 3),
                         order='BSpline',
                         in_out_planes_separately=True,
                         out_plane_order='NearestNeighbor',
                         copy_meta_data=True):
    '''
    Resample data with type sitk.
    :param data: data with type sitk
    :param new_spacing:
    :param order:
    :param in_out_planes_separately:
    :param out_plane_order:
    :return: data with type sitk
    '''
    assert order in InterpolationOrder.keys(), 'The \'order\' should be one of ( NearestNeighbor, Linear, BSpline ).'

    euler3d = sitk.Euler3DTransform()
    old_size = data.GetSize()
    old_spacing = data.GetSpacing()
    origin = data.GetOrigin()
    direction = data.GetDirection()

    new_size = [int((old_size[i]-1)*old_spacing[i]/new_spacing[i]) + 1 for i in range(3)]
    if in_out_planes_separately and order != out_plane_order:
        tmp_spacing = [new_spacing[0], new_spacing[1], old_spacing[2]]
        tmp_size = [new_size[0], new_size[1], old_size[2]]
        tmp_data = sitk.Resample(data, tmp_size, euler3d,
                                 InterpolationOrder[order],
                                 origin, tmp_spacing, direction)

        new_data = sitk.Resample(tmp_data, new_size, euler3d,
                                 InterpolationOrder[out_plane_order],
                                 origin, new_spacing, direction)
    else:
        new_data = sitk.Resample(data, new_size, euler3d,
                                 InterpolationOrder[order],
                                 origin, new_spacing, direction)
    if copy_meta_data:
        meta_keys = data.GetMetaDataKeys()
        for meta_key in meta_keys:
            meta_val = data.GetMetaData(meta_key)
            new_data.SetMetaData(key=meta_key, val=meta_val)

    return new_data


def resampling_for_lbl3d(data,
                         new_spacing=(1, 1, 3),
                         order='Linear',
                         in_out_planes_separately=True,
                         out_plane_order='NearestNeighbor',
                         to_one_hot=True,
                         copy_meta_data=False):
    '''
    Resample data with type sitk.
    :param data: data with type sitk
    :param new_spacing:
    :param order:
    :param in_out_planes_separately:
    :param out_plane_order:
    :param to_one_hot:
    :return: data with type sitk
    '''
    assert order in InterpolationOrder.keys(), 'The \'order\' should be one of ( NearestNeighbor, Linear, BSpline ).'

    euler3d = sitk.Euler3DTransform()
    old_size = data.GetSize()
    old_spacing = data.GetSpacing()
    origin = data.GetOrigin()
    direction = data.GetDirection()

    if to_one_hot:
        arr = sitk.GetArrayFromImage(data)
        dtype = arr.dtype
        max_val = arr.max()
    else:
        arr = None
        dtype = None
        max_val = 1

    data_one_hot = []
    if max_val > 1:
        arr_one_hot = np.zeros([max_val+1, *arr.shape], dtype=dtype)
        aranges = [np.arange(0, s) for s in arr.shape]
        indices = np.meshgrid(*aranges, indexing='ij')
        arr_one_hot[arr, indices[0], indices[1], indices[2]] = 1

        for ci in range(max_val+1):
            temp_data = sitk.GetImageFromArray(arr_one_hot[ci])
            temp_data.SetDirection(direction)
            temp_data.SetOrigin(origin)
            temp_data.SetSpacing(old_spacing)
            data_one_hot.append(temp_data)
    else:
        data_one_hot.append(data)

    new_size = [int((old_size[i] - 1) * old_spacing[i] / new_spacing[i]) + 1 for i in range(3)]
    new_datas = []
    for temp_data in data_one_hot:
        if in_out_planes_separately and order != out_plane_order:
            tmp_spacing = [new_spacing[0], new_spacing[1], old_spacing[2]]
            tmp_size = [new_size[0], new_size[1], old_size[2]]
            tmp_data = sitk.Resample(temp_data, tmp_size, euler3d,
                                     InterpolationOrder[order],
                                     origin, tmp_spacing, direction)

            new_data = sitk.Resample(tmp_data, new_size, euler3d,
                                     InterpolationOrder[out_plane_order],
                                     origin, new_spacing, direction)
        else:
            new_data = sitk.Resample(temp_data, new_size, euler3d,
                                     InterpolationOrder[order],
                                     origin, new_spacing, direction)
        new_datas.append(new_data)

    if max_val > 1:
        arrs = []
        for temp_data in new_datas:
            arrs.append(sitk.GetArrayFromImage(temp_data))
        new_arr = np.argmax(np.array(arrs), axis=0).astype(dtype)
        new_data = sitk.GetImageFromArray(new_arr)
        new_data.SetOrigin(origin)
        new_data.SetDirection(direction)
        new_data.SetSpacing(new_spacing)
    else:
        new_data = new_datas[0]
    if copy_meta_data:
        meta_keys = data.GetMetaDataKeys()
        for meta_key in meta_keys:
            meta_val = data.GetMetaData(meta_key)
            new_data.SetMetaData(key=meta_key, val=meta_val)

    return new_data




