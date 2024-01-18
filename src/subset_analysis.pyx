import cc3d
import numpy as np
import scipy
# import pandas as pd
# import utils
from time import perf_counter
cimport numpy as cnp
cimport cython
from cython.parallel cimport prange

cnp.import_array()

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False) # turn off negative index wrapping for this function
def find_porosity_visualization_interval_c(cnp.ndarray[cnp.uint8_t, ndim=3] image, int cube_size=100, int batch=100):
    """
    Finds the best cubic interval for visualizing the segmented dataset.

    cube_size: Size of the visualization cube, default is 100 (100x100x100).
    batch: Batch over which to calculate the stats, default is 100.
    """

    # scalar_data = deepcopy(image)

    # scalar_data[scalar_data == 0] = 199
    # scalar_data[scalar_data != 199] = 0
    # scalar_data[scalar_data == 199] = 1

    # size = image.shape[0] * image.shape[1] * image.shape[2]
    cdef float porosity, porosity_selected
    cdef int max_dim, counts_inside_sum, counts_outside_sum, best_index
    cdef Py_ssize_t i, mini, maxi, best_interval_index, inc
    cdef tuple best_interval
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] scalar_boot, scalar_boot_inner
    cdef cnp.ndarray[DTYPE_t, ndim=2, cast=True] stats_array = np.zeros(shape=(4, batch), dtype=DTYPE)
    cdef cnp.ndarray[cnp.uint32_t, ndim=3, cast=True] labels_out_outside
    # cdef cnp.ndarray[cnp.uint8_t, ndim=1, cast=True] labels_out_inside, index_inside, counts_inside
    cdef cnp.ndarray[cnp.uint32_t, ndim=1, cast=True] index_outside
    cdef cnp.ndarray[cnp.int64_t, ndim=1, cast=True] counts_outside
    cdef dict subset_dict
    cdef long start_time, end_time, running_time
    cdef short sample_flag = 0

    # Compute original porosity of image
    porosity = np.sum(image==1) / np.size(image)
    # Inner cube increment
    inc = cube_size - int(cube_size * 0.5)
    # One dimension of the given vector sample cube.
    max_dim = np.shape(image)[0]
    # cdef int batch_for_stats = max_dim - cube_size  # Max possible batch number

    # Or overwrite:
    # batch_for_stats = batch
    i = 0
    running_time = 0
    while sample_flag == 0:

        mini = np.random.randint(low=0, high=max_dim - cube_size)
        maxi = mini + cube_size
        scalar_boot = image[mini:maxi, mini:maxi, mini:maxi]
        scalar_boot_inner = image[mini + inc:maxi - inc, mini + inc:maxi - inc, mini + inc:maxi - inc]
        start_time = perf_counter()
        labels_out_outside = cc3d.largest_k(
            scalar_boot, k=1,
            connectivity=26, delta=0,
            return_N=False)
        end_time = perf_counter()
        running_time += end_time - start_time


        index_outside, counts_outside = np.unique(labels_out_outside, return_counts=True)
        counts_outside_sum = np.sum(counts_outside[1:])

        # labels_out_inside = cc3d.largest_k(
        #     scalar_boot_inner, k=1,
        #     connectivity=26, delta=0,
        #     return_N=False
        # )


        # index_inside, counts_inside = np.unique(labels_out_inside, return_counts=True)
        # print(index_inside, counts_inside)
        # counts_inside_sum = np.sum(counts_inside[1:])

        porosity_selected = (scalar_boot == 1).sum() / cube_size ** 3

        if (porosity_selected <= porosity * 1.2) & (porosity_selected >= porosity * 0.8):
            stats_array[0, i] = counts_outside_sum
            # stats_array[1, i] = counts_inside_sum
            stats_array[1, i] = porosity_selected
            stats_array[2, i] = mini
            stats_array[3, i] = scipy.stats.hmean([stats_array[0, i],
                                                   stats_array[1, i]])
            sample_flag = 1

        else:
            continue

    best_index = np.argmax(stats_array[3, :])
    best_interval_index = int(stats_array[2, best_index])

    print(f'Original Porosity: {round(porosity * 100, 2)} %\n' +
          f'Subset Porosity: {round(stats_array[1, best_index] * 100, 2)} %\n' +
          f'Competent Interval: [{best_interval_index}:{best_interval_index + cube_size},' +
          f'{best_interval_index}:{best_interval_index + cube_size},{best_interval_index}:{best_interval_index + cube_size}]')

    best_interval = (int(best_interval_index), int(best_interval_index + cube_size))
    print(f"{running_time = } seconds")
    subset_dict = {'Name': ['beadpack'],
                 'subset_start': [best_interval[0]],
                 'subset_end': [best_interval[1]]}
    # subset_df = pd.DataFrame({'Name': ['beadpack'],
    #                           'subset_start': [best_interval[0]],
    #                           'subset_end': [best_interval[1]]})

    return subset_dict


