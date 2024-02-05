import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import cc3d
import scipy
from src import utils
import os
import pandas as pd
import tqdm

# Morphology/Topology analysis packages
from quantimpy import minkowski as mk

# Heterogeneity analysis packages
from src.Vsi_new import Vsi
from scipy.ndimage import distance_transform_edt as dst
from scipy.ndimage import center_of_mass
import edt
import scipy.stats
import concurrent.futures
import multiprocessing as mp



class ImageQuantifier:
    def __init__(self, datapath):
        self.datapath = datapath
        self.image = utils.read_tiff(self.datapath)
        self.image_name = os.path.splitext(os.path.basename(self.datapath))[0]
        self.porosity = (self.image == 1).sum() / self.image.size

    def plot_slice(self, slice_num=None):

        if slice_num is None:
            slice_num = self.image.shape[0] // 2
        plt.figure(figsize=(10, 10))
        plt.imshow(self.image[slice_num, :, :], cmap='gray')
        plt.axis('off')
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    def run_analysis(self, heterogeneity_kwargs={}, ev_kwargs={}, write_results=True, save_path=None, to_file_kwargs={}):
        mf = self.get_quantimpy_mf()
        vf = self.heterogeneity_analysis(**heterogeneity_kwargs)
        interval = self.find_interval(**ev_kwargs)

        if save_path is None:
            save_path = os.path.dirname(self.datapath)
        if write_results:
            utils.write_results(mf, 'minkowski', directory_path=save_path, **to_file_kwargs)
            utils.write_results(vf, 'heterogeneity', directory_path=save_path, **to_file_kwargs)
            utils.write_results(interval, 'subsets', directory_path=save_path, **to_file_kwargs)


    def get_quantimpy_mf(self):
        """
        Returns the Minkowski functionals measured by the Quantimpy library.
        :return: Dataframe  of Minkowski functionals (Volume, Surface Area, Integral Mean Curvature, Euler Characteristic)
        """
        mf0, mf1, mf2, mf3 = mk.functionals(self.image.astype(bool))
        mf1 *= 8
        mf2 *= 2*np.pi**2
        mf3 *= 4*np.pi/3

        mf_df = pd.DataFrame({'Name': [self.image_name],
                              'Volume': [mf0],
                              'Surface Area': [mf1],
                              'Mean Curvature': [mf2],
                              'Euler Number': [mf3]})

        return mf_df

    def heterogeneity_analysis(self, **kwargs):
        """
        Performs a heterogeneity analysis on the image.
        :param image: 3D numpy array of the image with 0 as solid and 1 as pore.
        :returns: Vsi object with attributes variance and radii.
        """
        ds = edt.edt(self.image)
        mn_r = ds.max()  # maximum width of pores is used as minimum radius for moving windows
        mx_r = mn_r + 100  # maximum radius for moving windows
        # Instantiate a Vsi object
        vf = Vsi(self.image,
                 min_radius=mn_r, max_radius=mx_r, **kwargs)
        # Run heterogeneity analysis
        vf.get_heterogeneity()
        # Return dataframe with
        vf_df = vf.result()
        # heterogeneity_ratio = vf.rock_type()
        vf_df.insert(0, 'Name', self.image_name, allow_duplicates=True)
        # vf_df.insert(3, 'Heterogeneity Ratio', heterogeneity_ratio, allow_duplicates=True)

        return vf_df

    def find_interval(self, cube_size=100, batch=100, **kwargs):
        """
        Finds the best cubic interval for visualizing the segmented dataset.

        cube_size: Size of the visualization cube, default is 100 (100x100x100).
        batch: Batch over which to calculate the stats, default is 100.
        """
        # np.random.seed(1589061)
        scalar_data = deepcopy(self.image)



        # Inner cube increment
        inc = (cube_size - int(cube_size * 0.5)) // 2

        # One dimension of the given vector sample cube.
        max_dim = len(scalar_data)

        stats_array = np.zeros(shape=(batch, 5))

        # with concurrent.futures.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        #     for i in range(batch):
        #         stats_array[i, :] = executor.submit(self._get_stats_array,
        #                                             max_dim=max_dim,
        #                                             cube_size=cube_size,
        #                                             inc=inc, iteration=i).result()
        for i in range(batch):
            stats_array[i] = self._get_stats_array(max_dim=max_dim, cube_size=cube_size, inc=inc, iteration=i)
        print(stats_array)
        best_index = np.argmax(stats_array[:, 4])
        best_interval = int(stats_array[best_index, 3])

        # print(f'Original Porosity: {round(self.porosity * 100, 2)} %\n' +
        #       f'Subset Porosity: {round(stats_array[2, best_index] * 100, 2)} %\n' +
        #       f'Competent Interval: [{best_interval}:{best_interval + cube_size},' +
        #       f'{best_interval}:{best_interval + cube_size},{best_interval}:{best_interval + cube_size}]')

        best_interval = (int(best_interval), int(best_interval + cube_size))

        subset_df = pd.DataFrame({'Name': [self.image_name],
                                  'subset_start': [best_interval[0]],
                                  'subset_end': [best_interval[1]]})

        return subset_df

    # def _get_stats_array(self, max_dim: int, cube_size: int, inc: int, iteration: int) -> tuple:
    #     """
    #     Get the stats array for subset extraction
    #     Returns: np.ndarray of array stats
    #     """
    #     # Set a random seed
    #     np.random.seed(iteration*1742809)
    #     while True:
    #         mini = np.random.randint(low=0, high=max_dim - cube_size)
    #         maxi = mini + cube_size
    #
    #         scalar_boot = self.image[mini:maxi, mini:maxi, mini:maxi]
    #         scalar_boot_inner = self.image[mini + inc:maxi - inc, mini + inc:maxi - inc, mini + inc:maxi - inc]
    #
    #         labels_out_outside, _ = cc3d.largest_k(
    #             scalar_boot, k=1,
    #             connectivity=26, delta=0,
    #             return_N=True,
    #         )
    #
    #         index_outside, counts_outside = np.unique(labels_out_outside, return_counts=True)
    #         counts_outside_sum = np.sum(counts_outside[1:])
    #
    #         labels_out_inside, _ = cc3d.largest_k(
    #             scalar_boot_inner, k=1,
    #             connectivity=26, delta=0,
    #             return_N=True,
    #         )
    #
    #         index_inside, counts_inside = np.unique(labels_out_inside, return_counts=True)
    #         counts_inside_sum = np.sum(counts_inside[1:])
    #
    #         porosity_selected = (scalar_boot == 1).sum() / cube_size ** 3
    #
    #         if (porosity_selected <= self.porosity * 1.2) & (porosity_selected >= self.porosity * 0.8):
    #             harmonic_mean_metric = scipy.stats.hmean([counts_outside_sum, counts_inside_sum])
    #             return counts_outside_sum, counts_inside_sum, porosity_selected, mini, harmonic_mean_metric

    def _get_stats_array(self, max_dim: int, cube_size: int, inc: int, iteration: int) -> tuple:
        """
        Get the stats array for subset extraction
        Returns: np.ndarray of array stats
        """
        # Set a random seed
        np.random.seed(iteration*1742809)
        while True:
            mini = np.random.randint(low=0, high=max_dim - cube_size)
            maxi = mini + cube_size

            scalar_boot = self.image[mini:maxi, mini:maxi, mini:maxi]

            labels_out_outside = cc3d.largest_k(
                scalar_boot, k=1,
                connectivity=26, delta=0,
                return_N=False,
            )

            # Ensure percolation of largest component in z
            if not (1 in labels_out_outside[0] and 1 in labels_out_outside[-1]):
                continue

            counts_outside_sum = np.count_nonzero(labels_out_outside)
            # index_outside, counts_outside = np.unique(labels_out_outside, return_counts=True)
            # counts_outside_sum = np.sum(counts_outside[1:])

            # Compute center of mass
            com = np.asarray(center_of_mass(labels_out_outside))
            # Distance of the center of mass to the center of the cube
            dist = np.linalg.norm(com - [cube_size // 2]*3)

            porosity_selected = (scalar_boot == 1).sum() / cube_size ** 3

            if (porosity_selected <= self.porosity * 1.2) & (porosity_selected >= self.porosity * 0.8):
                harmonic_mean_metric = scipy.stats.hmean([counts_outside_sum, 1/dist])
                return counts_outside_sum, dist, porosity_selected, mini, harmonic_mean_metric






