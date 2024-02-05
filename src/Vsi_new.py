import numpy as np
import pandas as pd
import concurrent.futures
import multiprocessing as mp

class Vsi():
    def __init__(self, im, no_radii=50,
                 no_samples_per_radius=50,
                 min_radius=1, max_radius=100,
                 phase=None, grid=False):
        """
        This calculates porosity variance (scale independent), based on a moving window with various radii.

        Parameters:

            im: 3D segmented (with different phases labelled) or binary image (True for pores and False for solids)
            no_radii: number of radii to be used for moving windows
            no_samples_per_radius: number of windows per radius
            min_radius: minimum radius of windows
            max_radius: maximum radius of windows
            phase: label of phase of interest. default= None assuming binary with pores are ones/True
            grid: if True, the windows are distributed on a grid having same geometry as image (im). However, the number of centroids is
                controlled by "no_samples_per_radius". Inactive when auto_centers is True.

        returns
            variance & radii
        """
        self.im = im
        self.no_radii = no_radii
        self.no_samples_per_radius = no_samples_per_radius
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.phase = phase
        self.grid = grid
        self.radii = np.empty(self.no_radii, dtype=np.uint16)
        self.variance = np.empty_like(self.radii, dtype=np.float64)


    def result(self) -> pd.DataFrame:
        return pd.DataFrame({'Radii': [self.radii], 'Variance': [self.variance]})

    def get_heterogeneity(self) -> None:
        self.radii = np.linspace(self.min_radius, self.max_radius, self.no_radii, dtype='int')

        for i, r in enumerate(self.radii):
            cntrs = self.get_centers_3d(r)

            rr, cc, zz = cntrs[:, 0], cntrs[:, 1], cntrs[:, 2]
            mn = np.array([rr - r, cc - r, zz - r])
            rw_mx, col_mx, z_mx = np.array([rr + r + 1, cc + r + 1, zz + r + 1])
            rw_mn, col_mn, z_mn = (mn > 0) * mn

            porosity = np.empty((self.no_samples_per_radius,), dtype=np.float64)
            for j in range(self.no_samples_per_radius):
                porosity[j] = np.count_nonzero(self.im[rw_mn[j]:rw_mx[j], col_mn[j]:col_mx[j], z_mn[j]:z_mx[j]]) / (2*r+1)**3
            self.variance[i] = np.var(porosity)

    def get_centers_3d(self, r: int) -> np.ndarray:

        """
        im: 3D image
        r: radius of moving window
        no_centers: number of centers for the moving windows

        auto: (default= True) this makes sure that all the image is covered by the moving windows
                also makes sure that the number of generated centroids is >= no_centers.

                when false, random coordinates are generated where the number of generated centroids = no_centers.

        adjust_no_centers: when true, no_centers is adjusted to save running time.
                            So, in case of big window, the returned coordinates are <= no_centers, while windows cover all image

        max_no_centers: maximum number of centers to be returned. None returns all centers

        returns (n,3) centers for a cubic window with side = 2r

        """
        # -------adjust window's radius with image size------
        ss = np.array(self.im.shape)
        cnd = r >= ss / 2

        if sum(cnd) == 0:
            mn = np.array([r, r, r])
            mx = ss - r
        else:
            mn = (cnd * ss / 2) + np.invert(cnd) * r
            mx = (cnd * mn + cnd) + (np.invert(cnd) * (ss - r))

        rw_mn, col_mn, z_mn = mn.astype(int)
        rw_mx, col_mx, z_mx = mx.astype(int)
        # ----------------------------------------------------
        if self.grid:
            centers = self.grid_points(self.no_samples_per_radius, self.im)

        else:
            # ------random centroids----------------------
            rndx = np.random.randint(rw_mn, rw_mx, self.no_samples_per_radius)
            rndy = np.random.randint(col_mn, col_mx, self.no_samples_per_radius)
            rndz = np.random.randint(z_mn, z_mx, self.no_samples_per_radius)
            centers = np.array([rndx, rndy, rndz]).T

        return centers

    def get_grid_points(self) -> np.ndarray:

        """
        Gets indices of "no_points" voxels distributed in a grid within array

        Parameters:
            no_points: number of centeroids in the grid
            array: 3D array of the explored data. required for defining the grid geometry.
        return:
            Centroids
        """

        pts = 1000
        x, y, z = self.im.shape
        size = x * y * z

        if pts > size:
            pts = size

        f = 1 - (pts / size)
        s = np.ceil(np.array(self.im.shape) * f).astype('int')

        nx, ny, nz = s
        xs = np.linspace(0, x, nx, dtype='int', endpoint=True)
        ys = np.linspace(0, y, ny, dtype='int', endpoint=True)
        zs = np.linspace(0, z, nz, dtype='int', endpoint=True)

        rndx, rndy, rndz = np.meshgrid(xs, ys, zs)
        rndx = rndx.flatten()
        rndy = rndy.flatten()
        rndz = rndz.flatten()
        centers = np.array([rndx, rndy, rndz]).T[:pts]

        return centers