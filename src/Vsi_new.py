import numpy as np
import pandas as pd
import concurrent.futures
import multiprocessing as mp
import numba as nb

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
        # var = self.get_lvfv_3d()
        #
        # self.variance = var[0]
        # self.radii = var[1]

    def result(self) -> pd.DataFrame:
        return pd.DataFrame({'Radii': [self.radii], 'Variance': [self.variance]})

    def get_fast(self):
        radii = np.linspace(self.min_radius, self.max_radius, self.no_radii, dtype='int')

        for i, r in enumerate(radii):
            self.radii[i] = r
            cntrs = self.get_centers_3d(r)


            rr, cc, zz = cntrs[:, 0], cntrs[:, 1], cntrs[:, 2]
            mn = np.array([rr - r, cc - r, zz - r])
            mx = np.array([rr + r + 1, cc + r + 1, zz + r + 1])
            mn = (mn > 0) * mn
            rw_mn, col_mn, z_mn = mn
            rw_mx, col_mx, z_mx = mx  # if it exceeds the image size it doesn't change anything

            # subsets = np.empty((self.no_samples_per_radius, 2*r+1, 2*r+1, 2*r+1), dtype=np.uint8)
            porosity = np.empty((self.no_samples_per_radius,), dtype=np.float64)
            # with concurrent.futures.ProcessPoolExecutor(max_workers=mp.cpu_count()//2) as executor:
            #     for j in range(self.no_samples_per_radius):
            #         result = executor.submit(self.get_radius_porosity, rw_mn[j], rw_mx[j], col_mn[j],col_mx[j], z_mn[j], z_mx[j]).result()
            #         porosity[j] = result / (2*r+1)**3
            #     for j, result in enumerate(executor.map(self.get_radius_porosity, rw_mn, rw_mx, col_mn, col_mx, z_mn, z_mx, [r]*self.no_samples_per_radius)):
            #         porosity[j] = result

            for j in range(self.no_samples_per_radius):
                # porosity[j] = self.get_radius_porosity(self.im, rw_mn[j], rw_mx[j], col_mn[j], col_mx[j], z_mn[j], z_mx[j], r)
                porosity[j] = np.count_nonzero(self.im[rw_mn[j]:rw_mx[j], col_mn[j]:col_mx[j], z_mn[j]:z_mx[j]]) / (2*r+1)**3
            self.variance[i] = np.var(porosity)
            # self.variance[i] = self.get_radius_porosity(self.im, rw_mn, rw_mx, col_mn, col_mx, z_mn, z_mx, r, self.no_samples_per_radius)

    def get_radius_porosity(self, xmin, xmax, ymin, ymax, zmin, zmax):
        return np.count_nonzero(self.im[xmin:xmax, ymin:ymax, zmin:zmax])

    # @staticmethod
    # @nb.jit(nopython=True, parallel=True)
    # def get_radius_porosity(im, xmin, xmax, ymin, ymax, zmin, zmax, r, n_radii):
    #     porosity = np.empty((n_radii), dtype=np.float64)
    #     for j in nb.prange(n_radii):
    #         porosity[j] = np.count_nonzero(im[xmin[j]:xmax[j], ymin[j]:ymax[j], zmin[j]:zmax[j]])/ (2 * r + 1) ** 3
    #
    #     return np.var(porosity)
        #     if x == 1:
        #         pore_sum += 1
        #
        # return pore_sum / (2 * r + 1) ** 3



    def get_lvfv_3d(self):

        # set a range of r
        radii = np.linspace(self.min_radius, self.max_radius, self.no_radii, dtype='int')

        lvfv = np.empty_like(radii, dtype=np.float64)

        # with concurrent.futures.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        #     for i, result in enumerate(executor.map(self.vfv_3d, radii, chunksize=self.no_samples_per_radius//mp.cpu_count())):
        #         lvfv[i] = result
        for i, r in enumerate(radii):
            lvfv[i] = self.vfv_3d(r)
        return lvfv, radii

    def vfv_3d(self, r):
        """
        Calculates the volume fraction variance
        """
        cntrs = self.get_centers_3d(r)
        vfs = np.empty_like(cntrs, dtype=np.float64)
        # with concurrent.futures.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        #     for i, result in enumerate(executor.map(self.get_radius_porosity, cntrs, [r]*len(cntrs), chunksize=self.no_samples_per_radius // mp.cpu_count())):
        #         vfs[i] = result
        for i, cntr in enumerate(cntrs):
            vfs[i] = self.get_radius_porosity(cntr, r)
        var = np.var(vfs)
        return var

    # def get_radius_porosity(self, cntr, r):
    #     s = self.get_slice_3d(cntr, r)
    #     vf = self.phi(s)
    #     return vf

    def get_centers_3d(self, r):

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

    def get_slice_3d(self, center, r):

        """
        This slices the image with a cube whose center = center
        the cube's dimensions are corrected according to the image size
        """
        # ----check whether the window would exceeds the image boundaries----if so, cut window------
        rr, cc, zz = center
        mn = np.array([rr - r, cc - r, zz - r])
        mx = np.array([rr + r + 1, cc + r + 1, zz + r + 1])
        mn = (mn > 0) * mn

        rw_mn, col_mn, z_mn = mn
        rw_mx, col_mx, z_mx = mx  # if it exceeds the image size it doesn't change anything

        return self.im[rw_mn:rw_mx, col_mn:col_mx, z_mn:z_mx]

    def phi(self, subset, return_fraction=False):
        """
        calculates volume fraction (e.g., porosity) of
        a phase in an image im
        """
        if self.phase == None:  # assumes binary data
            sm = np.sum(subset)
        else:
            sm = len(subset[subset == self.phase])

        phi_ = sm / subset.size
        if return_fraction:
            return phi_, sm
        else:
            return phi_

    def get_grid_points(self):

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
        centers = (np.array([rndx, rndy, rndz]).T)[:pts]

        return centers