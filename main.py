import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from src import utils
from src.analysis import ImageQuantifier
import pandas as pd
from time import perf_counter_ns
from src import subset_analysis
import numpy as np


# if __name__ == "__main__":
#     # Download the data files
#     # utils.get_datafiles()
#
#     for image in ['beadpack']:#, 'castlegate', 'mtgambier', 'sandpack']:
#         # Check a slice of the image
#         img = ImageQuantifier(f"data/{image}.tif")
#         # img.plot_slice()
#         img.run_analysis(heterogeneity_kwargs={'no_radii': 20, 'no_samples_per_radius': 500}, ev_kwargs={'cube_size': 256},
#                          to_file_kwargs={'filetype': 'parquet'}, write_results=True)
#
#     minkowski = pd.read_parquet("data/image_characterization_results/minkowski.parquet")
#     # hetero = pd.read_parquet("data/image_characterization_results/heterogeneity.parquet")
#     subset = pd.read_parquet("data/image_characterization_results/subsets.parquet")
#
#     print(minkowski)
#     # print(hetero.head())
#     print(subset.head())

def subset_by_convolution(image, cube_size=256):
    kernel = np.ones((cube_size,cube_size,cube_size), dtype=np.uint8)

    # Porosity map
    # porosity_map = conv3D(image, kernel) / cube_size ** 3

    # Get batch_size largest porosity locations - partial sorting completes in O(n) time.
    #largest_porosity_indices = np.argpartition(porosity_map, batch_size)

    # Get batch_size largest porosity locations
    # porosity_map =

if __name__ == '__main__':
    np.random.seed(101325)
    image = utils.read_tiff("data/beadpack.tif")


    start = perf_counter_ns()
    subset_analysis.find_porosity_visualization_interval_c(image, cube_size=256)
    end = perf_counter_ns()
    print(f"Cython Time taken: {(end - start) / 1e9:0.05f}s")
    #
    np.random.seed(101325)
    img = ImageQuantifier(f"data/beadpack.tif")
    start = perf_counter_ns()
    img.find_porosity_visualization_interval()
    end = perf_counter_ns()
    print(f"Vanilla Python Time taken: {(end - start) / 1e9:0.05f}s")

