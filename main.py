import matplotlib.pyplot as plt
from src import utils
from src.analysis import ImageQuantifier
import pandas as pd
from src.Vsi_new import Vsi
import time
import edt

if __name__ == "__main__":
    # Download the data files
    # utils.get_datafiles()

    for image in ['beadpack', 'castlegate', 'mtgambier', 'sandpack']:
        # Check a slice of the image
        img = ImageQuantifier(f"/work/06898/bchang/DPM_IAP/data/{image}.tif")
        
        #start = time.perf_counter_ns()
        #img.find_porosity_visualization_interval()
        #end = time.perf_counter_ns()
        #print(f"Process took {(end - start)*1e-9:.5f}s")

        start = time.perf_counter_ns()
        img.find_interval(cube_size=100)
        end = time.perf_counter_ns()
        print(f"Process took {(end - start)*1e-9:.5f}s ")

        start = time.perf_counter_ns()
        ds = edt.edt(img.image, parallel=-1)
        mn_r = ds.max()
        mx_r = mn_r + 100
        vf = Vsi(img.image, min_radius=mn_r, max_radius=mx_r, no_radii=20, no_samples_per_radius=500)
        end = time.perf_counter_ns()
        print(f"Process took {(end - start)*1e-9:.5f}s ")
        # img.plot_slice()
        #img.run_analysis(heterogeneity_kwargs={'no_radii': 20, 'no_samples_per_radius': 500}, ev_kwargs={'cube_size': 256},
        #                 to_file_kwargs={'filetype': 'parquet'})

    #minkowski = pd.read_parquet("data/image_characterization_results/minkowski.parquet")
    #hetero = pd.read_parquet("data/image_characterization_results/heterogeneity.parquet")
    #subset = pd.read_parquet("data/image_characterization_results/subsets.parquet")

    #print(minkowski)
    #print(hetero.head())
    #print(subset.head())
