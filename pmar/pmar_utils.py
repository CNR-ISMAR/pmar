import numpy as np
import xarray as xr
from copy import deepcopy

def traj_distinct(bin_n, weight):
    print(f"bin_n: {bin_n.shape} | weight: {weight.shape}")
    w = deepcopy(weight)
    for t in range(1,len(bin_n)): # i leave the first element as 1, otherwise the loop would compare it to the last element.
        if bin_n[t] == bin_n[t-1]:
            w[t] = 0
    return w

def rasterhd2raster(raster_hd, grid):   
    '''
    Reproject raster onto a new grid. 
    #### NOTE: this method is not conservative. use and resampled use do not have the same integral. 
    So not using in concentration method, for now. 

    Parameters
    ----------
    raster_hd : DataArray
        the raster to reproject
    grid : DataArray
        the grid to match raster_hd to 
    '''
    raster = (raster_hd                                                                 
              .rio.reproject_match(grid, nodata=0, resampling=Resampling.sum)
              .where(~grid.isnull())
              .rio.write_nodata(0) # trying nodata = 0 to see if it fixes negative values problem
             )                                                                                                                                                                          
    return raster 
    

def check_particle_file(path_to_file):
    ds = xr.open_dataset(path_to_file)
    vars_to_check = ['x_sea_water_velocity', 'y_sea_water_velocity', 'x_wind', 'y_wind']
    for i in vars_to_check:
        if np.all(ds[i].load() == 0):
            print(f'ATTENTION: all 0 values detected for variable {i}.')
        else:
            print(f'all good: variable {i} has non-zero values.')
