import numpy as np
import xarray as xr
from copy import deepcopy
import cartopy.io.shapereader as shpreader
import geopandas as gpd
from rasterio.enums import Resampling

def get_marine_polygon(basin=None):
    '''
    Returns gdf with polygon of requested basin, to be used with OpenDrift's seed_from_polygon.
    Default (None) returns global ocean.
    '''
    # landmask from cartopy 
    shpfilename = shpreader.natural_earth(resolution='10m',
                            category='physical',
                            name='geography_marine_polys')
    
    gdf = gpd.read_file(shpfilename)

    if basin is None:
        marine_poly = gdf[['geometry']]
    else:
        marine_poly = gdf.set_index('label').loc[[str(basin)]][['geometry']]

    return marine_poly
    

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
            

def traj_distinct(bin_n, weight):
    print(f"bin_n: {bin_n.shape} | weight: {weight.shape}")
    w = deepcopy(weight)
    for t in range(1,len(bin_n)): # i leave the first element as 1, otherwise the loop would compare it to the last element.
        if bin_n[t] == bin_n[t-1]:
            w[t] = 0
    return w
    

def custom_plotfunc(ds, fig, tt, *args, **kwargs):
    '''
    For pmar.animate. WIP.
    it works but it writes a blank video to mp4.
    '''
    
    fig = plt.figure()
    ax = plt.axes(projection=cartopy.crs.PlateCarree())
    ax.coastlines('10m', zorder=12, color='k', linewidth=.7)
    ax.add_feature(cartopy.feature.LAND, facecolor='0.9', zorder=2)
    ax.add_feature(cartopy.feature.BORDERS, zorder=11, linewidth=.5, linestyle=':')
    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, linewidth=0, color='gray', linestyle='--')
    #levels=np.array([1,10,100,1000])
    gl.top_labels = False
    gl.right_labels = False

    # Colorlimits need to be fixed or your video is going to cause seizures.
    ds.isel(time=tt).plot(ax=ax, cmap=spectral_r, vmin=0, vmax=ds.max())
    ax.set_title('');
    return None, None