import numpy as np
import xarray as xr
import rioxarray as rxr
from copy import deepcopy
import cartopy
import cartopy.io.shapereader as shpreader
import geopandas as gpd
from rasterio.enums import Resampling
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely import box
from geocube.api.core import make_geocube
from functools import partial
from geocube.rasterize import rasterize_image
from rasterio.enums import MergeAlg
import logging
import matplotlib.pyplot as plt
from pathlib import Path
logger = logging.getLogger('pmar')


def make_poly(bounds, crs='4326', save_to=None):
    _df = pd.DataFrame()
    #_df['bounds'] = bounds
    _df['ID'] = ['seeding_polygon']
    _df['geometry'] = box(bounds[0], bounds[1], bounds[2], bounds[3])
    poly = gpd.GeoDataFrame(_df, geometry="geometry").set_crs(epsg=crs).to_crs(epsg='4326') # transform into geodataframe
    
    if save_to is not None:
        poly.to_file(save_to, driver='ESRI Shapefile')
        #self.poly_path = str(q)
    return poly

def _make_poly(lon, lat, crs='4326', save_to=None):
    """
    Creates shapely.Polygon where particles will be released homogeneously and writes it to a shapefile. 
    
    Parameters
    ----------
    lon : list or array
        Either a list giving 2 bounds or an array of lon coordinates.
    lat : list or array
        Either a list giving 2 bounds or an array of lat coordinates.
    crs : str, optional
        EPSG string for Polygon crs. Default is '4326'
    write : bool, optional
        Whether to save the polygon as a shapefile in the 's' directory. Default is True
    """
    
    
    #Path(self.basedir / 'polygons').mkdir(parents=True, exist_ok=True)
    #poly_path = f'polygon-crs_epsg:{crs}-lon_{np.round(lon[0],4)}_{np.round(lon[1],4)}-lat—{np.round(lat[0],4)}_{np.round(lat[1])}.shp'
    #q = self.basedir / 'polygons' / poly_path
    
    if len(np.array(lon)) == 2 and len(np.array(lat)) == 2:
        _df = pd.DataFrame() # create dataframe 
        _df['lon'] = lon # add lon and lat arrays
        _df['lat'] = lat
        _df['geometry'] = Polygon(zip([lon[0],lon[0],lon[1],lon[1]], [lat[0],lat[1],lat[1],lat[0]]))
        poly = gpd.GeoDataFrame(_df, geometry="geometry").set_crs(epsg=crs).to_crs(epsg='4326') # transform into geodataframe

    elif len(np.array(lon)) > 2 and len(np.array(lat)) > 2:
        x, y = np.meshgrid(lon, lat)

        _df = pd.DataFrame() # create dataframe 
        _df['lon'] = x.ravel() # add lon and lat arrays

        _df['lat'] = y.ravel()
        _df['geometry'] = _df.apply(lambda r: Point(r.lon, r.lat), axis=1) # create point geometries

        df = gpd.GeoDataFrame(_df, geometry="geometry") # transform into geodataframe

        # buffer. cap_style = 3 means a square buffer. (creates square around centroid with distance 'res' between centroid and side)
        buffer = df['geometry'].buffer(np.diff(lon).mean(), cap_style = 3)
        # add buffered polygon to geo df
        df['squares'] = buffer
        gds = gpd.GeoDataFrame(df, geometry="squares")
        # create polygon by dissolving squares into each other
        poly = gds.dissolve()
        # set crs and reproject to desired crs 
        poly = poly.set_crs(epsg=crs).to_crs(epsg='4326')
    else:
        raise ValueError("'lon' and 'lat' must have length larger than 2")
    
    if save_to is not None:
        poly.to_file(save_to, driver='ESRI Shapefile')
        #self.poly_path = str(q)

    return poly

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
#              .where(~grid.isnull())
              .rio.write_nodata(0) # trying nodata = 0 to see if it fixes negative values problem
             )                                                                                                                                                                          
    return raster 
    
def rasterize_points_add(point_data, like, measurements=None):
    '''
    Rasterize points on grid of resolution 'res' by adding values of variable 'measurements' in each cell.
    '''
    raster_data = make_geocube(vector_data=point_data, measurements=measurements, 
                    like=like,
                    #resolution=(-res, res), 
                    fill=0,
                    rasterize_function=partial(rasterize_image, merge_alg=MergeAlg.add, fill=0, all_touched=True, filter_nan=True))
    logger.warning('no "measurement" given in make_geocube, i.e. value to sum with mergealg.')
    return raster_data


def harmonize_use(use, res, study_area, like, tstep=None):
    '''
    Use can be a path to a shapefile or raster, a geopandas geodataframe or a xarray object.
    This method rasterizes use (if needed), it reprojects it to standard projection, it resamples it on given grid and within chosen bounding box. 
    Use is now compatible with PMAR run and ready to be assigned as weight to particles. 
    '''
    try:
        #Path(use).suffix
        if Path(use).suffix == '.shp':
            logger.debug('use is shapefile')
            vector_use = gpd.read_file(use)
            print('SHP USE TOTAL BOUNDS', vector_use.total_bounds)
        elif Path(use).suffix == '.tif':
            raster_use = rxr.open_rasterio(use).squeeze()
            print('TIF USE TOTAL BOUNDS', raster_use.x.min(), raster_use.x.max(), raster_use.y.min(), raster_use.y.max())            
            logger.debug('use is tif')

    except:     
        if type(use) is gpd.geodataframe.GeoDataFrame:
            logger.debug('use is geopandas dataframe')
            vector_use = use
            print('GPD USE TOTAL BOUNDS', vector_use.total_bounds)
        elif type(use) is xr.core.dataarray.DataArray:
            logger.debug('use is xarray')
            raster_use = use
            print('RXR USE TOTAL BOUNDS', raster_use.x.min(), raster_use.x.max(), raster_use.y.min(), raster_use.y.max()) 
        else:
            logger.debug('use is none of the above')

    if 'vector_use' in locals():
        if len(list(vector_use.columns.drop('geometry'))) > 1:
            raise ValueError('Too many columns in GeoDataFrame. Please provide shapefile with only one variable, other than geometry.')
        elif len(list(vector_use.columns.drop('geometry'))) < 1:
            raise ValueError('No columns found other than geometry. No values to burn into raster cells.')
        raster_use = rasterize_points_add(vector_use, like=like, measurements=list(vector_use.columns.drop('geometry'))).to_dataarray()
        logger.debug(f'rasterized vector_use to res = {res}')
        print('Rasterized USE TOTAL BOUNDS', raster_use.x.min(), raster_use.x.max(), raster_use.y.min(), raster_use.y.max()) 
        
    raster_use = raster_use.rio.reproject('epsg:4326', nodata=0).sortby('x').sortby('y').sel(x=slice(study_area[0], study_area[2]), y=slice(study_area[1], study_area[3])).fillna(0) # fillna is needed so i dont get nan values in the resampled sum
    logger.debug('use_raster successfully reprojected')
    print('Rasterized USE TOTAL BOUNDS', raster_use.x.min(), raster_use.x.max(), raster_use.y.min(), raster_use.y.max()) 
    # need a dataarray
    use = rasterhd2raster(raster_use, like) # resample the use raster on our grid, with user-defined res and crs
    print('Resampled USE TOTAL BOUNDS', use.x.min(), use.x.max(), use.y.min(), use.y.max()) 
    logger.debug('use_raster successfully resampled')
    if len(use.shape) > 2:
        use = use[0]
        
    use = use.where(use>=0,0) # rasterhd2raster sometimes gives small negative values when resampling. I am filling those with 0. 

    # convert use from quantity/day into quantity/timestep
    # timesteps_per_day = 24*60*60/tstep # tstep must be given in seconds 
    # use = use / timesteps_per_day
    
    logger.debug(f'final use has shape {use.shape}')
    return use


def check_particle_file(path_to_file):
    ds = xr.open_dataset(path_to_file)
    vars_to_check = ['x_sea_water_velocity', 'y_sea_water_velocity', 'x_wind', 'y_wind']
    for i in vars_to_check:
        if np.all(ds[i].load() == 0):
            logger.debug(f'ATTENTION: all 0 values detected for variable {i}.')
        else:
            logger.debug(f'all good: variable {i} has non-zero values.')


def traj_distinct(bin_n, weight):
    logger.debug(f"bin_n: {bin_n.shape} | weight: {weight.shape}")
    w = deepcopy(weight)
    for t in range(1,len(bin_n)): # i leave the first element as 1, otherwise the loop would compare it to the last element.
        if bin_n[t] == bin_n[t-1]:
            w[t] = 0
    return w
    

def plot_map(coastres='10m', figsize=[8,6], dpi=120, proj=cartopy.crs.PlateCarree(),  extent=None, title=''):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=proj)
    ax.coastlines(coastres, zorder=12, color='k', linewidth=.7)
    ax.add_feature(cartopy.feature.LAND, facecolor='0.9', zorder=2, alpha=0.5)
    ax.add_feature(cartopy.feature.BORDERS, zorder=11, linewidth=.5, linestyle=':')
    if extent is not None:
        ax.set_extent(extent, crs=cartopy.crs.PlateCarree()) # since some values are 1e36, if i dont set this I see an empty map
    
    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, linewidth=0, color='gray', linestyle='--')
    #levels=np.array([1,10,100,1000])
    gl.top_labels = False
    gl.right_labels = False
    
    ax.set_title(title)
    
    return fig, ax 
    

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