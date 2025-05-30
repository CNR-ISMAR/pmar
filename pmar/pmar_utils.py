import numpy as np
import xarray as xr
from copy import deepcopy
import cartopy.io.shapereader as shpreader
import geopandas as gpd
from rasterio.enums import Resampling
import pandas as pd
from shapely.geometry import Point, Polygon

def make_poly(lon, lat, crs='4326', save_to=None):
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
        poly.to_file(str(save_to), driver='ESRI Shapefile')
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