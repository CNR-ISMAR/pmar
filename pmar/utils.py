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
from shapely.ops import unary_union
from geocube.api.core import make_geocube
from functools import partial
from geocube.rasterize import rasterize_image
from rasterio.enums import MergeAlg
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from geocube.vector import vectorize 

logger = logging.getLogger('pmar')


SPATIAL_DOMAINS = {'global':
                {'readers':
                 {'currents': 'cmems_mod_glo_phy_my_0.083deg_P1D-m', # global physics reanalysis, daily, 1/12° horizontal resolution, 50 vertical levels, 1 Jan 1993 to 2021-06-30
                  'winds': 'cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H', # L4 sea surface wind and stress fields 0.125° res, hourly, 1 Jun 1994 to 21 Nov 2024
                  #'bathymetry':'cmems_mod_glo_phy_my_0.083deg_static', # global physics reanalysis static, same as currents
                  #'mixed-layer': 'cmems_mod_glo_phy_my_0.083deg_P1D-m', # same as currents
             #     'stokes': 'cmems_mod_glo_wav_my_0.2deg_PT3H-i', # global ocean waves reanalysis, 3-hourly, 1 Jan 1980 to 31 Jan 2025
              #    'spm': 'cmems_obs-oc_glo_bgc-transp_my_l4-multi-4km_P1M',
                 },
                'extent': [-180, -80, 179.92, 90],
                 'label': 'CMEMS_Global_Renalysis'
                },

                'med':
                {'readers':
                 {'currents': 'cmems_mod_med_phy-cur_my_4.2km_PT1H-m', # change H to D to get daily. Mediterranean Sea Physics Reanalysis, 1/24˚res, 141 z-levels, 1 Jan 1987 to 1 Feb 2025
                  'winds': 'cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H', # global L4
                  'bathymetry': 'cmems_mod_med_phy_my_4.2km_static', # Mediterranean Sea Physics Reanalysis
                  'mixed-layer': 'cmems_mod_med_phy-mld_my_4.2km_P1D-m', # Mediterranean Sea Physics Reanalysis
                  'stokes': 'cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H',   # Mediterranean Sea Waves Reanalysis, hourly, 1/24˚res, 1 Jan 1985 to 1 Mar 2025
                 },
                 'extent': [-6, 30.19, 36.29, 45.98],
                'label': 'CMEMS_Mediterranean_Renalysis'
                },

                'black sea':
                {'readers':
                 {'currents': 'cmems_mod_blk_phy-cur_my_2.5km_P1D-m', #Black Sea Physics Reanalysis, res 1/40º, 121 z-levels, 1 Jan 1993 to 1 Feb 2025
                  'winds': 'cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H', # global L4,
                  'bathymetry': 'cmems_mod_blk_phy_my_2.5km_static',  # Black Sea Physics Reanalysis
                  'mixed-layer': 'cmems_mod_blk_phy-mld_my_2.5km_P1D-m', # Black Sea Physics Reanalysis
                  'stokes': 'cmems_mod_blk_wav_my_2.5km_PT1H-i',  # Black Sea Waves Reanalysis
                 },
                'extent': [27.25, 40.5, 42, 47],
                'label': 'CMEMS_Black_Sea_Renalysis'
                },

                'baltic':
                {'readers':
                 {'currents': 'cmems_mod_bal_phy_my_P1D-m', #Baltic Sea Physics Reanalysis, 2x2km, 56 depth-levels, 1 Jan 1993 to 31 Dec 2023
                  'winds': 'cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H', # global L4,
                  'bathymetry': 'cmems_mod_bal_phy_my_static', #Baltic Sea Physics Reanalysis,
                  'mixed-layer':'cmems_mod_bal_phy_my_P1D-m', #Baltic Sea Physics Reanalysis, 2x2km, 56 depth-levels, 1 Jan 1993 to 31 Dec 2023
                  'stokes': 'cmems_mod_bal_wav_my_PT1H-i',
                 },
                 'extent': [9.04, 53.01, 30.21, 65.89],
                'label': 'CMEMS_Baltic_Sea_Renalysis'
                }
                                    
                  }


def get_copernicus(spatial_domain):
    """
    Return the Copernicus Marine (CMEMS) reader configuration for a given ocean basin.

    Parameters
    ----------
    spatial_domain : str
        Name of the ocean basin to retrieve Copernicus data for. One of 'global', 'med',
        'black sea' or 'baltic'.

    Returns
    -------
    dict
        Dictionary with keys 'readers' (a dict mapping variable name, e.g. 'currents',
        'winds', 'bathymetry', 'mixed-layer', 'stokes', to its Copernicus Marine dataset
        id), 'extent' (the basin's spatial extent as [xmin, ymin, xmax, ymax]) and 'label'
        (a human-readable name for the basin).

    Raises
    ------
    ValueError
        If spatial_domain is not one of the supported basins.
    """

    if spatial_domain not in SPATIAL_DOMAINS.keys():
        raise ValueError('Unsupported context. Please insert one of "global", "med", "black sea", "baltic".')

    return SPATIAL_DOMAINS[spatial_domain]


def select_spatial_domain(bounds):
    """
    Auto-select the Copernicus Marine basin(s) whose data covers a seeding bounding box.

    Compares `bounds` against every non-'global' basin's 'extent' in SPATIAL_DOMAINS and
    merges the readers of every basin whose extent intersects it (since OpenDrift can use
    multiple readers per variable). 'global' is added as a gap-filler alongside those
    regional readers whenever their combined extent doesn't fully cover `bounds` (this
    includes the case where no regional basin overlaps at all, in which case 'global' is
    used alone) -- regional data is always preferred where available, with 'global' only
    filling in what it doesn't cover.

    Parameters
    ----------
    bounds : list or array-like
        Seeding bounding box as [xmin, ymin, xmax, ymax].

    Returns
    -------
    dict
        Dictionary shaped like get_copernicus()'s return value: 'readers' (merged dict of
        dataset ids, keyed by f'{basin_name}:{variable}' and deduplicated by dataset id),
        'extent' (bounding box of the merged basins' extents) and 'label' (names of the
        merged basins).
    """
    seeding_box = box(*bounds)

    regional_matches = [
        name for name, domain in SPATIAL_DOMAINS.items()
        if name != 'global' and box(*domain['extent']).intersects(seeding_box)
    ]

    covered = False
    if regional_matches:
        union = unary_union([box(*SPATIAL_DOMAINS[name]['extent']) for name in regional_matches])
        covered = union.covers(seeding_box)

    matched_basins = list(regional_matches)
    if not covered:
        matched_basins.append('global')

    merged_readers = {}
    seen_dataset_ids = set()
    for basin_name in matched_basins:
        for var, dataset_id in SPATIAL_DOMAINS[basin_name]['readers'].items():
            if dataset_id in seen_dataset_ids:
                continue
            seen_dataset_ids.add(dataset_id)
            merged_readers[f'{basin_name}:{var}'] = dataset_id

    merged_extent = unary_union([box(*SPATIAL_DOMAINS[name]['extent']) for name in matched_basins]).bounds

    logger.info(f'auto-selected spatial_domain(s): {matched_basins}')

    return {
        'readers': merged_readers,
        'extent': list(merged_extent),
        'label': 'auto: ' + ' + '.join(matched_basins),
    }


def make_poly(bounds, crs='4326', save_to=None):
    """
    Create a single rectangular polygon covering the given bounds, as a GeoDataFrame with
    one row (ID 'seeding_polygon'), optionally saving it as a shapefile.

    Parameters
    ----------
    bounds : list or array-like
        Bounding box as [xmin, ymin, xmax, ymax].
    crs : str, optional
        EPSG code (without the 'EPSG:' prefix) for bounds and the output polygon. Default
        is '4326'.
    save_to : str, optional
        Path to save the polygon as an ESRI Shapefile. Default is None, i.e. the polygon
        is not saved.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with a single row containing the rectangular polygon.
    """
    df = pd.DataFrame()
    df['ID'] = ['seeding_polygon']
    df['geometry'] = box(bounds[0], bounds[1], bounds[2], bounds[3])
    poly = gpd.GeoDataFrame(df, geometry="geometry", crs = f"EPSG:{crs}") # transform into geodataframe
    
    if save_to is not None:
        poly.to_file(save_to, driver='ESRI Shapefile')
        #self.poly_path = str(q)
    return poly

# unused (only reference elsewhere is commented out in pmar.py) - kept for reference
# def _make_poly(lon, lat, crs='4326', save_to=None):
#     """
#     Creates shapely.Polygon where particles will be released homogeneously and writes it to a shapefile.
#
#     If lon and lat each have length 2, they are treated as [min, max] bounds and a single
#     rectangular polygon is created. If they have length greater than 2, they are treated as
#     coordinate arrays: a square buffer (of half the mean coordinate spacing) is built around
#     every point of the lon/lat meshgrid, producing a GeoDataFrame of adjoining square cells.
#
#     Parameters
#     ----------
#     lon : list or array
#         Either a list giving 2 bounds or an array of lon coordinates.
#     lat : list or array
#         Either a list giving 2 bounds or an array of lat coordinates.
#     crs : str, optional
#         EPSG code (without the 'EPSG:' prefix) that lon/lat are given in. Default is '4326'.
#     save_to : str, optional
#         Path to save the polygon(s) as an ESRI Shapefile. Default is None, i.e. not saved.
#
#     Returns
#     -------
#     geopandas.GeoDataFrame
#         GeoDataFrame (reprojected to EPSG:4326) containing either the single rectangular
#         polygon, or one square polygon per lon/lat grid point.
#
#     Raises
#     ------
#     ValueError
#         If lon and lat do not both have length 2, and do not both have length greater than 2.
#     """
#
#
#     #Path(self.basedir / 'polygons').mkdir(parents=True, exist_ok=True)
#     #poly_path = f'polygon-crs_epsg:{crs}-lon_{np.round(lon[0],4)}_{np.round(lon[1],4)}-lat—{np.round(lat[0],4)}_{np.round(lat[1])}.shp'
#     #q = self.basedir / 'polygons' / poly_path
#
#     if len(np.array(lon)) == 2 and len(np.array(lat)) == 2:
#         _df = pd.DataFrame() # create dataframe
#         _df['lon'] = lon # add lon and lat arrays
#         _df['lat'] = lat
#         _df['geometry'] = Polygon(zip([lon[0],lon[0],lon[1],lon[1]], [lat[0],lat[1],lat[1],lat[0]]))
#         poly = gpd.GeoDataFrame(_df, geometry="geometry").set_crs(epsg=crs).to_crs(epsg='4326') # transform into geodataframe
#
#     elif len(np.array(lon)) > 2 and len(np.array(lat)) > 2:
#         x, y = np.meshgrid(lon, lat)
#
#         _df = pd.DataFrame() # create dataframe
#         _df['lon'] = x.ravel() # add lon and lat arrays
#
#         _df['lat'] = y.ravel()
#         _df['geometry'] = _df.apply(lambda r: Point(r.lon, r.lat), axis=1) # create point geometries
#
#         df = gpd.GeoDataFrame(_df, geometry="geometry") # transform into geodataframe
#
#         # buffer. cap_style = 3 means a square buffer. (creates square around centroid with distance 'res' between centroid and side)
#         buffer = df['geometry'].buffer(np.diff(lon).mean()/2, cap_style = 3)
#         # add buffered polygon to geo df
#         df['squares'] = buffer
#         gds = gpd.GeoDataFrame(df, geometry="squares")
#         # create polygon by dissolving squares into each other
#         #poly = gds.dissolve()
#         # set crs and reproject to desired crs
#         #poly = poly.set_crs(epsg=crs).to_crs(epsg='4326')
#         poly = gds.set_crs(epsg=crs).to_crs(epsg='4326')
#
#     else:
#         raise ValueError("'lon' and 'lat' must have length larger than 2")
#
#     if save_to is not None:
#         poly.to_file(save_to, driver='ESRI Shapefile')
#         #self.poly_path = str(q)
#
#     return poly

def make_grid(res, study_area, crs='epsg:4326'):
    """
    Build a regular spatial grid over study_area at resolution res, masked to ocean cells,
    with each ocean cell assigned a unique sequential bin number.

    Parameters
    ----------
    res : float
        Grid cell size (in crs units), used for both x and y directions.
    study_area : list or array-like
        Bounding box of the grid as [xmin, ymin, xmax, ymax].
    crs : str, optional
        CRS to assign to the grid. Default is 'epsg:4326'.

    Returns
    -------
    xarray.Dataset
        Dataset with coordinates x_e/y_e (cell edges) and x_c/y_c (cell centres), and
        variables:
        - dx, dy : cell size in x/y (equal to res)
        - Xc, Yc : 2D meshgrid of cell-centre coordinates, dims ('y_c', 'x_c')
        - mask : 1 where the cell centre falls within the ocean (10m resolution Natural
          Earth ocean polygons), NaN elsewhere
        - bin_n : unique sequential integer id (0-based) for each ocean grid cell, NaN over
          land; used to bin/group trajectories spatially (see get_bin_n).
    """
    #if study_area is None:
      #  study_area = self.study_area

    shpfilename = shpreader.natural_earth(resolution='10m',
            category='physical',
            name='ocean')
    ocean = gpd.read_file(shpfilename)
        
    xmin, ymin, xmax, ymax = study_area
    cols = np.arange(xmin, xmax + res, res)            
    rows = np.arange(ymin, ymax + res, res)

    # outer cell edges
    x_e = cols 
    y_e = rows
    
    # cell centres
    x_c = (x_e + res/2)[:-1]
    y_c = (y_e + res/2)[:-1]     

    grid = xr.Dataset(coords={'x_e':x_e, 'y_e':y_e, 'x_c':x_c, 'y_c':y_c}).rio.write_crs(crs, inplace=True)

    grid['dx'] = grid['dy'] = res

    Xc, Yc = np.meshgrid(grid.x_c, grid.y_c)
    grid = grid.assign({'Xc': (('y_c', 'x_c'), Xc), 'Yc': (('y_c', 'x_c'), Yc)}) #dataarrays 

    mask = xr.ones_like(grid.Xc).rename({'x_c':'x','y_c':'y'}).rio.clip(ocean.geometry, ocean.crs, drop=False, all_touched=True).rename({'x':'x_c','y':'y_c'})
    bin_n = mask.where(mask!=1, mask.stack(cell=('y_c', 'x_c')).where(lambda x: x == 1).cumsum("cell").unstack())-1#np.arange(0,Xc.size).reshape((len(y_c), len(x_c))))
    
    grid['mask'] = mask
    grid['bin_n'] = bin_n#.astype(int)
    
    #grid = grid.assign({'bin_n': (('y_c','x_c'), bin_n.where(bin_n>0)),
     #                  'mask': (('y_c','x_c'), mask)})
    
    
    # grid = Grid(grid, 
    #             coords={"X": {"center": "x_c", "outer": "x_e"}, 
    #                    'Y': {"center": "y_c", "outer": "y_e"}}, 
    #             metrics = {
    #     ('X',): ['dx'], # X distances
    #     ('Y',): ['dy']},
    #            periodic=False)

    return grid

def make_vector_grid(raster_grid, buffer=None):
    """
    Convert a raster grid (e.g. the 'bin_n' or 'mask' variable of make_grid's output) into a
    vector GeoDataFrame of one polygon per cell, dropping empty cells.

    Parameters
    ----------
    raster_grid : xarray.DataArray or xarray.Dataset
        Raster grid to vectorize (via geocube.vector.vectorize).
    buffer : float, optional
        If given, an extra polygon is appended representing the ring between the dissolved
        grid boundary and that boundary buffered outward by this distance (in the grid's crs
        units), with a placeholder value of -1234 in the data column. Useful to add a fringe
        of "outside the grid" area around the vectorized grid. Default is None, i.e. no
        buffer polygon is added.

    Returns
    -------
    geopandas.GeoDataFrame
        Vectorized grid cells (plus, if buffer is given, one additional buffer-ring row).
    """
    vector_grid = vectorize(raster_grid).dropna()
    if buffer is not None:
        buff = vector_grid.dissolve().buffer(buffer)
        diff = gpd.GeoDataFrame({vector_grid.columns.drop('geometry')[0]: -1234, 'geometry': buff.difference(vector_grid.dissolve())}, crs=vector_grid.crs)
        vector_grid = pd.concat([vector_grid, diff]).reset_index(drop=True)
    return vector_grid

    
def get_marine_polygon(basin=None):
    '''
    Return a GeoDataFrame with the polygon of the requested marine region, to be used with
    OpenDrift's seed_from_polygon.

    Parameters
    ----------
    basin : str, optional
        Marine region label, matching the 'label' column of the Natural Earth
        'geography_marine_polys' dataset (e.g. a named sea or ocean basin). Default is None,
        i.e. all marine polygons (global ocean) are returned.

    Returns
    -------
    geopandas.GeoDataFrame
        Single-column GeoDataFrame ('geometry') with the requested polygon(s).
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

    Parameters
    ----------
    raster_hd : DataArray
        the raster to reproject
    grid : DataArray
        the grid to match raster_hd to

    Returns
    -------
    xarray.DataArray
        raster_hd reprojected and resampled (summed) onto grid's coordinates, with nodata
        set to 0.
    '''
    raster = (raster_hd                                                                 
              .rio.reproject_match(grid, nodata=0, resampling=Resampling.sum)
#              .where(~grid.isnull())
              .rio.write_nodata(0) # trying nodata = 0 to see if it fixes negative values problem
             )                                                                                                                                                                          
    return raster 
    
def _rasterize_points_add(point_data, like, measurements=None):
    '''
    Rasterize point data onto the grid of like by summing measurements within each cell.

    Parameters
    ----------
    point_data : geopandas.GeoDataFrame
        Point geometries with a column, given by measurements, to burn into the raster.
    like : xarray.DataArray
        Raster grid whose shape/resolution/crs the output is rasterized to match.
    measurements : str or list of str, optional
        Name(s) of the column(s) in point_data to sum into each raster cell. Default is
        None, which logs a warning since no value is specified to sum with MergeAlg.add.

    Returns
    -------
    xarray.Dataset
        Rasterized data on like's grid, with values in each cell equal to the sum of
        measurements from all points falling in (or touching, all_touched=True) that cell.
    '''
    raster_data = make_geocube(vector_data=point_data, measurements=measurements, 
                    like=like,
                    #resolution=(-res, res), 
                    fill=0,
                    rasterize_function=partial(rasterize_image, merge_alg=MergeAlg.add, fill=0, all_touched=True, filter_nan=True))
    if measurements is None:
        logger.warning('no "measurement" given in make_geocube, i.e. value to sum with mergealg.')
    return raster_data

def rasterize_conserve(vector_data, like, column_to_burn, all_touched=False):
    '''
    Rasterize vector_data onto the grid of like, burning column_to_burn while conserving its
    total sum: point values are summed per cell, and polygon values are rescaled by area
    before rasterizing so that the sum of the raster over a polygon's extent matches its
    original attribute value.

    Parameters
    ----------
    vector_data : geopandas.GeoDataFrame
        Vector data to rasterize. Must contain a single geometry type (either only Points or
        only Polygons).
    like : xarray.DataArray
        Raster grid whose shape/resolution/crs the output is rasterized to match.
    column_to_burn : str
        Name of the column in vector_data whose values are burned into the raster.
    all_touched : bool, optional
        For Point geometries, whether cells merely touched (not just containing the point)
        are also burned. Default is False. Not used for Polygon geometries, which are
        instead rasterized at 3x the resolution of like for area-conservation accuracy.

    Returns
    -------
    xarray.DataArray
        Rasterized column_to_burn on like's grid: for points, the sum of column_to_burn
        values falling in each cell; for polygons, the area-weighted value of column_to_burn
        integrated over each cell (so the raster's total sum approximates the sum of
        vector_data[column_to_burn]).

    Raises
    ------
    ValueError
        If vector_data contains more than one geometry type.
    '''
    if len(vector_data.geom_type.value_counts()) > 1:
        raise ValueError('please provide vector_data with a single geometry type, e.g. only polygons or only points.')
    
    elif vector_data.geom_type[0] == 'Point':
        merge_alg = MergeAlg.add
        measurements=[str(column_to_burn)]
        _like = like.copy()
    
    elif vector_data.geom_type[0] == 'Polygon':
        merge_alg = MergeAlg.replace
        vector_data[f'rescaled_{column_to_burn}'] = vector_data[str(column_to_burn)] / vector_data.geometry.area
        measurements=[f'rescaled_{column_to_burn}']
        _like = like.rio.reproject(dst_crs=like.rio.crs, resolution=[like.rio.resolution()[0]/3,like.rio.resolution()[1]/3 ], resampling=Resampling.average) # this gives a ~10% error on polygon edges

    raster_data = make_geocube(vector_data=vector_data, measurements=measurements, 
                    like=_like,
                    #resolution=(-res, res), 
                    fill=0,
                    rasterize_function=partial(rasterize_image, fill=0, merge_alg=merge_alg, all_touched=all_touched, filter_nan=True)).to_array().squeeze()

    if vector_data.geom_type[0] == 'Polygon':
        cell_area = abs(like.rio.resolution()[0])*abs(like.rio.resolution()[1])
        raster_data = raster_data.rio.reproject_match(like, resampling=Resampling.average)*cell_area
    
    return raster_data


def harmonize_use(use, res, study_area, like, tstep=None):
    '''
    Prepare a 'use' layer (representing e.g. human activity intensity) to be used as
    particle weight in a PMAR run.

    use can be a path to a shapefile or GeoTIFF, a geopandas GeoDataFrame or an xarray
    DataArray. This function rasterizes use if needed (via rasterize_conserve, conserving
    the total value for vector inputs), reprojects it to EPSG:4326, and resamples it onto
    like's grid within study_area. If tstep is given, use is also converted from a
    quantity/day rate into a quantity/timestep rate.

    Parameters
    ----------
    use : str, geopandas.GeoDataFrame or xarray.DataArray
        The use layer. If str, must be a path to a '.shp' or '.tif' file.
    res : float
        Resolution of the target grid, used only for logging.
    study_area : list or array-like
        Bounding box to clip the reprojected raster to, as [xmin, ymin, xmax, ymax].
    like : xarray.DataArray
        Raster grid whose shape/resolution/crs the output is resampled to match.
    tstep : float, optional
        Simulation time step in seconds. If given, use (assumed to be a quantity/day rate)
        is divided by the number of timesteps per day to give a quantity/timestep rate.
        Default is None, i.e. use is left as provided (per day).

    Returns
    -------
    xarray.DataArray
        use rasterized, reprojected and resampled onto like's grid, with negative values
        clipped to 0 and NaNs filled with 0.

    Raises
    ------
    ValueError
        If use is a GeoDataFrame with more than one non-geometry column (ambiguous which
        column to burn), or with no non-geometry column at all.
    '''
    try:
        #Path(use).suffix
        if Path(use).suffix == '.shp':
            logger.debug('use is shapefile')
            vector_use = gpd.read_file(use)
            #logger.debug('SHP USE TOTAL BOUNDS', vector_use.total_bounds)
        elif Path(use).suffix == '.tif':
            raster_use = rxr.open_rasterio(use).squeeze()
            #logger.debug('TIF USE TOTAL BOUNDS', raster_use.x.min().data, raster_use.x.max().data, raster_use.y.min().data, raster_use.y.max().data)            
            logger.debug('use is tif')

    except:     
        if type(use) is gpd.geodataframe.GeoDataFrame:
            logger.debug('use is geopandas dataframe')
            vector_use = use
            #logger.debug('GPD USE TOTAL BOUNDS', vector_use.total_bounds)
        elif type(use) is xr.core.dataarray.DataArray:
            logger.debug('use is xarray')
            raster_use = use
            #logger.debug('RXR USE TOTAL BOUNDS', raster_use.x.min().data, raster_use.x.max().data, raster_use.y.min().data, raster_use.y.max().data) 
        else:
            logger.debug('use is none of the above')

    if 'vector_use' in locals():
        if len(list(vector_use.columns.drop('geometry'))) > 1:
            raise ValueError('Too many columns in GeoDataFrame. Please provide shapefile with only one variable to burn into raster, other than geometry.')
        elif len(list(vector_use.columns.drop('geometry'))) < 1:
            raise ValueError('No columns found other than geometry. No values to burn into raster cells.')
        raster_use = rasterize_conserve(vector_use, like=like, column_to_burn=list(vector_use.columns.drop('geometry'))[0]).rio.reproject_match(like)
        logger.debug(f'rasterized vector_use to res = {res}')
        #logger.debug('Rasterized USE TOTAL BOUNDS', raster_use.x.min().data, raster_use.x.max().data, raster_use.y.min().data, raster_use.y.max().data) 

    raster_use = raster_use.rio.reproject('epsg:4326', nodata=0).sortby('x').sortby('y').sel(x=slice(study_area[0], study_area[2]), y=slice(study_area[1], study_area[3])).fillna(0) # fillna is needed so i dont get nan values in the resampled sum
    logger.debug('use_raster successfully reprojected')
    logger.debug('Rasterized USE TOTAL BOUNDS', raster_use.x.min().data, raster_use.x.max().data, raster_use.y.min().data, raster_use.y.max().data)         

    # need a dataarray
    raster_use = rasterhd2raster(raster_use, like) # resample the use raster on our grid, with user-defined res and crs
    logger.debug('Resampled USE TOTAL BOUNDS', raster_use.x.min().data, raster_use.x.max().data, raster_use.y.min().data, raster_use.y.max().data) 
    logger.debug('use_raster successfully resampled')
    # if len(use.shape) > 2:
    #     use = use[0]
        
    raster_use = raster_use.where(raster_use>=0,0) # rasterhd2raster sometimes gives small negative values when resampling. I am filling those with 0. 
    
    # reproject should be done at the end to ensure matching coords
    raster_use = raster_use.rio.reproject_match(like)
   
    # convert use from quantity/day into quantity/timestep
    if tstep is not None:
        logger.warning('Make sure use is provided per day.')
        timesteps_per_day = 24*60*60/tstep # tstep must be given in seconds 
        raster_use = raster_use / timesteps_per_day
    
    logger.debug(f'final use has shape {raster_use.shape}')
    return raster_use
    del vector_use, raster_use


def check_particle_file(path_to_file):
    """
    Sanity-check an OpenDrift output netcdf file for forcing variables that are all zero,
    which usually indicates a reader/forcing problem rather than genuine zero forcing.

    Checks the variables 'x_sea_water_velocity', 'y_sea_water_velocity', 'x_wind' and
    'y_wind', logging a debug warning for each one that is all zero across the whole
    dataset, and a debug confirmation otherwise. Does not return anything.

    Parameters
    ----------
    path_to_file : str or Path
        Path to the OpenDrift output netcdf file to check.
    """
    ds = xr.open_dataset(path_to_file)
    vars_to_check = ['x_sea_water_velocity', 'y_sea_water_velocity', 'x_wind', 'y_wind']
    for i in vars_to_check:
        if np.all(ds[i].load() == 0):
            logger.debug(f'ATTENTION: all 0 values detected for variable {i}.')
        else:
            logger.debug(f'all good: variable {i} has non-zero values.')


def traj_distinct(bin_n, weight):
    """
    Zero out weight at timesteps where a trajectory stays in the same spatial bin as the
    previous timestep, keeping weight only at the first timestep of each new bin "visit" -
    so a trajectory dwelling in one bin over several consecutive timesteps contributes
    weight only once per visit, rather than once per timestep.

    Meant to be applied per-trajectory along the time axis, e.g. via
    xr.apply_ufunc(traj_distinct, ds.bin_n, weight, input_core_dims=[['time'],['time']],
    output_core_dims=[['time']], vectorize=True).

    Parameters
    ----------
    bin_n : array-like
        1D array of bin ids for a single trajectory over time, in chronological order (e.g.
        the 'bin_n' variable of a PMAR dataset, for one trajectory).
    weight : array-like
        Weight values for the same trajectory over time, same shape as bin_n.

    Returns
    -------
    array-like
        Copy of weight with entries at timestep t set to 0 whenever bin_n[t] == bin_n[t-1]
        (the first timestep, t=0, is always kept as-is).
    """
    logger.debug(f"bin_n: {bin_n.shape} | weight: {weight.shape}")
    w = deepcopy(weight)
    for t in range(1,len(bin_n)): # i leave the first element as 1, otherwise the loop would compare it to the last element.
        if bin_n[t] == bin_n[t-1]:
            w[t] = 0
    return w
    