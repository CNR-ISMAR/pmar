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
import requests
import xml.etree.ElementTree as ET
import tempfile
import re
from concurrent.futures import ThreadPoolExecutor
from owslib.wfs import WebFeatureService

logger = logging.getLogger('pmar')

EMODNET_HA_WFS_URL = 'https://ows.emodnet-humanactivities.eu/wfs'
EMODNET_HA_WCS_URL = 'https://ows.emodnet-humanactivities.eu/wcs'
EMODNET_HITS_WORKERS = 20  # concurrency for list_emodnet_uses' per-layer feature-count checks


SPATIAL_DOMAINS = {'global':
                {'readers':
                 {'physics': 'cmems_mod_glo_phy_my_0.083deg_P1D-m', # global physics reanalysis, daily, 1/12° horizontal resolution, 50 vertical levels, 1 Jan 1993 to 2021-06-30
                  'winds': 'cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H', # L4 sea surface wind and stress fields 0.125° res, hourly, 1 Jun 1994 to 21 Nov 2024
                  #'bathymetry':'cmems_mod_glo_phy_my_0.083deg_static', # global physics reanalysis static, same as physics
                  'waves': 'cmems_mod_glo_wav_my_0.2deg_PT3H-i', # global ocean waves reanalysis, 3-hourly, 1 Jan 1980 to 31 Jan 2025
              #    'spm': 'cmems_obs-oc_glo_bgc-transp_my_l4-multi-4km_P1M',
                 },
                'extent': [-180, -80, 179.92, 90],
                 'label': 'CMEMS_Global_Renalysis'
                },

                'med':
                {'readers':
                 {'physics': 'cmems_mod_med_phy-cur_my_4.2km_PT1H-m', # change H to D to get daily. Mediterranean Sea Physics Reanalysis, 1/24˚res, 141 z-levels, 1 Jan 1987 to 1 Feb 2025
                  'winds': 'cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H', # global L4
                  'bathymetry': 'cmems_mod_med_phy_my_4.2km_static', # Mediterranean Sea Physics Reanalysis
                  'waves': 'cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H',   # Mediterranean Sea Waves Reanalysis, hourly, 1/24˚res, 1 Jan 1985 to 1 Mar 2025
                 },
                 'extent': [-6, 30.19, 36.29, 45.98],
                'label': 'CMEMS_Mediterranean_Renalysis'
                },

                'black sea':
                {'readers':
                 {'physics': 'cmems_mod_blk_phy-cur_my_2.5km_P1D-m', #Black Sea Physics Reanalysis, res 1/40º, 121 z-levels, 1 Jan 1993 to 1 Feb 2025
                  'winds': 'cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H', # global L4,
                  'bathymetry': 'cmems_mod_blk_phy_my_2.5km_static',  # Black Sea Physics Reanalysis
                  'waves': 'cmems_mod_blk_wav_my_2.5km_PT1H-i',  # Black Sea Waves Reanalysis
                 },
                'extent': [27.25, 40.5, 42, 47],
                'label': 'CMEMS_Black_Sea_Renalysis'
                },

                'baltic':
                {'readers':
                 {'physics': 'cmems_mod_bal_phy_my_P1D-m', #Baltic Sea Physics Reanalysis, 2x2km, 56 depth-levels, 1 Jan 1993 to 31 Dec 2023
                  'winds': 'cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H', # global L4,
                  'bathymetry': 'cmems_mod_bal_phy_my_static', #Baltic Sea Physics Reanalysis,
                  'waves': 'cmems_mod_bal_wav_my_PT1H-i',
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


def _wfs_has_features(name, bbox, wfs_url):
    """Check (via a cheap RESULTTYPE=hits request) whether a WFS layer has any features within bbox."""
    bbox_str = f'{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},EPSG:4326'
    try:
        resp = requests.get(wfs_url, params={
            'SERVICE': 'WFS', 'REQUEST': 'GetFeature', 'VERSION': '2.0.0',
            'TYPENAMES': name, 'BBOX': bbox_str, 'RESULTTYPE': 'hits',
        }, timeout=30)
        resp.raise_for_status()
        match = re.search(r'numberMatched="(\d+)"', resp.text)
        if match is None:
            logger.warning(f"could not parse feature count for '{name}'; keeping it in the list.")
            return True
        return int(match.group(1)) > 0
    except requests.RequestException as e:
        logger.warning(f"feature-count check failed for '{name}' ({e}); keeping it in the list.")
        return True


def list_emodnet_uses(bbox, wfs_url=EMODNET_HA_WFS_URL, wcs_url=EMODNET_HA_WCS_URL):
    """
    List EMODnet Human Activities datasets ("uses") available within a bounding box.

    Queries the EMODnet Human Activities WFS (vector datasets, e.g. aggregate extraction,
    cables, aquaculture, wind farms) and WCS (raster datasets, currently vessel density and
    route density, at various ship types/time periods) GetCapabilities documents, and lists
    every layer/coverage whose declared extent intersects bbox. Prints the result as a
    table (name, title, service, output_type) and also returns it, so a name can be passed
    directly to get_emodnet_use().

    WFS layers are further verified with a live feature-count check (RESULTTYPE=hits,
    run concurrently across candidates): some WFS layers -- mostly non-geographic
    "Advisory Council" governance layers (e.g. 'emodnet:blacksea', 'emodnet:baltic') --
    declare an inaccurate, world-spanning bounding box in GetCapabilities rather than
    their real features' extent, which would otherwise make them appear for any bbox.
    This adds a few seconds to the call but keeps the listing accurate. WCS coverages are
    not verified this way (their declared bbox is computed by the server directly from
    the real raster extent, so this particular issue doesn't apply, and checking actual
    coverage content would require fetching the raster data itself).

    Parameters
    ----------
    bbox : list or array-like
        Bounding box to search within, as [xmin, ymin, xmax, ymax] in EPSG:4326.
    wfs_url : str, optional
        EMODnet Human Activities WFS endpoint. Default is EMODNET_HA_WFS_URL.
    wcs_url : str, optional
        EMODnet Human Activities WCS endpoint. Default is EMODNET_HA_WCS_URL.

    Returns
    -------
    pandas.DataFrame
        Columns: 'name' (pass this to get_emodnet_use()), 'title', 'service' ('WFS' or 'WCS'),
        'output_type' ('vector' or 'raster', i.e. what get_emodnet_use() will return).
    """
    bbox_poly = box(*bbox)
    rows = []

    wfs = WebFeatureService(wfs_url, version='2.0.0')
    wfs_candidates = []
    for name, layer in wfs.contents.items():
        if layer.boundingBoxWGS84 is None:
            continue
        if box(*layer.boundingBoxWGS84).intersects(bbox_poly):
            wfs_candidates.append({'name': name, 'title': layer.title, 'service': 'WFS', 'output_type': 'vector'})

    with ThreadPoolExecutor(max_workers=EMODNET_HITS_WORKERS) as executor:
        has_features = list(executor.map(lambda c: _wfs_has_features(c['name'], bbox, wfs_url), wfs_candidates))
    rows.extend(c for c, keep in zip(wfs_candidates, has_features) if keep)

    # owslib's WCS 2.0.1 CoverageSummary parsing doesn't populate title/bbox for this
    # server, so GetCapabilities is parsed directly here instead.
    resp = requests.get(wcs_url, params={'SERVICE': 'WCS', 'REQUEST': 'GetCapabilities', 'VERSION': '2.0.1'}, timeout=60)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    ns = {'wcs': 'http://www.opengis.net/wcs/2.0', 'ows': 'http://www.opengis.net/ows/2.0'}
    for summary in root.findall('.//wcs:CoverageSummary', ns):
        cid = summary.find('wcs:CoverageId', ns)
        bbox_el = summary.find('ows:WGS84BoundingBox', ns)
        if cid is None or bbox_el is None:
            continue
        lc = [float(v) for v in bbox_el.find('ows:LowerCorner', ns).text.split()]
        uc = [float(v) for v in bbox_el.find('ows:UpperCorner', ns).text.split()]
        if box(lc[0], lc[1], uc[0], uc[1]).intersects(bbox_poly):
            title = summary.find('ows:Title', ns)
            rows.append({'name': cid.text, 'title': title.text if title is not None else cid.text, 'service': 'WCS', 'output_type': 'raster'})

    uses = pd.DataFrame(rows, columns=['name', 'title', 'service', 'output_type']).sort_values(['service', 'title']).reset_index(drop=True)
    print(uses.to_string(index=False))
    return uses


def get_emodnet_use(use_name, bbox, time=None, wfs_url=EMODNET_HA_WFS_URL, wcs_url=EMODNET_HA_WCS_URL):
    """
    Fetch an EMODnet Human Activities dataset ("use"), clipped to a bounding box.

    use_name is matched against WFS layer names (e.g. 'emodnet:aggregates') or WCS
    coverage ids (e.g. 'emodnet__vesseldensity_00avg') by their naming convention (':' for
    WFS, '__' for WCS) -- see list_emodnet_uses(bbox) for exact names to use.

    Parameters
    ----------
    use_name : str
        Exact WFS layer name or WCS coverage id, as returned by list_emodnet_uses().
    bbox : list or array-like
        Bounding box to clip the dataset to, as [xmin, ymin, xmax, ymax] in EPSG:4326.
    time : str, optional
        For time-dimensioned WCS coverages (e.g. vessel/route density), an ISO timestamp
        (as listed in the coverage's time positions) to request. Default is None, i.e. the
        coverage's own default time position (usually the most recent). Ignored for WFS
        layers.
    wfs_url : str, optional
        EMODnet Human Activities WFS endpoint. Default is EMODNET_HA_WFS_URL.
    wcs_url : str, optional
        EMODnet Human Activities WCS endpoint. Default is EMODNET_HA_WCS_URL.

    Returns
    -------
    geopandas.GeoDataFrame or xarray.DataArray
        For a WFS layer, a GeoDataFrame in EPSG:4326 with geometries clipped to bbox (not
        just features selected by intersecting it -- large-extent features, e.g. fishing
        quota zones, are cut down to bbox rather than returned whole), keeping all of the
        layer's original columns. get_ppi()/harmonize_use() expect a 'use' GeoDataFrame
        with exactly one non-geometry column, so select/derive one before passing this on,
        e.g. gdf[['permitted_amount_t', 'geometry']]. For a WCS coverage, a 2D
        xarray.DataArray in EPSG:4326 (fill values masked to NaN), ready to pass directly
        as 'use'.

    Raises
    ------
    ValueError
        If use_name's service can't be determined from its naming convention, or if the
        EMODnet server returns an error (e.g. an unknown name/id, or an invalid time).
    """
    if '__' in use_name:
        service = 'wcs'
    elif ':' in use_name:
        service = 'wfs'
    else:
        raise ValueError(
            f"Could not tell whether '{use_name}' is a WFS layer (name like 'emodnet:xxx') "
            "or a WCS coverage (id like 'emodnet__xxx'). Use the exact name/id returned by "
            "list_emodnet_uses()."
        )

    if service == 'wfs':
        bbox_str = f'{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},EPSG:4326'
        url = (f'{wfs_url}?SERVICE=WFS&REQUEST=GetFeature&VERSION=2.0.0'
               f'&TYPENAMES={use_name}&BBOX={bbox_str}&SRSNAME=EPSG:4326&OUTPUTFORMAT=json')
        gdf = gpd.read_file(url)
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs('EPSG:4326')

        if gdf.empty:
            logger.warning(f"No features found for '{use_name}' within bbox {bbox}.")
            return gdf

        # BBOX only selects features that *intersect* bbox -- it returns each feature's
        # whole geometry, which for large-extent features (e.g. fishing/quota management
        # zones) can extend far outside bbox. Clip to actually cut geometries down to it.
        gdf = gdf.clip(box(*bbox)).reset_index(drop=True)

        value_cols = [c for c in gdf.columns if c != 'geometry']
        logger.info(
            f"fetched {len(gdf)} features for '{use_name}'. Non-geometry columns: "
            f"{value_cols}. Select/derive a single value column before passing this as "
            "use= to get_ppi()/harmonize_use()."
        )
        return gdf

    # service == 'wcs'
    describe_resp = requests.get(wcs_url, params={
        'SERVICE': 'WCS', 'REQUEST': 'DescribeCoverage', 'VERSION': '2.0.1', 'COVERAGEID': use_name,
    }, timeout=60)
    describe_resp.raise_for_status()
    describe_root = ET.fromstring(describe_resp.content)
    time_domain = describe_root.find('.//{http://www.geoserver.org/wcsgs/2.0}TimeDomain')
    subset_time = None
    if time_domain is not None:
        subset_time = time if time is not None else time_domain.get('default')

    params = [
        ('SERVICE', 'WCS'), ('REQUEST', 'GetCoverage'), ('VERSION', '2.0.1'),
        ('COVERAGEID', use_name), ('FORMAT', 'image/tiff'),
        ('SUBSETTINGCRS', 'http://www.opengis.net/def/crs/EPSG/0/4326'),
        ('OUTPUTCRS', 'http://www.opengis.net/def/crs/EPSG/0/4326'),
        ('SUBSET', f'Long({bbox[0]},{bbox[2]})'),
        ('SUBSET', f'Lat({bbox[1]},{bbox[3]})'),
    ]
    if subset_time is not None:
        params.append(('SUBSET', f'time("{subset_time}")'))

    resp = requests.get(wcs_url, params=params, timeout=120)
    resp.raise_for_status()
    if resp.content[:5] == b'<?xml':
        root = ET.fromstring(resp.content)
        msg = ' '.join(root.itertext()).strip()
        raise ValueError(f"EMODnet WCS request for '{use_name}' failed: {msg}")

    with tempfile.NamedTemporaryFile(suffix='.tif') as tmp:
        tmp.write(resp.content)
        tmp.flush()
        da = rxr.open_rasterio(tmp.name, masked=True).squeeze()
        da.load()  # force read into memory before the temp file is deleted

    if da.rio.crs is not None and da.rio.crs.to_epsg() != 4326:
        da = da.rio.reproject('EPSG:4326')

    return da


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
    