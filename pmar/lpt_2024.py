import logging
import rioxarray as rxr
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta, date
from shapely.geometry import Point, Polygon
from opendrift.models.oceandrift import OceanDrift
from opendrift.models.openoil import OpenOil
import copernicusmarine
from opendrift.readers.reader_netCDF_CF_generic import Reader
import opendrift
from pydap.client import open_url
from pydap.cas.get_cookies import setup_session
import time as T
from pathlib import Path
from xhistogram.xarray import histogram
import seaborn as sns
spectral_r = sns.color_palette("Spectral_r", as_cmap=True)
import cartopy.crs as ccrs
import cartopy
from matplotlib.ticker import LogFormatter, PercentFormatter
from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib.colors as col
import os
from opendrift.readers import reader_shape
import cartopy.io.shapereader as shpreader
import netrc
import random
import rasterio
import tempfile
from cachetools import LRUCache
from netCDF4 import Dataset
import hashlib
import json
from flox.xarray import xarray_reduce # for xarray grouping over multiple variables
os.environ['PROJ_LIB'] = '/var/miniconda3/envs/opendrift/share/proj/'
from functools import partial
from rasterio.enums import Resampling
import glob

#from dask.distributed import Client

logger = logging.getLogger("PMAR")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

#client = Client(n_workers=4, threads_per_worker=2)

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
              .rio.write_nodata(np.nan)
             )                                                                                                                                                                          
    return raster 

class PMARCache(object):
    def __init__(self, cachedir): # cachedir è una sottodirectory di basedir
        self.cachedir = Path(cachedir)
        self.cachedir.mkdir(exist_ok=True)
        
    def get_data_file(self, extension, **kwargs):
        _data_file = hashlib.md5(str(sorted(kwargs.items())).encode('utf-8')).hexdigest()
        data_file = f"{_data_file}.{str(extension)}" # chiave della cache e nome del file, generalizzata sia per calculate_trajectories che particle_raster
        path_data_file = Path(self.cachedir) / data_file 
        return path_data_file
    
    def set_metadata(self, extension, **kwargs):
        path_data_file = self.get_data_file(extension, **kwargs)
        path_metadata_file = str(path_data_file) + '_metadata' #TODO rendere più robusto
        with open(path_metadata_file,'w') as fi:
            json.dump(kwargs,fi,default=str)
            
    def particle_cache(self, poly_path, pnum, start_time, season, duration_days, s_bounds, z, tstep, hdiff, termvel, crs):
        cache_key = {'poly_path': poly_path, 'pnum': pnum, 'start_time': start_time.strftime("%Y-%m-%d"), 'season': season, 'duration_days': duration_days, 's_bounds': s_bounds, 'z': z, 'tstep': tstep, 'hdiff': hdiff, 'termvel': termvel, 'crs': crs}
        path_data_file = self.get_data_file('nc', **cache_key) # chiave della cache e nome del file
        self.set_metadata('nc', **cache_key) #TODO spostare
        logger.error('particle cache = '+str(cache_key))
        return path_data_file, path_data_file.exists()
        
    def old_raster_cache(self, res, poly_path, pnum, ptot, duration_days, start_time, reps, tshift, s_bounds, z, tstep, hdiff, termvel, crs, tinterp, r_bounds, use_path, decay_coef, aggregate, depth_layer, z_bounds, particle_status, traj_dens):
        cache_key = {'res': res, 'poly_path': poly_path, 'pnum': pnum, 'ptot': ptot, 'duration_days': duration_days, 'start_time': start_time, 'reps': reps, 'tshift': tshift, 's_bounds': s_bounds, 'z': z, 'tstep': tstep, 'hdiff': hdiff, 'termvel': termvel, 'crs': crs, 'tinterp': tinterp, 'r_bounds': r_bounds, 'use_path': use_path, 'decay_coef': decay_coef, 'aggregate':aggregate, 'depth_layer': depth_layer, 'z_bounds': z_bounds, 'particle_status': particle_status, 'traj_dens' : traj_dens}
        path_data_file = self.get_data_file('tif', **cache_key) # chiave della cache e nome del file
        self.set_metadata('tif', **cache_key) #TODO spostare
        logger.error('raster cache = '+str(cache_key))
        return path_data_file, path_data_file.exists()

    def raster_cache(self, res, poly_path, pnum, ptot, duration, start_time, reps, tshift, use_path, use_label, decay_coef, r_bounds):
        cache_key = {'res': res, 'poly_path': poly_path, 'pnum': pnum, 'ptot': ptot, 'duration': duration, 'start_time': start_time, 'reps': reps, 'tshift': tshift, 'use_path': use_path, 'use_label': use_label, 'decay_coef': decay_coef, 'r_bounds': r_bounds}
        path_data_file = self.get_data_file('tif', **cache_key) # chiave della cache e nome del file
        #path_data_file = Path(str(_path_data_file).split('.tif')[0]+'_use-RES-TIME.tif')
        self.set_metadata('tif', **cache_key) #TODO spostare
        logger.error('raster cache = '+str(cache_key))
        return path_data_file, path_data_file.exists()

    def c_cache(self, res, poly_path, pnum, ptot, duration, start_time, reps, tshift, use_path, use_label, decay_coef, r_bounds):
        cache_key = {'res': res, 'poly_path': poly_path, 'pnum': pnum, 'ptot': ptot, 'duration': duration, 'start_time': start_time, 'reps': reps, 'tshift': tshift, 'use_path': use_path, 'use_label': use_label, 'decay_coef': decay_coef, 'r_bounds': r_bounds}
        _path_data_file = self.get_data_file('tif', **cache_key) # chiave della cache e nome del file
        path_data_file = Path(str(_path_data_file).split('.tif')[0]+'_use-'+use_label+'.tif')
        self.set_metadata('tif', **cache_key) #TODO spostare
        logger.error('raster cache = '+str(cache_key))
        return path_data_file, path_data_file.exists()
        
        
class PMAR(object): # rename this? it's long and confusing
    """
    Developed at CNR ISMAR in Venice.

    ...

    Attributes
    ----------
    depth : bool
        boolean stating whether particle simulation is 2D (depth==False) or 3D (depth==True)
    o : OceanDrift
        OpenDrift object producing particle simulation
    poly_path : str
        path to shapefile containing polygon used for particle seeding
    bathy_path : str
        path to netcdf file for bathymetry (GEBCO)
    particle_path : str
        path to netcdf file containing output of OpenDrift simulation
    raster : xarray
        xarray object containing output of particle_raster() method

    Methods
    -------
    run()
        runs OpenDrift simulation and produces desired raster
    make_poly()
        if no polygon is given for seeding, creates and writes polygon based on given lon lat
    calculate_trajectories()
        runs OpenDrift simulation
    particle_raster()
        computes 2D histogram of particle concentration and writes to tif (if requested)
    plot()
        plots output of particle_raster()
    scatter()
        plots trajectories over time 
    animate()
        WIP

    """


    def __init__(self, context, pressure='general', basedir='lpt_output', localdatadir = None, poly_path = None, uv_path='cmems', wind_path='cmems', mld_path='cmems', bathy_path=None, particle_path=None, depth=False, netrc_path=None):
        """
        Parameters
        ---------- 
        context : str
            String defining the context of the simulation i.e., the ocean model output to be used for particle forcing. Options are 'med-cmems', 'bs-cmems' and 'bridge-bs'. 
        pressure : str, optional
            
        basedir : str, optional
            path to the base directory where all output will be stored. Default is to create a directory called 'lpt output' in the current directory.
        localdatadir : str, optional
            path to directory where input data (ocean, atmospheric) is stored. Default is None
        poly_path : str, optional
            path to shapefile containing polygon to be used for seeding of particles.
        uv_path : str, optional
            path to the netcdf file containing ocean current data. Default is 'cmems', meaning CMEMS data will be streamed from Copernicus.
        wind_path : str, optional
            path to the netcdf file containing wind velocity data. Default is 'cmems', meaning CMEMS data will be streamed from Copernicus.        
        mld_path : str, optional
            path to the netcdf file containing mixed layer depth data. Default is 'cmems', meaning CMEMS data will be streamed from Copernicus.  
        bathy_path : str
            Path to bathymetry file to be used in 3D simulations. Default is None. An error is raised if depth is True and bathy_path is None. 
        particle_path : str, optional
            path to netcdf file containing output of OpenDrift simulation. Default is None. If a particle_path is given in initialisation,    
        depth : bool, optional
            boolean stating whether particle simulation is 2D (depth==False) or 3D (depth==True). Default is False
        """

        Path(basedir).mkdir(parents=True, exist_ok=True)
        self.uv_path = None
        self.wind_path = None
        self.mld_path = None 
        self.bathy_path = bathy_path 
        self.basedir = Path(basedir)
        self.particle_path = particle_path # i can import an existing particle_path
        #self.ds = None. commented otherwise try: self.ds except: does not work
        self.o = None
        self.poly_path = poly_path 
        self.raster = None
        self.origin_marker = 0
        self.netrc_path = netrc_path
        self.tstep = None
        self.pnum = None
        self.depth = depth
        self.termvel = 1e-3
        self.decay_coef = 0
        self.context = context
        self.outputdir = None
        self.pressure = pressure
        self.localdatadir = localdatadir
        self.particle_status = None
        self.reps = 1
        self.tshift = None
        self.cache = PMARCache(Path(basedir) / 'cachedir')
        self.raster_path = None
        #self._polygon_grid = None
        self._x_e = None
        self._y_e = None
        self._x_c = None
        self._y_c = None
        self.res = None
        self.weight = None
        self.r_bounds = None
        self.res = None

        # This should be a separate method
        pres_list = ['general', 'microplastic', 'bacteria']
        pressures = pd.DataFrame(columns=['pressure', 'termvel', 'decay_coef'], 
                    data = {'pressure': pres_list, 
                            'termvel': [0, 1e-3, 0], 
                            'decay_coef': [0, 0, 1]})
        
        
        if pressure in pres_list:
            self.termvel = pressures[pressures['pressure'] == f'{pressure}']['termvel'].values[0]
            self.decay_coef = pressures[pressures['pressure'] == f'{pressure}']['decay_coef'].values[0]
        
        elif pressure == 'oil':
            pass
        
        else:
            pass

        # this should be a separate method
        # if a path to a shapefile is given to be used for seeding, read it and save it in the lpt_output/polygons dir
        if poly_path is not None: 
            Path(self.basedir / 'polygons').mkdir(parents=True, exist_ok=True)            
            poly = gpd.read_file(poly_path).to_crs('epsg:4326')
            bds = np.round(poly.total_bounds, 4) 
            local_poly = f'polygon-crs_epsg:{poly.crs}-lon_{bds[0]}_{bds[2]}-lat—{bds[1]}_{bds[3]}.shp'
            q = self.basedir / 'polygons' / local_poly
            poly.to_file(str(q), driver='ESRI Shapefile')
            self.poly_path = str(q)
        else:
            if 'med' in self.context:
                self.poly_path = f'{DATA_DIR}/polygon-med-full-basin.shp'
            elif 'bs' in self.context:
                self.poly_path = f'{DATA_DIR}/polygon-bs-full-basin.shp'
            else:
                pass
            

        # this (if still needed?) should be a separate method
        # if particle_path is given, retrieve number of reps and load ds
        if self.particle_path is not None: 

            # gather number of reps from origin marker. THIS IS A PROBLEM IF THERE ARE MORE BATCHES IN SAME FOLDER. 
            if type(self.particle_path) is not str: 
                _reps = np.zeros(len(self.particle_path))
                for idx, f in enumerate(self.particle_path):
                    _reps[idx] = xr.open_dataset(f).origin_marker.maxval

                self.reps = len(np.unique(_reps))
                #self.reps = int(np.max(_reps))+1
                #if len(_reps) != len(np.unique(_reps)):
                #    logger.warning(f'Repetition of origin_marker value detected. Please specify number of reps.')

            
            #self.get_ds # this should not be automatic, sometimes it is too heavy and useless  
            
            # if a particle_path is given, meaning a run with those configs already exists, the poly_path contained in the file's attributes "wins" over poly_path
            #if self.ds.poly_path is not None:
             #   self.poly_path = str(self.ds.poly_path)            
    
    def get_userinfo(self, machine):
        try:
            secrets = netrc.netrc(self.netrc_path)
        except FileNotFoundError:
            ''
        auth = secrets.authenticators(machine)
        if auth is None:
            return ''
        return f'{auth[0]}:{auth[2]}@'

    
    def x_grid(self):
        if self._x_e is None:
            raise Error('polygon_grid needs to be called before using this method')
        return self._x_e

    def y_grid(self):
        if self._y_e is None:
            raise Error('polygon_grid needs to be called before using this method')
        return self._y_e
        
    def polygon_grid(self, res, r_bounds=None):
        '''
        Make grid from polygon of domain area.
        '''
        #try:
         #   self._polygon_grid
         #   print('polygon_grid: _polygon_grid was previously calculated.')
            
        if res != self.res: 
            print(f'polygon_grid: calculating new polygon_grid with resolution = {res}.')
            #res = self.res 
            crs = 'epsg:4326' # need to use EPSG:4326 because the output of opendrift is in this epsg and i want to use this grid for the histogram of the opendrift output
            
            if r_bounds is not None: # if r_bounds are given, meaning we are calculating the raster on a different region than seeding, create new polygon to use for aggregation / visualisation
                poly = self.make_poly(lon=[r_bounds[0], r_bounds[2]], lat=[r_bounds[1], r_bounds[3]], write=False).to_crs('epsg:4326').buffer(distance=res*3)
                print('polygon_grid: use r_bounds to make new polygon_grid')
            else:
                poly = gpd.read_file(self.poly_path).to_crs(crs).buffer(distance=res*3) # the buffer is added because of the non-zero radius when seeding, otherwise some particles might be left out
            
            xmin, ymin, xmax, ymax = poly.total_bounds
            
            cols = list(np.arange(xmin, xmax + res, res))               
            rows = list(np.arange(ymin, ymax + res, res)) 
            
            polygons = [] 
            #print('polygon_grid: starting for loop...')
            for y in rows[:-1]:                
                for x in cols[:-1]:    
                    polygons.append(Polygon([(x,y),
                                             (x+res, y),
                                             (x+res, y+res),
                                             (x, y+res)]))
            #print('polygon_grid: for loop done!')
            
            grid = gpd.GeoDataFrame({'geometry':polygons}, crs=crs)  
    
            #print('intersecting grid with poly')
            intersect = grid[grid.intersects(poly.geometry[0])].reset_index()
            self._polygon_grid = intersect
            
            self._x_e = np.array(cols) # outer edge coordinates
            self._y_e = np.array(rows)
            self._x_c = np.unique(self._polygon_grid.centroid.x.values.round(4)) # centroid coordinates
            self._y_c = np.unique(self._polygon_grid.centroid.y.values.round(4))
        else:
            self._polygon_grid
            print(f'polygon_grid: _polygon_grid was previously calculated with resolution = {self.res}.')
            
        #print('polygon_grid: done.')
        self.res = res
        print(f'updated self.res = {self.res}')
        return self._polygon_grid

    @property
    def get_ds(self):
        try:
            self.ds
            print('get_ds: returning previously calculated ds.')
        except:
            print('get_ds: retrieving ds from particle_path.')
            if type(self.particle_path) is str or len(self.particle_path) == 1:
                ds = xr.open_dataset(self.particle_path, chunks={'trajectory': 10000, 'time':1000})
            
            else:
                partial_func = partial(self._preprocess, correct_len = self.find_correct_len())
                ds = xr.open_mfdataset(self.particle_path, preprocess=partial_func, concat_dim='trajectory', combine='nested', join='override', parallel=True, chunks={'trajectory': 10000, 'time':1000})
                logger.error(f'lat = {ds.lat.shape}, lon={ds.lon.shape}, time={ds.time.shape}')
                # if the run contained reps, ensure trajectories have unique IDs for convenience
                logger.error(f'self.reps = {self.reps}')
                ds['trajectory'] = np.arange(0, len(ds.trajectory)) 
            
            self.ds = ds
            print('get_ds: done.')

        return self.ds

    # Given multiple particle_paths, it returns the maximum time length. 
    # Used for preprocessing function in get_ds to make sure all ds's have same time length, even if opendrift ended early because of all particles beaching.
    def find_correct_len(self):
        '''
        Sometimes opendrift runs end early because e.g. all particles have beached, resulting in reps with different time lengths. This method finds the maximum time lenght of all reps.
        '''
        #print([filename for filename in self.particle_path])
        lens = np.array([len(xr.open_dataset(filename).time) for filename in self.particle_path])
        print(f'giving all datasets the same time length of {lens} tsteps...')
        return lens.max()

    def _preprocess(self, ds, correct_len):
        '''
        Sometimes opendrift runs end early because e.g. all particles have beached, resulting in reps with different time lengths. This method pads all reps so that they all have same time length (determined with find_correct_len).
        '''
        return ds.pad(pad_width={'time': (0, correct_len-len(ds.time))}, mode='edge')
         

    # doublecheck this. very old, maybe could be improved. 
    def make_poly(self, lon, lat, crs='4326', write=True):
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
        
        
        Path(self.basedir / 'polygons').mkdir(parents=True, exist_ok=True)
        poly_path = f'polygon-crs_epsg:{crs}-lon_{np.round(lon[0],4)}_{np.round(lon[1],4)}-lat—{np.round(lat[0],4)}_{np.round(lat[1])}.shp'
        q = self.basedir / 'polygons' / poly_path
        
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
            raise ValueError("'lon' and 'lat' must have lenght larger than 2")
        
        if write is True:
            poly.to_file(str(q), driver='ESRI Shapefile')
            self.poly_path = str(q)
        else:
            return poly
    
# [pnum, start_time, season, duration_days, s_bounds, z, tstep, hdiff, termvel, crs]
    def calculate_trajectories(self, pnum, start_time='2019-01-01', season=None, duration_days=30, s_bounds=None, z=-0.5, tstep=timedelta(hours=4), hdiff=10, termvel=None, crs='4326', seeding_radius=2e3, loglevel=40):
        """
        Calculate trajectories using Oceandrift module by OpenDrift (MET Norway).
        Uses OceanDrift module. 
                
        This method is currently optimised for use over the Mediterranean and Black Sea only. 
        Forcing data is streamed from Copernicus (ocean currents, wind, mixed layer depth), while GEBCO bathymetry is stored locally. 
        
        Particles are seeded homogeneously over polygon created using make_poly() method. 
        
        
        Parameters
        ----------
        context : str
            String defining the context of the simulation i.e., the model data to be used for particle forcing. Options are 'med-cmems', 'bs-cmems' and 'bridge-bs'. 
        pnum : int 
            The number of particles to be seeded
        start_time : str, optional
            The start time of the simulation. Default is '2019-01-01'
        season : str, optional
            Season in which simulation should be run. Defines start_time automatically. Default is None
        duration_days : int, optional
            Integer defining the length, or duration, of the particle simulation in days. Default is 30 (to be used for test run)
        s_bounds : list, optional
            Spatial bounds for seeding written as bounds=[x1,y1,x2,y2]. Default is None, i.e. the whole basin is covered
        z : float, optional
            Depth at which to seed particles in [m]. Default is -0.5m. 
        tstep : timedelta, optional
            Time step used for OpenDrift simulation. Default is 6 hours
        hdiff : float, optional
            Horizontal diffusivity of particles, in [m2/s]. Default is 10m2/s
        termvel : float, optional
            Terminal velocity representing buoyancy of particles, in [m/s]. Default is None, meaning termvel is defined in __init__      
        crs : str, optional
            EPSG string for polygon. Default is 4326   
        loglevel : int, optional
            OpenDrift loglevel. Set to 0 (default) to retrieve all debug information.
            Provide a higher value (e.g. 20) to receive less output.
        """
        
        t_0 = T.time()   
        
        context = self.context 
 
        if termvel is None:
            termvel = self.termvel
        

        # context only makes sense for tools4msp implementation. 
        # raise error if unsupported context is requested
        avail_contexts = ['bridge-bs', 'med-cmems', 'bs-cmems']
        if context not in avail_contexts:
            raise ValueError(f"Unsupported context given. Context variable must be one of {avail_contexts}")
        self.context = context
        
        ### time settings ####
        ssns = {'summer': datetime(2019,6,1),
               'winter': datetime(2019,12,1),
               'spring': datetime(2019,3,1),
               'autumn': datetime(2019,9,1)}

        if season in ssns.keys():
            start_time = ssns[season]  
        else:
            start_time = datetime.strptime(start_time, '%Y-%m-%d')
        end_time = start_time + timedelta(days=duration_days) # this is printed in particle_path
        time_step = tstep

        #### REMOVING TSEED #### APRIL 2024
        #if self.tseed is None:
         #   self.tseed = timedelta(days=(duration_days * 20 / 100)) #tseed is 20% of total duration
            
        #tseed = self.tseed
        # tseed gets added to the total duration, then removed and everything is realigned. 
        #duration = timedelta(days=duration_days)+tseed-timedelta(days=1) # true duration of the run. the tseed time period is then deleted.

        duration = timedelta(days=duration_days)
        
        self.tstep = tstep
        
        # polygon used for seeding of particles (poly_path). if s_bounds are given, a new polygon is created using those bounds. 
        if s_bounds is not None:
            lon = s_bounds[0::2]
            lat = s_bounds[1::2]
            self.make_poly(lon, lat, crs=crs)
        
        # path to write particle simulation file. also used for our 'cache'    
        poly = gpd.read_file(self.poly_path)
        bds = np.round(poly.total_bounds, 4) # only used in particle_path
        t0 = start_time.strftime('%Y-%m-%d')
        t1 = end_time.strftime('%Y-%m-%d')
        
        # initialise OpenDrift object
        if self.pressure == 'oil':
            self.o = OpenOil(loglevel=loglevel)
        else:
            self.o = OceanDrift(loglevel=loglevel) 
        
        
        file_path, file_exists = self.cache.particle_cache(poly_path=self.poly_path, pnum=pnum, start_time=start_time, season=season, duration_days=duration_days, s_bounds=s_bounds, z=z, tstep=tstep, hdiff=hdiff, termvel=termvel, crs=crs)
        
        # if a file with that name already exists, simply import it  
        if file_exists == True:
            #self.o.io_import_file(str(q)) # this is sometimes too heavy
            ps = xr.open_mfdataset(file_path)
            print(f'NOTE: Opendrift file with these configurations already exists within {self.basedir} and has been imported. Please delete the existing file to produce a new simulation.') ### this might be irrelevant now with cachedir

        # otherwise, run requested simulation
        elif file_exists == False:
            print('adding landmask...')
            # landmask from cartopy (from "use shapefile as landmask" example on OpenDrift documentation)
            shpfilename = shpreader.natural_earth(resolution='10m',
                                    category='cultural',
                                    name='admin_0_countries')
            reader_landmask = reader_shape.Reader.from_shpfiles(shpfilename)
            
            self.o.add_reader([reader_landmask])
            self.o.set_config('general:use_auto_landmask', False)  # Disabling the automatic GSHHG landmask
            
            # bridge-bs context is VERY OLD, NEEDS FIXING. import relevant readers based on context
            if context == 'bridge-bs': # local WP2 data
                
                if self.localdatadir is None:
                    raise ValueError('bridge-bs data not found. Please provide absolute path to directory containing oceanographic / atmospheric data from bridge-bs, i.e. localdatadir')
                else:    
                    bridge_dir = Path(self.localdatadir) ## ??????
                
                dates = pd.date_range(start_time.strftime("%Y-%m-%d"),(start_time+duration).strftime("%Y-%m-%d"),freq='d').strftime("%Y-%m-%d").tolist()


                # separate method add_readers?
                uvpaths={}
                mldpaths={}
                windpaths = {}
                
                #for idx in range(start_time.year, end_time.year+1):
                for idx, d in enumerate(dates):
                    date = d.replace('-', '')
                    uvpaths[idx] = str(bridge_dir / f'BS*{date}*UV.nc')
                    mldpaths[idx] = str(bridge_dir / f'BS*{date}*MLD.nc') 
                
                uv_path = list(uvpaths.values())
                mld_path = list(mldpaths.values())
                
                for i in range(start_time.year, end_time.year+1):
                    windpaths[i] = str(bridge_dir / f'era5_y{i}.nc')
                
                wind_path = list(windpaths.values())
                bathy_path = str(bridge_dir / 'bs_bathymetry.nc')
                
            elif context == 'bs-cmems': # copernicus Black Sea data (remote connection)
                ocean_id = "cmems_mod_blk_phy-cur_my_2.5km_P1D-m"
                wind_id = "cmems_obs-wind_glo_phy_my_l4_P1M" # this is a global reanalysis product, so same in all contexts
                mld_id = 'cmems_mod_blk_phy-mld_my_2.5km_P1D-m'
                bathy_id = 'cmems_mod_glo_phy_my_0.083deg_static' # this is a global reanalysis product, so same in all contexts
                stokes_id = 'cmems_mod_blk_wav_my_2.5km_PT1H-i'

            # maybe useful to print paths of readers i am using, so one knows which product is being used when they simply use a context. 
            elif context == 'med-cmems': # copernicus Med Sea data (remote connection)
                ocean_id = "med-cmcc-cur-rean-h"
                #wind_id = "cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H"
                wind_id = "cmems_obs-wind_glo_phy_my_l4_P1M" # this is a global reanalysis product, so same in all contexts
                mld_id = 'med-cmcc-mld-rean-d'
                bathy_id = 'cmems_mod_glo_phy_my_0.083deg_static' # this is a global reanalysis product, so same in all contexts

            
            else:
                raise ValueError("Unsupported context. Please choose one of 'bridge-bs', 'bs-cmems' or 'med-cmems'.")

            reader_ids = [ocean_id, wind_id]#, stokes_id] # list of copernicus product ids, including bathymetry and mld if simulation is 3D
            if self.depth == True:
                reader_ids.append(mld_id)
                reader_ids.append(bathy_id)
      

            readers = [] # list of opendrift readers created from those copernicus products
            end_datetime = datetime.strptime((end_time + timedelta(days=1)).date().strftime('%Y-%m-%d'), '%Y-%m-%d') # rounding up to the next day, so it works even when duration is given as a float (such as in SUA)
            for path in reader_ids: 
                data = copernicusmarine.open_dataset(dataset_id = path,
                                                   start_datetime = start_time, 
                                                   end_datetime = end_datetime) 
                r = Reader(data)
                readers.append(r)

            print('adding readers...')
            self.o.add_reader(readers)

            ### what is this????
            if context == 'bridge-bs':
                for k,r in self.o.readers.items(): 
                    r.always_valid = True                  
            

            # separate method with opendrift where i set configs and do seeding and run
            # some OpenDrift configurations
            self.o.set_config('general:coastline_action', 'previous') # behaviour at coastline. 'stranding' means beaching of particles is allowed
            self.o.set_config('drift:horizontal_diffusivity', hdiff)  # horizontal diffusivity
            self.o.set_config('drift:advection_scheme', 'euler') # advection schemes (default is 'euler'). other options are 'runge-kutta', 'runge-kutta4'
            
            # uncertainty
            #self.o.set_config('drift:current_uncertainty', .1)
            #self.o.set_config('drift:wind_uncertainty', 1)

            ##### SEEDING #####
            print('seeding particles...')
            np.random.seed(None) # to avoid seeding in same exact position each time             

            # if simulation is 3D, set 3D parameters (terminal velocity, vertical mixing, seafloor action) and seed particles over polygon
            if self.depth == True:
                self.o.set_config('seed:terminal_velocity', termvel) # terminal velocity
                self.o.seed_from_shapefile(shapefile=str(self.poly_path), number=pnum, time=[start_time, start_time],#+self.tseed], 
                                           terminal_velocity=termvel, z=z, origin_marker=self.origin_marker, radius=seeding_radius)
                #self.o.set_config('vertical_mixing:diffusivitymodel', 'windspeed_Large1994')
                self.o.set_config('general:seafloor_action', 'deactivate')
                self.o.set_config('drift:vertical_mixing', True)
            
            # if simulation is 2D, simply seed particles over polygon
            else:
                self.o.seed_from_shapefile(shapefile=str(self.poly_path), number=pnum, time=[start_time, start_time],#+self.tseed],
                                           origin_marker=self.origin_marker, radius=seeding_radius)
            
            # run simulation and write to temporary file
            #with tempfile.TemporaryDirectory("particle", dir=self.basedir) as qtemp:
            qtemp = tempfile.TemporaryDirectory("particle", dir=self.basedir)
            temp_outfile = qtemp.name + f'/temp_particle_file_marker-{self.origin_marker}.nc'


            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")

            self.o.run(duration=duration, #end_time=end_time, 
                       time_step=time_step, #time_step_output=timedelta(hours=24), 
                       outfile=temp_outfile, export_variables=['lon', 'lat', 'z', 'status', 'age_seconds', 'origin_marker'])#, 'sea_floor_depth_below_sea_level', 'ocean_mixed_layer_thickness',])# 'x_sea_water_velocity', 'y_sea_water_velocity', 'x_wind', 'y_wind'])

            elapsed = (T.time() - t_0)
            print("total simulation runtime %s" % timedelta(minutes=elapsed/60)) 


            #### CHECK HERE IF READERS WERE PROCESSED CORRECTLY #### # separate method?
            if hasattr(self.o, 'discarded_readers'):
                logger.warning(f'Readers {self.o.discarded_readers} were discarded. Particle transport will be affected')

            #### A BIT OF POST-PROCESSING ####
            # separate method 
            print('writing to netcdf...')

            _ps = xr.open_dataset(temp_outfile) # open temporary file
            #print('time len before processing', len(_ps.time))

            # keep 'inactive' particles visible (i.e. particles that have beached or gotten stuck on seafloor)
            ps = _ps.where(_ps.status>=0).ffill('time') 
            #print('time len after ffill inactive', len(ps.time))

            # write useful attributes
            ps = ps.assign_attrs({'total_bounds': poly.total_bounds, 'start_time': t0, 'duration_days': duration_days, 'pnum': pnum, 'hdiff': hdiff,
                                  #'tseed': self.tseed.total_seconds(), 
                                  'tstep': tstep.total_seconds(), 'termvel': termvel, 'depth': str(self.depth),
                                  'poly_path': str(self.poly_path), 'opendrift_log': str(self.o)}) 
                                    #removing this attribute as i already have the check for discarded readers elsewhere. 
                                    #This way I can compare attributes more easily for new caching method


            ps.to_netcdf(str(file_path))
            print(f"done. NetCDF file '{str(file_path)}' created successfully.")

        
        self.particle_path = str(file_path)

        self.ds = ps
        
        if 'qtemp' in locals():
            Path(temp_outfile).unlink()
            os.rmdir(qtemp.name)

        
        logger.error(f'particle simulation lat = {ps.lat.shape}, lon={ps.lon.shape}, time={ps.time.shape}')

        
        pass
        
        
    def filter_by_status(self, status):
        """
        Filtering trajectories by status (active, stranded or seafloor).
        """
        available_status = {'active': 0, 'stranded': 1, 'seafloor': 2}

        try:
            ds = self.ds
        except:
            ds = self.get_ds

        try:
            traj = ds.trajectory.where(ds.isel(time=-1).status==available_status[status]).dropna('trajectory').data
            filtered_ds = ds.sel(trajectory=traj)
        except:
            print("Unrecognised status. Status must be one of 'active', 'stranded', 'seafloor'. No filtering by status was carried out.")    
        print(f'Rasterizing only {status} particles.')
        return filtered_ds

    def interpolate_time(self, new_timestep):
        """
        Interpolate time on ds by giving a new timestep (in hours).
        """
        try:
            ds = self.ds
        except:
            ds = self.get_ds
            
        new_time = np.arange(pd.to_datetime(ds.time[0].values), pd.to_datetime(ds.time[-1].values),timedelta(hours=new_timestep)) #new time variables used for interpolation
        ds = ds.interp(time=new_time, method='slinear') # interpolate dataset 
        
        print(f'Interpolating time on ds. New timestep = {new_timestep} hours...')
        return ds

    def assign_weight(self, weight=1):
        """
        Add a weight variable to ds. If 
        """
        ds = self.get_ds

        print(f'Adding weight variable to ds...')
        w = np.ones_like(self.ds.lon)*weight # longitude is always going to be available as a variable, so taking it as reference for the shape
        ds = ds.assign({'weight': (('trajectory', 'time'), w.data)}) 
        
        return ds

    
    def decay_rate(self, k):
        '''
        Exponential decay rate function based on given decay coefficient k. 
        All trajectories decay in the same way. 
        
        Parameteres
        -----------
        k : float
            decay coefficient for exponential decay function
        '''
        y = np.exp(-k*(self.ds.time-self.ds.time.min()).astype(int)/60/60/1e9/24) #decay function
        return y

    # make into property?
    def use_by_traj(self, use_path, res, r_bounds=None):
        '''
        Select value of use raster at the trajectories' starting positions. 
        Returns the use_value as a DataArray.

        Parameters
        ----------
        use_path : str
            Path of .tif file representing density of human activity acting as pressure source. Used to assign  'weights' to trajectories in histogram calculation.   
        res : float
            Resolution at which to bin use raster.
        r_bounds : list, optional
            Spatial bounds for computation of raster over a subregion, written as [x1,y1,x2,y2]. Default is None (bounds are taken from self.)
        '''
        print('applying use weight...')
        grid = self.polygon_grid(res, r_bounds=r_bounds)
        poly_bounds = grid.total_bounds
        
        _use = rxr.open_rasterio(use_path).rio.reproject('epsg:4326', nodata=0).sortby('x').sortby('y').isel(band=0).drop('band').sel(x=slice(poly_bounds[0], poly_bounds[2]), y=slice(poly_bounds[1], poly_bounds[3])).fillna(0) # what was fillna for?

        # create reference grid
        # we already have a function that creates a grid called polygon grid?
        _gr = xr.DataArray(np.zeros((len(self._x_c), len(self._y_c))), coords={'x': self._x_c, 'y': self._y_c})

        gr = (
            xr.DataArray(_gr) # need to transpose it because xhistogram does that for some reason
            .rio.write_nodata(np.nan)
            .rio.write_crs('epsg:4326')
            .rio.write_coordinate_system())

        use = rasterhd2raster(_use, gr) # resample the use raster on our grid, with user-defined res and crs

        # NB: there is a bug in the rasterhd2raster function (no conservation)
        # so the sum of my raster won't match the sum of the use raster (but it will match the sum of the resampled one)
        # questo use weight in realtà è solo il valore dell'uso nella posizione iniziale di ogni traiettoria. forse si potrebbe chiamare in un altro modo
        use_value = _use.sel(x=self.ds.isel(time=0).lon, y=self.ds.isel(time=0).lat,  method='nearest')#, tolerance=self.res*2) ### TODO capire bene tolerance. with the new resampling method, could do tolerance=res/2?
        
        print(f'number of trajectories with use value = 0 {use_value.where(use_value==0).count().data}')
        print(f'number of trajectories with use value != 0 {use_value.where(use_value!=0).count().data}')
        
        return use_value 

    def get_bin_n(self, res, t=0, r_bounds=None):
        '''
        Add new variable to ds ('bin_n_t0') containing its "bin number" at timestep t, i.e. a unique identifier corresponding to a specific spatial grid-cell. 
        
        Parameters
        ----------
        t : int
            index of timestep to consider
        '''
      
        grid = self.polygon_grid(res, r_bounds=r_bounds)
        ds = self.get_ds
        # calculate bin number of each trajectory at time 0, assign it to variable bin_n_t0
        _bins = np.zeros((grid.shape)) 
        #print(_bins.shape)
        _bins[:,0] = np.array(grid.centroid.x) 
        _bins[:,1] = np.array(grid.centroid.y)  
        print(f'calculating bin number at timestep {t}')
        func = lambda plon, plat: np.abs(_bins-[plon,plat]).sum(axis=1).argmin()
        #print('bin_n ufunc')
        ds['bin_n_t0'] = xr.apply_ufunc(func, ds.isel(time=t).lon, ds.isel(time=t).lat, vectorize=True, dask='parallelized')

        return ds

    def get_histogram(self, res, r_bounds=None, weighted=False):
        '''
        res : float
            cell size, aka desired resolution of output map
        weighted : bool, optional
            whether the histogram should be weighted or not
        Parameters
        ----------
        
        '''
        self.polygon_grid(res, r_bounds=r_bounds)
        xbin = self._x_e
        ybin = self._y_e
        
        if weighted:
            print('calculating weighted histogram...')
            h = histogram(self.ds.lon, self.ds.lat, bins=[xbin, ybin], dim=['trajectory', 'time'], weights=self.ds.weight, block_size='auto')
        else:
            print('no weight variable detected, calculating unweighted histogram...')
            h = histogram(self.ds.lon, self.ds.lat, bins=[xbin, ybin], dim=['trajectory', 'time'], block_size='auto')

        h = h.transpose() # important to not have it where(h>0) otherwise it misses values when summing
        
        return h

    def write_tiff(self, h, crs=4326, path=None, description=None, output_dir=None):# instead of use label, could do 'assign_attrs' for any attributes, including use_label, i might want to add
        '''
        Georeferences histogram and writes it as tif. 
        
        Parameters
        ----------
        crs : int, optional
            epsg code
        write_tif = bool, optional
            Whether to write raster as a tif file. 
        '''
        r = (
            xr.DataArray(h) 
            .rio.write_nodata(np.nan)
            .rio.write_crs('epsg:'+str(crs))
            .rio.write_coordinate_system())

        if output_dir: 
            path = output_dir+'/'+str(path).split('/')[-1]
            
        if description:
            path = str(path).replace('.tif', f'_use-{description}.tif')
            #path = str(path).replace('.tif', f'_use-{description}.nc') # FOR DEBUGGING

        r.rio.to_raster(path, nodata=0) 
        #h.to_netcdf(path)
        
        print(f'Wrote tiff to {path}.')
        pass

    def residence_time(self, res, r_bounds):#, r_bounds, status, tinterp):
        '''
        Estimates the residence time per cell (in hours) by multiplying the count of trajectories in a cell by the timestep of the run. 

        Parameters
        ----------
        res : float
            cell size, aka desired resolution of output map
        '''
        # for residence time, the weight is simply the timestep in hours 
        w = datetime.strptime(self.get_ds.time_step_output, '%H:%M:%S').hour
        
        self.ds = self.assign_weight(weight=w)
        # could add attributes to the variable explaining what the weight is in each case
        
        return self.get_histogram(res, r_bounds=r_bounds, weighted=True)  

        
    def concentration(self, use_path, res, r_bounds=None, decay_coef=0):
        '''
        Calculates concentration of pressure per grid-cell at any given time, based on given raster of use density. 
        Concentration is calculated by assigning to each trajectory a weight that is equal to the use intensity at the trajectory's starting position.
        To ensure conservation over time, the weight at each timestep is equal to value_of_use/number_of_timesteps/number_of_trajectories_starting_from_that_cell. 

        Parameters
        ----------
        use_path : str
            Path of .tif file representing density of human activity acting as pressure source. Used to assign  'weights' to trajectories in histogram calculation. 
        '''
        self.get_ds

        # extract value of use at starting position of each trajectory
        use_value = self.use_by_traj(use_path=use_path, res=res, r_bounds=r_bounds)
        self.ds = self.get_bin_n(res=res, t=0, r_bounds=r_bounds)
        weight_by_bin = use_value.groupby(self.ds.isel(time=0).bin_n_t0)
        counts_per_bin = self.ds.isel(time=0).bin_n_t0.groupby(self.ds.isel(time=0).bin_n_t0).count()
        w = weight_by_bin/counts_per_bin/len(self.ds.time) # here, i am dividing weight by the number of particles in each cell at starting pos

        # decay rate. default is no decay, but this is still useful to give weight the correct shape
        print(f'computing decay rate function with decay coefficient k = {self.decay_coef}...')
        y = self.decay_rate(k=decay_coef)
        weight = w*y
        
        self.ds = self.assign_weight(weight)

        r = self.get_histogram(res, r_bounds=r_bounds, weighted=True).assign_attrs({'use_path': use_path})#.rename({'histogram_lon_lat': 'r0'})

        return r
        

    def single_run(self, pnum, start_time, duration, res, r_bounds=None, decay_coef=0, use_path=None, use_label=None, output_dir=None):
        '''
        Wrapper ? method computing trajectories and producing raster maps of residence time and (if required) concentration.
        
        Parameters
        ----------
        pnum : int
            Number of particles to seed, i.e. trajectories to calculate.
        start_time : string
            Start date of the run, in YYYY-MM-DD format.
        duration : float
            Duration of the run, in days. 
        res : float
            Resolution of output rasters, in lon/lat degrees.
        use_path : string, optional
            Path to map of human activities (.tiff) considered as pressure source. Default is None
        decay_coef : float, optional
            Coefficient k of exponential decay function. Default is 0 (no decay)
        output_dir : dict, optional
        
        '''
        self.rt_cache = PMARCache(Path(self.basedir) / 'residence-time')
        
        rt_path, rt_exists = self.rt_cache.raster_cache(res=res, poly_path=self.poly_path, pnum=pnum, ptot=None, start_time=start_time, duration=duration, reps=None, tshift=None, use_path=use_path, use_label=use_label, decay_coef=decay_coef, r_bounds=r_bounds)#, aggregate=aggregate, depth_layer=depth_layer, z_bounds=z_bounds, particle_status=particle_status, traj_dens=traj_dens)
        print(f'##################### rt_exists = {rt_exists}')
        print(f'##################### rt_path = {rt_path}')
        self.calculate_trajectories(start_time=start_time, pnum=pnum, duration_days=duration)

        try:
            rt_output_dir = output_dir['temp_rt_output']
            c_output_dir = output_dir['temp_c_output']
        except:
            rt_output_dir = c_output_dir = None
            
        if rt_exists == True:
            self.rt = rxr.open_rasterio(rt_path)
        else:
            self.rt = self.residence_time(res=res, r_bounds=r_bounds)
            self.write_tiff(self.rt, path=rt_path, output_dir=rt_output_dir)
            
        if use_path:
            self.c_cache = PMARCache(Path(self.basedir) / f'concentration-{use_label}')
    
            c_path, c_exists = self.c_cache.raster_cache(res=res, poly_path=self.poly_path, pnum=pnum, ptot=None, start_time=start_time, duration=duration, reps=None, tshift=None, use_path=use_path, use_label=use_label, decay_coef=decay_coef, r_bounds=r_bounds)#, aggregate=aggregate, depth_layer=depth_layer, z_bounds=z_bounds, particle_status=particle_status, traj_dens=traj_dens)
            print(f'##################### c_exists = {c_exists}')
            print(f'##################### c_path = {c_path}')

            if c_exists == True:
                self.c = rxr.open_rasterio(c_path)
            else:
                self.c = self.concentration(use_path=use_path, res=res, r_bounds=r_bounds, decay_coef=decay_coef)
                
                self.write_tiff(self.c, path=c_path, output_dir=c_output_dir) # this will have to be use label
        else:
            print('No use_path provided. Returning residence time only.')
        
    
    def sum_reps(self, rep_path):
        '''
        Sum concentration rasters of single reps, return summed raster. 
    
        Parameters 
        ----------
        rep_path : list
            list of paths to each rep 
        '''
        for idx, rep in enumerate(rep_path):
            if idx == 0:
                r1 = r0 = rxr.open_rasterio(rep) # should be open rasterio 
            else:
                r0 = r1
                r1 = r0 + rxr.open_rasterio(rep)

            #fig, ax = plt.subplots()
            #r1.histogram_lon_lat.where(r1.histogram_lon_lat>0).plot(ax=ax)
        
        return r1
    
    def run(self, ptot, reps, tshift, duration=30, start_time='2019-01-01', res=0.04, r_bounds=None, use_path=None, use_label=None, decay_coef=0):
        '''
        Wrapper ? method computing trajectories and producing residence time and concentration raster over required number of reps. 
    
        Parameters
        ----------
        ptot : int
    
        reps : int
    
        tshift : int
    
        duration : int, optional
    
        start_time : string, optional
    
        res : float, optional

        r_bounds : list, optional
    
        use_path : string, optional

        use_label : string, optional
    
        decay_coef : float, optional
        
        '''
        self.rt_cache = PMARCache(Path(self.basedir) / 'residence-time')
        self.c_cache = PMARCache(Path(self.basedir) / f'concentration-{use_label}')


        rt_path, rt_exists = self.rt_cache.raster_cache(res=res, poly_path=self.poly_path, pnum=None, ptot=ptot, start_time=start_time, duration=duration, reps=reps, tshift=tshift, use_path=use_path, use_label=use_label, decay_coef=decay_coef, r_bounds=r_bounds)#, aggregate=aggregate, depth_layer=depth_layer, z_bounds=z_bounds, particle_status=particle_status, traj_dens=traj_dens)
        c_path, c_exists = self.c_cache.raster_cache(res=res, poly_path=self.poly_path, pnum=None, ptot=ptot, start_time=start_time, duration=duration, reps=reps, tshift=tshift, use_path=use_path, use_label=use_label, decay_coef=decay_coef, r_bounds=r_bounds)#, aggregate=aggregate, depth_layer=depth_layer, z_bounds=z_bounds, particle_status=particle_status, traj_dens=traj_dens)

        temp_rt_output_dir = tempfile.TemporaryDirectory(dir=self.rt_cache.cachedir)
        temp_c_output_dir = tempfile.TemporaryDirectory(dir=self.c_cache.cachedir)

        if rt_exists == True:
            print('raster rt file with these configurations already exists')
            rt = rxr.open_rasterio(rt_path)
            #c = rxr.open_rasterio(rep_c_path)
        
        if c_exists == True:
            print('raster c file with these configurations already exists')
            c = rxr.open_rasterio(c_path)
            
        if rt_exists == False or c_exists == False:
            for n in range(0, reps):
                #print(f'Starting rep #{n+1}...')
                start_time_dt = datetime.strptime(start_time, '%Y-%m-%d')+timedelta(days=tshift)*n #convert start_time into datetime to add tshift
                rep_start_time = start_time_dt.strftime("%Y-%m-%d") # bring back to string to feed to opendrift
                
                logger.warning(f'Starting rep #{n} with start_time = {rep_start_time}')
                
                rep_id = n # rep ID is maybe a better name than origin_marker! # self.rep_id
                # this will have to go as an attribute in ds too, useful for plotting
                
                pnum = int(np.round(ptot/reps)) #  ptot should be split among the reps
                
                self.single_run(pnum=pnum, duration=duration, start_time=rep_start_time, res=res, r_bounds=r_bounds, use_path=use_path, use_label=use_label, decay_coef=decay_coef, output_dir = {'temp_rt_output': temp_rt_output_dir.name, 'temp_c_output': temp_c_output_dir.name})
                logger.warning(f'Done with rep #{n}.')

            # the below could be improved...
            rep_rt_path = glob.glob(f'{temp_rt_output_dir.name}/*.tif')
            print(f'############################## rep_rt_path = {rep_rt_path}')
            rep_c_path = glob.glob(f'{temp_c_output_dir.name}/*.tif')
            print(f'############################## rep_c_path = {rep_c_path}')
    
            ### FOR DEBUGGING
            #rep_rt_path = glob.glob(f'{temp_output_dir.name}/*RES-TIME*.nc')
            #rep_c_path = glob.glob(f'{temp_output_dir.name}/*{use_label}*.nc')
            #### 
            rt = self.sum_reps(rep_rt_path)
            self.write_tiff(rt, path=rt_path)
            
            c = self.sum_reps(rep_c_path)
            self.write_tiff(c, path=c_path)
            print('DONE.')

        return rt, c
        
    
    def plot(self, r=None, xlim=None, ylim=None, cmap=spectral_r, shading='flat', vmin=None, vmax=None, norm=None, coastres='10m', proj=ccrs.PlateCarree(), dpi=120, figsize=[10,7], rivers=False, title=None, save_fig='thumbnail'):
        """
        Plot particle_raster outputs.
        
        Parameters
        ----------
        xlim, ylim : array-like, optional
            Specify *x*- and *y*-axis limits.
        cmap : matplotlib colormap name or colormap, optional
            Plot colormap. Default is seaborn's 'spectral' map, reversed. 
        shading : {'flat', 'nearest', 'gouraud', 'auto'}, optional
            The fill style for the quadrilateral; defaults to
            'flat' or :rc:`pcolor.shading`. Possible values:

            - 'flat': A solid color is used for each quad. The color of the
              quad (i, j), (i+1, j), (i, j+1), (i+1, j+1) is given by
              ``C[i, j]``. The dimensions of *X* and *Y* should be
              one greater than those of *C*; if they are the same as *C*,
              then a deprecation warning is raised, and the last row
              and column of *C* are dropped.
            - 'nearest': Each grid point will have a color centered on it,
              extending halfway between the adjacent grid centers.  The
              dimensions of *X* and *Y* must be the same as *C*.
            - 'gouraud': Each quad will be Gouraud shaded: The color of the
              corners (i', j') are given by ``C[i', j']``. The color values of
              the area in between is interpolated from the corner values.
              The dimensions of *X* and *Y* must be the same as *C*. When
              Gouraud shading is used, *edgecolors* is ignored.
            - 'auto': Choose 'flat' if dimensions of *X* and *Y* are one
              larger than *C*.  Choose 'nearest' if dimensions are the same.

            See :doc:`/gallery/images_contours_and_fields/pcolormesh_grids`
            for more description.
        vmin, vmax : float, optional
            Values to anchor the colormap, otherwise they are inferred from the
            data and other keyword arguments. When a diverging dataset is inferred,
            setting one of these values will fix the other by symmetry around
            ``center``. Setting both values prevents use of a diverging colormap.
            If discrete levels are provided as an explicit list, both of these
            values are ignored. 
        norm : matplotlib.colors.Normalize, optional
            If ``norm`` has ``vmin`` or ``vmax`` specified, the corresponding
            kwarg must be ``None``.
        coastres : str or :class:`cartopy.feature.Scaler`, optional
            A named resolution to use from the Natural Earth
            dataset. Currently can be one of "auto" (default), "110m", "50m",
            and "10m", or a Scaler object.
        proj : {None, 'aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear', str}, optional
            The projection type of the `~.axes.Axes`. *str* is the name of
            a custom projection, see `~matplotlib.projections`. The default
            None results in a 'rectilinear' projection.
            Default is cartopy.crs.PlateCarree().
        dpi : float, default: :rc:`figure.dpi`
            The resolution of the figure in dots-per-inch.
        figsize : tuple, optional
            A tuple (width, height) of the figure in inches.   
        rivers : bool, optional
            Whether to plot rivers. Default is False
        save_fig : bool, optional
            Whether to save figure as png. Default is True
        """
                
        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 16
        
        if self.raster is None: 
            raise ValueError("No raster has been calculated yet. Please launch particle_raster method first.")
        
        if r is None:
            r = self.raster[list(self.raster.data_vars)[-1]]
        
        else:
            pass
        
        ### this is actually creating problems in plots, commenting for now
        # drop nan values to have cleaner thumbnails
        #lon_var = [varname for varname in r.coords if "lon" in varname][0] # find name of lon variable
        #lat_var = [varname for varname in r.coords if "lat" in varname][0] # find name of lat variable

        r = r.where(r>0)
        #r = r.dropna(str(lon_var), 'all').dropna(str(lat_var), 'all')
        
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=proj)
        ax.coastlines(coastres, zorder=11, color='k', linewidth=.5)
        ax.add_feature(cartopy.feature.LAND, zorder=0, facecolor='0.9') #'#FFE9B5'
        ax.add_feature(cartopy.feature.BORDERS, zorder=11, linewidth=.5, linestyle=':')
        if rivers is True:
            ax.add_feature(cartopy.feature.RIVERS, zorder=12)
        gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, linewidth=.5, color='gray', linestyle='--')
        gl.top_labels = False
        gl.right_labels = False    

        im = r.plot(vmin=vmin, vmax=vmax, norm=norm, shading=shading, cmap=cmap, add_colorbar=False)
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        cbar = plt.colorbar(im, cax=cax, extend='max')
        cbar.set_label('uom from input layer', rotation=90)

        if title is not None:
            ax.set_title(title, fontsize=12)
        else:
            ax.set_title(f'use_label: {r.use_label}\n use_path: {r.use_path}', fontsize=8)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)


        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        if save_fig is not None:
            #Path(self.basedir / 'thumbnails').mkdir(parents=True, exist_ok=True) # useful for geoplatform to keep them in same dir as raster
            plt.savefig(str(save_fig), dpi=160, bbox_inches='tight')

        return fig, ax

    
    def scatter(self, t=None, xlim=None, ylim=None, s=1, alpha=1, c='age', cmap='rainbow', coastres='10m', proj=ccrs.PlateCarree(), dpi=120, figsize=[10,7], rivers=False, save_to=None):
        """
        Visualise particle trajectories over time [days elapsed], defined by age_seconds. 
        
        
        Parameters
        ----------
        t : int, optional
            If a t is given, the scatter plot will represent the selected instant in time (int) or time interval (slice). Otherwise, it will show the entire trajectory (default). 
        xlim, ylim : array-like, optional
            Specify *x*- and *y*-axis limits.
        s : float, optional
            Marker size for scatter plot. Default is 1
        c : str, optional
            String defining variable used for colormap. Options are 'z' or 'age' (Default)
        cmap : matplotlib colormap name or colormap, optional
            Plot colormap. Default is seaborn's 'spectral' map, reversed. 
        coastres : str or :class:`cartopy.feature.Scaler`, optional
            A named resolution to use from the Natural Earth
            dataset. Currently can be one of "auto" (default), "110m", "50m",
            and "10m", or a Scaler object.
        proj : {None, 'aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear', str}, optional
            The projection type of the `~.axes.Axes`. *str* is the name of
            a custom projection, see `~matplotlib.projections`. The default
            None results in a 'rectilinear' projection.
            Default is cartopy.crs.PlateCarree().
        dpi : float, default: :rc:`figure.dpi`
            The resolution of the figure in dots-per-inch.
        figsize : tuple, optional
            A tuple (width, height) of the figure in inches.   
        rivers : bool, optional
            Whether to plot rivers. Default is False
        """        
        
        O = np.floor(np.log10(len(self.ds.time)))
        
        if t is None:
            if O >= 3:
                ds = self.ds.isel(time=slice(None,None,int(len(self.ds.time)/10**(O-2))))
            else:
                ds = self.ds
        else:
            ds = self.ds.isel(time=t)
        
        
        if c == 'age': 
            c = ds.age_seconds/60/60/24
            #cmap = 'rainbow' #'copper'
            cbar_label = 'elapsed days'
        elif c == 'z':
            c = ds.z
            #cmap = 'viridis'
            cbar_label = 'depth [m]'
        elif c == 'marker':
            c = ds.origin_marker
            cbar_label = 'origin marker (#rep)'
        
        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 16
        
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=proj)
        ax.coastlines(coastres, zorder=11, color='k', linewidth=.5)
        ax.add_feature(cartopy.feature.LAND, facecolor='0.9', zorder=10) #'#FFE9B5'
        ax.add_feature(cartopy.feature.BORDERS, zorder=10, linewidth=.5, linestyle=':')
        if rivers is True:
            ax.add_feature(cartopy.feature.RIVERS, zorder=12)
        gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, linewidth=.5, color='gray', linestyle='--')
        gl.top_labels = False
        gl.right_labels = False    

        im = ax.scatter(ds.lon, ds.lat, s=s, c=c, cmap=cmap, alpha=alpha)
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(cbar_label, rotation=90)
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        ax.set_title('Particle trajectories over time', fontsize=10);

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
        if save_to is not None:
            plt.savefig(save_to, dpi=160, bbox_inches='tight')
        
        return fig, ax
    
    
    def animate(self):
        """
        WIP. Histogram animation using xmovie
        
        """
        
        pass


    
def check_particle_file(path_to_file):
    ds = xr.open_dataset(path_to_file)
    vars_to_check = ['x_sea_water_velocity', 'y_sea_water_velocity', 'x_wind', 'y_wind']
    for i in vars_to_check:
        if np.all(ds[i].load() == 0):
            print(f'ATTENTION: all 0 values detected for variable {i}.')
        else:
            print(f'all good: variable {i} has non-zero values.')
