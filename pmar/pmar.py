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
from opendrift.models.chemicaldrift import ChemicalDrift
#import copernicusmarine
from opendrift.readers.reader_copernicusmarine import Reader
#from opendrift.readers.reader_netCDF_CF_generic import Reader
import opendrift
from pydap.client import open_url
from pydap.cas.get_cookies import setup_session
import time as T
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
#import netrc
import random
import rasterio
import tempfile
from netCDF4 import Dataset
from pathlib import Path
import hashlib
import json
#from flox.xarray import xarray_reduce # for xarray grouping over multiple variables
os.environ['PROJ_LIB'] = '/var/miniconda3/envs/opendrift/share/proj/'
from functools import partial
from rasterio.enums import Resampling
import glob
from copy import deepcopy
from geocube.api.core import make_geocube
from pmar.pmar_cache import PMARCache
from pmar.pmar_utils import traj_distinct, rasterhd2raster, check_particle_file, get_marine_polygon

logger = logging.getLogger("PMAR")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
        
        
class PMAR(object): 
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
    get_trajectories()
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


    def __init__(self, context, pressure='general', chemical_compound=None, basedir='lpt_output', localdatadir = None, seeding_shapefile = None, poly_path = None, uv_path=None, wind_path=None, mld_path=None, bathy_path=None, particle_path=None, depth=False, netrppi_path=None):
        """
        Parameters
        ---------- 
        context : str
            String defining the context of the simulation i.e., the ocean model output to be used for particle forcing. Options are 'med-cmems', 'bs-cmems' and 'bridge-bs'. 
        pressure : str, optional

        chemical_compound : str, optional
            if pressure is 'chemical', a chemical_compound can be specified to be initialised in ChemicalDrift
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
        self.uv_path = uv_path
        self.wind_path = wind_path
        self.mld_path = mld_path 
        self.bathy_path = bathy_path 
        self.basedir = Path(basedir)
        self.particle_path = particle_path # i can import an existing particle_path
        #self.ds = None. commented otherwise try: self.ds except: does not work
        self.o = None
        self.poly_path = poly_path 
#        self.raster = None unused
        self.origin_marker = 0
        #self.netrc_path = netrc_path
        self.tstep = None
        self.pnum = None
        self.depth = depth
        self.termvel = 1e-3
        self.decay_coef = 0
        self.context = self.get_context(str(context))
        self.outputdir = None
        self.pressure = pressure
        self.chemical_compound = chemical_compound
        self.localdatadir = localdatadir
        self.particle_status = None
        self.reps = 1
        self.tshift = None
        self.cache = PMARCache(Path(basedir) / 'cachedir')
        self.raster_path = None
        self._polygon_grid = None
        self._x_e = None
        self._y_e = None
        self._x_c = None
        self._y_c = None
        self.res = None
        self.weight = None
        self.r_bounds = None
        self.res = None

        self.study_area = None # this will substitude r_bounds 
        self.seeding_shapefile = seeding_shapefile # this will substitude poly_path, and is only to be used for seeding. can be point, line or polygon. if point or line, buffer needs to be added
        self.seed_within_bounds = None # this will substitude s_bounds. if no seeding_shapefile is given, user can choose to give lon, lat bounds to seed within

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

        # if a path to a shapefile is given to be used for seeding, read it and save it in the lpt_output/polygons dir
        if poly_path is not None: 
            Path(self.basedir / 'polygons').mkdir(parents=True, exist_ok=True)            
            poly = gpd.read_file(poly_path).to_crs('epsg:4326')
            bds = np.round(poly.total_bounds, 4) # only used for poly file name
            local_poly = f'polygon-crs_epsg:{poly.crs}-lon_{bds[0]}_{bds[2]}-lat—{bds[1]}_{bds[3]}.shp'
            q = self.basedir / 'polygons' / local_poly
            poly.to_file(str(q), driver='ESRI Shapefile')
            self.poly_path = str(q)
        #else:
         #   if 'med' in self.context:
          #      self.poly_path = f'{DATA_DIR}/polygon-med-full-basin.shp'
           # elif 'bs' in self.context:
            #    self.poly_path = f'{DATA_DIR}/polygon-bs-full-basin.shp'
           # else:
           #     pass
            

        # this (if still needed?) should be a separate method
        # if particle_path is given, retrieve number of reps and load ds
        if self.particle_path is not None: 

            # gather number of reps from origin marker. THIS IS A PROBLEM IF THERE ARE MORE BATCHES IN SAME FOLDER. 
            if type(self.particle_path) is list: 
                self.reps = len(particle_path)
                

    def get_context(self, context):
        """
        Parameters
        ---------- 
        context : str
            One of 'global', 'med', 'black sea', 'baltic'. Returns strings to relevant Copernicus products to add as OpenDrift readers. 
        """
        
        contexts = {'global': 
                        {'readers': 
                         {'currents': 'cmems_mod_glo_phy_my_0.083deg_P1D-m', # global physics reanalysis, daily, 1/12° horizontal resolution, 50 vertical levels, 1 Jan 1993 to 25 Feb 2025
                          'winds': 'cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H', # L4 sea surface wind and stress fields 0.125° res, hourly, 1 Jun 1994 to 21 Nov 2024
                          'bathymetry':'cmems_mod_glo_phy_my_0.083deg_static', # global physics reanalysis static, same as currents
                          #'mixed-layer': 'cmems_mod_glo_phy_my_0.083deg_P1D-m', # same as currents
                          'stokes': 'cmems_mod_glo_wav_my_0.2deg_PT3H-i', # global ocean waves reanalysis, 3-hourly, 1 Jan 1980 to 31 Jan 2025
                          'spm': 'cmems_obs-oc_glo_bgc-transp_my_l4-multi-4km_P1M',
                         },
                        'extent': [-180, -80, 179.92, 90],
                        'polygon': get_marine_polygon()},
                        
                        'med': 
                        {'readers':
                         {'currents': 'med-cmcc-cur-rean-d', # change d to h to get hourly. Mediterranean Sea Physics Reanalysis, 1/24˚res, 141 z-levels, 1 Jan 1987 to 1 Feb 2025
                          'winds': 'cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H', # global L4
                          'bathymetry': 'cmems_mod_med_phy_my_4.2km_static', # Mediterranean Sea Physics Reanalysis
                          'mixed-layer': 'med-cmcc-mld-rean-d', # Mediterranean Sea Physics Reanalysis
                          'stokes': 'cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H',   # Mediterranean Sea Waves Reanalysis, hourly, 1/24˚res, 1 Jan 1985 to 1 Mar 2025
                         },
                         'extent': [-6, 30.19, 36.29, 45.98],
                         'polygon': None,
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
                        'polygon': None
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
                         'polygon': None,
                        }
            }

        if context not in contexts.keys():
            raise ValueError('Unsupported context. Please insert one of "global", "med", "black sea", "baltic".')
            
        self.extent = contexts[context]['extent']
        
        return contexts[context]

    
    def x_grid(self):
        if self._x_e is None:
            raise ValueError('polygon_grid needs to be called before using this method')
        return self._x_e

    def y_grid(self):
        if self._y_e is None:
            raise ValueError('polygon_grid needs to be called before using this method')
        return self._y_e

    #@property
    #def polygon_grid(self):
     #   return self._polygon_grid

    #@polygon_grid.setter
    def polygon_grid(self, res, r_bounds=None):
        '''
        Create grid of given resolution (res) and intersect it with poly_path.
        '''
        
        # if res == self.res and self._polygon_grid is not None: # should test if r_bounds matches too!
        #     print(f'polygon_grid: _polygon_grid was previously calculated with resolution = {self.res}.')
        # else:
        logger.info(f'polygon_grid: calculating new polygon_grid with resolution = {res} and r_bounds = {r_bounds}.')
        #res = self.res 
        crs = 'epsg:4326' # need to use EPSG:4326 because the output of opendrift is in this epsg and i want to use this grid for the histogram of the opendrift output
        
        if r_bounds is not None: # if r_bounds are given, meaning we are calculating the raster on a different region than seeding, create new polygon to use for aggregation / visualisation
            poly = self.make_poly(lon=[r_bounds[0], r_bounds[2]], lat=[r_bounds[1], r_bounds[3]], write=False).to_crs('epsg:4326')#.buffer(distance=res*3)
            logger.info(f'making new polygon_grid using r_bounds = {r_bounds}')
        else:
            poly = gpd.read_file(self.poly_path).to_crs(crs)#.buffer(distance=res*3) # the buffer is added because of the non-zero radius when seeding, otherwise some particles might be left out
        
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
        self._x_c = np.unique(self._polygon_grid.centroid.x.values) # centroid coordinates .round(4)
        self._y_c = np.unique(self._polygon_grid.centroid.y.values)
            
        #print('polygon_grid: done.')
        self.res = res
        print(f'updated self.res = {self.res}')
        return self._polygon_grid

    def raster_grid(self, res=None, r_bounds=None):
        logger.warning(f'rasterizing grid with extent = {r_bounds}')
        grid = self.polygon_grid(res=res, r_bounds=r_bounds)
        grid['weight'] = 1
        _raster_grid = make_geocube(vector_data=self._polygon_grid, measurements=["weight"], resolution=(-res, res))
        self._raster_grid = _raster_grid.weight
        return self._raster_grid

    @property
    def get_ds(self):
        # this doesnt actually work with reps, because self.ds always exists, but then it is only the last rep.
        # should discriminate e.g. by comparing self.ptot with self.ds.trajectory.
        # if they don't match, i am probably only looking at the last rep. 
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
        lens = np.array([len(xr.open_dataset(filename).time) for filename in self.particle_path])
        print(f'giving all datasets the same time length of {lens} tsteps...')
        return lens.max()

    def _preprocess(self, ds, correct_len):
        '''
        Sometimes opendrift runs end early because e.g. all particles have beached, resulting in reps with different time lengths. This method pads all reps so that they all have same time length (determined with find_correct_len).
        '''
        return ds.pad(pad_width={'time': (0, correct_len-len(ds.time))}, mode='edge')
         

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
            raise ValueError("'lon' and 'lat' must have length larger than 2")
        
        if write is True:
            poly.to_file(str(q), driver='ESRI Shapefile')
            self.poly_path = str(q)
        else:
            return poly
    
    def get_trajectories(self, pnum, start_time='2019-01-01', season=None, duration_days=30, s_bounds=None, z=-0.5, tstep=timedelta(hours=4), hdiff=10, termvel=None, crs='4326', seeding_radius=0, beaching=False, stokes_drift=False, loglevel=40):
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
         
        if termvel is None:
            termvel = self.termvel
        
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

        duration = timedelta(days=duration_days)
        
        self.tstep = tstep
        
        # polygon used for seeding of particles (poly_path). if s_bounds are given, a new polygon is created using those bounds. 
        # this creates a square polygon even if a poly_path is given. not what we want
        #if s_bounds is None:
         #   s_bounds = self.extent
        #lon = s_bounds[0::2]
        #lat = s_bounds[1::2]
        #self.make_poly(lon, lat, crs=crs)
        ### INSTEAD
        if s_bounds is not None:
            lon = s_bounds[0::2]
            lat = s_bounds[1::2]
            self.make_poly(lon, lat, crs=crs)
        
        # path to write particle simulation file. also used for our 'cache'    
        #poly = gpd.read_file(self.poly_path)
        bds = np.round(gpd.read_file(self.seeding_shapefile).total_bounds, 4) # only used in particle_path # PUT SEEDING SHAPEFILE RATHER THAN POLY PATH
        t0 = start_time.strftime('%Y-%m-%d')
        t1 = end_time.strftime('%Y-%m-%d')
        
        
        file_path, file_exists = self.cache.particle_cache(context=self.context, seeding_shapefile=self.seeding_shapefile, poly_path=self.poly_path, pnum=pnum, start_time=start_time, season=season, duration_days=duration_days, s_bounds=s_bounds, seeding_radius=seeding_radius, beaching=beaching, z=z, tstep=tstep, hdiff=hdiff, termvel=termvel, crs=crs, stokes_drift=stokes_drift)
        
        # if a file with that name already exists, simply import it  
        if file_exists == True:
            ps = xr.open_mfdataset(file_path)
            logger.error(f'Opendrift file with these configurations already exists within {self.basedir} and has been imported.') ### this might be irrelevant with cachedir

        # otherwise, run requested simulation
        elif file_exists == False:
            
            # initialise OpenDrift object
            if self.pressure == 'oil':
                self.o = OpenOil(loglevel=loglevel)
            elif self.pressure == 'chemical':
                self.o = ChemicalDrift(loglevel=loglevel)
            else:
                self.o = OceanDrift(loglevel=loglevel) 
                
            
            # some OpenDrift configurations
            if beaching:
                self.o.set_config('general:coastline_action', 'stranding') # behaviour at coastline. 'stranding' means beaching of particles is allowed
            else:
                self.o.set_config('general:coastline_action', 'previous') # behaviour at coastline. 'previous' means particles that reach the coast do not get stuck
            
            self.o.set_config('drift:horizontal_diffusivity', hdiff)  # horizontal diffusivity
            self.o.set_config('drift:advection_scheme', 'euler') # advection schemes (default is 'euler'). other options are 'runge-kutta', 'runge-kutta4'
            
            # ChemicalDrift configs -- values are only an example for now
            ## DO NOT CHANGE ORDER OF CONFIGS ## 
            if self.pressure == 'chemical':
                self.o.set_config('drift:vertical_mixing', True) # OpenDrift default is False, should be True for ChemicalDrift
                self.o.set_config('vertical_mixing:diffusivitymodel', 'windspeed_Large1994')
                self.o.set_config('vertical_mixing:background_diffusivity',0.0001)
                #self.o.set_config('vertical_mixing:timestep', 60) commenting because it's the default value anyway
                #o.set_config('drift:horizontal_diffusivity', 10)
                
                self.o.set_config('chemical:particle_diameter',25.e-6)  # m
                self.o.set_config('chemical:particle_diameter_uncertainty',1.e-7) # m
                
                # Parameters from radionuclides (Magne Simonsen 2019)
                self.o.set_config('chemical:sediment:resuspension_depth',1.)
                self.o.set_config('chemical:sediment:resuspension_depth_uncert',0.1)
                self.o.set_config('chemical:sediment:resuspension_critvel',0.15)
                self.o.set_config('chemical:sediment:desorption_depth',1.)
                self.o.set_config('chemical:sediment:desorption_depth_uncert',0.1)
                
                # 
                self.o.set_config('chemical:transformations:volatilization', True) # not always true, e.g. for Cd
                self.o.set_config('chemical:transformations:degradation', True) # not always true, e.g. for Cd
                self.o.set_config('chemical:transformations:degradation_mode', 'OverallRateConstants')
                
                # Chemical properties ### these are already set up in init_chemical_compound, at least for PAHs
                #self.o.set_config('chemical:transfer_setup','organics')
                #self.o.set_config('chemical:transformations:dissociation','nondiss')
                
                self.o.init_chemical_compound(self.chemical_compound) # works for a selection of PAHs
                logger.info(f'initialising chemical compound {self.chemical_compound}')

                #o.set_config('seed:LMM_fraction',.995)
                #o.set_config('seed:particle_fraction',.005)
                self.o.set_config('seed:LMM_fraction',.5)
                self.o.set_config('seed:particle_fraction',.5)

                # these have to be here in this order otherwise it gives error
                self.o.init_species() 
                self.o.init_transfer_rates()
                                

            print('adding landmask...')
            # landmask from cartopy (from "use shapefile as landmask" example on OpenDrift documentation)
            shpfilename = shpreader.natural_earth(resolution='10m',
                                    category='cultural',
                                    name='admin_0_countries')
            reader_landmask = reader_shape.Reader.from_shpfiles(shpfilename)
            
            self.o.add_reader([reader_landmask])
            self.o.set_config('general:use_auto_landmask', False)  # Disabling the automatic GSHHG landmask
            
            print('adding readers...')            
            readers = []
            for var in self.context['readers']:
                readers.append(Reader(dataset_id=self.context['readers'][var]))
            self.o.add_reader(readers) # add all readers for that context.
            #self.o.add_readers_from_list(self.context['readers'].values()) # this will add readers lazily, and only read them if useful. 
            
            # uncertainty
            #self.o.set_config('drift:current_uncertainty', .1)
            #self.o.set_config('drift:wind_uncertainty', 1)

            ##### SEEDING #####
            print('seeding particles...')
            np.random.seed(None)            

            # poly = gpd.read_file(self.poly_path)
            # if np.unique(poly.geometry.type) == 'Point':
            #     poly['geometry'] = poly.geometry.buffer(.01)
            #     seed_poly_path = Path(self.basedir / 'polygons' / ('buffered_'+self.poly_path.split('polygons/')[1]))
            #     poly.to_file(seed_poly_path)
            # else:
            #     seed_poly_path = self.poly_path
                
            seeding_poly = gpd.read_file(self.seeding_shapefile)
            if np.unique(seeding_poly.geometry.type) == 'Point':
                seeding_poly['geometry'] = seeding_poly.geometry.buffer(.01)
                new_seeding_shapefile = Path(self.basedir / 'polygons' / ('buffered_'+self.seeding_shapefile.split('/')[-1]))
                seeding_poly.to_file(new_seeding_shapefile)
                self.seeding_shapefile = str(new_seeding_shapefile)
                logger.info(f'Added buffer to {np.unique(seeding_poly.geometry.type)} type geometry in self.seeding_shapefile to allow seed_from_shapefile')
           # else:
            #    seed_poly_path = self.seeding_shapefile
            
            
            # if simulation is 3D, set 3D parameters (terminal velocity, vertical mixing, seafloor action) and seed particles over polygon
            if self.depth == True:
                self.o.set_config('seed:terminal_velocity', termvel) # terminal velocity
                self.o.seed_from_shapefile(shapefile=str(self.seeding_shapefile), number=pnum, time=start_time, 
                                           terminal_velocity=termvel, z=z, origin_marker=self.origin_marker, radius=seeding_radius)
                #self.o.set_config('vertical_mixing:diffusivitymodel', 'windspeed_Large1994')
                self.o.set_config('general:seafloor_action', 'deactivate')
                self.o.set_config('drift:vertical_mixing', True)
            
            # if simulation is 2D, simply seed particles over polygon
            else:
                self.o.seed_from_shapefile(shapefile=str(self.seeding_shapefile), number=pnum, time=start_time,
                                           origin_marker=self.origin_marker, radius=seeding_radius) # 
            
            # run simulation and write to temporary file
            #with tempfile.TemporaryDirectory("particle", dir=self.basedir) as qtemp:
            qtemp = tempfile.TemporaryDirectory("particle", dir=self.basedir)
            temp_outfile = qtemp.name + f'/temp_particle_file_marker-{self.origin_marker}.nc'

            print('running opendrift...')
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")

            self.o.run(duration=duration, #end_time=end_time, 
                       time_step=time_step, #time_step_output=timedelta(hours=24), 
                       outfile=temp_outfile, export_variables=['lon', 'lat', 'z', 'status', 'age_seconds', 'origin_marker'])#, 'sea_floor_depth_below_sea_level', 'ocean_mixed_layer_thickness',])# 'x_sea_water_velocity', 'y_sea_water_velocity', 'x_wind', 'y_wind'])

            elapsed = (T.time() - t_0)
            print("total simulation runtime %s" % timedelta(minutes=elapsed/60)) 

            if hasattr(self.o, 'discarded_readers'):
                logger.warning(f'Readers {self.o.discarded_readers} were discarded. Particle transport will be affected')

            #### A BIT OF POST-PROCESSING ####
            print('writing to netcdf...')

            _ps = xr.open_dataset(temp_outfile) # open temporary file
            #print('time len before processing', len(_ps.time))

            # keep 'inactive' particles visible (i.e. particles that have beached or gotten stuck on seafloor)
            ps = _ps.where(_ps.status>=0).ffill('time') 
            #print('time len after ffill inactive', len(ps.time))

            # write useful attributes
            ps = ps.assign_attrs({'total_bounds': seeding_poly.total_bounds, 'start_time': t0, 'duration_days': duration_days, 'pnum': pnum, 'hdiff': hdiff,
                                  #'tseed': self.tseed.total_seconds(), 
                                  'tstep': tstep.total_seconds(), 'termvel': termvel, 'depth': str(self.depth), 'seeding_shapefile': str(self.seeding_shapefile),
                                  'poly_path': str(self.poly_path), 'opendrift_log': str(self.o)}) 


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

    # TODO: when there are discrepancies in the grids, the integral does not match 
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
        
        _use = rxr.open_rasterio(use_path).rio.reproject('epsg:4326', nodata=0).sortby('x').sortby('y').isel(band=0).drop('band').sel(x=slice(poly_bounds[0], poly_bounds[2]), y=slice(poly_bounds[1], poly_bounds[3])).fillna(0) # fillna is needed so i dont get nan values in the resampled sum

        gr = self.raster_grid(res=res, r_bounds=r_bounds)
        
        use = rasterhd2raster(_use, gr) # resample the use raster on our grid, with user-defined res and crs
        use = use.where(use>=0,0) # rasterhd2raster sometimes gives small negative values when resampling. I am filling those with 0. 

        use_value = use.sel(x=self.ds.isel(time=0).lon, y=self.ds.isel(time=0).lat,  method='nearest')
        
        print(f'number of trajectories with use value = 0 {use_value.where(use_value==0).count().data}')
        print(f'number of trajectories with use value != 0 {use_value.where(use_value!=0).count().data}')
        
        return use_value 

    def assign_weight(self, weight=1):
        """
        Add a weight variable to ds. If 
        """
        # TODO. produce weight variable from floats, arrays, shapefiles, rasters
        
        ds = self.get_ds

        print(f'Adding weight variable to ds...')
        w = np.ones_like(self.ds.lon)*weight # longitude is always going to be available as a variable, so taking it as reference for the shape
        ds = ds.assign({'weight': (('trajectory', 'time'), w.data)}) 
        
        return ds

    def normalize_weight(self, weight, res, r_bounds=None):
        # dividing each trajectory weight by the number of particles that were in the same bin at t0
        self.ds = self.get_bin_n(res=res, t=0, r_bounds=r_bounds)
        print('weight_by_bin...')
        weight_by_bin = weight.groupby(self.ds.isel(time=0).bin_n_t0)
        print('done.')
        counts_per_bin = self.ds.isel(time=0).bin_n_t0.groupby(self.ds.isel(time=0).bin_n_t0).count()
        print('counts_per_bin...')
        normalized_weight = weight_by_bin/counts_per_bin 
        print('done.')
        return normalized_weight

    def set_weights(self, res=None, r_bounds=None, weight=1, use_path=None, emission=None, decay_coef=0, normalize=False, assign=False):
        
        if use_path is not None:
            # extract value of use at starting position of each trajectory
            use_value = self.use_by_traj(use_path=use_path, res=res, r_bounds=r_bounds)
            weight = use_value#/len(self.ds.time) # dividing use weight by the number of timesteps for conservation

        if emission is not None:
            self.emission = emission * self.tstep.seconds / timedelta(days=1).total_seconds() # convert to amount of pressure per my timestep
            logger.warning(f'Converted emission from {emission} per day to {self.emission} per timestep.')
            weight = weight*self.emission
        
        if decay_coef != 0:
        # decay rate. default is no decay, but this is still useful to give weight the correct shape
            print(f'computing decay rate function with decay coefficient k = {self.decay_coef}...')
            y = self.decay_rate(k=decay_coef) # default value = 0 means there is no decay
            weight = weight*y

        # broadcasting to correct shape
        weight = xr.DataArray(weight).broadcast_like(self.ds.lon)
        
        if normalize is True:
            weight = self.normalize_weight(weight, res=res, r_bounds=r_bounds)
            print('weight normalized by number of particles in bin at t0.')

        # ASSIGN needs to be LAST, otherwise would be assigning the wrong weight
        if assign is True:
            self.ds = self.assign_weight(weight)

        return weight
    


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
        _bins = np.zeros((len(grid),2)) 
        #print(_bins.shape)
        _bins[:,0] = np.array(grid.centroid.x) 
        _bins[:,1] = np.array(grid.centroid.y)  
        func = lambda plon, plat: np.abs(_bins-[plon,plat]).sum(axis=1).argmin()
        #print('bin_n ufunc')
        try:
            print(f'calculating bin number at timestep {t}')
            ds['bin_n_t0'] = xr.apply_ufunc(func, ds.isel(time=t).lon, ds.isel(time=t).lat, vectorize=True, dask='parallelized')
            print('bin_n_t0 done.')
        except:
            logger.warning(f'Calculating bin_n at all timesteps {t}.')
            ds['bin_n'] = xr.apply_ufunc(func, ds.lon.chunk({'trajectory': len(ds.trajectory)/10}), ds.lat.chunk({'trajectory': len(ds.trajectory)/10}), vectorize=True, dask='parallelized')
            
        return ds

    def get_histogram(self, res, r_bounds=None, weight=1, normalize=False, assign=False, dim=['trajectory', 'time'], block_size='auto', use_path=None, emission=None, decay_coef=0):
        '''
        "Density trend (DT). The particle DT reflects the number of particles that have visited each grid cell during a certain time interval."
        from Stanev et al., 2019
        
        res : float
            cell size, aka desired resolution of output map
        weighted : bool, optional
            whether the histogram should be weighted or not
        Parameters
        ----------
        
        '''

        if self.reps > 1:
            logger.warning(f'this run contains {self.reps} reps. to get histogram of aggregated reps, use .run() method. get_histogram() will only work on the last rep.')

        self.polygon_grid(res, r_bounds=r_bounds)
        xbin = self._x_e
        ybin = self._y_e

        weights = self.set_weights(res=res, r_bounds=r_bounds, weight=weight, normalize=normalize, use_path=use_path, emission=emission, decay_coef=decay_coef, assign=assign)
        print('set_weights done.')
        
        #NOTE: NaNs in weights will make the weighted sum as nan. To avoid this, call .fillna(0.) on your weights input data before calling histogram().
        h = histogram(self.ds.lon, self.ds.lat, bins=[xbin, ybin], dim=dim, weights=weights.fillna(0.), block_size=block_size) 
        # block_size='auto' was giving division by 0 error, which is a known bug: https://github.com/xgcm/xhistogram/issues/16
        #NOTE: NaNs in weights will make the weighted sum as nan. To avoid this, call .fillna(0.) on your weights input data before calling histogram().
            
        h = h.transpose().rename({'lon_bin':'x', 'lat_bin':'y'}) # important to not have it where(h>0) otherwise it misses values when summing
        
        return h


    def ppi(self, res, use_path=None, emission=None, r_bounds=None, decay_coef=0, normalize=True): 
        '''
        Calculates ppi of pressure per grid-cell at any given time, based on given raster of use density. 
        ppi is calculated by assigning to each trajectory a weight that is equal to the use intensity at the trajectory's starting position.
        To ensure conservation over time, the weight at each timestep is equal to value_of_use/number_of_timesteps/number_of_trajectories_starting_from_that_cell. 

        Parameters
        ----------
        res : float
        
        use_path : str
            Path of file representing density of human activity acting as pressure source. Used to assign  'weights' to trajectories in histogram calculation. 
        emission : float, optional
            amount of pressure released per day by use in use_path 
        r_bounds : list, optional

        decay_coef : float, optional

        normalize : bool, optional
        
        '''
        if r_bounds is None:
            r_bounds = self.extent # take whole basin
        
        if use_path is None:
            logger.info('no use_path provided. calculating ppi from unity-use.')
            self.raster_grid(res=res, r_bounds=r_bounds).rio.to_raster(self.basedir / 'unity-use.tif') # unity weight use grid
            self.use_path = Path (self.basedir / 'unity-use.tif')
        
        # if emission is not None:
        #     self.emission = emission * self.tstep.seconds / timedelta(days=1).total_seconds() # convert to amount of pressure per my timestep
        #     logger.warning(f'Converted emission from {emission} per day to {self.emission} per timestep.')
            
        #weights = self.set_weights(res=res, r_bounds=r_bounds, use_path=use_path, decay_coef=decay_coef, normalize=normalize, assign=True)

        r = self.get_histogram(res=res, r_bounds=r_bounds, normalize=False, block_size=len(self.ds.time), use_path=use_path, emission=emission).assign_attrs({'use_path': use_path, 'emission':emission})

        return r


    # this is a histogram with a "distinct" on trajectories. i.e. if a particle stays in same cell for multiple timesteps, it doesn't get doublecounted.
    # note that this method fails if a trajectory goes back to same cell after a period of time. 
    
    def _traj_per_bin(self, res, r_bounds=None, use_path=None):#, weighted=False):
        counts_per_cell = self.get_histogram(res, weight=1, dim=['time']) # this gives me, for each trajectory, the count of how many tsteps it has spent in each cell
        
        if use_path is not None:
            weights = self.set_weights(res=res, r_bounds=r_bounds, use_path=use_path, normalize=False, assign=True)

        # this way, it will take the weight 
        try:
            h = self.get_histogram(res, weight=self.ds.weight.fillna(0.), dim=['time'], block_size=len(self.ds.trajectory)) 
        except:
            h = self.get_histogram(res, dim=['time'], block_size=len(self.ds.trajectory)) 
            logger.warning('No weight variable was found. Calculating unweighted histogram.')
            #h = self.get_histogram(res, weight=self.ds.weight.fillna(0.), dim=['time'])
        
        tpb = h/counts_per_cell



        return tpb.fillna(0.).sum('trajectory')

    def traj_density(self, res, r_bounds=None): # similar to emodnet route density
        w = self.set_weights(1)
        self.ds = self.get_bin_n(res=res, t='all')
        w_distinct = xr.apply_ufunc(traj_distinct, self.ds.bin_n.chunk(chunks={'time': -1, 'trajectory': int(len(self.ds.trajectory)/10)}), w.chunk(chunks={'time': -1, 'trajectory': int(len(self.ds.trajectory)/10)}), input_core_dims = [['time'],['time']], output_core_dims = [['time']], vectorize=True, dask='parallelized')
        h_distinct = self.get_histogram(res=res, r_bounds=r_bounds, weight=w_distinct, block_size='auto')
        return h_distinct


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

        print(path)
        r.rio.to_raster(path, nodata=0) 
        #h.to_netcdf(path)
        
        print(f'Wrote tiff to {path}.')
        pass
    

    def single_run(self, pnum, start_time, duration, res, tstep=timedelta(hours=4), r_bounds=None, s_bounds=None, seeding_radius=0, beaching=False, hdiff=10, decay_coef=0, use_path=None, use_label='unity-use', output_dir=None, save_tiffs=False, thumbnail=None, loglevel=40, make_dt=True, make_td=True):
        '''
        Compute trajectories and producing raster maps of ppi.
        
        Parameters
        ----------
        pnum : intf
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
        self.res = res

        # this method runs its own cache 
        self.get_trajectories(pnum=pnum, start_time=start_time, tstep=tstep, duration_days=duration, s_bounds=s_bounds, seeding_radius=seeding_radius, beaching=beaching, hdiff=hdiff, loglevel=loglevel)

        # # create dataset where all outputs will be stored
        self.output = xr.Dataset()
        
        self.ppi_cache = PMARCache(Path(self.basedir) / f'ppi-{use_label}')
        if output_dir is not None:
            self.ppi_cache = PMARCache(output_dir['temp_ppi_output'])
        ppi_path, ppi_exists = self.ppi_cache.raster_cache(context=self.context, res=res, poly_path=self.poly_path, pnum=pnum, ptot=None, start_time=start_time, duration=duration, reps=None, tshift=None, use_path=use_path, use_label=use_label, decay_coef=decay_coef, r_bounds=r_bounds)#, aggregate=aggregate, depth_layer=depth_layer, z_bounds=z_bounds, particle_status=particle_status, traj_dens=traj_dens)
        print(f'##################### ppi_exists = {ppi_exists}')
        print(f'##################### ppi_path = {ppi_path}')
        # calculate ppi

        if ppi_exists == True:
            self.output['ppi'] = rxr.open_rasterio(ppi_path).isel(band=0)
        else:
            ppi = self.ppi(use_path=use_path, res=res, r_bounds=r_bounds, decay_coef=decay_coef)

            self.output['ppi'] = ppi.rename({'x':'lon', 'y':'lat'}) # changing coordinate names because there was an issue with precision. original dataset coords have higher precision than coords in raster 

            self.write_tiff(ppi, path=ppi_path)

        return self.output

    
    def sum_reps(self, rep_path):
        '''
        Sum ppi rasters of single reps, return summed raster. 
    
        Parameters 
        ----------
        rep_path : list
            list of paths to each rep 
        '''
        if self.reps == 1:
            r1 = rxr.open_rasterio(rep_path)
        else:
            for idx, rep in enumerate(rep_path):
                if idx == 0:
                    r1 = r0 = rxr.open_rasterio(rep) 
                else:
                    r0 = r1
                    r1 = r0 + rxr.open_rasterio(rep)
        
        return r1
    
    def run(self, ptot, reps, tshift, duration=30, start_time='2019-01-01', tstep=timedelta(hours=4), res=0.04, r_bounds=None, s_bounds=None, seeding_radius=0, beaching=False, use_path=None, use_label='unity-use', hdiff=10, decay_coef=0, loglevel=40, make_dt=True, make_td=True):
        '''
        Compute trajectories and produce ppi raster over required number of reps. 
    
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

        ppi_output_dir = Path(self.basedir) / f'ppi-{use_label}'

        self.ppi_cache = PMARCache(ppi_output_dir)
             
        ppi_path, ppi_exists = self.ppi_cache.raster_cache(context=self.context, res=res, poly_path=self.poly_path, pnum=None, ptot=ptot, start_time=start_time, duration=duration, reps=reps, tshift=tshift, use_path=use_path, use_label=use_label, decay_coef=decay_coef, r_bounds=r_bounds)#, aggregate=aggregate, depth_layer=depth_layer, z_bounds=z_bounds, particle_status=particle_status, traj_dens=traj_dens)

        self.output = xr.Dataset()
        
        if ppi_exists == True:
            print('raster ppi file with these configurations already exists. see self.output')
            self.output['ppi'] = rxr.open_rasterio(ppi_path)

        for n in range(0, reps):
            #print(f'Starting rep #{n+1}...')
            start_time_dt = datetime.strptime(start_time, '%Y-%m-%d')+timedelta(days=tshift)*n #convert start_time into datetime to add tshift
            rep_start_time = start_time_dt.strftime("%Y-%m-%d") # bring back to string to feed to opendrift
            
            logger.warning(f'Starting rep #{n} with start_time = {rep_start_time}')
            
            rep_id = n # rep ID is maybe a better name than origin_marker! # self.rep_id
            # this will have to go as an attribute in ds too, useful for plotting
            
            pnum = int(np.round(ptot/reps)) #  ptot should be split among the reps
            
            self.single_run(pnum=pnum, duration=duration, tstep=tstep, start_time=rep_start_time, res=res, r_bounds=r_bounds, s_bounds=s_bounds, seeding_radius=seeding_radius, beaching=beaching, use_path=use_path, use_label=use_label, hdiff=hdiff, decay_coef=decay_coef, save_tiffs=True, make_dt=make_dt, make_td=make_td)#output_dir = {'dt_output': dt_output_dir, 'rt_output': rt_output_dir, 'c_output': c_output_dir}, loglevel=loglevel)
            logger.warning(f'Done with rep #{n}.')

        #if use_path:
        if reps>1:
            rep_ppi_path = glob.glob(f'{ppi_output_dir}/*.tif')
            print(f'############################## rep_ppi_path = {rep_ppi_path}')
            self.output['ppi'] = ppi = self.sum_reps(rep_ppi_path)
        
        self.write_tiff(self.output['ppi'], path=ppi_path)
        thumb_ppi_path = str(ppi_path).split('.tif')[0]+'.png'
        self.plot(self.output['ppi'], title=use_label, savepng=thumb_ppi_path)
        print('DONE.')
        # else:
        #     print('No use_path provided. Returning residence time only.')

        pass
        
    
    def plot(self, raster, title=None, xlim=None, ylim=None, cmap=spectral_r, shading='flat', vmin=None, vmax=None, norm=None, coastres='10m', proj=cartopy.crs.epsg(3857), transform=cartopy.crs.PlateCarree(), dpi=120, figsize=[10,7], rivers=False, savepng=None):
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
        
        ### this is actually creating problems in plots, commenting for now
        # drop nan values to have cleaner thumbnails
        #lon_var = [varname for varname in r.coords if "lon" in varname][0] # find name of lon variable
        #lat_var = [varname for varname in r.coords if "lat" in varname][0] # find name of lat variable

        r = raster.where(raster>0)
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

        im = r.plot(vmin=vmin, vmax=vmax, norm=norm, shading=shading, cmap=cmap, add_colorbar=False, transform=transform)
        #cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        cax = fig.add_axes([.92, 0.2, 0.02, 0.6])
        cbar = plt.colorbar(im, cax=cax, extend='max')
        cbar.set_label('uom from input layer', rotation=90)

        ax.set_title(title)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)


        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        if savepng is not None:
            #Path(self.basedir / 'thumbnails').mkdir(parents=True, exist_ok=True) # useful for geoplatform to keep them in same dir as raster
            plt.savefig(str(savepng), dpi=160, bbox_inches='tight')

        return fig, ax

    
    def scatter(self, t=None, xlim=None, ylim=None, s=1, alpha=1, c='age', cmap='rainbow', coastres='10m', proj=cartopy.crs.epsg(3857), dpi=120, figsize=[10,7], rivers=False, save_to=None):
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

        im = ax.scatter(ds.lon, ds.lat, s=s, c=c, cmap=cmap, alpha=alpha, transform=cartopy.crs.PlateCarree())
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


    
