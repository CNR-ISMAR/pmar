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
from opendrift.models.plastdrift import PlastDrift
from opendrift.models.chemicaldrift import ChemicalDrift
#import copernicusmarine
from opendrift.readers.reader_copernicusmarine import Reader
from opendrift.readers.reader_netCDF_CF_generic import Reader as GenericReader
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
from pmar.pmar_utils import traj_distinct, check_particle_file, get_marine_polygon, make_poly, harmonize_use
#from xgcm import Grid

logger = logging.getLogger('pmar')
#logger.setLevel(logging.DEBUG)
# sh = logging.StreamHandler()
# sf = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
# sh.setFormatter(sf)
# logger.addHandler(sh)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
        
        
class PMAR(object): 
    """
    Developed at CNR ISMAR in Venice.

    ...

    Attributes
    ----------


    Methods
    -------


    """


    def __init__(self, context='global', readers=None, pressure='generic', chemical_compound=None, basedir='pmar_output', localdatadir = None, particle_path=None, netrppi_path=None, loglevel=10):
        """
        Parameters
        ---------- 
        context : str or list, optional
            If str, an ocean basin for which CMEMS readers are available. Otherwise, a list of paths of files to be imported as OpenDrift readers. Default is 'global'
        pressure : str, optional

        chemical_compound : str, optional
            if pressure is 'chemical', a chemical_compound can be specified to be initialised in ChemicalDrift
        basedir : str, optional
            path to the base directory where all output will be stored. Default is to create a directory called 'pmar_output' in the current directory.
        localdatadir : str, optional
            path to directory where input data (ocean, atmospheric) is stored. Default is None
        particle_path : str, optional
            path to netcdf file containing output of OpenDrift simulation. Default is None. If a particle_path is given in initialisation,    
        """


        Path(basedir).mkdir(parents=True, exist_ok=True)
        self.basedir = Path(basedir)
        if particle_path is None:
            self.particle_path = [] # i can import an existing particle_path
        else:
            self.particle_path = particle_path
        #self.ds = None. commented otherwise try: self.ds except: does not work
        self.o = None
        self.seeding_id = 0 # gives error in opendrift if set to None
        #self.netrc_path = netrc_path
        self.tstep = None
        self.pnum = None
        self.termvel = 0 #1e-3
        self.decay_coef = 0
        self.outputdir = None
        #self.pressure = pressure
        self.chemical_compound = chemical_compound
        self.localdatadir = localdatadir
        self.particle_status = None
        self.seedings = 1
        self.tshift = None
        self.cache = PMARCache(Path(basedir) / 'cachedir')
        self.raster_path = None
        self.grid = None
        self.res = None
        self.weight = None
        self.study_area = None
        self.res = None
        self.ds = None
        self.ppi_path = []
        self.seeding_shapefile = None # can be point, line or polygon
        self.seed_within_bounds = None # if no seeding_shapefile is given, user can choose to give lon, lat bounds to seed within
        self.loglevel = loglevel
        self.readers = readers
        
        # set up logger 
        logformat = '%(asctime)s %(levelname)-7s %(name)s:%(lineno)d: %(message)s'
        datefmt = '%H:%M:%S'        
        sh = logging.StreamHandler()
        sf = logging.Formatter(logformat)
        sh.setFormatter(sf)

        if (logger.hasHandlers()):
            logger.handlers.clear()        
        logger.addHandler(sh)
        if loglevel < 10:  # 0 is NOTSET, giving no output
            loglevel = 10
        logger.setLevel(loglevel)

        self.set_pressure(pressure)

        # define context
        if context is not None:
            try: 
                self.context = self.get_context(str(context))
            except:
                # alternatively, provide paths to readers manually
                self.context = {'readers': dict(zip(range(0,len(context)),context))}
        
        # if a path to a shapefile is given to be used for seeding, read it and save it in the pmar_output/polygons dir
        # why save it locally????
        Path(self.basedir / 'polygons').mkdir(parents=True, exist_ok=True)            
        
        # this (if still needed?) should be a separate method
        # if particle_path is given, retrieve number of seedings and load ds
            # gather number of seedings from origin marker. THIS IS A PROBLEM IF THERE ARE MORE BATCHES IN SAME FOLDER. 
        if type(self.particle_path) is list: 
                if self.particle_path:
                    self.seedings = len(particle_path)
                


    def set_pressure(self, pressure):
        pressures = {'generic': {'opendrift_module': OceanDrift},
                     'plastic': {'opendrift_module': PlastDrift},
                  #   'bacteria': {'opendrift_module': OceanDrift, 'termvel':0, 'decay_coef':1},
                     'chemical': {'opendrift_module': ChemicalDrift},
                     'oil': {'opendrift_module': OpenOil},
                     'metal': {'opendrift_module': ChemicalDrift}}        

        self.pressure = pressure
        self.opendrift_module = pressures[pressure]['opendrift_module']
        logger.info(f'initializing pressure {self.pressure} with opendrift module {self.opendrift_module}')

    def get_context(self, context):
        """
        Parameters
        ---------- 
        context : str
            One of 'global', 'med', 'black sea', 'baltic'. Returns strings to relevant Copernicus products to add as OpenDrift readers. 
        """
        
        contexts = {'global': 
                        {'readers': 
                         {'currents': 'cmems_mod_glo_phy_my_0.083deg_P1D-m', # global physics reanalysis, daily, 1/12° horizontal resolution, 50 vertical levels, 1 Jan 1993 to 2021-06-30
                          'winds': 'cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H', # L4 sea surface wind and stress fields 0.125° res, hourly, 1 Jun 1994 to 21 Nov 2024
                          'bathymetry':'cmems_mod_glo_phy_my_0.083deg_static', # global physics reanalysis static, same as currents
                          #'mixed-layer': 'cmems_mod_glo_phy_my_0.083deg_P1D-m', # same as currents
                          'stokes': 'cmems_mod_glo_wav_my_0.2deg_PT3H-i', # global ocean waves reanalysis, 3-hourly, 1 Jan 1980 to 31 Jan 2025
                          'spm': 'cmems_obs-oc_glo_bgc-transp_my_l4-multi-4km_P1M',
                         },
                        'extent': [-180, -80, 179.92, 90],
                        'polygon': None# get_marine_polygon()
                        },
                        
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


    def xgrid(self, res, study_area=None, crs='epsg:4326'):
        xmin, ymin, xmax, ymax = study_area
        cols = np.arange(xmin, xmax + res, res)            
        rows = np.arange(ymin, ymax + res, res)

        # outer cell edges
        x_e = cols 
        y_e = rows
        
        # cell centres
        x_c = (x_e + res/2)[:-1]
        y_c = (y_e + res/2)[:-1]     

        xgrid = xr.Dataset(coords={'x_e':x_e, 'y_e':y_e, 'x_c':x_c, 'y_c':y_c})

        xgrid['dx'] = xgrid['dy'] = res

        Xc, Yc = np.meshgrid(xgrid.x_c, xgrid.y_c)
        xgrid = xgrid.assign({'Xc': (('y_c', 'x_c'), Xc), 'Yc': (('y_c', 'x_c'), Yc)}) #dataarrays 

        xgrid = xgrid.assign({'bin_n': (('y_c','x_c'), np.arange(0,Xc.size).reshape((len(y_c), len(x_c)))
)})
        
        xgrid.rio.write_crs(crs, inplace=True)
        
        # grid = Grid(xgrid, 
        #             coords={"X": {"center": "x_c", "outer": "x_e"}, 
        #                    'Y': {"center": "y_c", "outer": "y_e"}}, 
        #             metrics = {
        #     ('X',): ['dx'], # X distances
        #     ('Y',): ['dy']},
        #            periodic=False)

        return xgrid

    @property
    def get_ds(self):
        # this doesnt actually work with seedings, because self.ds always exists, but then it is only the last rep.
        # should discriminate e.g. by comparing self.ptot with self.ds.trajectory.
        # if they don't match, i am probably only looking at the last rep. 
        #try:
         #   self.ds
            #logger.info('get_ds: returning previously calculated ds.')
        #except:
        
        #if not self.ds:
        if self.particle_path:
            logger.info('get_ds: retrieving ds from particle_path.')
            
            if type(self.particle_path) is str or len(self.particle_path) == 1:
            #    ds = xr.open_dataset(self.particle_path, chunks={'trajectory': 10000, 'time':1000})
            
                partial_func = None
            else:
                partial_func = partial(self._preprocess, correct_len = self.find_correct_len())
                
            ds = xr.open_mfdataset(self.particle_path, preprocess=partial_func, concat_dim='trajectory', combine='nested', parallel=True, chunks={'trajectory': 10000, 'time':1000}) # add join='ovverride' to have them realigned
            logger.debug(f'lat = {ds.lat.shape}, lon={ds.lon.shape}, time={ds.time.shape}')
            # if the run contained multiple seedings, ensure trajectories have unique IDs for convenience
            logger.debug(f'self.seedings = {self.seedings}')
            ds['trajectory'] = np.arange(0, len(ds.trajectory)) 
                       
            self.ds = ds
        else:
            logger.error('particle_path is None. Unable to retrieve ds.')
        
        return self.ds

    # Given multiple particle_paths, it returns the maximum time length. 
    # Used for preprocessing function in get_ds to make sure all ds's have same time length, even if opendrift ended early because of all particles beaching.
    def find_correct_len(self):
        '''
        Sometimes opendrift runs end early because e.g. all particles have beached, resulting in seedings with different time lengths. This method finds the maximum time lenght of all seedings.
        '''
        lens = np.array([len(xr.open_dataset(filename).time) for filename in self.particle_path])
        logger.info(f'giving all datasets the same time length of {lens} tsteps...')
        return lens.max()

    def _preprocess(self, ds, correct_len):
        '''
        Sometimes opendrift runs end early because e.g. all particles have beached, resulting in seedings with different time lengths. This method pads all seedings so that they all have same time length (determined with find_correct_len).
        '''
        return ds.pad(pad_width={'time': (0, correct_len-len(ds.time))}, mode='edge')
         

    def _make_poly(self, lon, lat, crs='4326', save_to=True):
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
        
        if save_to is not None:
            poly.to_file(str(save_to), driver='ESRI Shapefile')
            #self.poly_path = str(q)
    
        return poly
    
    def get_trajectories(self, pnum, start_time, season=None, duration_days=30, study_area=None, seeding_shapefile = None, seed_within_bounds=None, z=-0.5, tstep=timedelta(hours=1), hdiff=10, termvel=0, crs='4326', seeding_radius=0, beaching=False, stokes_drift=False):
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
        start_time : str
            The start date of the simulation, in string format 'dd-mm-yyyy'
        season : str, optional
            Season in which simulation should be run. Defines start_time automatically. Default is None
        duration_days : int, optional
            Integer defining the length, or duration, of the particle simulation in days. Default is 30 (to be used for test run)
        seeding_shapefile : str, optional
            Path to a shapefile containing geometries to seed within. Default is None,  i.e. the whole basin is covered.
        seed_within_bounds : list, optional
            Spatial bounds for seeding written as bounds=[x1,y1,x2,y2]. Default is None, i.e. the whole basin is covered
        z : float, optional
            Depth at which to seed particles [m]. Default is -0.5m. 
        tstep : timedelta, optional
            Time step used for OpenDrift simulation. Default is 6 hours
        hdiff : float, optional
            Horizontal diffusivity of particles, in [m2/s]. Default is 10m2/s
        termvel : float, optional
            Terminal velocity representing buoyancy of particles, in [m/s]. Default is 0. Positive means rising up, negative sinking down
        seeding_radius : float, optional
            Opendrift "radius".
        beaching : bool, optional

        stokes_drift : bool, optional
        
        crs : str, optional
            EPSG string for polygon. Default is 4326   
        loglevel : int, optional
            OpenDrift loglevel. Set to 0 (default) to retrieve all debug information.
            Provide a higher value (e.g. 20) to receive less output.
        """
        
        t_0 = T.time()   
         
        # if termvel is None:
        #     termvel = self.termvel
        
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

        if seed_within_bounds is not None:
            lon = seed_within_bounds[0::2]
            lat = seed_within_bounds[1::2]  
            poly_path = f'polygon-crs_epsg:{crs}-lon_{np.round(lon[0],4)}_{np.round(lon[1],4)}-lat—{np.round(lat[0],4)}_{np.round(lat[1])}.shp'
            q = self.basedir / 'polygons' / poly_path
            make_poly(lon, lat, crs=crs, save_to=str(q))
            self.seeding_shapefile = str(q)
            logger.info(f'seeding within bounds {lon},{lat}')
        else: 
            # not needed anymore. seed_from_shapefile accepts point features
            # if the seeding shapefile contains points, add a small buffer to turn them into polygons, to use opendrift.seed_from_shapefile
            self.seeding_shapefile = seeding_shapefile
            if np.unique(gpd.read_file(self.seeding_shapefile).geometry.type) == 'Point':
                seeding_poly = gpd.read_file(self.seeding_shapefile)
                seeding_poly['geometry'] = seeding_poly.geometry.to_crs('3857').geometry.buffer(2000).to_crs(4326)
                new_seeding_shapefile = Path(self.basedir / 'polygons' / ('buffered_'+self.seeding_shapefile.split('/')[-1]))
                seeding_poly.to_file(new_seeding_shapefile)
                self.seeding_shapefile = str(new_seeding_shapefile)
                logger.debug(f'Added buffer to {np.unique(seeding_poly.geometry.type)} type geometry in self.seeding_shapefile to allow seed_from_shapefile')

            logger.info(f'seeding particles evenly within shapefile {self.seeding_shapefile}')
            #self.seeding_shapefile = seeding_shapefile
        
        # path to write particle simulation file. also used for our 'cache'    
        t0 = start_time.strftime('%Y-%m-%d')
        t1 = end_time.strftime('%Y-%m-%d')
        
            
        # initialise OpenDrift object
        self.o = self.opendrift_module(loglevel=40)
            
        
        # some OpenDrift configurations
        if beaching:
            self.o.set_config('general:coastline_action', 'stranding') # behaviour at coastline. 'stranding' means beaching of particles is allowed
        else:
            self.o.set_config('general:coastline_action', 'previous') # behaviour at coastline. 'previous' means particles that reach the coast do not get stuck
        
        self.o.set_config('drift:horizontal_diffusivity', hdiff)  # horizontal diffusivity
        self.o.set_config('drift:advection_scheme', 'euler') # advection schemes (default is 'euler'). other options are 'runge-kutta', 'runge-kutta4'
        
        # ChemicalDrift configs - ## DO NOT CHANGE ORDER OF CONFIGS ## 
        if self.pressure in ['chemical', 'metal']:
            self.o.set_config('drift:vertical_mixing', True) # OpenDrift default is False, should be True for ChemicalDrift
            self.o.set_config('vertical_mixing:diffusivitymodel', 'windspeed_Large1994')
            self.o.set_config('vertical_mixing:background_diffusivity',0.0001)
            #self.o.set_config('vertical_mixing:timestep', 60) is default
            #o.set_config('drift:horizontal_diffusivity', 10)

            # leaving default values
            #self.o.set_config('chemical:particle_diameter',25.e-6)  # m
            #self.o.set_config('chemical:particle_diameter_uncertainty',1.e-7) # m
            
            # Parameters from radionuclides (Magne Simonsen 2019)
            #self.o.set_config('chemical:sediment:resuspension_depth',1.) is default
            self.o.set_config('chemical:sediment:resuspension_depth_uncert',0.1) # there are default values for these 
            self.o.set_config('chemical:sediment:resuspension_critvel',0.15)
            self.o.set_config('chemical:sediment:desorption_depth',1.)
            self.o.set_config('chemical:sediment:desorption_depth_uncert',0.1)
            
            #if self.pressure == 'metal':
              #  self.o.set_config('chemical:transformations:volatilization', False) default is False
                #self.o.set_config('chemical:transformations:degradation', False)  default is False
            if self.pressure == 'chemical':
                self.o.set_config('chemical:transformations:volatilization', True) 
                self.o.set_config('chemical:transformations:degradation', True) 
                self.o.set_config('chemical:transformations:degradation_mode', 'OverallRateConstants')
            
            self.o.init_chemical_compound(self.chemical_compound) # includes a selection of PAHs and metals
            logger.info(f'initialising chemical compound {self.chemical_compound}')

            # by default, all is dissolved at seeding. maybe parametrise in future
            LMM_fraction = 1
            self.o.set_config('seed:LMM_fraction',LMM_fraction)
            self.o.set_config('seed:particle_fraction',1-LMM_fraction)

            # these have to be here in this order otherwise it gives error
            self.o.init_species() 
            self.o.init_transfer_rates()
                            

        logger.info('adding landmask...')
        # landmask from cartopy (from "use shapefile as landmask" example on OpenDrift documentation)
        shpfilename = shpreader.natural_earth(resolution='10m',
                                category='cultural',
                                name='admin_0_countries')
        reader_landmask = reader_shape.Reader.from_shpfiles(shpfilename)
        
        self.o.add_reader([reader_landmask])
        self.o.set_config('general:use_auto_landmask', False)  # Disabling the automatic GSHHG landmask
        
        logger.info('adding readers...')            
        
        if self.readers:
            # add local datasets as readers. added for sl application
            for id,r in enumerate(self.readers):
                reader = GenericReader(r, name=f'cmems_{id}')
                self.o.add_reader(reader)#, variables=['x_sea_water_velocity', 'y_sea_water_velocity'])

        else:
        #if self.context:
            readers = []
            # add predefined readers from context
            for var in self.context['readers']:
                readers.append(Reader(dataset_id=self.context['readers'][var]))
            self.o.add_reader(readers) # add all readers for that context.
            #self.o.add_readers_from_list(self.context['readers'].values()) # this will add readers lazily, and only read them if useful. 
        
        # uncertainty
        #self.o.set_config('drift:current_uncertainty', .1)
        #self.o.set_config('drift:wind_uncertainty', 1)

        ##### SEEDING #####
        logger.info('seeding particles...')
        np.random.seed(None)            

        Path(self.basedir / 'polygons').mkdir(parents=True, exist_ok=True)
        
        # if simulation is 3D, set 3D parameters (terminal velocity, vertical mixing, seafloor action) and seed particles over polygon
        #logger.debug(f'self.seeding_shapefile = {self.seeding_shapefile}')

        self.termvel = termvel
        self.o.set_config('vertical_mixing:diffusivitymodel', 'windspeed_Large1994')
        #self.o.set_config('general:seafloor_action', 'deactivate') # not applicable in chemical drift
        self.o.set_config('drift:vertical_mixing', True)        
        self.o.seed_from_shapefile(shapefile=self.seeding_shapefile, number=pnum, time=start_time, 
                                   terminal_velocity=termvel, z=z, origin_marker=self.seeding_id, radius=seeding_radius)

        
        # run simulation and write to temporary file
        #with tempfile.TemporaryDirectory("particle", dir=self.basedir) as qtemp:
        qtemp = tempfile.TemporaryDirectory("particle", dir=self.basedir)
        temp_outfile = qtemp.name + f'/temp_particle_file_marker-{self.seeding_id}.nc' # probably not needed anymore

        logger.info('running opendrift...')
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")

        self.o.run(duration=duration, #end_time=end_time, 
                   time_step=time_step, #time_step_output=timedelta(hours=24), 
                   outfile=temp_outfile, 
                   export_variables=['lon', 'lat', 'z', 'status', 'age_seconds', 'origin_marker', 'specie', 'sea_floor_depth_below_sea_level'])#, 'ocean_mixed_layer_thickness', 'x_sea_water_velocity', 'y_sea_water_velocity', 'x_wind', 'y_wind'])

        elapsed = (T.time() - t_0)
        logger.info("total simulation runtime %s" % timedelta(minutes=elapsed/60)) 

        if hasattr(self.o, 'discarded_readers'):
            logger.warning(f'Readers {self.o.discarded_readers} were discarded. Particle transport will be affected')

        #### A BIT OF POST-PROCESSING ####
        logger.info('writing to netcdf...')

        _ps = xr.open_dataset(temp_outfile) # open temporary file
        #_ps = self.o.result # writing this to netcdf gives strange error TypeError: Invalid value for attr 'dtype': <class 'numpy.int32'>.
        #print('time len before processing', len(_ps.time))

        # keep 'inactive' particles visible (i.e. particles that have beached or gotten stuck on seafloor)
        ps = _ps.where(_ps.status>=0).ffill('time') 
        #print('time len after ffill inactive', len(ps.time))

        # write useful attributes
        ps = ps.assign_attrs({#'extent': self.extent, 
                              'start_time': t0, 'duration_days': duration_days, 'pnum': pnum, 'hdiff': hdiff,
                              #'tseed': self.tseed.total_seconds(), 
                              'tstep': tstep.total_seconds(), 'termvel': termvel, 'seeding_shapefile': self.seeding_shapefile,
                              #'poly_path': self.poly_path, 
                              'opendrift_log': str(self.o)}) 


        #ps.to_netcdf(str(file_path))
        #logger.info(f"done. NetCDF file '{str(file_path)}' created successfully.")

        
        self.ds = ps
        
        if 'qtemp' in locals():
            Path(temp_outfile).unlink()
            os.rmdir(qtemp.name)

        
        logger.info(f'particle simulation lat = {ps.lat.shape}, lon={ps.lon.shape}, time={ps.time.shape}')

        
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
            logger.warning("Unrecognised status. Status must be one of 'active', 'stranded', 'seafloor'. No filtering by status was carried out.")    
        logger.info(f'Rasterizing only {status} particles.')
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
        
        logger.info(f'Interpolating time on ds. New timestep = {new_timestep} hours...')
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
    def use_by_traj(self, use, res, study_area=None):
        '''
        Select value of use raster at the trajectories' starting positions. 
        Returns the use_value as a DataArray.

        Parameters
        ----------
        use : str
            Path of .tif file representing density of human activity acting as pressure source. Used to assign  'weights' to trajectories in histogram calculation.   
        res : float
            Resolution at which to bin use raster.
        study_area : list, optional
            Spatial bounds for computation of raster over a subregion, written as [x1,y1,x2,y2]. Default is None (bounds are taken from self.)
        '''
        logger.info("sampling use value at particles' starting locations..")
        self.use = use
        
        use_value = use.sel(x=self.ds.isel(time=0).lon, y=self.ds.isel(time=0).lat,  method='nearest')
        
        #logger.info(f'number of trajectories without use value  {use_value.where(use_value==0).count().data}')
        logger.info(f'number of trajectories with non-zero use value  {use_value.where(use_value!=0).count().data}/{len(self.ds.trajectory)}')
        
        return use_value 

    # to deprecate
    def _assign_weight(self, weight=1):
        """
        Add a weight variable to ds. 
        """
        # TODO. produce weight variable from floats, arrays, shapefiles, rasters
        
        ds = self.get_ds

        logger.info(f'Adding weight variable to ds...')
        w = np.ones_like(self.ds.lon)*weight # longitude is always going to be available as a variable, so taking it as reference for the shape
        ds = ds.assign({'weight': (('trajectory', 'time'), w.data)}) 
        logger.info('updating weight variable in ds')
        
        return ds

    def normalize_weight(self, weight, res, study_area=None):
        # dividing each trajectory weight by the number of particles that were in the same bin at t0
        #bin_n_t0 = self.get_bin_n(res=res, t=0, study_area=study_area)
        bin_n = self.get_bin_n(res=res, study_area=study_area)        
        #self.ds['bin_n_t0'] = bin_n_t0
        self.ds['bin_n'] = bin_n
        logger.info('weight_by_bin...')
        weight_by_bin = weight.groupby(self.ds.isel(time=0).bin_n.load()) # added load otherwise groupby raises error
        #logger.info('done.') 
        counts_per_bin = self.ds.isel(time=0).bin_n.groupby(self.ds.isel(time=0).bin_n).count()
        logger.info('counts_per_bin...')
        normalized_weight = weight_by_bin/counts_per_bin 
        #logger.info('done.')
        return normalized_weight

    def set_weights(self, res=None, study_area=None, weight=1, use=None, emission=None, decay_coef=0, normalize=False, assign=False):
        '''
        Associate weights with each trajectory. 
        ORDER OF OPERATIONS MATTERS.
        Weights depend on one or more of the following components: use value in each cell (use must be provided as quantity/day), emission (quantity/day), decay coefficient, number of trajectories starting in the same cell (normalize parameter).  
        '''
            
        if use is not None:
            # extract value of use at starting position of each trajectory
            use_value = self.use_by_traj(use=use, res=res, study_area=study_area)
            weight = use_value#/len(self.ds.time) # dividing use weight by the number of timesteps for conservation

        if emission is not None:
            self.emission = emission * self.ds.tstep / timedelta(days=1).total_seconds() # convert to amount of pressure per my timestep
            logger.info(f'Converted emission from {emission} per day to {self.emission} per timestep.')
            weight = weight*self.emission

        if normalize is True:
            weight = self.normalize_weight(weight=weight, res=res, study_area=study_area)
            logger.info('weight normalized by number of particles in bin at t0.')        
            
        # decay rate. default is no decay, but this is still useful to give weight the correct shape
        logger.info(f'computing decay rate function with decay coefficient k = {self.decay_coef}...')
        y = self.decay_rate(k=decay_coef) # default value = 0 means there is no decay
        weight = weight*y

        if assign is True:
            self.ds = self.ds.assign({'weight': (('trajectory', 'time'), weight.data)}) 
            logger.info('updating weight variable in ds')

        return weight
    


    def get_bin_n(self, res, t=None, study_area=None):
        '''
        Add new variable to ds ('bin_n_t0') containing its "bin number" at timestep t, i.e. a unique identifier corresponding to a specific spatial grid-cell. 
        
        Parameters
        ----------
        t : int
            index of timestep to consider
        '''
      
        ds = self.ds
        if t is not None:
            bin_n = self.grid.bin_n.sel(x_c=self.ds.isel(time=t).lon, y_c=self.ds.isel(time=t).lat, method='nearest')
            logger.info(f'calculated bin number at timestep {t}.')
        else:
            bin_n = self.grid.bin_n.sel(x_c=self.ds.lon, y_c=self.ds.lat, method='nearest')
            logger.info(f'calculated bin number at all timesteps.')
        
        return bin_n

    def get_histogram(self, res, study_area=None, weight=1, normalize=False, assign=True, dim=['trajectory', 'time'], block_size='auto', use=None, emission=None, decay_coef=0):
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

        if self.seedings > 1:
            logger.warning(f'this run contains {self.seedings} seedings. to get histogram of aggregated seedings, use .run() method. get_histogram() will only work on the last rep.')

        xbin = self.grid.x_e
        ybin = self.grid.y_e
        
        #self.get_ds
        
        weights = self.set_weights(res=res, study_area=study_area, weight=weight, normalize=normalize, use=use, emission=emission, decay_coef=decay_coef, assign=assign)
        logger.info('set_weights done.')
        
        #NOTE: NaNs in weights will make the weighted sum as nan. To avoid this, call .fillna(0.) on your weights input data before calling histogram().
        h = histogram(self.ds.lon, self.ds.lat, bins=[xbin, ybin], dim=dim, weights=weights.fillna(0.), block_size=block_size) 
        # block_size='auto' was giving division by 0 error, which is a known bug: https://github.com/xgcm/xhistogram/issues/16
        #NOTE: NaNs in weights will make the weighted sum as nan. To avoid this, call .fillna(0.) on your weights input data before calling histogram().
            
        h = h.transpose().rename({'lon_bin':'x', 'lat_bin':'y'}) # important to not have it where(h>0) otherwise it misses values when summing
        
        return h

    # to be deprecated
    def ppi(self, res, use=None, emission=None, study_area=None, decay_coef=0, normalize=True): 
        '''
        Calculates ppi of pressure per grid-cell at any given time, based on given raster of use density. 
        ppi is calculated by assigning to each trajectory a weight that is equal to the use intensity at the trajectory's starting position.
        To ensure conservation over time, the weight at each timestep is equal to value_of_use/number_of_timesteps/number_of_trajectories_starting_from_that_cell. 

        Parameters
        ----------
        res : float
        
        use : str
            Path of file representing density of human activity acting as pressure source. Used to assign  'weights' to trajectories in histogram calculation. 
        emission : float, optional
            amount of pressure released per day by use in use 
        study_area : list, optional

        decay_coef : float, optional

        normalize : bool, optional
        
        '''
        # if study_area is None:
        #     study_area = self.extent # take whole basin
        
#        if use is None:
 #           logger.info('no use provided. calculating ppi from unity-use.')
  #          self.use = xr.ones_like(self.grid.Xc)
        
        # if emission is not None:
        #     self.emission = emission * self.tstep.seconds / timedelta(days=1).total_seconds() # convert to amount of pressure per my timestep
        #     logger.warning(f'Converted emission from {emission} per day to {self.emission} per timestep.')
            
        #weights = self.set_weights(res=res, study_area=study_area, use_path=use_path, decay_coef=decay_coef, normalize=normalize, assign=True)

        #### NEED TO DO THIS, BUT ONLY FOR LAST REP, IN A LOOP
        # self._particle_path ? 
        #self.ds = self.get_ds
        
        r = self.get_histogram(res=res, study_area=study_area, normalize=True, assign=True, block_size=len(self.ds.time), use=use, decay_coef=decay_coef, emission=emission)#.assign_attrs({'use_path': use_path, 'emission':emission})

        return r


    # this is a histogram with a "distinct" on trajectories. i.e. if a particle stays in same cell for multiple timesteps, it doesn't get doublecounted.
    # note that this method fails if a trajectory goes back to same cell after a period of time. 
    
    def _traj_per_bin(self, res, study_area=None, use=None):#, weighted=False):
        counts_per_cell = self.get_histogram(res, weight=1, dim=['time']) # this gives me, for each trajectory, the count of how many tsteps it has spent in each cell
        
        if use is not None:
            weights = self.set_weights(res=res, study_area=study_area, use=use, normalize=False, assign=True)

        # this way, it will take the weight 
        try:
            h = self.get_histogram(res, weight=self.ds.weight.fillna(0.), dim=['time'], block_size=len(self.ds.trajectory)) 
        except:
            h = self.get_histogram(res, dim=['time'], block_size=len(self.ds.trajectory)) 
            logger.warning('No weight variable was found. Calculating unweighted histogram.')
            #h = self.get_histogram(res, weight=self.ds.weight.fillna(0.), dim=['time'])
        
        tpb = h/counts_per_cell



        return tpb.fillna(0.).sum('trajectory')

    def traj_density(self, res, study_area=None): # similar to emodnet route density
        w = self.set_weights(1)
        self.ds = self.get_bin_n(res=res, t='all')
        w_distinct = xr.apply_ufunc(traj_distinct, self.ds.bin_n.chunk(chunks={'time': -1, 'trajectory': int(len(self.ds.trajectory)/10)}), w.chunk(chunks={'time': -1, 'trajectory': int(len(self.ds.trajectory)/10)}), input_core_dims = [['time'],['time']], output_core_dims = [['time']], vectorize=True, dask='parallelized')
        h_distinct = self.get_histogram(res=res, study_area=study_area, weight=w_distinct, block_size='auto')
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

        #logger.info(path)
        r.rio.to_raster(path, nodata=0) 
        
        logger.info(f'Wrote tiff to {path}.')
        pass
    

    def single_run(self, pnum, start_time, duration, res, tstep=timedelta(hours=1), study_area=None, seeding_shapefile=None, seed_within_bounds=None, z=-0.5, seeding_radius=0, beaching=False, hdiff=10, termvel=0, stokes_drift=False, decay_coef=0, use=None, use_label=None, emission=None, output_dir=None, save_tiffs=False, thumbnail=None, crs='4326'):
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
        use : string, optional
            Path to map of human activities (.tiff) considered as pressure source. Default is None
        decay_coef : float, optional
            Coefficient k of exponential decay function. Default is 0 (no decay)
        output_dir : dict, optional

        '''
        self.res = res

        #### SPATIAL DOMAIN
        # establish study area
        if seeding_shapefile is not None and seed_within_bounds is not None:
            raise ValueError('Please provide either seeding_shapefile or seed_within_bounds, not both.')        
        elif seeding_shapefile is None:
            if study_area is None:
                if seed_within_bounds is None:
                    logger.warning('No study area provided. Taking the whole basin as spatial domain. This may take up a lot of memory.')
                    seed_within_bounds = self.extent
                study_area = seed_within_bounds
            elif study_area is not None:
                if seed_within_bounds is None:
                    seed_within_bounds = study_area
        elif seeding_shapefile is not None:
            study_area = gpd.read_file(seeding_shapefile).total_bounds
        
        self.study_area = study_area
        self.seeding_shapefile = seeding_shapefile 
        self.seed_within_bounds = seed_within_bounds
        logger.debug(f'study_area = {self.study_area}, seed_within_bounds = {self.seed_within_bounds}, seeding_shapefile = {self.seeding_shapefile}')
        
        file_path, file_exists = self.cache.particle_cache(context=self.context, pressure=self.pressure, chemical_compound=self.chemical_compound, seeding_shapefile=seeding_shapefile, pnum=pnum, start_time=start_time, duration_days=duration, seed_within_bounds=seed_within_bounds, seeding_radius=seeding_radius, beaching=beaching, z=z, tstep=tstep, hdiff=hdiff, termvel=termvel, stokes_drift=stokes_drift, seeding_id=self.seeding_id)

        logger.info(f'particle_path exists = {file_exists}, {file_path}')
        logger.debug(f'seeding_shapefile = {self.seeding_shapefile}')        

        self.particle_path.append(file_path) 

        self.grid = self.xgrid(res=res, study_area = study_area) # should be in init?

        
        #if use_path is a shapefile, rasterize it using geocube (NB: only tested with points)
        # if use_path and Path(use_path).suffix == '.shp':
        #     vector_use = gpd.read_file(use_path)
        #     raster_use = rasterize_points_add(vector_use, res=res)
        #     use_path = str(self.basedir)+use_path.split('/')[-1].split('.shp')[0]+'.tif'
        #     raster_use.rio.to_raster(use_path)
        #     logger.debug(f'Rasterized use: {use_path}')
        
        # if a file with that name already exists, simply import it  
        if file_exists == True:
            logger.info(f'Opendrift file with these configurations already exists within {self.basedir}. Importing.') 
            #self.particle_path.append(file_path)
            #self.ds = self.get_ds
            self.seeding_shapefile = xr.open_dataset(self.particle_path[-1]).seeding_shapefile #take only last particle path
            self.ds = xr.open_dataset(self.particle_path[-1])
            #self.poly_path = self.ds.poly_path
            # add weight to ds 
            # this should only be done if the traj file already exists. so i'm assigning the weight to be able to see it but not recalculating it. 


        # otherwise, run requested simulation
        else:
            self.get_trajectories(pnum=pnum, start_time=start_time, tstep=tstep, duration_days=duration, seeding_shapefile=seeding_shapefile, seed_within_bounds=seed_within_bounds, z=z, seeding_radius=seeding_radius, beaching=beaching, hdiff=hdiff)
            self.ds.to_netcdf(str(file_path))
            logger.info(f"done. NetCDF file '{str(file_path)}' created successfully.") 

        if use is None:
            logger.info('no use provided. calculating ppi from unity-use.')
            use = xr.ones_like(self.grid.Xc).rename({'x_c':'x', 'y_c':'y'})
        else:
            use = harmonize_use(use, res, study_area=study_area, like=self.grid.Xc.rename({'x_c':'x', 'y_c':'y'}), tstep=self.ds.tstep) # renaming otherwise geocube does not recognise coords
        self.use = use

        # this should run whether or not there is already a particle or raster cache. so i can see the weight variable in ds.
        self.set_weights(res=res, study_area=study_area, use=use, emission=emission, decay_coef=decay_coef, normalize=True, assign=True)   
        
        # # create dataset where all outputs will be stored
        self.output = xr.Dataset()

        # if use_label is None and use_path is not None:
        #     use_label = str(use_path).split('/')[-1][:10]
            
        self.ppi_cache = PMARCache(Path(self.basedir) / f'ppi-{use_label}')
        
        if output_dir is not None:
            self.ppi_cache = PMARCache(output_dir['temp_ppi_output'])

#        if not study_area:
#            if seed_within_bounds:
#                study_area = seed_within_bounds
#            elif seeding_shapefile:
#                study_area = gpd.read_file(seeding_shapefile).total_bounds
#            else:
#                study_area = self.extent # take whole basin

        
        ppi_path, ppi_exists = self.ppi_cache.raster_cache(context=self.context, pressure=self.pressure, chemical_compound=self.chemical_compound, seeding_shapefile=self.seeding_shapefile, seeding_radius=seeding_radius, res=res, pnum=pnum, ptot=None, start_time=start_time, duration=duration, seedings=self.seedings, seeding_id=self.seeding_id, tshift=None, use=use, use_label=use_label, emission=emission, decay_coef=decay_coef, study_area=study_area)#, aggregate=aggregate, depth_layer=depth_layer, z_bounds=z_bounds, particle_status=particle_status, traj_dens=traj_dens)
        self.ppi_path.append(ppi_path)
        
        logger.info(f'ppi_exists = {ppi_exists}, {ppi_path}')
        # calculate ppi

        if ppi_exists == True:
            self.output['ppi'] = rxr.open_rasterio(ppi_path).isel(band=0)
        else:
            ppi = self.ppi(use=use, emission=emission, res=res, study_area=study_area, decay_coef=decay_coef)
            ppi = self.get_histogram(res=res, study_area=study_area, normalize=False, assign=True, block_size=len(self.ds.time), weight=self.ds.weight) # weight have been assigned above, so no need to normalize or apply decay etc.
            
            self.output['ppi'] = ppi.rename({'x':'lon', 'y':'lat'}) # changing coordinate names because there was an issue with precision. original dataset coords have higher precision than coords in raster 
                   

            self.write_tiff(ppi, path=ppi_path)
        #thumb_ppi_path = str(ppi_path).split('.tif')[0]+'.png'
        #self.plot(self.output['ppi'], title=use_label, savepng=thumb_ppi_path)

        pass

    
    def sum_seedings(self, rep_path):
        '''
        Sum ppi rasters of single seedings, return summed raster. 
    
        Parameters 
        ----------
        rep_path : list
            list of paths to each rep 
        '''
        if self.seedings == 1:
            r1 = rxr.open_rasterio(rep_path)
        else:
            for idx, rep in enumerate(rep_path):
                if idx == 0:
                    r1 = r0 = rxr.open_rasterio(rep) 
                else:
                    r0 = r1
                    r1 = r0 + rxr.open_rasterio(rep)
        
        return r1
    
    def run(self, ptot, seedings=1, tshift=0, duration=30, start_time='2019-01-01', tstep=timedelta(hours=1), seeding_shapefile=None, seed_within_bounds=None, z=-0.5, seeding_radius=0, beaching=False, res=0.04, study_area=None, use=None, use_label=None, emission=None, hdiff=10, decay_coef=0, make_dt=True, make_td=True):
        '''
        Compute trajectories and produce ppi raster over required number of seedings. 
    
        Parameters
        ----------
        ptot : int
    
        seedings : int
    
        tshift : int
    
        duration : int, optional
    
        start_time : string, optional
    
        res : float, optional

        study_area : list, optional
    
        use : string, optional

        use_label : string, optional
    
        decay_coef : float, optional
        
        '''

        self.seedings = seedings
        
        ppi_output_dir = Path(self.basedir) / f'ppi-{use_label}'

        self.ppi_cache = PMARCache(ppi_output_dir)
             
        ppi_path, ppi_exists = self.ppi_cache.raster_cache(context=self.context, pressure=self.pressure, chemical_compound=self.chemical_compound, seeding_shapefile=seeding_shapefile, seeding_radius=seeding_radius, res=res, pnum=None, ptot=ptot, start_time=start_time, duration=duration, seedings=seedings, seeding_id=self.seeding_id, tshift=tshift, use=use, use_label=use_label, emission=emission, decay_coef=decay_coef, study_area=study_area)#, aggregate=aggregate, depth_layer=depth_layer, z_bounds=z_bounds, particle_status=particle_status, traj_dens=traj_dens)
        
        self.output = xr.Dataset()
        
        if ppi_exists == True:
            logger.info('PPI with requested configurations found in cache.')
            self.output['ppi'] = rxr.open_rasterio(ppi_path)
        else:
            for n in range(0, seedings):

                self.seeding_id = n
                
                start_time_dt = datetime.strptime(start_time, '%Y-%m-%d')+timedelta(days=tshift)*n #convert start_time into datetime to add tshift
                rep_start_time = start_time_dt.strftime("%Y-%m-%d") # bring back to string to feed to opendrift
                
                logger.info(f'Starting rep #{self.seeding_id} with start_time = {rep_start_time}')
                
                #rep_id = n # rep ID is maybe a better name than origin_marker! # self.rep_id
                # this will have to go as an attribute in ds too, useful for plotting
                
                pnum = int(np.round(ptot/seedings)) #  ptot should be split among the seedings
                
                self.single_run(pnum=pnum, duration=duration, tstep=tstep, start_time=rep_start_time, res=res, seeding_shapefile=seeding_shapefile, study_area=study_area, seed_within_bounds=seed_within_bounds, z=z, seeding_radius=seeding_radius, beaching=beaching, use=use, use_label=use_label, emission=emission, hdiff=hdiff, decay_coef=decay_coef, save_tiffs=True)#output_dir = {'dt_output': dt_output_dir, 'rt_output': rt_output_dir, 'c_output': c_output_dir}, loglevel=loglevel)
                logger.info(f'Done with rep #{n}.')
    
            #if use_path:
            if seedings>1:
                #rep_ppi_path = glob.glob(f'{ppi_output_dir}/*.tif') # this is problematic because it takes all tifs in that dir. could be older ones. # need to save the actual list of seeding files. 
                logger.debug(f'ppi_path = {self.ppi_path}')
                self.output['ppi'] = ppi = self.sum_seedings(self.ppi_path)
            
            self.write_tiff(self.output['ppi'], path=ppi_path)
        thumb_ppi_path = str(ppi_path).split('.tif')[0]+'.png'
        self.plot(self.output['ppi'], title=use_label, savepng=thumb_ppi_path)

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

    
    def scatter(self, t=None, xlim=None, ylim=None, s=1, alpha=1, c='age', cmap='rainbow', coastres='10m', proj=cartopy.crs.PlateCarree(), transform=None, dpi=120, figsize=[10,7], rivers=False, save_to=None):
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

        # subset of time
        if t is None:
            if O >= 3:
                ds = self.ds.isel(time=slice(None,None,int(len(self.ds.time)/10**(O-2))))
            else:
                ds = self.ds
        else:
            ds = self.ds.isel(time=t)

        # subset of trajectories
        O = np.floor(np.log10(len(ds.trajectory)))
        if O > 3:
            ds = ds.isel(trajectory=slice(None,None,int(len(ds.trajectory)/10**(O-1))))
        else:
            ds = ds
      
        
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

        im = ax.scatter(ds.lon, ds.lat, s=s, c=c, cmap=cmap, alpha=alpha, transform=transform)
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
        WIP. Histogram animation using xmovie.
        import custom_plotfunc from pmar_utils.
        
        """
        
        
        pass


    
