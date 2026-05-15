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
from opendrift.models.larvalfish import LarvalFish
#import copernicusmarine
from opendrift.readers.reader_copernicusmarine import Reader
from opendrift.readers.unstructured import shyfem
from opendrift.readers.reader_netCDF_CF_generic import Reader as GenericReader
import opendrift
#from pydap.client import open_url
#from pydap.cas.get_cookies import setup_session
import time as T
from xhistogram.xarray import histogram
import os
from opendrift.readers import reader_shape
import cartopy.io.shapereader as shpreader
#import netrc
#import random
#import rasterio
#import tempfile
#from netCDF4 import Dataset
from pathlib import Path
#from flox.xarray import xarray_reduce # for xarray grouping over multiple variables
os.environ['PROJ_LIB'] = '/var/miniconda3/envs/opendrift/share/proj/'
from functools import partial
from rasterio.enums import Resampling
#import glob
#from copy import deepcopy
#from geocube.api.core import make_geocube
import networkx as nx
from pmar.cache import PMARCache
from pmar.utils import traj_distinct, check_particle_file, get_marine_polygon, make_poly, harmonize_use, make_grid, make_vector_grid, get_copernicus
#from cmap import Colormap
#from pmar.plotting import plot_map
from scipy.sparse import coo_array
#tol_inc = Colormap('tol:incandescent').to_mpl()
#tol_muted = Colormap('tol:muted').to_mpl()
#rainbow = Colormap('tol:rainbow_PuRd').to_mpl()
#tol_div = Colormap('tol:BuRd').to_mpl()
#from xgcm import Grid
from scipy.sparse import diags
from scipy.sparse.linalg import matrix_power
from scipy.io import mmwrite
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


    def __init__(self, spatial_domain='global', readers=None, reader_type=None, pressure='generic', chemical_compound=None, basedir='pmar_output', localdatadir = None, particle_path=None, netrppi_path=None, loglevel=10):
        """
        Parameters
        ---------- 
        spatial_domain : str or list, optional
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
        self.seedings = None
        self.tshift = None
        self.cache = PMARCache(Path(basedir) / 'cachedir')
        self.raster_path = None
        self.grid = None
        self.res = None
        self.weight = None
        self.study_area = None
        self.ds = None
        self.ppi_list = []
        self.seeding_shapefile = None # can be point, line or polygon
        self.seed_within_bounds = None # if no seeding_shapefile is given, user can choose to give lon, lat bounds to seed within
        self.loglevel = loglevel
        self.readers = readers
        self.reader_type = reader_type
        self.particle_list = []
        self.output = None
        
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

        # define spatial_domain
        if spatial_domain is not None:
            try: 
                self.spatial_domain = get_copernicus(str(spatial_domain))
                self.extent = self.spatial_domain['extent']

            except:
                # alternatively, provide paths to readers manually
                self.spatial_domain = {'readers': dict(zip(range(0,len(spatial_domain)),spatial_domain))}
        
        # if a path to a shapefile is given to be used for seeding, read it and save it in the pmar_output/polygons dir
        # why save it locally????
        Path(self.basedir / 'polygons').mkdir(parents=True, exist_ok=True)            
        
        # this (if still needed?) should be a separate method
        # if particle_path is given, retrieve number of seedings and load ds
            # gather number of seedings from origin marker. THIS IS A PROBLEM IF THERE ARE MORE BATCHES IN SAME FOLDER. 
        # if type(self.particle_path) is list: 
        #         if self.particle_path:
        #             self.seedings = len(particle_path)
                


    def set_pressure(self, pressure):
        pressures = {'generic': {'opendrift_module': OceanDrift},
                     'plastic': {'opendrift_module': PlastDrift},
                  #   'bacteria': {'opendrift_module': OceanDrift, 'termvel':0, 'decay_coef':1},
                     'chemical': {'opendrift_module': ChemicalDrift},
                     'oil': {'opendrift_module': OpenOil},
                     'metal': {'opendrift_module': ChemicalDrift},
                    'larvae': {'opendrift_module': LarvalFish}}        

        self.pressure = pressure
        self.opendrift_module = pressures[pressure]['opendrift_module']
        logger.info(f'initializing pressure {self.pressure} with opendrift module {self.opendrift_module}')

    
    def _make_grid(self, res, study_area=None, crs='epsg:4326'):
        
        if study_area is None:
            study_area = self.study_area
            
        xmin, ymin, xmax, ymax = study_area
        cols = np.arange(xmin, xmax + res, res)            
        rows = np.arange(ymin, ymax + res, res)

        # outer cell edges
        x_e = cols 
        y_e = rows
        
        # cell centres
        x_c = (x_e + res/2)[:-1]
        y_c = (y_e + res/2)[:-1]     

        grid = xr.Dataset(coords={'x_e':x_e, 'y_e':y_e, 'x_c':x_c, 'y_c':y_c})

        grid['dx'] = grid['dy'] = res

        Xc, Yc = np.meshgrid(grid.x_c, grid.y_c)
        grid = grid.assign({'Xc': (('y_c', 'x_c'), Xc), 'Yc': (('y_c', 'x_c'), Yc), 'ones': (('y_c','x_c'), np.ones((len(grid.y_c), len(grid.x_c))))}) #dataarrays 

        grid = grid.assign({'bin_n': (('y_c','x_c'), np.arange(0,Xc.size).reshape((len(y_c), len(x_c)))
)})
        
        grid.rio.write_crs(crs, inplace=True)

        return grid

    
    def import_trajectories(self, files_to_import): #, align_by = 'time'):
        if type(files_to_import) is str:
            self.particle_path = files_to_import
            ds = xr.open_dataset(self.particle_path)
            print('string')
            
        elif type(files_to_import) is list:
            self.particle_list = files_to_import
            self.seedings = len(self.particle_list)
            
            ds = xr.open_mfdataset(self.particle_list, parallel=True, combine='nested', join='outer', concat_dim='seeding') # this will give all seedings the same dimensions and pad them with nans
            print('list')
            
        elif type(files_to_import) is xr.core.dataset.Dataset:
            ds = files_to_import 
            print('ds')
        
        elif type(files_to_import) is list:
            raise ValueError('Please use PMARmulti class to import multiple trajectory files.')
        else:
            raise ValueError('Type of files_to_import unrecognised. please provide path(s) to file(s) or xarray Dataset.')
    
        self.ds = ds[['lon','lat','z','age_seconds', 'origin_marker']]#.ffill('time') # is this what we want?
        # giving all traj unique id?
        pass

    def import_ppi(self, files_to_import):
        r = xr.open_mfdataset(files_to_import, engine="rasterio", concat_dim='seeding', combine='nested').squeeze().band_data

        self.ppi_list = files_to_import
        return r
    

    # Given multiple particle_paths, it returns the maximum time length. 
    # Used for preprocessing function in get_ds to make sure all ds's have same time length, even if opendrift ended early because of all particles beaching.
    def find_correct_len(self):
        '''
        Sometimes opendrift runs end early because e.g. all particles have beached, resulting in seedings with different time lengths. This method finds the maximum time lenght of all seedings.
        '''
        lens = np.array([len(xr.open_dataset(filename).time) for filename in self.particle_list])
        logger.info(f'giving all datasets the same time length of {lens} tsteps...')
        return lens.max()

    def _preprocess(self, ds, correct_len):
        '''
        Sometimes opendrift runs end early because e.g. all particles have beached, resulting in seedings with different time lengths. This method pads all seedings so that they all have same time length (determined with find_correct_len).
        '''
        return ds.pad(pad_width={'time': (0, correct_len-len(ds.time))}, mode='edge')
         
    
    def get_trajectories(self, pnum, start_time, seed_time=timedelta(days=0), duration_days=30, study_area=None, seeding_shapefile = None, seed_within_bounds=None, z=-0.5, tstep=timedelta(hours=1), hdiff=10, termvel=0, crs='4326', seeding_radius=0, beaching=False, stokes_drift=False, pnum_per_shape=False):
        """
        Calculate trajectories using Oceandrift module by OpenDrift (MET Norway).
        Uses OceanDrift module. 
                
        This method is currently optimised for use over the Mediterranean and Black Sea only. 
        Forcing data is streamed from Copernicus (ocean currents, wind, mixed layer depth), while GEBCO bathymetry is stored locally. 
        
        Particles are seeded homogeneously over polygon created using make_poly() method. 
        
        
        Parameters
        ----------
        spatial_domain : str
            String defining the spatial_domain of the simulation i.e., the model data to be used for particle forcing. Options are 'med-cmems', 'bs-cmems' and 'bridge-bs'. 
        pnum : int 
            The number of particles to be seeded
        start_time : str
            The start date of the simulation, in string format 'dd-mm-yyyy'
        seed_time : timedelta, optional
            for continuous release of particles
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
        
        # needed for cache
        self.seeding_radius = seeding_radius
        self.pnum = pnum
        self.start_time = start_time
        self.seed_time = seed_time
        self.duration = duration_days

        #### SPATIAL DOMAIN
        # establish study area
        if seeding_shapefile is not None and seed_within_bounds is not None:
             raise ValueError('Please provide either seeding_shapefile or seed_within_bounds, not both.')        
            
        if seed_within_bounds is None and seeding_shapefile is None:
            seed_within_bounds = self.spatial_domain['extent']
        
        if seed_within_bounds is not None:
            poly_path = f'polygon-crs_epsg:{crs}-lon_{np.round(seed_within_bounds[0],4)}_{np.round(seed_within_bounds[2],4)}-lat—{np.round(seed_within_bounds[1],4)}_{np.round(seed_within_bounds[3])}.shp'
            q = self.basedir / 'polygons' / poly_path
            make_poly(seed_within_bounds, crs=crs, save_to=str(q))
            # COULD REPLACE THIS WITH VECTOR GRID
            seeding_shapefile = str(q)
            logger.info(f'created seeding_shapefile with bounds {seed_within_bounds}')

        if study_area is None:
            study_area = self.spatial_domain['extent']

        self.study_area = study_area
        self.seeding_shapefile = seeding_shapefile 
        self.seed_within_bounds = seed_within_bounds
        logger.debug(f'study_area = {self.study_area}, seed_within_bounds = {self.seed_within_bounds}, seeding_shapefile = {self.seeding_shapefile}')

        file_path, file_exists = self.cache.particle_cache(context=self.spatial_domain, pressure=self.pressure, chemical_compound=self.chemical_compound, seeding_shapefile=seeding_shapefile, pnum=pnum, start_time=start_time, seed_time=seed_time, duration_days=duration_days, seed_within_bounds=seed_within_bounds, seeding_radius=seeding_radius, beaching=beaching, z=z, tstep=tstep, hdiff=hdiff, termvel=termvel, stokes_drift=stokes_drift, seeding_id=self.seeding_id)

        logger.info(f'particle_path exists = {file_exists}, {file_path}')
        logger.debug(f'seeding_shapefile = {self.seeding_shapefile}')        

        self.particle_path = file_path
        #self.particle_list.append(file_path)
        
        if file_exists == True:
            logger.info(f'Opendrift file with these configurations already exists within {self.basedir}. Importing.') 
            #logger.info(f'self.ds has shape {self.ds.dims}')
        
        else:
         
            # if termvel is None:
            #     termvel = self.termvel
            
            ### time settings ####
            start_time = datetime.strptime(start_time, '%Y-%m-%d')
            
            #time_step = tstep
    
            duration = timedelta(days=duration_days)

            self.tstep = tstep
    
            
            if 'Point' in list(gpd.read_file(self.seeding_shapefile).geometry.type):
                seeding_poly = gpd.read_file(self.seeding_shapefile)
                seeding_poly['geometry'] = seeding_poly.geometry.to_crs('3857').geometry.buffer(2000).to_crs(4326)
                new_seeding_shapefile = Path(self.basedir / 'polygons' / ('buffered_'+self.seeding_shapefile.split('/')[-1]))
                seeding_poly.to_file(new_seeding_shapefile)
                self.seeding_shapefile = str(new_seeding_shapefile)
                logger.debug(f'Added buffer to {np.unique(seeding_poly.geometry.type)} type geometry in self.seeding_shapefile to allow seed_from_shapefile')
    
                logger.info(f'seeding particles evenly within shapefile {self.seeding_shapefile}')
                #self.seeding_shapefile = seeding_shapefile
            
                
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
                    # if self.reader_type == 'shyfem':
                    #     reader = shyfem.Reader(r)
                    # else:
                    #     reader = GenericReader(r, name=f'cmems_{id}')
                    self.o.add_reader(r)#, variables=['x_sea_water_velocity', 'y_sea_water_velocity'])
    
            else:
            #if self.spatial_domain:
                readers = []
                # add predefined readers from spatial_domain
                for var in self.spatial_domain['readers']:
                    readers.append(Reader(dataset_id=self.spatial_domain['readers'][var]))
                self.o.add_reader(readers) # add all readers for that spatial_domain.
                #self.o.add_readers_from_list(self.spatial_domain['readers'].values()) # this will add readers lazily, and only read them if useful. 
        
    
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
    
            # TODO 
            if pnum_per_shape is True:
                seeding_polygon = gpd.read_file(self.seeding_shapefile)
                for shape in range(len(seeding_polygon)):
                    self.o.seed_from_geopandas(geodataframe=seeding_polygon.iloc[[shape]], number=pnum, time=[start_time, start_time + seed_time], terminal_velocity=termvel, z=z, origin_marker=self.seeding_id)
            else:
                self.o.seed_from_shapefile(shapefile=self.seeding_shapefile, number=pnum, time=[start_time, start_time + seed_time], terminal_velocity=termvel, z=z, origin_marker=self.seeding_id, radius=seeding_radius)
    
    
            logger.info('running opendrift...')
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
    
            # check outfile
            if type(self.particle_path) is list:
                if len(self.particle_path)>1:
                    raise ValueError('opendrift outfile must be string')
            
            self.o.run(duration=duration, 
                       time_step=tstep, #time_step_output=timedelta(hours=24), 
                       outfile=str(self.particle_path), #temp_outfile, 
                       export_variables=['lon', 'lat', 'z', 'status', 'age_seconds', 'origin_marker', 'specie', 'sea_floor_depth_below_sea_level', 'weight', 'length']) # 'ocean_mixed_layer_thickness', 'x_sea_water_velocity', 'y_sea_water_velocity', 'x_wind', 'y_wind',
    
            elapsed = (T.time() - t_0)
            logger.info("total simulation runtime %s" % timedelta(minutes=elapsed/60)) 
    
            if hasattr(self.o, 'discarded_readers'):
                logger.warning(f'Readers {self.o.discarded_readers} were discarded. Particle transport will be affected')
    
            #ds = self.o.result
            #ds['lon'] = ds['lon'].where(ds.status>=0).ffill('time')
            #ds['lat'] = ds['lat'].where(ds.status>=0).ffill('time')
            #if 'z' in ds.data_vars:
             #   ds['z'] = ds['z'].where(ds.status>=0).ffill('time')
            
            #self.ds = ds 
        self.import_trajectories(str(file_path))

        logger.info(f'particle simulation lat = {self.ds.lat.shape}, lon={self.ds.lon.shape}, time={self.ds.time.shape}')
        logger.info(f"done. NetCDF file '{str(file_path)}' created successfully.") 
        
        pass

    def plot_trajectories(self, t=None, filename=None, s=None, c='age', cmap='viridis', alpha=None, proj=None):
        from pmar.plotting import plot_map
        import matplotlib.pyplot as plt
        import cartopy
        if proj is None:
            proj = cartopy.crs.PlateCarree()
            
        ds = self.ds
        O = np.floor(np.log10(len(ds.time)))

        # subset of time
        if t is None:
            if O >= 3:
                ds = ds.isel(time=slice(None,None,int(len(ds.time)/10**(O-2))))
            else:
                ds = ds
        else:
            ds = ds.isel(time=t)
    
        # subset of trajectories
        O = np.floor(np.log10(len(ds.trajectory)))
        if O > 3:
            ds = ds.isel(trajectory=slice(None,None,int(len(ds.trajectory)/10**(O-1))))
        else:
            ds = ds
      
        
        if c == 'age': 
            c = ds.age_seconds/60/60/24
            cmap = 'rainbow' #'copper'
            cbar_label = 'elapsed days'
        elif c == 'z':
            c = ds.z
            cmap = 'viridis'
            cbar_label = 'depth [m]'
        elif c == 'marker':
            c = ds.origin_marker
            cbar_label = 'origin marker (#rep)'

        fig, ax = plot_map(filename=filename, proj=proj)
        im = ax.scatter(ds.lon, ds.lat, s=s, c=c, cmap=cmap, alpha=alpha, transform=cartopy.crs.PlateCarree())
        cax = fig.add_axes([ax.get_position().x1+0.05,ax.get_position().y0,0.02,ax.get_position().height])
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(cbar_label, rotation=90)
        
    def filter_by_status(self, status):
        """
        Filtering trajectories by status (active, stranded or seafloor).
        """
        available_status = {'active': 0, 'stranded': 1, 'seafloor': 2}

#        try:
        ds = self.ds
#        except:
#            ds = self.get_ds

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
        #try:
        ds = self.ds
        #except:
        #    ds = self.get_ds
            
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


    def normalize_weight(self, weight, res, study_area=None):
        # dividing each trajectory weight by the number of particles that were in the same bin at t0
        
        bin_n_t0 = self.get_bin_n(res=res, t=0, study_area=study_area)
        self.ds['bin_n_t0'] = bin_n_t0

        weight_by_bin = weight.groupby(self.ds.bin_n_t0.load()) # added load otherwise groupby raises error
        counts_per_bin = bin_n_t0.groupby(self.ds.bin_n_t0).count()
        
        logger.info('counts_per_bin...')
        normalized_weight = weight_by_bin/counts_per_bin 

        return normalized_weight

    def set_weights(self, res=None, study_area=None, weight=1, use=None, emission=None, decay_coef=0, normalize=False, assign=True):
        '''
        Associate weights with each trajectory. 
        ORDER OF OPERATIONS MATTERS.
        Weights depend on one or more of the following components: use value in each cell (use must be provided as quantity/day), emission (quantity/day), decay coefficient, number of trajectories starting in the same cell (normalize parameter).  
        '''
        
            
        if study_area is None:
            study_area = self.study_area
        
        if res is not None or study_area is not None:
            # if a new res or study area is required, recalculate grid
            self.grid = make_grid(res=res, study_area=study_area)
        if use is None:
            logger.info('no use provided. calculating ppi from unity use.')            

        # could merge use and weight and just say it can be int or array
        if type(weight) is int:
            weight = xr.ones_like(self.ds.lon)*weight
            logger.warning(weight.shape)

        if use is not None:
            use = harmonize_use(use, res, study_area=study_area, like=self.grid.Xc.rename({'x_c':'x', 'y_c':'y'}))#, tstep=self.ds.tstep) # renaming otherwise geocube does not recognise coords
            self.use = use
            use_value = self.use_by_traj(use=self.use, res=res, study_area=study_area).broadcast_like(self.ds.lon)
            weight = weight*use_value#/len(self.ds.time) # dividing use weight by the number of timesteps for conservation
            # (trajectory)
        #else:
         #   weight=xr.ones_like(self.ds.lon) # needed to give weights the correct shape in all possible cases
            logger.warning(weight.shape)

        if emission is not None:
            self.emission = emission * self.ds.tstep / timedelta(days=1).total_seconds() # convert to amount of pressure per my timestep
            logger.info(f'Converted emission from {emission} per day to {self.emission} per timestep.')
            weight = weight*self.emission
        # (time)
            logger.warning(weight.shape)
        
        if normalize is True:
            weight = self.normalize_weight(weight=weight, res=res, study_area=study_area).drop({'x_c','y_c','bin_n_t0'})
            logger.info('weight normalized by number of particles in bin at t0.')        
            logger.warning(weight.shape)

        if decay_coef != 0:
        # decay rate. default is no decay, but this is still useful to give weight the correct shape
            logger.info(f'computing decay rate function with decay coefficient k = {self.decay_coef}...')
            y = self.decay_rate(k=decay_coef) # default value = 0 means there is no decay
            weight = weight*y
            logger.warning(weight.shape)

        # give weight correct shape
        #weight = weight.broadcast_like(self.ds.lon)
        
        if assign is True:
            #self.ds = self.ds.assign({'weight': (('trajectory', 'time'), weight.data)}) 
            self.ds['weight'] = weight
            logger.info('updating weight variable in ds')
        else:
            return weight
    


    def get_bin_n(self, res, t=None, study_area=None, ds=None):
        '''
        Add new variable to ds ('bin_n_t0') containing its "bin number" at timestep t, i.e. a unique identifier corresponding to a specific spatial grid-cell. 
        
        Parameters
        ----------
        t : int
            index of timestep to consider
        '''

        # if i opened multiple seedings with mfdataset, this does not work, because most particles are inactive at the first timestep. 
        #if 'seeding' in list(self.ds.dims):
         #   ds = self.ds.bfill('time')
        #else:
         #   ds = self.ds
        if ds is None:
            ds = self.ds
        
        if t is not None:
            bin_n = self.grid.bin_n.sel(x_c=ds.isel(time=t).lon, y_c=ds.isel(time=t).lat, method='nearest')
            logger.info(f'calculated bin number at timestep {t}.')
        else:
            bin_n = self.grid.bin_n.sel(x_c=ds.lon, y_c=ds.lat, method='nearest')
            logger.info(f'calculated bin number at all timesteps.')
        
        return bin_n

    def get_histogram(self, res, study_area=None, weighted=True, dim=['trajectory', 'time'], block_size='auto'):
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

#        if self.seedings > 1:
#            logger.warning(f'this run contains {self.seedings} seedings. to get histogram of aggregated seedings, use .run() method. get_histogram() will only work on the last rep.')

        if study_area is None:
            study_area = self.study_area
            
        self.grid = make_grid(res=res, study_area=study_area)
        xbin = self.grid.x_e.data
        ybin = self.grid.y_e.data
        
        #self.get_ds
        
        # weights = self.set_weights(res=res, study_area=study_area, weight=weight, normalize=normalize, use=use, emission=emission, decay_coef=decay_coef, assign=assign)
        # logger.info('set_weights done.')
        
        #NOTE: NaNs in weights will make the weighted sum as nan. To avoid this, call .fillna(0.) on your weights input data before calling histogram().
        if weighted == False:
            weights = None
        else:
            weights = self.ds.weight.fillna(0.)
        h = histogram(self.ds.lon, self.ds.lat, bins=[xbin, ybin], dim=dim, weights=weights, block_size=block_size) 
        # block_size='auto' was giving division by 0 error, which is a known bug: https://github.com/xgcm/xhistogram/issues/16
        #NOTE: NaNs in weights will make the weighted sum as nan. To avoid this, call .fillna(0.) on your weights input data before calling histogram().
            
        h = h.transpose().rename({'lon_bin':'x', 'lat_bin':'y'}) # important to not have it where(h>0) otherwise it misses values when summing

        r = (
            xr.DataArray(h) 
            .rio.write_nodata(0)#np.nan)
            .rio.write_crs('epsg:4326')
            .rio.write_coordinate_system())

        return r    
    
    # this is not used in run. it is more if one wishes to create a ppi with different weights.
    def get_ppi(self, res, use=None, use_label=None, emission=None, study_area=None, decay_coef=0): 
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
        
        '''
            
        self.ppi_cache = PMARCache(Path(self.basedir) / f'ppi-{use_label}')

        ppi_path, ppi_exists = self.ppi_cache.raster_cache(context=self.spatial_domain, pressure=self.pressure, chemical_compound=self.chemical_compound, seeding_shapefile=self.seeding_shapefile, seeding_radius=self.seeding_radius, res=res, pnum=self.pnum, ptot=None, start_time=self.start_time, duration=self.duration, seedings=self.seedings, seeding_id=self.seeding_id, tshift=None, use=use, use_label=use_label, emission=emission, decay_coef=decay_coef, study_area=study_area)#, aggregate=aggregate, depth_layer=depth_layer, z_bounds=z_bounds, particle_status=particle_status, traj_dens=traj_dens)
        #self.ppi_list.append(ppi_path)
        
        logger.info(f'ppi_exists = {ppi_exists}, {ppi_path}')
        # calculate ppi
        if not self.output:
            self.output = xr.Dataset()
            
        if ppi_exists == True:
            # use import raster
            self.output['ppi'] = rxr.open_rasterio(ppi_path).squeeze()
        else:
            self.set_weights(res=res, study_area=study_area, use=use, decay_coef=decay_coef, normalize=True, assign=True)
            
            r = self.get_histogram(res=res, study_area=study_area, block_size=len(self.ds.time)).rio.reproject_match(self.grid.mask) #.assign_attrs({'use_path': use_path, 'emission':emission})
    
            ppi = r / len(self.ds.time)
                
            self.output['ppi'] = ppi.rio.reproject_match(self.grid.mask)#.rename({'x':'lon', 'y':'lat'}) # changing coordinate names because there was an issue with precision. original dataset coords have higher precision than coords in raster 

            self.write_tiff(ppi, path=ppi_path)
        
        self.ppi_path = ppi_path
            
        pass

    def plot_ppi(self, proj=None, cmap=None, filename=None):
        from pmar.plotting import plot_map
        import cartopy
        if proj is None:
            proj = cartopy.crs.epsg(4326)

        transform = cartopy.crs.PlateCarree()
        
        Path(self.basedir / 'figures').mkdir(parents=True, exist_ok=True)  
        if filename is not None:
            filename=str(self.basedir)+f'/figures/{filename}'
        fig, ax = plot_map(proj=proj, filename=filename)
        self.output.ppi.where(self.output.ppi>0).plot(ax=ax, cmap=cmap, transform=transform)

    # this does not include weights
    def _get_CM(self, cm_grid=None, res=None, study_area=None, t0=0, t1=-1, filename=None):#, apply_landmask=False):
        
        if study_area is None:
            study_area = self.study_area
            
        ds = self.ds.isel(time=[t0,t1]) 
        if cm_grid is None: # if no polygons are given, consider entire grid
            if res is not None: # if a new resolution is given, create new grid
                grid = make_grid(res=res, study_area=study_area)
                self.grid = grid
            else: # otherwise consider existing grid defined in .run()
                grid = self.grid
            #cm_grid = _make_poly(lon=grid.x_c, lat=grid.y_c) 
            cm_grid = make_vector_grid(grid.bin_n.astype(float))

        
        p_df = ds[['lon','lat']].to_dataframe().reset_index()
        p_gdf = gpd.GeoDataFrame(
                                p_df,
                                geometry=gpd.points_from_xy(p_df.lon, p_df.lat),
                                crs=cm_grid.crs
                                )
        
        ll = {0:{},1:{}}

        for t in range(2):
            for i, polys in enumerate(cm_grid.geometry):
                poly = cm_grid.geometry.iloc[i]
                is_in_poly = p_gdf.sort_values(['time','trajectory'])[0+t*len(ds.trajectory):len(ds.trajectory)+t*len(ds.trajectory)].set_index('trajectory').geometry.within(poly)
                traj_in_poly = list(is_in_poly[is_in_poly].index)
                ll[t][i] = traj_in_poly
        
        _Nij_sparse = []
        _Pij_sparse = []
        _Pji_sparse = []
        row = []
        col = []
        
        for i in range(len(cm_grid)):
            for j in range(len(cm_grid)):
                N = len(set(ll[0][i]).intersection(ll[1][j]))
                if N > 0:
                    _Nij_sparse.append(N)
                    _Pij_sparse.append(N / len(ll[0][i]))
                    _Pji_sparse.append(N / len(ll[1][j]))
                    row.append(i)
                    col.append(j)
        
        Nij_sparse = coo_array((_Nij_sparse, (row, col)), shape=((len(cm_grid), len(cm_grid))))
        Pij_sparse = coo_array((_Pij_sparse, (row, col)), shape=((len(cm_grid), len(cm_grid))))
        Pji_sparse = coo_array((_Pji_sparse, (row, col)), shape=((len(cm_grid), len(cm_grid)))).T
        
        self.cm_grid = cm_grid 
        self.CM = Pij_sparse

        if filename is not None:
            mmwrite(filename, Pij_sparse)
        
        return Nij_sparse, Pij_sparse, Pji_sparse



    def _get_Nij(self, cm_grid=None, res=None, study_area=None, weighted=False, t0=0, t1=-1, buffer=None):
        
        if study_area is None:
            study_area = self.study_area
            
        ds = self.ds.ffill('time').isel(time=[t0,t1]) # V IMPORTANT, otherwise i miss beached particles and sum of CM rows is never equal to 1 , whereas bfill is needed for datasets with continuous seeding
        
        
        if cm_grid is None: # if no polygons are given, consider entire grid
            if res is not None: # if a new resolution is given, create new grid
                grid = make_grid(res=res, study_area=study_area)
                self.grid = grid
            else: # otherwise consider existing grid defined in .run()
                grid = self.grid
            #cm_grid = _make_poly(lon=grid.x_c, lat=grid.y_c) 
            cm_grid = make_vector_grid(grid.bin_n.astype(float))
        
        if buffer is not None:
            buff = cm_grid.dissolve().buffer(buffer)
            diff = gpd.GeoDataFrame({cm_grid.columns.drop('geometry')[0]: -1234, 'geometry': buff.difference(cm_grid.dissolve())}, crs=cm_grid.crs)
            cm_grid = pd.concat([cm_grid, diff]).reset_index(drop=True)
        
        self.cm_grid = cm_grid

        df_cols = ['lon','lat']
        if 'weighted' == True:
            if 'weight' in self.ds.data_vars:
                cols.append('weight')
            else:
                raise ValueError('no weight variable found in ds')
            
        p_df = ds[df_cols].to_dataframe().reset_index()
        p_gdf = gpd.GeoDataFrame(
                                p_df,
                                geometry=gpd.points_from_xy(p_df.lon, p_df.lat),
                                crs=cm_grid.crs
                                )
        
        ll = {0:{},1:{}}
        # ll contains for each time (0,1), and for each polygon in cm_grid, the ids of the trajectories contained in it

        for t in range(2):
            for i, polys in enumerate(cm_grid.geometry):
                poly = cm_grid.geometry.iloc[i]
                is_in_poly = p_gdf.sort_values(['time','trajectory'])[0+t*len(ds.trajectory):len(ds.trajectory)+t*len(ds.trajectory)].set_index('trajectory').geometry.within(poly)
                traj_in_poly = list(is_in_poly[is_in_poly].index)
                if 'weight' in p_gdf.columns:
                    ll[t][i] = dict(zip(traj_in_poly, list(p_gdf.iloc[traj_in_poly]['weight'])))
                else:
                    ll[t][i] = traj_in_poly

        return ll
    
    def get_Pij(self, cm_grid=None, res=None, study_area=None, t0=0, t1=-1, buffer=None, filename=None, weighted=False):        

        ll = self._get_Nij(cm_grid=cm_grid, res=res, study_area=study_area, t0=t0, t1=t1, buffer=buffer, weighted=weighted)
        # _Nij_sparse = []
        _Pij_sparse = []
        # _Pji_sparse = []
        row = []
        col = []

        for i in range(len(self.cm_grid)):
            for j in range(len(self.cm_grid)):
                N = len(set(ll[0][i]).intersection(ll[1][j]))
                if N > 0:
                    # _Nij_sparse.append(N)
                    _Pij_sparse.append(N / len(ll[0][i]))
                    # _Pji_sparse.append(N / len(ll[1][j]))
                    row.append(i)
                    col.append(j)
        
        # Nij_sparse = coo_array((_Nij_sparse, (row, col)), shape=((len(cm_grid), len(cm_grid))))
        Pij_sparse = coo_array((_Pij_sparse, (row, col)), shape=((len(self.cm_grid), len(self.cm_grid))))
        # Pji_sparse = coo_array((_Pji_sparse, (row, col)), shape=((len(cm_grid), len(cm_grid)))).T
        
        #self.cm_grid = cm_grid 
        self.CM = Pij_sparse

        if filename is not None:
            mmwrite(filename, Pij_sparse)
            logger.info(f'wrote file {filename}')
        
        return Pij_sparse

    def get_Pji(self, cm_grid=None, res=None, study_area=None, t0=0, t1=-1, filename=True):        

        ll = self._get_Nij(cm_grid=cm_grid, res=res, study_area=study_area, t0=t0, t1=t1)
        # _Nij_sparse = []
        # _Pij_sparse = []
        _Pji_sparse = []
        row = []
        col = []
        
        for i in range(len(self.cm_grid)):
            for j in range(len(self.cm_grid)):
                N = len(set(ll[0][i]).intersection(ll[1][j]))
                if N > 0:
                    # _Nij_sparse.append(N)
                    # _Pij_sparse.append(N / len(ll[0][i]))
                    _Pji_sparse.append(N / len(ll[1][j]))
                    row.append(i)
                    col.append(j)
        
        # Nij_sparse = coo_array((_Nij_sparse, (row, col)), shape=((len(cm_grid), len(cm_grid))))
        # Pij_sparse = coo_array((_Pij_sparse, (row, col)), shape=((len(cm_grid), len(cm_grid))))
        Pji_sparse = coo_array((_Pji_sparse, (row, col)), shape=((len(self.cm_grid), len(self.cm_grid)))).T
        
        #self.cm_grid = cm_grid 
        #self.CM = Pij_sparse

        if filename is not None:
            mmwrite(filename, Pji_sparse)
        
        return Pji_sparse
    

    def plot_CM(self, tick_labels=None, figsize=[8,6], annot=False, linewidths=0, vmin=0, vmax=None, norm=None, cmap='Greys', mask=None):
        # use cm_grid for tick_labels
        from pmar.plotting import plot_array
        import matplotlib.pyplot as plt
        CM = self.CM.toarray() # my CM is now by default a sparse array. to plot it, must convert it to a dense array 
        
        if mask is not None:
            mask = CM<=mask

        if annot == True:
            CM = np.round(CM,2)
            
        fig, ax = plt.subplots(figsize=figsize)
        plot_array(CM, tick_labels=tick_labels, annot=annot, linewidths=linewidths, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, mask=mask, ax=ax)

    def dispersal_distance(self, CM=None, cm_grid=None):
        '''
        "as the crow flies" dispersal distance from connectiviy matrix
        '''
        if cm_grid is None:
            cm_grid = self.cm_grid

        if CM is None:
            CM = self.CM
            
        cm_grid = cm_grid.to_crs(32601)
        
        disp_dist = np.zeros(len(cm_grid))
        for i in range(len(cm_grid)):
            CM_row = CM.tocsr()[i,:] # convert to scipy.csr so i can index by row
            if CM_row.sum()>0:
                D_col = np.array([cm_grid.geometry.centroid.iloc[i].distance(cm_grid.geometry.centroid.iloc[j])*1e-3 for j in range(len(cm_grid))])
                disp_dist[i] = CM_row@D_col
            else:
                disp_dist[i] = np.nan

        disp_dist = xr.DataArray(disp_dist,
                  coords = {'bin':self.grid.bin_n.stack(bin=('y_c','x_c')).dropna('bin').bin}).unstack().sortby('x_c').sortby('y_c').rio.write_crs(4326)#.rio.reproject_match(self.grid.mask)
        
        self.disp_dist = disp_dist
        return disp_dist
    

    def plot_origin(self, Pji_sparse, destination_idx, annotate=False):
        '''
        Given my initial condition (i.e. particle distribution at start time), what is the probability that a particle found in cell {destination_idx} originated from the other cells?
        '''
        cm_grid = self.cm_grid
        cm_grid[f'origin_of_{destination_idx}'] = Pji_sparse.toarray()[destination_idx]*100 
        # check sum of probabilities is 100
        print(cm_grid[f'origin_of_{destination_idx}'].sum(), '%')

        fig, ax = plot_map()
        im=cm_grid.plot(column=f'origin_of_{destination_idx}', ax=ax, legend=True, cmap=tol_inc, legend_kwds={'format':'%.0f%%'})
        ax.set_title('probability of each cell being the origin, \ngiven initial condition')
        cm_grid.iloc[[destination_idx]].boundary.plot(color='k', linewidth=1, label='destination cell', ax=ax, linestyle='--')
        plt.legend()
        if annotate is True:
            if len(cm_grid) < 400: # arbitrary
                cm_grid.boundary.plot(ax=ax, linewidth=0.5, color='grey')
                for i in range(len(cm_grid)):
                    ax.annotate(i, xy=[cm_grid.iloc[[i]].geometry.centroid.x, cm_grid.iloc[[i]].geometry.centroid.y], fontsize=8)

        return cm_grid[f'origin_of_{destination_idx}']
            

    def plot_destination(self, Pij_sparse, origin_idx, annotate=False):
        '''
        Given my initial condition (i.e. particle distribution at start time), what is the probability that a particle seeded in cell {origin_idx} will end up in the other cells?
        '''
        cm_grid = self.cm_grid
        cm_grid[f'destination_of_{origin_idx}'] = Pij_sparse.toarray()[origin_idx]*100 
        # check sum of probabilities is 100
        print(cm_grid[f'destination_of_{origin_idx}'].sum(), '%')

        fig, ax = plot_map()
        im=cm_grid.plot(column=f'destination_of_{origin_idx}', ax=ax, legend=True, cmap=tol_inc, legend_kwds={'format':'%.0f%%'})
        ax.set_title('probability of each cell being the destination, \ngiven initial condition')
        cm_grid.iloc[[origin_idx]].boundary.plot(color='k', linewidth=1, label='origin cell', ax=ax, linestyle='--')
        plt.legend()
        if annotate is True:
            if len(cm_grid) < 400: # arbitrary
                cm_grid.boundary.plot(ax=ax, linewidth=0.5, color='grey')
                for i in range(len(cm_grid)):
                    ax.annotate(i, xy=[cm_grid.iloc[[i]].geometry.centroid.x, cm_grid.iloc[[i]].geometry.centroid.y], fontsize=8)

        return cm_grid[f'destination_of_{origin_idx}']
        
    
    def _predict_with_CM_np(self, initial_condition, CM=None):
    # only works when CM is a grid covering the whole domain.
    # if initial_condition has shape (m,n), the CM must have shape (mxn, mxn)
        if CM is None:
            CM = self.CM

        _cm_prediction = np.matmul(initial_condition.ravel(), CM).reshape(initial_condition.shape)

        cm_prediction = (xr.DataArray(_cm_prediction,
                                      coords=dict(y_c=self.grid.y_c, x_c=self.grid.x_c, ))
                         .rio.write_crs('epsg:4326'))
        
        self.cm_prediction = cm_prediction
        
        return cm_prediction
        
    def _predict_with_CM_xr(self, initial_condition, initial_condition_crs='epsg:4326', CM=None):
    # only works when CM is a grid covering the whole domain.
    # if initial_condition has shape (m,n), the CM must have shape (mxn, mxn)
        if CM is None:
            CM = self.CM.toarray() # my CM is now by default a sparse array, turn it into the full array to make predictions

        initial_condition = initial_condition.load().rio.set_crs(initial_condition_crs).rio.reproject('epsg:4326').sortby('y').sortby('x').rename({'y':'y_c', 'x':'x_c'})
        
        if initial_condition.data.ravel().shape[0] != CM.shape[0]:
            logger.warning('initial_condition and CM dimensions do not match. clipping initial_condition to match cm_grid.')
            initial_condition = initial_condition.rio.clip(self.cm_grid.geometry, self.cm_grid.geometry.crs, drop=True, all_touched=True)
            initial_condition = initial_condition.where(initial_condition>=0)
        else:
            logger.info('initial_condition and CM dimensions match.')
        
        initial_condition_stack = initial_condition.stack(origin=('y_c','x_c')).dropna('origin')        
        
        CM_xr = xr.DataArray(CM, 
                             coords = dict(origin=initial_condition_stack.origin, destination=initial_condition_stack.rename({'origin':'destination', 'x_c':'x_c_end', 'y_c':'y_c_end'}).destination))
        
        cm_prediction = CM_xr.dot(initial_condition_stack).unstack().rio.set_spatial_dims(x_dim='x_c_end',y_dim='y_c_end')
        
        self.cm_prediction = cm_prediction
        
        return cm_prediction


    def predict_with_CM(self, initial_condition, initial_condition_crs='epsg:4326', CM=None, k=None, sum=False):
    # only works when CM is a grid covering the whole domain.
    # if initial_condition has shape (m,n), the CM must have shape (mxn, mxn)
        if CM is None:
            CM = self.CM # my CM is now by default a sparse array

        initial_condition = initial_condition.load().rio.set_crs(initial_condition_crs).rio.reproject_match(self.grid.mask).sortby('y').sortby('x').rename({'y':'y_c', 'x':'x_c'})

        if initial_condition.data[~np.isnan(initial_condition.data)].shape[0] != CM.shape[0]:
            logger.warning('initial_condition and CM dimensions do not match')#. clipping initial_condition to match cm_grid.')
            initial_condition = initial_condition.rio.clip(self.cm_grid[self.cm_grid[list(self.cm_grid.columns.drop('geometry'))[0]] >=0].geometry, self.cm_grid.geometry.crs, drop=False, all_touched=True)
            initial_condition = initial_condition.where(initial_condition>=0)
        else:
            logger.info('initial_condition and CM dimensions match.')
        
        initial_condition_stack = initial_condition.stack(origin=('y_c','x_c')).dropna('origin')        
        print(initial_condition_stack.shape)
        ## this would be for the case where an extra node was added to denote "everywhere else" outside of cm_grid.
        ## but unstack obviously does not work, so can't return a datarray in lon, lat. only a sparse matrix. 
        maxbin = int(self.grid.bin_n.max().data)
        if len(self.cm_grid)-1 > maxbin:
            logger.warning('adding extra node to network')
            _initial_condition_stack = xr.concat([initial_condition_stack, xr.DataArray(data=[[1]], dims={'x_c':0, 'y_c':0}).stack(origin=('y_c','x_c'))], dim='origin')
            print(_initial_condition_stack.shape)
        else:
            _initial_condition_stack = initial_condition_stack
            print(_initial_condition_stack.shape)
        
        ic_sparse = diags(_initial_condition_stack.data, shape=(CM.shape), format='coo')

        if k is not None:
            if sum is True:
                # like in Chamberlain et al. (2023), "the probability of a float sampling a given region within a range of time steps"
                #CM = np.sum(matrix_power(CM, i) for i in range(1,k+1))
                CM = self.CM
                for i in range(2,k+1):
                        CM += matrix_power(self.CM,i)
            
            else:
                CM = matrix_power(CM, k)
    
            
        cm_prediction_sparse = ic_sparse.dot(CM).sum(0)

        # adding maxbin to exclude "dummy" node before reprojection
        cm_prediction = xr.DataArray(np.array(cm_prediction_sparse)[0][:maxbin+1],
                              coords = dict(destination=initial_condition_stack.rename({'origin':'destination', 'x_c':'x', 'y_c':'y'}).destination)).unstack().sortby('x').sortby('y').rio.write_crs(4326).rio.reproject_match(self.grid.mask)
        # reproject match is needed to avoid tiny discrepancies in precision of x,y variables which cause the grids to mismatch
        
        self.cm_prediction = cm_prediction
        self._cm_prediction_sparse = cm_prediction_sparse # in case it's needed to check what goes outside of system, or in "dummy" node
        
        return cm_prediction

    
    def get_network(self, CM=None, node_names=None):
        # use cm_grid['layer'].to_dict() for node_names
        self.node_names = node_names
        if CM is None:
            CM = self.CM
        G = nx.from_scipy_sparse_array(CM, create_using=nx.MultiDiGraph, edge_attribute='weight', parallel_edges=True) # important to specify DiGraph (directed graph) and parallel edges
        if node_names is not None:
            nx.set_node_attributes(G, values=node_names, name='node_name')
        self.network = G
        return G

    def plot_network(self, node_shapes=None, show_map=False, rescale_edge=None, width=2, figsize=[7,7], extent=None, cmap=None):
        from pmar.plotting import _plot_network
        if node_shapes is True:
            node_shapes = self.cm_grid
            
        _plot_network(self.network, node_shapes=node_shapes, show_map=show_map, width=width, rescale_edge=rescale_edge, figsize=figsize, extent=extent, cmap=cmap)
        
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
    

    def run(self, pnum, start_time, duration, res, tstep=timedelta(hours=1), study_area=None, seeding_shapefile=None, seed_within_bounds=None, z=-0.5, seeding_radius=0, pnum_per_shape=False, beaching=False, hdiff=10, termvel=0, stokes_drift=False, decay_coef=0, use=None, use_label=None, emission=None, save_tiffs=False, crs='4326', output_dir=None):
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

        self.get_trajectories(pnum=pnum, start_time=start_time, tstep=tstep, duration_days=duration, seeding_shapefile=seeding_shapefile, seed_within_bounds=seed_within_bounds, z=z, seeding_radius=seeding_radius, beaching=beaching, hdiff=hdiff, pnum_per_shape=pnum_per_shape)

            
        ppi = self.get_ppi(res=res, study_area=study_area, use=use, use_label=use_label, emission=emission, decay_coef=decay_coef)#, block_size=len(self.ds.time)) # weight have been assigned above, so no need to normalize or apply decay etc.
            

        #thumb_ppi_path = str(ppi_path).split('.tif')[0]+'.png'
        #self.plot(self.output['ppi'], title=use_label, savepng=thumb_ppi_path)

        # other outputs
        # _h = self.get_histogram(res=res, study_area=study_area, block_size=len(self.ds.time), dim=['trajectory'])
        
        # self.output['sum'] = _h.sum('time').rio.reproject_match(self.grid.mask)
        # self.output['max'] = _h.max('time').rio.reproject_match(self.grid.mask)
        # self.output['q9'] = _h.quantile(q=.9, dim='time').rio.write_crs('epsg:4326').rio.reproject_match(self.grid.mask) # quantile method remove attributes for some reason, even with keep_attrs = True, so have to re-write crs.
        
        pass

    
    


class PMARmulti(PMAR): 
    """
    Developed at CNR ISMAR in Venice.

    ...

    Attributes
    ----------


    Methods
    -------


    """
        


    def get_trajectories(self, pnum, start_time, seed_time=timedelta(days=0), seedings=1, tshift=0, end_time=None, duration_days=30, study_area=None, seeding_shapefile = None, seed_within_bounds=None, z=-0.5, tstep=timedelta(hours=1), hdiff=10, termvel=0, crs='4326', seeding_radius=0, beaching=False, stokes_drift=False, pnum_per_shape=False):

        self.seedings = seedings
        self.tshift = tshift
        self.seeding_radius = seeding_radius
        
        for n in range(0, seedings):
    
            self.seeding_id = n

            start_time_dt = datetime.strptime(start_time, '%Y-%m-%d')+timedelta(days=tshift)*n #convert start_time into datetime to add tshift
            rep_start_time = start_time_dt.strftime("%Y-%m-%d") # bring back to string to feed to opendrift
            if end_time is not None:
                rep_duration = (datetime.strptime(end_time, '%Y-%m-%d') - start_time_dt).days
            else:
                rep_duration = duration_days
            
            logger.info(f'Starting rep #{self.seeding_id} with start_time = {rep_start_time}, end_time: {end_time}, and duration: {rep_duration}')
            
            #pnum = int(np.round(ptot/seedings)) #  ptot should be split among the seedings
            
            super().get_trajectories(pnum=pnum, start_time=rep_start_time, seed_time=seed_time, duration_days=rep_duration, study_area=study_area, seeding_shapefile=seeding_shapefile, seed_within_bounds=seed_within_bounds, z=z, tstep=tstep, hdiff=hdiff, termvel=termvel, crs=crs, seeding_radius=seeding_radius, beaching=beaching, stokes_drift=stokes_drift, pnum_per_shape=pnum_per_shape)
            logger.info(f'Done with rep #{n}.')
            self.particle_list.append(self.particle_path)

        self.import_trajectories(self.particle_list)

    def get_ppi(self, res, use=None, use_label=None, emission=None, study_area=None, decay_coef=0): 
        
        ppi_output_dir = Path(self.basedir) / f'ppi-{use_label}'

        self.ppi_cache = PMARCache(ppi_output_dir)
             
        ppi_path, ppi_exists = self.ppi_cache.raster_cache(context=self.spatial_domain, pressure=self.pressure, chemical_compound=self.chemical_compound, seeding_shapefile=self.seeding_shapefile, seeding_radius=self.seeding_radius, res=res, pnum=None, ptot=self.pnum, start_time=self.start_time, duration=self.duration, seedings=self.seedings, seeding_id=self.seeding_id, tshift=self.tshift, use=use, use_label=use_label, emission=emission, decay_coef=decay_coef, study_area=study_area)#, aggregate=aggregate, depth_layer=depth_layer, z_bounds=z_bounds, particle_status=particle_status, traj_dens=traj_dens)
        
        self.output = xr.Dataset()
        self.grid = make_grid(res=res, study_area=study_area)
        
        if ppi_exists == True:
            logger.info('PPI with requested configurations found in cache.')
            self.output['ppi'] = rxr.open_rasterio(ppi_path)
        else:
        
            for pfile in self.particle_list:
                temp_self = PMAR()
                temp_self.__dict__ = self.__dict__
                
                temp_self.import_trajectories(pfile)
                if hasattr(self, 'use') and self.use is not None: # to avoid re-rasterizing use for every seeding
                    use = self.use
                temp_self.get_ppi(res=res, use=use, use_label=use_label, emission=emission, study_area=study_area, decay_coef=decay_coef)
                self.ppi_list.append(temp_self.ppi_path)
                if hasattr(temp_self, 'use') and temp_self.use is not None: # to avoid errors if ppi already exists
                    self.use = temp_self.use
                  #  self.use_label = temp_self.use_label
                del temp_self

            self.import_trajectories(self.particle_list)
            
            self.output['ppi'] = self.sum_seedings(self.ppi_list).squeeze().rio.reproject_match(self.grid.mask.rename({'y_c':'y','x_c':'x'})).rename({'x':'lon','y':'lat'})/len(self.ds.seeding)
            self.write_tiff(self.output['ppi'], path=ppi_path)
            self.ppi_path = ppi_path
        
    

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
                    r1 = r0 + rxr.open_rasterio(rep)#.rio.reproject_match(r0, crs=r0.rio.crs, all_touched=True)
        
        return r1

    # TODO import trajectories from particle_list even if multippi already exists
    # can get particle_list from particle_cache
    def run(self, pnum, start_time, duration, res, seedings=1, tshift=0, end_time=None, tstep=timedelta(hours=1), study_area=None, seeding_shapefile=None, seed_within_bounds=None, z=-0.5, seeding_radius=0, pnum_per_shape=False, beaching=False, hdiff=10, termvel=0, stokes_drift=False, decay_coef=0, use=None, use_label=None, emission=None, output_dir=None, save_tiffs=False, thumbnail=None, crs='4326'):
        '''
        Compute trajectories and produce ppi raster over required number of seedings. 
    
        Parameters
        ----------
        pnum : int
    
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
             
        ppi_path, ppi_exists = self.ppi_cache.raster_cache(spatial_domain=self.spatial_domain, pressure=self.pressure, chemical_compound=self.chemical_compound, seeding_shapefile=seeding_shapefile, seeding_radius=seeding_radius, res=res, pnum=None, ptot=pnum, start_time=start_time, duration=duration, seedings=seedings, seeding_id=self.seeding_id, tshift=tshift, use=use, use_label=use_label, emission=emission, decay_coef=decay_coef, study_area=study_area)#, aggregate=aggregate, depth_layer=depth_layer, z_bounds=z_bounds, particle_status=particle_status, traj_dens=traj_dens)
        
        self.output = xr.Dataset()
        
        if ppi_exists == True:
            logger.info('PPI with requested configurations found in cache.')
            self.output['ppi'] = rxr.open_rasterio(ppi_path)
        else:
            self.get_trajectories(pnum=pnum, start_time=start_time, seedings=seedings, tshift=tshift, end_time=end_time, duration_days=duration, study_area=study_area, seeding_shapefile=seeding_shapefile, seed_within_bounds=seed_within_bounds, z=z, tstep=tstep, hdiff=hdiff, termvel=termvel, crs=crs, seeding_radius=seeding_radius, beaching=beaching, stokes_drift=stokes_drift, pnum_per_shape=pnum_per_shape)
            self.get_ppi(res=res, use=use, use_label=use_label, emission=emission, study_area=study_area, decay_coef=decay_coef)
            
        pass


    def get_Pij(self, cm_grid=None, res=None, study_area=None, t0=0, t1=-1, buffer=None, filenames=None):
        '''
        Average transition matrix 
        '''
        for i, file in enumerate(self.particle_list):
            logger.info(f'Calculating average CM out of {len(self.particle_list)} particle files.')
            ds = xr.open_dataset(file).isel(time=slice(t0,t1))
            temp_self = PMAR()
            temp_self.__dict__ = self.__dict__
            temp_self.import_trajectories(ds)
            if hasattr(self, 'use') and self.use is not None: # to avoid re-rasterizing use for every seeding
                temp_self.set_weights(use=self.use, res=res, study_area=study_area)
                weighted=True
            else:
                weighted=False

            if filenames is not None:
                filename = filenames[i]
                
            temp_self.get_Pij(cm_grid=cm_grid, res=res, study_area=study_area, t0=t0, t1=t1, buffer=buffer, filename=filename, weighted=weighted)
            if i == 0:
                Pij = temp_self.CM/len(self.particle_list)
            else:
                Pij = Pij + temp_self.CM/len(self.particle_list)
            del temp_self
            logger.info(f'Done calculating CM for particle file {i}.')
        self.CM = Pij
