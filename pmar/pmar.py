import logging
import rioxarray as rxr
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import shapely
from datetime import datetime, timedelta, date
from shapely.geometry import Point, Polygon
from opendrift.models.oceandrift import OceanDrift
from opendrift.models.openoil import OpenOil
from opendrift.models.plastdrift import PlastDrift
from opendrift.models.chemicaldrift import ChemicalDrift
from opendrift.models.larvalfish import LarvalFish
from opendrift.readers.reader_copernicusmarine import Reader
from opendrift.readers.unstructured import shyfem
from opendrift.readers.reader_netCDF_CF_generic import Reader as GenericReader
import opendrift
import time as T
from xhistogram.xarray import histogram
import os
from opendrift.readers import reader_shape
import cartopy.io.shapereader as shpreader
from pathlib import Path
#from flox.xarray import xarray_reduce # for xarray grouping over multiple variables
os.environ['PROJ_LIB'] = '/var/miniconda3/envs/opendrift/share/proj/'
from functools import partial
from rasterio.enums import Resampling
#import glob
#from geocube.api.core import make_geocube
import networkx as nx
from pmar.cache import PMARCache
from pmar.utils import traj_distinct, check_particle_file, get_marine_polygon, make_poly, harmonize_use, make_grid, make_vector_grid, get_copernicus, select_spatial_domain, get_emodnet_use
#from cmap import Colormap
from pmar.plotting import plot_map
from scipy.sparse import coo_array, diags
from scipy.sparse.linalg import matrix_power
from scipy.io import mmwrite
logger = logging.getLogger('pmar')

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
        
        
class PMAR(object):
    """
    Particle-based Modelling and Analysis for MARine Decision Support.

    Developed at CNR ISMAR in Venice. Drives OpenDrift (MET Norway) to simulate particle
    trajectories representing a given tracer (generic, plastic, chemical, oil, metal or
    larvae) released from a given source.

    Attributes
    ----------
    ds : xr.Dataset or None
        Imported particle trajectory data (lon, lat, z, age_seconds, origin_marker, ...).
    o : opendrift.models.basemodel.OpenDriftSimulation or None
        The active OpenDrift simulation object while a run is in progress.
    grid : xr.Dataset or None
        Regular spatial grid (cell edges/centers, bin indices) used for rasterization.
    cm_grid : geopandas.GeoDataFrame or None
        Vector (polygon) version of grid, used for connectivity matrix calculations.
    cm : scipy.sparse array or None
        Connectivity matrix between polygons in cm_grid.
    output : xr.Dataset or None
        Computed rasterized outputs (e.g. 'ppi', 'SUM', 'MAX', quantile indicators).
    cache : pmar.cache.PMARCache
        On-disk cache used to avoid recomputing trajectories/rasters for the same config.
    basedir : pathlib.Path
        Root output directory for this instance's simulation/cache/figure files.

    Methods
    -------
    See method docstrings below for the full pipeline: get_trajectories/import_trajectories,
    get_ppi/get_indicators, get_cm/get_Nij/predict_with_cm, and run() as the top-level
    single-seeding pipeline (get_trajectories -> get_ppi -> get_indicators).
    """


    def __init__(self, spatial_domain='auto', readers=None, reader_type=None, pressure='generic', chemical_compound=None, basedir='pmar_output', localdatadir = None, particle_path=None, netrppi_path=None, loglevel=10, log_to_file=False, _internal=False):
        """
        Initialize a PMAR instance: set up output/cache directories, default state, logging,
        the pressure/OpenDrift module, and the spatial domain/readers.

        `_internal` is for internal use only (see PMARmulti.get_ppi/get_cm's `temp_self`): when
        True, this call skips (re)configuring the shared module logger entirely, so a nested
        helper PMAR instance never clobbers the outer/real instance's log handlers or level.

        Parameters
        ----------
        spatial_domain : str or list, optional
            If str, an ocean basin for which CMEMS readers are available ('global', 'med',
            'black sea' or 'baltic'), or 'auto' to defer basin selection until
            get_trajectories() knows the seeding location (seed_within_bounds or
            seeding_shapefile), at which point the basin(s) whose Copernicus Marine data
            covers that seeding area are selected automatically (see
            pmar.utils.select_spatial_domain). Otherwise, a list of paths of files to be
            imported as OpenDrift readers. Default is 'global'
        readers : list, optional
            Local reader datasets/paths to add to OpenDrift directly (added as-is via
            o.add_reader), instead of readers derived from spatial_domain. Default is None.
        reader_type : str, optional
            Type of the local readers passed in `readers` (e.g. 'shyfem'). Currently unused
            by get_trajectories, kept for future reader-type-specific handling. Default is None.
        pressure : str, optional
            Type of pressure to simulate; one of 'generic', 'plastic', 'chemical', 'oil',
            'metal', 'larvae'. Selects the OpenDrift module via set_pressure(). Default is 'generic'.
        chemical_compound : str, optional
            if pressure is 'chemical', a chemical_compound can be specified to be initialised in ChemicalDrift
        basedir : str, optional
            path to the base directory where all output will be stored. Default is to create a directory called 'pmar_output' in the current directory.
        localdatadir : str, optional
            path to directory where input data (ocean, atmospheric) is stored. Default is None
        particle_path : str, optional
            path to netcdf file containing output of OpenDrift simulation. Default is None.
            If a particle_path is given in initialisation, it is stored on self.particle_path
            for later import (self.particle_path defaults to an empty list otherwise).
        netrppi_path : str, optional
            Currently unused (reserved). Default is None.
        loglevel : int, optional
            Logging level for the module logger (values below 10 are clamped to 10/NOTSET-safe).
            Default is 10 (DEBUG).
        log_to_file : bool, optional
            If True (default), in addition to the console, also write logger output to a
            timestamped text file in `basedir` (pmar_<timestamp>.log), at the same level as
            the console. Set to False to log to the console only.
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
        self.seeding_radius = None
        self.start_time = None
        
        # set up logger
        if not _internal:
            logformat = '%(asctime)s %(levelname)-7s %(name)s:%(lineno)d: %(message)s'
            datefmt = '%H:%M:%S'
            sf = logging.Formatter(logformat)

            if (logger.hasHandlers()):
                logger.handlers.clear()
            sh = logging.StreamHandler()
            sh.setFormatter(sf)
            logger.addHandler(sh)

            if log_to_file:
                log_filename = f"pmar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                fh = logging.FileHandler(self.basedir / log_filename)
                fh.setFormatter(sf)
                logger.addHandler(fh)

            if loglevel < 10:  # 0 is NOTSET, giving no output
                loglevel = 10
            logger.setLevel(loglevel)

        self.set_pressure(pressure)

        # define spatial_domain
        if spatial_domain == 'auto':
            # resolved later in get_trajectories(), once the seeding location is known
            self.spatial_domain = 'auto'
        elif spatial_domain is not None:
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
        """
        Select and store the OpenDrift simulation module for a given pressure type.

        Parameters
        ----------
        pressure : str
            One of 'generic', 'plastic', 'chemical', 'oil', 'metal', 'larvae'. Sets
            self.pressure and self.opendrift_module (the OpenDrift model class to
            instantiate in get_trajectories).
        """
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
        """
        Build a regular rectangular xr.Dataset grid over study_area at resolution res.

        Parameters
        ----------
        res : float
            Cell size (in crs units, typically degrees).
        study_area : list, optional
            Spatial bounds [xmin, ymin, xmax, ymax]. Default is None, i.e. self.study_area.
        crs : str, optional
            CRS to write on the resulting grid. Default is 'epsg:4326'.

        Returns
        -------
        xr.Dataset
            Grid with cell edges (x_e, y_e), cell centers (x_c, y_c), cell size (dx, dy),
            meshgrid coordinates (Xc, Yc), a ones array, and a per-cell index array (bin_n).
        """
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

    
    def import_trajectories(self, files_to_import, align_by_age=False): #, align_by = 'time'):
        """
        Load OpenDrift trajectory output into self.ds.

        Parameters
        ----------
        files_to_import : str or list of str or xr.Dataset
            A single path to an OpenDrift output netCDF file, a list of paths (opened and
            concatenated along a new 'seeding' dimension via xr.open_mfdataset, padding
            seedings of different lengths with NaNs), or an already-open xr.Dataset.
        align_by_age : bool, optional
            If True, roll each trajectory's time series (via the nested roll_by_release
            helper) so its first valid timestep is at time index 0, then drop trailing NaNs.
            Useful when trajectories were seeded at different times. Default is False.

        Notes
        -----
        Sets self.ds to the ['lon', 'lat', 'z', 'age_seconds', 'origin_marker'] subset of the
        opened dataset. When files_to_import is a list, also sets self.particle_list and
        self.seedings.
        """
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

        if align_by_age == True:
            logger.warning('rolling trajectories')
            
            def roll_by_release(da):
                """
                Roll a (trajectory, time) DataArray so that the first valid value
                of each trajectory is at time index 0.
                """
            
                # Index of first non-NaN value along time
                shift = da.isnull().argmin("time").values
            
                nt = da.sizes["time"]
            
                idx = (np.arange(nt)[None, :] + shift[:, None]) % nt
            
                return xr.DataArray(
                    np.take_along_axis(da.values, idx, axis=1),
                    dims=da.dims,
                    coords=da.coords,
                    attrs=da.attrs,
                    name=da.name,
                )

            for var in self.ds.data_vars:
                self.ds[var] = roll_by_release(self.ds[var])
            # drop tail of nans
            self.ds = self.ds.dropna('time')
            # add some tests to check it all went smoothly: age_seconds should be 0 for all trajectories at time 0 , e.g.
            # np.unique(self.ds.age_seconds.isel(time=0)) == 0
            
            # swap dimensions time and age
            #age = self.ds.age_seconds.isel(trajectory=0)
            #self.ds = self.ds.assign_coords(age_seconds=("time", age.data))
            #self.ds = self.ds.swap_dims({"time": "age_seconds"}).drop('time')
            
        pass

    def import_ppi(self, files_to_import):
        """
        Load one or more previously-computed PPI raster file(s) as a DataArray.

        Parameters
        ----------
        files_to_import : str or list of str
            Path(s) to raster file(s) to open (via rasterio), concatenated along a new
            'seeding' dimension if multiple are given. Also stored on self.ppi_list.

        Returns
        -------
        xr.DataArray
            The 'band_data' variable of the opened (and squeezed) dataset.
        """
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
         
    
    def get_trajectories(self, pnum, start_time, seed_time=timedelta(days=0), duration_days=30, seeding_shapefile = None, seed_within_bounds=None, seed_from_file=None, z=-0.5, tstep=timedelta(hours=1), hdiff=10, termvel=0, crs='4326', seeding_radius=0, beaching=False, stokes_drift=False, pnum_per_shape=False):
        """
        Calculate trajectories using Oceandrift module by OpenDrift (MET Norway).
        Uses OceanDrift module. 
                
        This method is currently optimised for use over the Mediterranean and Black Sea only. 
        Forcing data is streamed from Copernicus (ocean currents, wind, mixed layer depth), while GEBCO bathymetry is stored locally. 
        
        Particles are seeded homogeneously over polygon created using make_poly() method. 
        
        
        Parameters
        ----------
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
            If self.spatial_domain is 'auto', this (or seed_within_bounds) is required, and
            its total_bounds is used to auto-select the covering Copernicus basin(s).
        seed_within_bounds : list, optional
            Spatial bounds for seeding written as bounds=[x1,y1,x2,y2]. Default is None, i.e. the whole basin is covered.
            If self.spatial_domain is 'auto', this (or seeding_shapefile) is required, and
            is used to auto-select the covering Copernicus basin(s) (see
            pmar.utils.select_spatial_domain).
        seed_from_file : str, optional
            Path to an OpenDrift output netcdf file to seed particles from (via seed_from_file()), e.g. to continue a previous simulation. Default is None.
        z : float, optional
            Depth at which to seed particles [m]. Default is -0.5m.
        tstep : timedelta, optional
            Time step used for OpenDrift simulation. Default is 1 hour
        hdiff : float, optional
            Horizontal diffusivity of particles, in [m2/s]. Default is 10m2/s
        termvel : float, optional
            Terminal velocity representing buoyancy of particles, in [m/s]. Default is 0. Positive means rising up, negative sinking down
        crs : str, optional
            EPSG string for polygon. Default is 4326
        seeding_radius : float, optional
            Opendrift "radius".
        beaching : bool, optional
            Whether particles reaching the coastline are allowed to beach ('stranding'). Default is False, i.e. particles that reach the coast do not get stuck ('previous').
        stokes_drift : bool, optional
            Whether to include Stokes drift in the simulation. Default is False.
        pnum_per_shape : bool, optional
            If True, seed pnum particles within each individual shape of seeding_shapefile (via seed_from_geopandas()), rather than pnum particles distributed across the whole shapefile. Default is False.
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
            if self.spatial_domain == 'auto':
                raise ValueError("spatial_domain='auto' requires seed_within_bounds or seeding_shapefile to be given, since there is no seeding location to auto-detect a basin from.")
            seed_within_bounds = self.spatial_domain['extent']

        if self.spatial_domain == 'auto':
            seeding_bounds = seed_within_bounds if seed_within_bounds is not None else gpd.read_file(seeding_shapefile).total_bounds
            self.spatial_domain = select_spatial_domain(seeding_bounds)
            self.extent = self.spatial_domain['extent']
            logger.info(f"auto-selected spatial_domain: {self.spatial_domain['label']}")

        if seed_within_bounds is not None:
            poly_path = f'polygon-crs_epsg:{crs}-lon_{np.round(seed_within_bounds[0],4)}_{np.round(seed_within_bounds[2],4)}-lat—{np.round(seed_within_bounds[1],4)}_{np.round(seed_within_bounds[3])}.shp'
            q = self.basedir / 'polygons' / poly_path
            make_poly(seed_within_bounds, crs=crs, save_to=str(q))
            # COULD REPLACE THIS WITH VECTOR GRID
            seeding_shapefile = str(q)
            logger.info(f'created seeding_shapefile with bounds {seed_within_bounds}')


        self.seeding_shapefile = seeding_shapefile 
        self.seed_within_bounds = seed_within_bounds
        logger.debug(f'seed_within_bounds = {self.seed_within_bounds}, seeding_shapefile = {self.seeding_shapefile}')

        file_path, file_exists = self.cache.particle_cache(context=self.spatial_domain, pressure=self.pressure, chemical_compound=self.chemical_compound, seeding_shapefile=seeding_shapefile, pnum=pnum, start_time=start_time, seed_time=seed_time, duration_days=duration_days, seed_within_bounds=seed_within_bounds, seed_from_file=seed_from_file, seeding_radius=seeding_radius, beaching=beaching, z=z, tstep=tstep, hdiff=hdiff, termvel=termvel, stokes_drift=stokes_drift, seeding_id=self.seeding_id)

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
            
            self.o.set_config('environment:constant:horizontal_diffusivity', hdiff)  # horizontal diffusivity
            self.o.set_config('drift:advection_scheme', 'runge-kutta4') # advection schemes (default is 'euler'). other options are 'runge-kutta', 'runge-kutta4'
            
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
                                
    
            #logger.info('adding landmask...')
            # landmask from cartopy (from "use shapefile as landmask" example on OpenDrift documentation)
            #shpfilename = shpreader.natural_earth(resolution='10m',
             #                       category='cultural',
              #                      name='admin_0_countries')
            #reader_landmask = reader_shape.Reader.from_shpfiles(shpfilename)
            
            #self.o.add_reader([reader_landmask])
            self.o.set_config('general:use_auto_landmask', True)  # False disables the automatic GSHHG landmask
            
            logger.info('adding readers...')            
            
            if self.readers:
                # add local datasets as readers. added for sl application
                for id,r in enumerate(self.readers):
                    #if self.reader_type == 'shyfem':
                      #  reader = shyfem.Reader(r)
                    #else:
                     #   reader = GenericReader(r, name=f'cmems_{id}')
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
            if seed_from_file is not None:
                self.o.seed_from_file(seed_from_file)
            elif pnum_per_shape is True:
                seeding_polygon = gpd.read_file(self.seeding_shapefile)
                for shape in range(len(seeding_polygon)):
                    self.o.seed_from_geopandas(geodataframe=seeding_polygon.iloc[[shape]], number=pnum, time=[start_time - seed_time, start_time], terminal_velocity=termvel, z=z, origin_marker=self.seeding_id)
            else:
                self.o.seed_from_shapefile(shapefile=self.seeding_shapefile, number=pnum, time=[start_time - seed_time, start_time], terminal_velocity=termvel, z=z, origin_marker=self.seeding_id, radius=seeding_radius)
    
            
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

    def _continue_trajectories(self, seed_from_file, duration_days):
        '''
        Continue a previous OpenDrift run by re-seeding particles from its output file.

        Recovers pnum, start_time, tstep, termvel and beaching config from the attrs/data of
        an existing OpenDrift output file, then calls get_trajectories(seed_from_file=...) to
        extend the simulation for a further duration_days. USE WITH CAUTION: only the settings
        recovered here are reused; other get_trajectories parameters fall back to defaults.

        Parameters
        ----------
        seed_from_file : str
            Path to an existing OpenDrift output netCDF file to resume from.
        duration_days : int
            Additional duration, in days, to run the continued simulation for.

        Returns
        -------
        The return value of get_trajectories() (None; trajectories are stored on self.ds).
        '''
        ds = xr.open_dataset(seed_from_file)
        pnum = len(ds.trajectory)
        start_time = str(ds.time.max().data)[:10]
        tstep = pd.to_timedelta(ds.time_step_calculation).to_pytimedelta()
        termvel = ds.attrs['config_seed:terminal_velocity']
        if ds.attrs['config_general:coastline_action'] == 'stranding':
            beaching = True
        else:
            beaching = False
        return self.get_trajectories(pnum=pnum, start_time=start_time, tstep=tstep, duration_days=duration_days, termvel=termvel, beaching=beaching, seed_from_file=seed_from_file, hdiff=10)

    
    def plot_trajectories(self, t=None, filename=None, s=1, c='age', cmap='viridis', alpha=None, proj=None):
        """
        Scatter-plot particle positions from self.ds on a map.

        Parameters
        ----------
        t : int or slice, optional
            Timestep index (or slice) to plot. Default is None, i.e. a subsampled range of
            timesteps (and, for large datasets, trajectories) is plotted for readability.
        filename : str, optional
            Path to save the figure to. Default is None (figure is not saved).
        s : float, optional
            Marker size for the scatter plot. Default is 1.
        c : {'age', 'z', 'marker'}, optional
            What to color points by: elapsed time in days ('age'), depth ('z'), or
            origin_marker ('marker'). Default is 'age'.
        cmap : str, optional
            Colormap; overridden internally based on `c`. Default is 'viridis'.
        alpha : float, optional
            Marker transparency. Default is None.
        proj : cartopy.crs.Projection, optional
            Map projection. Default is None, i.e. cartopy.crs.PlateCarree().
        """
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
        """
        Normalize each trajectory's weight by dividing it by the number of particles that were
        seeded within the same spatial bin at the first timestep (t=0).

        This corrects for uneven seeding density in space, so that weight differences between
        bins reflect trajectory behaviour (e.g. use, emission, decay) rather than simply how
        many particles happened to start in that bin.

        Parameters
        ----------
        weight : xr.DataArray
            Per-trajectory weight to be normalized, with dimensions matching self.ds (e.g.
            'trajectory', 'time').
        res : float
            Cell size, i.e. the resolution of the grid used to bin trajectories at t=0. Passed
            to get_bin_n().
        study_area : optional
            Spatial extent used to build the grid. Default is None, i.e. the whole domain.

        Returns
        -------
        xr.DataArray
            weight divided by the number of particles seeded in the same bin at t=0, grouped
            by bin_n_t0.
        """
        # dividing each trajectory weight by the number of particles that were in the same bin at t0

        bin_n_t0 = self.get_bin_n(res=res, t=0, study_area=study_area)
        self.ds['bin_n_t0'] = bin_n_t0

        _, inverse, counts = np.unique(bin_n_t0.values, return_inverse=True, return_counts=True)
        counts_per_bin = bin_n_t0.copy(data=counts[inverse])

        logger.info('counts_per_bin...')
        normalized_weight = weight / counts_per_bin
        normalized_weight = normalized_weight.assign_coords(bin_n_t0=self.ds['bin_n_t0'])

        return normalized_weight

    def set_weights(self, res=None, study_area=None, weight=1, use=None, emission=None, decay_coef=0, normalize=False, assign=True):
        '''
        Associate weights with each trajectory.
        ORDER OF OPERATIONS MATTERS.
        Weights depend on one or more of the following components: use value in each cell (use must be provided as quantity/day), emission (quantity/day), decay coefficient, number of trajectories starting in the same cell (normalize parameter).
        use may also be given as a string, in which case it is treated as an EMODnet
        Human Activities dataset identifier (as returned by pmar.utils.list_emodnet_uses())
        and fetched via pmar.utils.get_emodnet_use(use, bbox=study_area) before being
        passed on to harmonize_use(); this requires study_area (or self.study_area) to be
        set, and raises ValueError if the identifier isn't recognised or returns no data
        within study_area.
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
            if isinstance(use, str):
                if study_area is None:
                    raise ValueError(f"use='{use}' is an EMODnet dataset identifier, which requires study_area (or self.study_area) to be set, to know where to fetch it from.")
                use_name = use
                logger.info(f"use is a string; fetching '{use_name}' from EMODnet Human Activities within study_area {study_area}...")
                use = get_emodnet_use(use_name, bbox=study_area)
                use_empty = use.empty if hasattr(use, 'empty') else bool(np.isnan(use).all())
                if use_empty:
                    raise ValueError(f"EMODnet dataset '{use_name}' returned no data within study_area {study_area}.")
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
        Look up each trajectory's grid-cell "bin number" at timestep t, i.e. a unique
        identifier corresponding to a specific spatial grid-cell, via nearest-neighbor
        lookup of (lon, lat) into self.grid.bin_n.

        Parameters
        ----------
        res : float
            Currently unused directly (grid resolution is taken from self.grid); kept for
            API consistency with sibling methods.
        t : int, optional
            Index of timestep to consider. Default is None, i.e. bin numbers are computed
            for every timestep.
        study_area : optional
            Currently unused directly; kept for API consistency with sibling methods.
        ds : xr.Dataset, optional
            Trajectory dataset to look up positions from. Default is None, i.e. self.ds.

        Returns
        -------
        xr.DataArray
            Bin number(s) matching the trajectory (and, if t is None, time) dimensions of ds.
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

    def get_histogram(self, res, study_area=None, weighted=True, dim=['trajectory', 'time'], t=None, block_size='auto'):
        '''
        Compute a 2D (lon, lat) histogram of particle positions on a grid of resolution res.

        "Density trend (DT). The particle DT reflects the number of particles that have
        visited each grid cell during a certain time interval." (Stanev et al., 2019)

        Parameters
        ----------
        res : float
            Cell size, aka desired resolution of output map. Used to (re)build self.grid.
        study_area : list, optional
            Spatial bounds [xmin, ymin, xmax, ymax] for the grid. Default is None, i.e.
            self.study_area.
        weighted : bool, optional
            Whether to weight the histogram by self.ds.weight (NaNs filled with 0). If
            False, an unweighted (count) histogram is computed. Default is True.
        dim : list of str, optional
            Dimensions of self.ds to reduce over when binning. Default is
            ['trajectory', 'time'] (i.e. sum over both).
        t : int, optional
            If given, restrict to this single timestep index before binning (and dim is
            forced to None). Default is None, i.e. all timesteps are used.
        block_size : optional
            Passed through to xhistogram.xarray.histogram for chunked computation. Default
            is 'auto'.

        Returns
        -------
        xr.DataArray
            Histogram of particle counts (or weighted sums) per grid cell, with dims
            renamed to ('x', 'y') and CRS/nodata written.
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
        


        if t is not None:
            ds = self.ds.isel(time=t)
            dim = None #['trajectory','time']
            print(f'calculating histogram at time {t}')
        else:
            ds = self.ds 
        
        #NOTE: NaNs in weights will make the weighted sum as nan. To avoid this, call .fillna(0.) on your weights input data before calling histogram().
        if weighted == False:
            weights = None
        else:
            weights = ds.weight.fillna(0.)
        
        h = histogram(ds.lon, ds.lat, bins=[xbin, ybin], dim=dim, weights=weights, block_size=block_size) 
        # block_size='auto' was giving division by 0 error, which is a known bug: https://github.com/xgcm/xhistogram/issues/16
        #NOTE: NaNs in weights will make the weighted sum as nan. To avoid this, call .fillna(0.) on your weights input data before calling histogram().
            
        h = h.transpose().rename({'lon_bin':'x', 'lat_bin':'y'}) # important to not have it where(h>0) otherwise it misses values when summing

        r = (
            xr.DataArray(h) 
            .rio.write_nodata(0)#np.nan)
            .rio.write_crs('epsg:4326')
            .rio.write_coordinate_system())

        return r    

    def get_indicators(self, res, study_area, quantile=.9):
        """
        Compute summary indicator rasters (SUM, MAX, and a quantile) from the per-timestep
        trajectory histogram, and store them in self.output.

        Parameters
        ----------
        res : float
            Resolution used to bin trajectories (passed to get_histogram).
        study_area : list
            Spatial bounds [xmin, ymin, xmax, ymax] for the grid.
        quantile : float, optional
            Quantile (over time) to compute in addition to SUM/MAX. Default is 0.9, stored
            as self.output['Q90'].

        Notes
        -----
        Sets self.output['SUM'], self.output['MAX'], and self.output[f'Q{quantile*100}'],
        each reprojected to match self.grid.mask.
        """
        # other indicators
        if hasattr(self.ds, 'weight'):
            weighted = True
        else:
            weighted = False
            
        _h = self.get_histogram(res=res, study_area=study_area, block_size=len(self.ds.time), dim=['trajectory'], weighted=weighted)

        if not self.output:
            self.output = xr.Dataset()
        
        self.output['SUM'] = _h.sum('time').rio.reproject_match(self.grid.mask)
        self.output['MAX'] = _h.max('time').rio.reproject_match(self.grid.mask)
        self.output[f'Q{str(int(quantile*100))}'] = _h.quantile(q=quantile, dim='time').rio.write_crs('epsg:4326').rio.reproject_match(self.grid.mask) # quantile method remove attributes for some reason, even with keep_attrs = True, so have to re-write crs.
    
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
    
            ppi = r / len(self.ds.time) # divide by number of timesteps so that it is independent of temporal resolution (timestep)
                
            self.output['ppi'] = ppi.rio.reproject_match(self.grid.mask)#.rename({'x':'lon', 'y':'lat'}) # changing coordinate names because there was an issue with precision. original dataset coords have higher precision than coords in raster 

            self.write_tiff(ppi, path=ppi_path)
        
        self.ppi_path = ppi_path
            
        pass

    def plot_ppi(self, proj=None, cmap=None, filename=None):
        """
        Plot self.output.ppi (masked to values > 0) on a map.

        Parameters
        ----------
        proj : cartopy.crs.Projection, optional
            Map projection. Default is None, i.e. cartopy.crs.epsg(4326).
        cmap : str, optional
            Colormap for the PPI raster. Default is None (matplotlib default).
        filename : str, optional
            If given, filename (relative to self.basedir/figures) to save the figure to.
            Default is None (figure is not saved).
        """
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

    def get_particles_by_cell(self, t, ds=None, cm_grid=None, res=None, study_area=None, weighted=False, buffer=None):
        """
        Create dictionary ll containing, for each polygon in cm_grid, the list of particles located in that polygon at time t.
        """
        if study_area is None:
            study_area = self.study_area
            
        # create "cm_grid"
        if cm_grid is None: # if no polygons are given, consider entire grid
            if res is not None: # if a new resolution is given, create new grid
                grid = make_grid(res=res, study_area=study_area)
                self.grid = grid
            else: # otherwise consider existing grid defined in .run()
                grid = self.grid
            #cm_grid = _make_poly(lon=grid.x_c, lat=grid.y_c) 
            cm_grid = make_vector_grid(grid.bin_n.astype(float))

        # add buffer 
        if buffer is not None:
            buff = cm_grid.dissolve().buffer(buffer)
            diff = gpd.GeoDataFrame({cm_grid.columns.drop('geometry')[0]: -1234, 'geometry': buff.difference(cm_grid.dissolve())}, crs=cm_grid.crs)
            cm_grid = pd.concat([cm_grid, diff]).reset_index(drop=True)
        
        self.cm_grid = cm_grid

        if ds is None:
            ds = self.ds
        
        ll = {}
        # select which columns to include in geodataframe (whether only lat, lon or also weight)
        df_cols = ['lon','lat']
        if 'weighted' == True:
            if 'weight' in ds.data_vars:
                cols.append('weight')
            else:
                raise ValueError('no weight variable found in ds')

        # # create geodataframe with particle positions at time t
        # p_df = ds.isel(time=t)[df_cols].to_dataframe().reset_index()
        # p_gdf = gpd.GeoDataFrame(
        #                         p_df,
        #                         geometry=gpd.points_from_xy(p_df.lon, p_df.lat),
        #                         crs=cm_grid.crs
        #                         )

        # create shapely points
        points = shapely.points(ds.lon.isel(time=t).values.ravel(),
                        ds.lat.isel(time=t).values.ravel())


        # create dictionary ll containing list of trajectory ids present in each polygon at time t
        # uses a spatial index (STRtree) to do the point-in-polygon assignment in one vectorized
        # query instead of looping shapely.within(points, poly) once per cell (O(n_cells x n_traj))
        n_cells = len(cm_grid)
        tree = shapely.STRtree(cm_grid.geometry.values)
        point_idx, cell_idx = tree.query(points, predicate='within')
        order = np.argsort(cell_idx, kind='stable')
        cell_idx_sorted = cell_idx[order]
        point_idx_sorted = point_idx[order]
        unique_cells, starts = np.unique(cell_idx_sorted, return_index=True)
        groups = np.split(point_idx_sorted, starts[1:])
        ll = {i: np.array([], dtype=int) for i in range(n_cells)}
        for c, g in zip(unique_cells, groups):
            ll[int(c)] = g
        return ll

    def _ll_to_indicator(self, ll, n_cells, n_traj):
        """
        Build a sparse (n_cells x n_traj) 0/1 matrix from a get_particles_by_cell-style {cell: trajectory_ids} dict,
        so that pairwise trajectory co-occurrence counts between two such dicts can be computed with
        a single sparse matrix multiply (A0 @ A1.T) instead of an O(n_cells^2) loop with set intersections.
        """
        if ll:
            rows = np.concatenate([np.full(len(ids), i) for i, ids in ll.items()])
            cols = np.concatenate([np.asarray(ids) for ids in ll.values()])
        else:
            rows = cols = np.array([], dtype=int)
        data = np.ones(len(rows))
        return coo_array((data, (rows, cols)), shape=(n_cells, n_traj)).tocsr()

    def get_Nij(self, t, cm_grid=None, res=None, study_area=None, weighted=False, buffer=None):
        """
        Calculate sparse adjacency matrix Nij from particle positions at t=0 and t=t.

        Returns
        -------
        Nij_sparse : scipy.sparse array
            Adjacency matrix of particle counts from cell i at t=0 to cell j at t.
        Ni0 : numpy.ndarray
            Number of particles actually found in each cell i at t=0 (A0.sum(1)). Exposed
            so get_cm can normalize rows by the true departing cohort of each cell rather
            than by Nij's own row-sum, which undercounts whenever particles leave cm_grid
            entirely by time t and are therefore absent from every destination column.
        """
        ll0 = self.get_particles_by_cell(t=0, cm_grid=cm_grid, res=res, study_area=study_area, buffer=buffer, weighted=weighted)
        ll1 = self.get_particles_by_cell(t=t, cm_grid=cm_grid, res=res, study_area=study_area, buffer=buffer, weighted=weighted)

        n_cells = len(self.cm_grid) # needs to be self.cm_grid otherwise it does not take buffer
        n_traj = self.ds.sizes['trajectory']
        A0 = self._ll_to_indicator(ll0, n_cells, n_traj)
        A1 = self._ll_to_indicator(ll1, n_cells, n_traj)
        Nij_sparse = coo_array(A0 @ A1.T)
        Ni0 = np.asarray(A0.sum(1)).ravel()
        return Nij_sparse, Ni0

    def get_cm(self, t=-1, cm_grid=None, res=None, study_area=None, weighted=False, buffer=None, filename=None):
        """
        Calculate the connectivity matrix (cm): the Markov transition matrix obtained by
        normalizing each row i of adjacency matrix Nij by Ni, i.e. the actual number of
        particles found in cell i at t=0.
        Entries of this matrix contain the probability of going from cell i to cell j after exactly t timesteps from t0.
        Rows only sum to 1 if every particle found in cell i at t=0 is still found somewhere in
        cm_grid (domain + buffer, if used) at time t; particles that leave cm_grid entirely by
        time t are not found in any destination cell, so they correctly show up as a row sum < 1
        (leaked probability mass) rather than being silently dropped from the normalization.
        If buffer is given, the buffer cell (last row of cm_grid, appended by get_particles_by_cell) is forced
        into an absorbing state: P(buffer -> buffer) = 1 and P(buffer -> j) = 0 for every other
        cell j, so particles that leak out of the tracked domain into the buffer are considered
        permanently gone from the system rather than being able to transition back.
        """
        Nij, Ni = self.get_Nij(t=t, cm_grid=cm_grid, res=res, study_area=study_area, buffer=buffer, weighted=weighted)
        cm = np.zeros_like(Ni, dtype=float)
        # mask to avoid division by 0
        mask = Ni != 0
        # divide Nij by Ni, i.e. number of particles coming from cell i
        cm[mask] = 1.0 / Ni[mask]
        cm_sparse = diags(cm) @ Nij # matrix multipli Nij with diagonal matrix with reciprocal of Ni

        if buffer is not None:
            # buffer cell is always the last row of cm_grid (see get_particles_by_cell); make it absorbing
            buffer_idx = cm_sparse.shape[0] - 1
            cm_sparse = cm_sparse.tolil()
            cm_sparse[buffer_idx, :] = 0
            cm_sparse[buffer_idx, buffer_idx] = 1
            cm_sparse = cm_sparse.tocsr()

        if filename is not None:
            mmwrite(filename, cm_sparse)
            logger.info(f'wrote file {filename}')

        self.cm = cm_sparse
        return cm_sparse

    def get_cm_within_range(self, t_max, cm_grid=None, res=None, study_area=None, weighted=False, buffer=None, filename=None):
        """
        Calculate the Lagrangian connectivity matrix (cm) containing the probability of passing through cell j from cell i within t_max timesteps, starting from t=0.

        If buffer is given, the buffer cell (last row of cm_grid, appended by get_particles_by_cell) is forced
        into an absorbing state: P(buffer -> buffer) = 1 and P(buffer -> j) = 0 for every other
        cell j, so particles that leak out of the tracked domain into the buffer are considered
        permanently gone from the system rather than being able to transition back.
        """
        # generate list of trajectories in each polygon at time 0 ll0
        ll0 = self.get_particles_by_cell(0, res=res, study_area=study_area, cm_grid=cm_grid, buffer=buffer, weighted=weighted)
        # generate list of trajectories in each polygon at all other timesteps llt
        for t in range(1, t_max):
            _llt = self.get_particles_by_cell(t, res=res, study_area=study_area, cm_grid=cm_grid, buffer=buffer, weighted=weighted)
            if t == 1:
                llt = _llt
            else:
                llt = {k: list(set(llt[k]) | set(_llt[k])) for k in range(len(self.cm_grid))} # union

        # count the intersection between ll0 and llt for each polygon pair, and divide it by ll0 in the source polygon (number of particles starting from that polygon)
        n_cells = len(self.cm_grid)
        n_traj = self.ds.sizes['trajectory']
        A0 = self._ll_to_indicator(ll0, n_cells, n_traj)
        At = self._ll_to_indicator(llt, n_cells, n_traj)
        Nij = A0 @ At.T
        Ni0 = np.asarray(A0.sum(1)).ravel()
        cm = np.zeros_like(Ni0, dtype=float)
        mask = Ni0 != 0
        cm[mask] = 1.0 / Ni0[mask]
        cm_sparse = coo_array(diags(cm) @ Nij)

        if buffer is not None:
            # buffer cell is always the last row of cm_grid (see get_particles_by_cell); make it absorbing
            buffer_idx = cm_sparse.shape[0] - 1
            cm_sparse = cm_sparse.tolil()
            cm_sparse[buffer_idx, :] = 0
            cm_sparse[buffer_idx, buffer_idx] = 1
            cm_sparse = coo_array(cm_sparse)

        if filename is not None:
            mmwrite(filename, cm_sparse)
            logger.info(f'wrote file {filename}')

        return cm_sparse


    
    def _get_Pij(self, t=-1, cm_grid=None, res=None, study_area=None, buffer=None, filename=None, weighted=False):
        """
        Legacy/alternate implementation of get_cm: calculate the Markov transition matrix
        Pij by normalizing the adjacency matrix Nij (co-occurrence of trajectories in cell i
        at t=0 and cell j at t) by Ni, the number of particles coming from cell i.

        Parameters
        ----------
        t : int, optional
            Timestep index (relative to t=0) to compute transitions to. Default is -1.
        cm_grid : geopandas.GeoDataFrame, optional
            Polygon grid to use; default is None, i.e. built from self.grid via get_particles_by_cell.
        res : float, optional
            Resolution for building a new grid, if cm_grid is None. Default is None.
        study_area : list, optional
            Spatial bounds for the grid. Default is None, i.e. self.study_area.
        buffer : float, optional
            Buffer distance added around cm_grid as an extra "outside" node. Default is None.
        filename : str, optional
            If given, path to write the resulting sparse matrix to (Matrix Market format).
            Default is None.
        weighted : bool, optional
            Currently unused directly in this method body; kept for API consistency with
            get_particles_by_cell. Default is False.

        Returns
        -------
        scipy.sparse array
            The transition matrix Pij_sparse (also stored on self.CM). If buffer is given,
            the buffer cell (last row of cm_grid, appended by get_particles_by_cell) is forced into an
            absorbing state: P(buffer -> buffer) = 1 and P(buffer -> j) = 0 for every other
            cell j, so particles that leak out of the tracked domain into the buffer are
            considered permanently gone from the system rather than being able to transition
            back.
        """
        ll0 = self.get_particles_by_cell(t=0, cm_grid=cm_grid, res=res, study_area=study_area, buffer=buffer, weighted=weighted)
        ll1 = self.get_particles_by_cell(t=t, cm_grid=cm_grid, res=res, study_area=study_area, buffer=buffer, weighted=weighted)

        n_cells = len(self.cm_grid)
        n_traj = self.ds.sizes['trajectory']
        A0 = self._ll_to_indicator(ll0, n_cells, n_traj)
        A1 = self._ll_to_indicator(ll1, n_cells, n_traj)
        Nij = A0 @ A1.T
        Ni0 = np.asarray(A0.sum(1)).ravel()
        Pij = np.zeros_like(Ni0, dtype=float)
        mask = Ni0 != 0
        Pij[mask] = 1.0 / Ni0[mask]
        Pij_sparse = coo_array(diags(Pij) @ Nij)

        if buffer is not None:
            # buffer cell is always the last row of cm_grid (see get_particles_by_cell); make it absorbing
            buffer_idx = Pij_sparse.shape[0] - 1
            Pij_sparse = Pij_sparse.tolil()
            Pij_sparse[buffer_idx, :] = 0
            Pij_sparse[buffer_idx, buffer_idx] = 1
            Pij_sparse = coo_array(Pij_sparse)

        #self.cm_grid = cm_grid
        self.CM = Pij_sparse

        if filename is not None:
            mmwrite(filename, Pij_sparse)
            logger.info(f'wrote file {filename}')
        
        return Pij_sparse

    # to update
    def get_Pji(self, cm_grid=None, res=None, study_area=None, t0=0, t1=-1, filename=True):
        """
        Calculate reverse transition probabilities Pji: the probability that a particle
        found in cell j at t1 originated from cell i at t0, for each pair of cells.

        Parameters
        ----------
        cm_grid : geopandas.GeoDataFrame, optional
            Polygon grid to use. Default is None.
        res : float, optional
            Resolution for building a new grid, if cm_grid is None. Default is None.
        study_area : list, optional
            Spatial bounds for the grid. Default is None, i.e. self.study_area.
        t0 : int, optional
            Origin timestep index. Default is 0.
        t1 : int, optional
            Destination timestep index. Default is -1.
        filename : str or bool, optional
            If given (and not None), path to write the resulting sparse matrix to (Matrix
            Market format). Default is True.

        Returns
        -------
        scipy.sparse array
            The reverse transition matrix Pji_sparse.
        """
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
    

    def plot_cm(self, tick_labels=None, figsize=[8,6], annot=False, linewidths=0, vmin=0, vmax=None, norm=None, cmap='Greys', mask=None):
        """
        Plot self.cm (densified) as a matrix heatmap.

        Parameters
        ----------
        tick_labels : list, optional
            Labels for matrix rows/columns (e.g. from cm_grid). Default is None.
        figsize : list, optional
            Figure size [width, height]. Default is [8, 6].
        annot : bool, optional
            Whether to annotate cells with their (rounded) values. Default is False.
        linewidths : float, optional
            Width of lines between cells. Default is 0.
        vmin, vmax : float, optional
            Color scale bounds. Default is 0 and None.
        norm : optional
            Matplotlib color normalization. Default is None.
        cmap : str, optional
            Colormap. Default is 'Greys'.
        mask : float, optional
            If given, cells with value <= mask are masked out of the plot. Default is None.
        """
        # use cm_grid for tick_labels
        from pmar.plotting import plot_matrix
        import matplotlib.pyplot as plt
        cm = self.cm.toarray() # my cm is now by default a sparse array. to plot it, must convert it to a dense array

        if mask is not None:
            mask = cm<=mask

        if annot == True:
            cm = np.round(cm,2)

        fig, ax = plt.subplots(figsize=figsize)
        plot_matrix(cm, tick_labels=tick_labels, annot=annot, linewidths=linewidths, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, mask=mask, ax=ax)

    def dispersal_distance(self, cm=None, cm_grid=None):
        '''
        "as the crow flies" dispersal distance from connectiviy matrix
        '''
        if cm_grid is None:
            cm_grid = self.cm_grid

        if cm is None:
            cm = self.cm

        cm_grid = cm_grid.to_crs(32601)

        disp_dist = np.zeros(len(cm_grid))
        for i in range(len(cm_grid)):
            cm_row = cm.tocsr()[i,:] # convert to scipy.csr so i can index by row
            if cm_row.sum()>0:
                D_col = np.array([cm_grid.geometry.centroid.iloc[i].distance(cm_grid.geometry.centroid.iloc[j])*1e-3 for j in range(len(cm_grid))])
                disp_dist[i] = cm_row@D_col
            else:
                disp_dist[i] = np.nan

        disp_dist = xr.DataArray(disp_dist,
                  coords = {'bin':self.grid.bin_n.stack(bin=('y_c','x_c')).dropna('bin').bin}).unstack().sortby('x_c').sortby('y_c').rio.write_crs(4326)#.rio.reproject_match(self.grid.mask)
        
        self.disp_dist = disp_dist
        return disp_dist


    def connectivity_analysis(self, node_name='index', self_loops=True, plot=False):
        '''
        Compute per-cell network connectivity metrics from the connectivity matrix.

        Builds a directed graph from self.cm with edge weight 'p' = transition probability and 
        self.cm_grid's polygons as nodes.
        Computes and plots, for each node: in-strength, out-strength and betweenness centrality.
        Betweenness is computed on edge weights transformed as in Costa et al., (2017).

        Requires self.cm and self.cm_grid to already be set (e.g. via get_cm()).

        Parameters
        ----------
        node_name : str, optional
            Column in self.cm_grid to use as each node's identifier/label, attached to the graph nodes and included in the returned GeoDataFrame.
            Default is 'index', i.e. self.cm_grid's (reset) integer index is used instead,
            stored as node attribute 'ID'.
        self_loops : bool, optional
            Whether to keep self-loop edges (a cell's probability of transitioning to
            itself) when computing the degree/centrality metrics below. Default is True.
        plot : bool, optional
            If True, plot each computed metric ('in_strength', 'out_strength',
            'betweenness') as a separate choropleth map via pmar.plotting.plot_map().
            Default is False.

        Returns
        -------
        geopandas.GeoDataFrame
            self.cm_grid's node_name and geometry columns, plus 'in_strength',
            'out_strength' and 'betweenness' for each cell.
        '''
        if self.cm_grid[self.cm_grid.columns.drop('geometry')].iloc[-1][0] == -1234.0:
            cm = self.cm.tocsr()[:-1,:-1] # exclude buffer 
            cm_grid = self.cm_grid[:-1]
        else:
            cm = self.cm
            cm_grid = self.cm_grid
        
        G = nx.from_scipy_sparse_array(cm, create_using=nx.DiGraph, edge_attribute='p', parallel_edges=False) # important to specify DiGraph (directed graph) 
        
        if node_name == 'index':
            cm_grid = cm_grid.reset_index()
            nx.set_node_attributes(G, values=cm_grid.index, name='ID')
        else:
            nx.set_node_attributes(G, values=cm_grid[str(node_name)], name=str(node_name))
    
        if self_loops == False:
            G.remove_edges_from(nx.selfloop_edges(G))
            
        coordinates = np.column_stack((cm_grid.geometry.centroid.x, cm_grid.geometry.centroid.y))
    
        connectivity_df = cm_grid[[str(node_name), 'geometry']]
        
        connectivity_df['in_strength'] = dict(G.in_degree(weight='p')).values()
        connectivity_df['out_strength'] = dict(G.out_degree(weight='p')).values()
            
        logprob = np.log(1/np.array(list(dict(nx.get_edge_attributes(G, 'p')).values()))) 
        nx.set_edge_attributes(G, dict(zip(dict(nx.get_edge_attributes(G, 'p')).keys(), logprob)), 'p_dist') 
        connectivity_df['betweenness'] = dict(nx.betweenness_centrality(G, weight='p_dist', endpoints=True)).values()
    
        if plot == True:
            for col in connectivity_df.columns.drop([str(node_name),'geometry']):
                fig, ax = plot_map()
                connectivity_df.plot(column=col, ax=ax, zorder=1, legend=True)#, legend_kwds={'label': col})
                ax.set_title(col)
            
        return connectivity_df

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


    def predict_with_cm(self, initial_condition, initial_condition_crs='epsg:4326', cm=None, k=None, sum=False):
        """Predict the probability distribution k Markov steps ahead of initial_condition.

        Only works when cm is a grid covering the whole domain: if initial_condition has
        shape (m,n), cm must have shape (mxn, mxn).

        Parameters
        ----------
        initial_condition : xr.DataArray
            Initial probability/occupancy field, reprojected and matched onto self.grid.mask.
        initial_condition_crs : str
            CRS of initial_condition, default 'epsg:4326'.
        cm : sparse array, optional
            Connectivity/transition matrix to use; defaults to self.cm.
        k : int, optional
            Number of Markov steps to propagate forward. If None, a single step is applied.
        sum : bool
            If True (and k is not None), sum the distribution over all steps 0..k (as in
            Chamberlain et al. (2023): "the probability of a float sampling a given region
            within a range of time steps") instead of returning only the k-th step.

        Returns
        -------
        xr.DataArray
            Predicted distribution over self.grid.mask.

        Notes
        -----
        The distribution is propagated incrementally as a vector (v <- v @ cm, k times,
        accumulated when sum=True) rather than by forming the full matrix power cm**k via
        matrix_power. cm itself never changes, so this avoids the sparsity fill-in that makes
        repeated squaring of a sparse matrix progressively denser (and slower) as k grows;
        cost becomes O(k * nnz(cm)) instead of superlinear in k.
        """
    # only works when cm is a grid covering the whole domain.
    # if initial_condition has shape (m,n), the cm must have shape (mxn, mxn)
        if cm is None:
            cm = self.cm # my cm is now by default a sparse array

        initial_condition = initial_condition.load().rio.set_crs(initial_condition_crs).rio.reproject_match(self.grid.mask).sortby('y').sortby('x').rename({'y':'y_c', 'x':'x_c'})

        if initial_condition.data[~np.isnan(initial_condition.data)].shape[0] != cm.shape[0]:
            logger.warning('initial_condition and cm dimensions do not match')#. clipping initial_condition to match cm_grid.')
            #initial_condition = initial_condition.rio.clip(self.cm_grid[self.cm_grid[list(self.cm_grid.columns.drop('geometry'))[0]] >=0].geometry, self.cm_grid.geometry.crs, drop=False, all_touched=True)
            initial_condition = initial_condition.where(initial_condition>=0)
            initial_condition = initial_condition.where(self.grid.mask>0)
        else:
            logger.info('initial_condition and cm dimensions match.')

        initial_condition_stack = initial_condition.stack(origin=('y_c','x_c')).dropna('origin')
        print(initial_condition_stack.shape)

        ## this would be for the case where an extra node was added to denote "everywhere else" outside of cm_grid.
        ## but unstack obviously does not work, so can't return a datarray in lon, lat. only a sparse matrix.
        maxbin = int(self.grid.bin_n.max().data)
        if len(self.cm_grid)-1 > maxbin:
            logger.warning('adding extra node to network')
            _initial_condition_stack = xr.concat([initial_condition_stack, xr.DataArray(data=[[0]], dims={'x_c':0, 'y_c':0}).stack(origin=('y_c','x_c'))], dim='origin')
            print(_initial_condition_stack.shape)
        else:
            _initial_condition_stack = initial_condition_stack
            print(_initial_condition_stack.shape)

        # ic_sparse = diags(_initial_condition_stack.data, shape=(cm.shape), format='coo')
        # replaced by a sparse row vector: ic_sparse.dot(cm).sum(0) was only ever computing
        # v @ cm for the vector v held on the diagonal, so building the full n x n diagonal
        # matrix was unnecessary.
        n_ic = len(_initial_condition_stack.data)
        ic_vec = coo_array(
            (_initial_condition_stack.data, (np.zeros(n_ic, dtype=int), np.arange(n_ic))),
            shape=(1, cm.shape[0])
        ).tocsr()

        if k is not None:
            cm_csr = cm.tocsr()
            if sum is True:
                # like in Chamberlain et al. (2023), "the probability of a float sampling a given region within a range of time steps"
                #cm = np.sum(matrix_power(cm, i) for i in range(1,k+1))
                #cm = self.cm
                # single_cm = cm
                # new_cm = 0
                # for i in range(0,k+1): #k+1 otherwise range goes up to k-1
                #     new_cm += matrix_power(single_cm,i) # this is wrong when there is a dummy node with high probability!!!
                # cm = new_cm
                # incremental vector propagation instead of recomputing matrix_power(cm, i)
                # from scratch for every i (same formula/result, O(k) instead of O(k log k)
                # with progressive sparsity fill-in): total = sum_{i=0}^{k} v @ cm**i
                current = ic_vec.copy()
                total = ic_vec.copy()  # i=0 term: cm**0 = I, so v @ I = v
                for _ in range(1,k):
                    current = current.dot(cm_csr)
                    print('current sum =', current.sum(), 'current sum excluding leakage cell =', current[:,:-1].sum())
                    total = total + current
                    print('total sum =', total.sum(), 'current sum excluding leakage cell =', total[:,:-1].sum())
                cm_prediction_sparse = total
                print('total sum =', total.sum(), 'current sum excluding leakage cell =', total[:,:-1].sum())
            else:
                # cm = matrix_power(cm, k)
                # incremental vector propagation instead of forming cm**k as a full matrix
                # (avoids the fill-in that makes sparse matrix_power progressively slower for
                # larger k): current = v @ cm**k
                current = ic_vec.copy()
                for _ in range(k): # 1, k+1? otherwise range gets only up k-1?
                    current = current.dot(cm_csr)
                    #print('total sum =', current.sum(), 'sum excluding leakage cell =', current[:,:-1].sum())
                cm_prediction_sparse = current

        else:
            cm_prediction_sparse = ic_vec.dot(cm)
            
        # adding maxbin to exclude "dummy" node before reprojection
        cm_prediction = xr.DataArray(cm_prediction_sparse.toarray()[0][:maxbin+1],
                              coords = dict(destination=initial_condition_stack.rename({'origin':'destination', 'x_c':'x', 'y_c':'y'}).destination)).unstack().sortby('x').sortby('y').rio.write_crs(4326).rio.reproject_match(self.grid.mask)
        print(cm_prediction.sum().data)
        # reproject match is needed to avoid tiny discrepancies in precision of x,y variables which cause the grids to mismatch
        
        self.cm_prediction = cm_prediction
        self._cm_prediction_sparse = cm_prediction_sparse # in case it's needed to check what goes outside of system, or in "dummy" node
        
        return cm_prediction

    
    def get_network(self, cm=None, node_names=None):
        """
        Build a directed networkx graph from the connectivity matrix (cm).

        Parameters
        ----------
        cm : scipy.sparse array, optional
            Connectivity/transition matrix to use as the graph's adjacency matrix. Default
            is None, i.e. self.cm.
        node_names : dict, optional
            Mapping of node index to a name, attached to each node as 'node_name' (e.g. use
            cm_grid['layer'].to_dict()). Default is None.

        Returns
        -------
        networkx.DiGraph
            The resulting directed graph (also stored on self.network).
        """
        # use cm_grid['layer'].to_dict() for node_names
        self.node_names = node_names
        if cm is None:
            cm = self.cm

        if self.cm_grid[self.cm_grid.columns.drop('geometry')].iloc[-1][0] == -1234.0:
            cm = self.cm.tocsr()[:-1,:-1]
            
        G = nx.from_scipy_sparse_array(cm, create_using=nx.DiGraph, edge_attribute='p', parallel_edges=False) # important to specify DiGraph (directed graph) and parallel edges; 'p' for transition probability, matching connectivity_analysis()
        if node_names is not None:
            nx.set_node_attributes(G, values=node_names, name='node_name')
        self.network = G
        return G

    def plot_network(self, node_shapes=None, show_map=False, rescale_edge=None, width=2, figsize=[7,7], extent=None, cmap=None):
        """
        Plot self.network (built by get_network) via pmar.plotting._plot_network.

        Parameters
        ----------
        node_shapes : bool or geopandas.GeoDataFrame, optional
            If True, use self.cm_grid as the node shapes/positions; alternatively pass a
            GeoDataFrame directly. Default is None (nodes plotted without shapes).
        show_map : bool, optional
            Whether to overlay a basemap. Default is False.
        rescale_edge : optional
            Edge-width rescaling option passed to _plot_network. Default is None.
        width : float, optional
            Base edge width. Default is 2.
        figsize : list, optional
            Figure size [width, height]. Default is [7, 7].
        extent : list, optional
            Map extent to restrict the plot to. Default is None.
        cmap : str, optional
            Colormap for edges/nodes. Default is None.
        """
        from pmar.plotting import _plot_network
        if node_shapes is True:
            node_shapes = self.cm_grid
            
        _plot_network(self.network, node_shapes=node_shapes, show_map=show_map, width=width, rescale_edge=rescale_edge, figsize=figsize, extent=extent, cmap=cmap)

    def export_network(self, out_path=None, node_shapes=None, weight_attr='p', curvature=0.1, self_loop_radius=None, n_points=30, include_nodes=True, style='both', width_range=(1, 8), p_range=(0, 1), cmap=None, draw_arrows=True, self_loop_arrow=True, arrow_length_range=None, arrow_width_range=None, target_gap=None, centrality_self_loops=True):
        """
        Export self.network (built by get_network) to a GeoJSON FeatureCollection of
        curved, weighted, directed edges (+ node centrality attributes) via
        pmar.utils.network_to_geojson, for rendering on an interactive web map.

        Parameters
        ----------
        out_path : str, optional
            If given, the GeoJSON is also written to this path. Default is None (only
            returned).
        node_shapes : geopandas.GeoDataFrame, optional
            Polygon grid giving node positions. Default is None, i.e. self.cm_grid, or
            self.cm_grid[:-1] if self.cm_grid has one more row than self.network has
            nodes (the buffer cell added when get_cm/get_network were called with
            buffer=...).
        weight_attr, curvature, self_loop_radius, n_points, include_nodes, style,
        width_range, p_range, cmap, draw_arrows, self_loop_arrow, arrow_length_range,
        arrow_width_range, target_gap :
            Passed through unchanged to pmar.utils.network_to_geojson(); see its
            docstring (in particular, `style` controls whether 'stroke-width'/'stroke'
            are baked into each edge feature, and `draw_arrows` whether a triangular
            arrowhead feature is added at each edge's target end, so the map shows
            correct thickness/color/direction with no app-side styling code beyond a
            one-time property-to-style mapping).
        centrality_self_loops : bool, optional
            Whether self-loop edges (a cell's probability of transitioning to itself)
            count toward the in_strength/out_strength/betweenness node attributes
            computed below, passed to connectivity_analysis()'s self_loops= (default
            True, matching the fact that self.network always includes self-loop edges).
            Ignored if include_nodes is False.

        Returns
        -------
        dict
            GeoJSON FeatureCollection.
        """
        from pmar.utils import network_to_geojson

        if node_shapes is None:
            node_shapes = self.cm_grid
            if len(node_shapes) == self.network.number_of_nodes() + 1:
                node_shapes = node_shapes[:-1] # drop the buffer cell, absent from self.network

        G = self.network
        if include_nodes:
            connectivity_df = self.connectivity_analysis(node_name='index', self_loops=centrality_self_loops, plot=False)
            metrics = connectivity_df[['in_strength', 'out_strength', 'betweenness']].to_dict('index')
            G = self.network.copy() # avoid mutating self.network with these attributes
            for col in ('in_strength', 'out_strength', 'betweenness'):
                nx.set_node_attributes(G, {n: m[col] for n, m in metrics.items()}, col)

        return network_to_geojson(G, node_shapes, out_path=out_path, weight_attr=weight_attr, curvature=curvature, self_loop_radius=self_loop_radius, n_points=n_points, include_nodes=include_nodes, style=style, width_range=width_range, p_range=p_range, cmap=cmap, draw_arrows=draw_arrows, self_loop_arrow=self_loop_arrow, arrow_length_range=arrow_length_range, arrow_width_range=arrow_width_range, target_gap=target_gap)

    # this is a histogram with a "distinct" on trajectories. i.e. if a particle stays in same cell for multiple timesteps, it doesn't get doublecounted.
    # note that this method fails if a trajectory goes back to same cell after a period of time. 
    def _traj_per_bin(self, res, study_area=None, use=None):#, weighted=False):
        """
        Compute, per grid cell, the (optionally weighted) fraction of a trajectory's total
        time steps that were spent in that cell, summed over all trajectories.

        Parameters
        ----------
        res : float
            Resolution used to bin trajectories (passed to get_histogram).
        study_area : list, optional
            Spatial bounds for the grid. Default is None.
        use : str, optional
            Path to a use raster; if given, weights are (re)computed via set_weights before
            binning. Default is None (uses self.ds.weight if present, else unweighted).

        Returns
        -------
        xr.DataArray
            Per-cell time-fraction, summed over trajectories.
        """
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
        """
        Compute a "distinct trajectory density" (route density) raster, similar to EMODnet
        route density: repeated same-cell visits by a single trajectory are deduplicated
        before binning, so a trajectory lingering in a cell doesn't inflate its density.

        Parameters
        ----------
        res : float
            Resolution used to bin trajectories.
        study_area : list, optional
            Spatial bounds for the grid. Default is None.

        Returns
        -------
        xr.DataArray
            Histogram of deduplicated per-trajectory weights per grid cell.
        """
        w = self.set_weights(1)
        self.ds = self.get_bin_n(res=res, t='all')
        w_distinct = xr.apply_ufunc(traj_distinct, self.ds.bin_n.chunk(chunks={'time': -1, 'trajectory': int(len(self.ds.trajectory)/10)}), w.chunk(chunks={'time': -1, 'trajectory': int(len(self.ds.trajectory)/10)}), input_core_dims = [['time'],['time']], output_core_dims = [['time']], vectorize=True, dask='parallelized')
        h_distinct = self.get_histogram(res=res, study_area=study_area, weight=w_distinct, block_size='auto')
        return h_distinct


    def write_tiff(self, h, crs=4326, path=None, description=None, output_dir=None):# instead of use label, could do 'assign_attrs' for any attributes, including use_label, i might want to add
        '''
        Georeference a histogram/raster array and write it to disk as a GeoTIFF.

        Parameters
        ----------
        h : array-like
            Raster data to write (e.g. a histogram from get_histogram).
        crs : int, optional
            EPSG code to write on the raster. Default is 4326.
        path : str, optional
            Output file path. Default is None; required unless output_dir is used to build
            a path from an existing `path`.
        description : str, optional
            If given, appended to the output filename as f'_use-{description}.tif'. Default
            is None.
        output_dir : str, optional
            If given, directory to write the file into instead of the directory implied by
            `path` (the filename is kept). Default is None.
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
        Top-level single-seeding pipeline: compute trajectories, then the PPI raster, then
        summary indicators. Equivalent to calling get_trajectories(), get_ppi() and
        get_indicators() in sequence.

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
        tstep : timedelta, optional
            OpenDrift simulation time step. Default is timedelta(hours=1).
        study_area : list, optional
            Spatial bounds [xmin, ymin, xmax, ymax] for rasterization. Default is None.
        seeding_shapefile : str, optional
            Path to a shapefile to seed particles within. Default is None.
        seed_within_bounds : list, optional
            Spatial bounds to seed within, if no seeding_shapefile is given. Default is None.
        z : float, optional
            Depth at which to seed particles [m]. Default is -0.5.
        seeding_radius : float, optional
            OpenDrift seeding "radius". Default is 0.
        pnum_per_shape : bool, optional
            If True, seed pnum particles within each shape of seeding_shapefile rather than
            across the whole shapefile. Default is False.
        beaching : bool, optional
            Whether particles reaching the coastline can beach ('stranding'). Default is False.
        hdiff : float, optional
            Horizontal diffusivity of particles [m2/s]. Default is 10.
        termvel : float, optional
            Currently unused directly by this method body (get_trajectories is called
            without it, so its own default of 0 applies); kept for API consistency.
            Default is 0.
        stokes_drift : bool, optional
            Currently unused directly by this method body; kept for API consistency.
            Default is False.
        decay_coef : float, optional
            Coefficient k of exponential decay function. Default is 0 (no decay)
        use : string, optional
            Path to map of human activities (.tiff) considered as pressure source. Default is None
        use_label : str, optional
            Label identifying the `use` raster, used in cache keys/output filenames. Default
            is None.
        emission : float, optional
            Amount of pressure released per day by the source in `use`. Default is None.
        save_tiffs : bool, optional
            Currently unused directly by this method body; kept for API consistency.
            Default is False.
        crs : str, optional
            Currently unused directly by this method body; kept for API consistency.
            Default is '4326'.
        output_dir : str, optional
            Currently unused directly by this method body; kept for API consistency.
            Default is None.
        '''

        self.res = res

        self.get_trajectories(pnum=pnum, start_time=start_time, tstep=tstep, duration_days=duration, seeding_shapefile=seeding_shapefile, seed_within_bounds=seed_within_bounds, z=z, seeding_radius=seeding_radius, beaching=beaching, hdiff=hdiff, pnum_per_shape=pnum_per_shape)

            
        self.get_ppi(res=res, study_area=study_area, use=use, use_label=use_label, emission=emission, decay_coef=decay_coef)#, block_size=len(self.ds.time)) # weight have been assigned above, so no need to normalize or apply decay etc.
        
        self.get_indicators(res=res, study_area=study_area)
        
        pass

    


class PMARmulti(PMAR):
    """
    Multi-seeding variant of PMAR: repeats trajectory generation over several seedings
    (optionally time-shifted via tshift) and aggregates the resulting PPI rasters and
    connectivity matrices across seedings.

    Developed at CNR ISMAR in Venice. Inherits PMAR's single-seeding machinery and overrides
    get_trajectories, get_ppi, run, and get_cm to loop over self.seedings repetitions
    (using a temporary single-seeding PMAR instance, `temp_self`, sharing this object's
    state, to compute per-seeding results before aggregating them). Adds sum_seedings() to
    sum per-seeding PPI rasters.

    Attributes
    ----------
    seedings : int or None
        Number of seedings/repetitions for the current run.
    tshift : int or None
        Time shift (in days) applied between successive seedings.
    ppi_list : list
        Paths to each seeding's individual PPI raster, aggregated by sum_seedings().

    See PMAR's class docstring for the remaining inherited attributes/methods.
    """
        


    def get_trajectories(self, pnum, start_time, seed_time=timedelta(days=0), seedings=1, tshift=0, end_time=None, duration_days=30, study_area=None, seeding_shapefile = None, seed_within_bounds=None, z=-0.5, tstep=timedelta(hours=1), hdiff=10, termvel=0, crs='4326', seeding_radius=0, beaching=False, stokes_drift=False, pnum_per_shape=False):
        """
        Compute trajectories for `seedings` repetitions, each start-time-shifted by tshift
        days from the previous one, and combine them all into self.ds.

        Parameters
        ----------
        pnum, seed_time, seeding_shapefile, seed_within_bounds, z, tstep, hdiff, termvel,
        crs, seeding_radius, beaching, stokes_drift, pnum_per_shape :
            Passed through unchanged to PMAR.get_trajectories() for each seeding; see its
            docstring for details.
        start_time : string
            Start date of the first seeding, in YYYY-MM-DD format. Each subsequent seeding n
            starts at start_time + n * tshift days.
        seedings : int, optional
            Number of repeated seedings to run. Default is 1.
        tshift : int, optional
            Time shift, in days, applied between successive seedings' start times.
            Default is 0.
        end_time : string, optional
            If given, overrides duration_days: each seeding's duration is computed as
            (end_time - that seeding's shifted start_time). Default is None.
        duration_days : int, optional
            Duration, in days, of each seeding (ignored if end_time is given). Default is 30.

        Notes
        -----
        Sets self.seedings, self.tshift, self.seeding_radius, appends each seeding's output
        path to self.particle_list, and finally calls self.import_trajectories() to load all
        seedings combined into self.ds.
        """
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
            
            super().get_trajectories(pnum=pnum, start_time=rep_start_time, seed_time=seed_time, duration_days=rep_duration, seeding_shapefile=seeding_shapefile, seed_within_bounds=seed_within_bounds, z=z, tstep=tstep, hdiff=hdiff, termvel=termvel, crs=crs, seeding_radius=seeding_radius, beaching=beaching, stokes_drift=stokes_drift, pnum_per_shape=pnum_per_shape)
            logger.info(f'Done with rep #{n}.')
            self.particle_list.append(self.particle_path)

        self.import_trajectories(self.particle_list)
        

    def get_ppi(self, res, use=None, use_label=None, emission=None, study_area=None, decay_coef=0):
        """
        Compute the PPI raster aggregated across all seedings in self.particle_list.

        For each seeding's particle file, builds a temporary single-seeding PMAR instance
        sharing this object's state, computes that seeding's PPI via PMAR.get_ppi(), then
        sums all per-seeding PPI rasters (via sum_seedings) and divides by the number of
        seedings to get the averaged multi-seeding PPI.

        Parameters
        ----------
        res : float
            Resolution of the output PPI raster.
        use : str, optional
            Path to map of human activities (.tiff) considered as pressure source. Reused
            across seedings once computed for the first one that needs it. Default is None.
        use_label : str, optional
            Label identifying the `use` raster, used in cache keys/output filenames. Default
            is None.
        emission : float, optional
            Amount of pressure released per day by the source in `use`. Default is None.
        study_area : list, optional
            Spatial bounds [xmin, ymin, xmax, ymax] for rasterization. Default is None.
        decay_coef : float, optional
            Coefficient k of exponential decay function. Default is 0 (no decay).

        Notes
        -----
        Checks a raster cache keyed on the multi-seeding config first; if found, loads the
        cached PPI instead of recomputing. Sets self.output['ppi'], self.ppi_list, and
        self.ppi_path.
        """
        ppi_output_dir = Path(self.basedir) / f'ppi-{use_label}'

        self.ppi_cache = PMARCache(ppi_output_dir)
             
        ppi_path, ppi_exists = self.ppi_cache.raster_cache(context=self.spatial_domain, pressure=self.pressure, chemical_compound=self.chemical_compound, seeding_shapefile=self.seeding_shapefile, seeding_radius=self.seeding_radius, res=res, pnum=None, ptot=self.pnum, start_time=self.start_time, duration=self.duration, seedings=self.seedings, seeding_id=self.seeding_id, tshift=self.tshift, use=use, use_label=use_label, emission=emission, decay_coef=decay_coef, study_area=study_area)#, aggregate=aggregate, depth_layer=depth_layer, z_bounds=z_bounds, particle_status=particle_status, traj_dens=traj_dens)

        if not self.output:
            self.output = xr.Dataset()
        self.grid = make_grid(res=res, study_area=study_area)
        
        if ppi_exists == True:
            logger.info('PPI with requested configurations found in cache.')
            self.output['ppi'] = rxr.open_rasterio(ppi_path)
        else:
        
            for pfile in self.particle_list:
                temp_self = PMAR(_internal=True)
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
            
            self.output['ppi'] = self.sum_seedings(self.ppi_list).squeeze().rio.reproject_match(self.grid.mask.rename({'y_c':'y','x_c':'x'})).rename({'x':'lon','y':'lat'})/len(self.ds.seeding) # divide by n of seedings
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
        Top-level multi-seeding pipeline: compute trajectories (over `seedings` repetitions)
        and produce the aggregated PPI raster and summary indicators. Equivalent to calling
        get_trajectories(), get_ppi() and get_indicators() in sequence, but first checks a
        raster cache and skips recomputation entirely if a matching PPI already exists.

        Parameters
        ----------
        pnum : int
            Number of particles to seed per repetition.
        start_time : string
            Start date of the first seeding, in YYYY-MM-DD format.
        duration : float
            Duration of each seeding, in days (ignored if end_time is given).
        res : float
            Resolution of output rasters, in lon/lat degrees.
        seedings : int, optional
            Number of repeated seedings to run. Default is 1.
        tshift : int, optional
            Time shift, in days, applied between successive seedings' start times. Default is 0.
        end_time : string, optional
            If given, overrides `duration`: each seeding's duration is computed from its
            shifted start_time to end_time. Default is None.
        tstep : timedelta, optional
            OpenDrift simulation time step. Default is timedelta(hours=1).
        study_area : list, optional
            Spatial bounds [xmin, ymin, xmax, ymax] for rasterization. Default is None.
        seeding_shapefile : str, optional
            Path to a shapefile to seed particles within. Default is None.
        seed_within_bounds : list, optional
            Spatial bounds to seed within, if no seeding_shapefile is given. Default is None.
        z : float, optional
            Depth at which to seed particles [m]. Default is -0.5.
        seeding_radius : float, optional
            OpenDrift seeding "radius". Default is 0.
        pnum_per_shape : bool, optional
            If True, seed pnum particles within each shape of seeding_shapefile rather than
            across the whole shapefile. Default is False.
        beaching : bool, optional
            Whether particles reaching the coastline can beach ('stranding'). Default is False.
        hdiff : float, optional
            Horizontal diffusivity of particles [m2/s]. Default is 10.
        termvel : float, optional
            Terminal velocity of particles [m/s]. Default is 0.
        stokes_drift : bool, optional
            Whether to include Stokes drift. Default is False.
        decay_coef : float, optional
            Coefficient k of exponential decay function. Default is 0 (no decay).
        use : string, optional
            Path to map of human activities (.tiff) considered as pressure source. Default is None.
        use_label : string, optional
            Label identifying the `use` raster, used in cache keys/output filenames. Default
            is None.
        emission : float, optional
            Amount of pressure released per day by the source in `use`. Default is None.
        output_dir : str, optional
            Currently unused directly by this method body; kept for API consistency.
            Default is None.
        save_tiffs : bool, optional
            Currently unused directly by this method body; kept for API consistency.
            Default is False.
        thumbnail : optional
            Currently unused directly by this method body; kept for API consistency.
            Default is None.
        crs : str, optional
            EPSG string for seeding polygons, passed through to get_trajectories.
            Default is '4326'.
        '''

        self.seedings = seedings
        
        ppi_output_dir = Path(self.basedir) / f'ppi-{use_label}'

        self.ppi_cache = PMARCache(ppi_output_dir)
             
        ppi_path, ppi_exists = self.ppi_cache.raster_cache(context=self.spatial_domain, pressure=self.pressure, chemical_compound=self.chemical_compound, seeding_shapefile=seeding_shapefile, seeding_radius=seeding_radius, res=res, pnum=None, ptot=pnum, start_time=start_time, duration=duration, seedings=seedings, seeding_id=self.seeding_id, tshift=tshift, use=use, use_label=use_label, emission=emission, decay_coef=decay_coef, study_area=study_area)#, aggregate=aggregate, depth_layer=depth_layer, z_bounds=z_bounds, particle_status=particle_status, traj_dens=traj_dens)

        if not self.output:
            self.output = xr.Dataset()
        
        if ppi_exists == True:
            logger.info('PPI with requested configurations found in cache.')
            self.output['ppi'] = rxr.open_rasterio(ppi_path)
        else:
            self.get_trajectories(pnum=pnum, start_time=start_time, seedings=seedings, tshift=tshift, end_time=end_time, duration_days=duration, study_area=study_area, seeding_shapefile=seeding_shapefile, seed_within_bounds=seed_within_bounds, z=z, tstep=tstep, hdiff=hdiff, termvel=termvel, crs=crs, seeding_radius=seeding_radius, beaching=beaching, stokes_drift=stokes_drift, pnum_per_shape=pnum_per_shape)
            self.get_ppi(res=res, use=use, use_label=use_label, emission=emission, study_area=study_area, decay_coef=decay_coef)

        self.get_indicators(res=res, study_area=study_area)
            
        pass


    def get_cm(self, cm_grid=None, res=None, study_area=None, t0=0, t1=-1, buffer=None, filenames=None):
        '''
        Compute the connectivity matrix (cm) averaged across all seedings in self.particle_list.

        For each seeding's particle file, builds a temporary single-seeding PMAR instance
        sharing this object's state, imports that seeding's trajectories (sliced to
        [t0, t1)), optionally applies weights (if self.use is set), computes that seeding's
        transition matrix via PMAR.get_cm(), and accumulates the mean across all seedings.

        Parameters
        ----------
        cm_grid : geopandas.GeoDataFrame, optional
            Polygon grid to use. Default is None.
        res : float, optional
            Resolution for building a new grid, if cm_grid is None. Default is None.
        study_area : list, optional
            Spatial bounds for the grid. Default is None.
        t0 : int, optional
            Start timestep index (inclusive) to slice each seeding's trajectories to.
            Default is 0.
        t1 : int, optional
            End timestep index (exclusive) to slice each seeding's trajectories to, and the
            timestep at which transitions are evaluated. Default is -1.
        buffer : float, optional
            Buffer distance added around cm_grid as an extra "outside" node. Default is None.
        filenames : list of str, optional
            Per-seeding paths to write each seeding's individual transition matrix to.
            Default is None.

        Returns
        -------
        scipy.sparse array
            The averaged transition matrix (also stored on self.cm).

        Notes
        -----
        The per-seeding temp_self shares self.__dict__, so importing each seeding's
        trajectories overwrites self.ds along the way; self.ds is re-imported from
        self.particle_list at the end to restore the full multi-seeding dataset (with the
        'seeding' dimension) before returning.
        '''
        for i, file in enumerate(self.particle_list):
            logger.info(f'Calculating average cm out of {len(self.particle_list)} particle files.')
            ds = xr.open_dataset(file).isel(time=slice(t0,t1))
            temp_self = PMAR(_internal=True)
            temp_self.__dict__ = self.__dict__
            temp_self.import_trajectories(ds)
            if hasattr(self, 'use') and self.use is not None: # to avoid re-rasterizing use for every seeding
                temp_self.set_weights(use=self.use, res=res, study_area=study_area)
                weighted=True
            else:
                weighted=False

            if filenames is not None:
                filename = filenames[i]
            else:
                filename = None
                
            temp_self.get_cm(cm_grid=cm_grid, res=res, study_area=study_area, t=-1, buffer=buffer, filename=filename, weighted=weighted)
            if i == 0:
                cm = temp_self.cm/len(self.particle_list)
            else:
                cm = cm + temp_self.cm/len(self.particle_list)
            del temp_self
            logger.info(f'Done calculating cm for particle file {i}.')
        self.import_trajectories(self.particle_list) # restore self.ds to the full multi-seeding dataset, since temp_self shares self.__dict__ and overwrote it with each seeding's single-seeding ds above
        self.cm = cm
        return cm
