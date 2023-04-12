import logging
import rioxarray as rxr
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from shapely.geometry import Point, Polygon
from datetime import timedelta
from opendrift.models.oceandrift import OceanDrift
from opendrift.models.openoil import OpenOil
from opendrift.readers import reader_netCDF_CF_generic
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
from datetime import date, timedelta
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
#from dask.distributed import Client
#client = Client(n_workers=7, threads_per_worker=2)

logger = logging.getLogger("LagrangianDispersion")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')


class LagrangianDispersion(object): 
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
    particle_simulation()
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
        poly_path : str, optional
            path to shapefile containing polygon to be used for seeding of particles
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
        self.bathy_path = None 
        self.basedir = Path(basedir)
        self.particle_path = particle_path # i can import an existing particle_path
        self.ds = None
        self.o = None
        self.poly_path = poly_path 
        self.raster = None
        self.origin_marker = 0
        self.netrc_path = netrc_path
        self.tstep = None
        self.tseed = None
        self.pnum = None
        self.depth = depth
        self.termvel = 1e-3
        self.decay_rate = 0
        self.context = context
        self.outputdir = None
        self.pressure = pressure
        self.localdatadir = localdatadir
        
        pres_list = ['general', 'microplastic', 'bacteria']
        pressures = pd.DataFrame(columns=['pressure', 'termvel', 'decay_rate'], 
                    data = {'pressure': pres_list, 
                            'termvel': [0, 1e-3, 0], 
                            'decay_rate': [0, 0, 1]})
        
        
        if pressure in pres_list:
            self.termvel = pressures[pressures['pressure'] == f'{pressure}']['termvel'].values[0]
            self.decay_rate = pressures[pressures['pressure'] == f'{pressure}']['decay_rate'].values[0]
        
        elif pressure == 'oil':
            pass
        
        else:
            pass
    
        # if particle_path is given, retrieve attributes from filename and load ds
        
        if self.particle_path is not None: 
            
            avail_contexts = ['med', 'bs', 'bridge-bs', 'med-cmems', 'bs-cmems']
            
            for c in avail_contexts:
                if c in self.particle_path:
                    self.context = c
                
            self.depth = self.particle_path.split('depth_')[1][0] == 'T'  
            
            if self.particle_path.find('tseed') != -1:
                self.tseed = timedelta(seconds=float(self.particle_path.split('tseed_')[1].split('s')[0]))
            
            self.tstep = timedelta(seconds=float(self.particle_path.split('tstep_')[1].split('s')[0]))
            
            self.pnum = int(self.particle_path.split('pnum_')[1].split('-')[0])
            
            # load opendrift dataset.  
            if type(self.particle_path) == dict: 
                for i in self.particle_path:
                    path_list = self.particle_path
                    path_list[i] = str(path_list[i])
                
                paths = path_list.values()
            else:
                paths = str(self.particle_path)
            
            ds = xr.open_mfdataset(paths, concat_dim='trajectory', combine='nested', join='override', parallel=True, chunks={'trajectory': 10000, 'time':1000})

            ds['trajectory'] = np.arange(0, len(ds.trajectory)) # give trajectories unique ID  
    
            self.ds = ds
            
            if self.ds.poly_path is not None:
                self.poly_path = str(self.ds.poly_path)
            
        pass

    def get_userinfo(self, machine):
        try:
            secrets = netrc.netrc(self.netrc_path)
        except FileNotFoundError:
            ''
        auth = secrets.authenticators(machine)
        if auth is None:
            return ''
        return f'{auth[0]}:{auth[2]}@'
    
    def run(self, reps=1, tshift = timedelta(days=28), pnum=100, start_time='2019-01-01', season=None, duration_days=30, s_bounds=None, z=-0.5, tstep=timedelta(hours=6), hdiff=10, termvel=None, raster=True, res=3000, crs='4326', tinterp=None, r_bounds=None, use_path='even', decay_rate=None, aggregate='mean', depth_layer='full_depth', z_bounds=[10,100], loglevel=40, save_to=None, plot=True, particle_status='active_only'):         
        """
        Launches methods particle_simulation and particle_raster. 
        
        
        Parameters 
        ----------
        reps = int, optional
            Number of desired iterations for the particle simulation. Default is 1
        pnum : int, optional 
            The number of particles to be seeded. Default is 100 (to be used for a test run)
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
        raster : bool, optional
            Boolean stating whether to launch 'particle_raster' method. Default is True 
        res : float, optional
            Spatial resolution for raster in [m]. Default is 3km
        crs : str, optional
            EPSG string for raster. Default is 4326        
        tinterp : float, optional
            Timestep (in hours) used for interpolation of trajectories in particle_raster method. Default is None
        r_bounds : list, optional
            Spatial bounds for computation of histogram (particle_raster) written as bounds=[x1,y1,x2,y2]. Default is None (full basin)
        use_path : str, optional
            Path to .tif file representing density of human use of marine environment, used for 'weights' of particles in histogram calculation. 
            If no use_path is given, a weight of 1 is given to all particles ('even' for even distribution).
        decay_rate : float, optional
            Decay rate of particles in [days-1]. Default is None, meaning decay_rate is defined in __init__
        aggregate : str, optional
            String indicating whether trajectories should be aggregated by calculating their maximum ('max') or 'mean' over time. Default is 'mean'
        depth_layer : str, optional
            Depth layer chosen for computing histogram. One of 'full_depth', 'water_column', 'surface' or 'seafloor'. Default is 'full_depth'
        z_bounds : list, optional
            Two parameters, given as z_bounds=[z_surface, z_bottom], determining the depth layers' thickness in [m]. 
            The first represents vertical distance from the ocean surface (z=0), whhile the second represents vertical distance from the ocean bottom, given by the bathymetry. 
            Default is z_bounds=[10,100].
        loglevel : int, optional
            OpenDrift loglevel. Set to 0 (default) to retrieve all debug information.
            Provide a higher value (e.g. 20) to receive less output.
        save_to : str, optional
            Name of the folder to save raster figures into. If None (default), rasters are saved within the rasters directory in basedir. 
        
        """
        # raise error if particle_path is already given -> DEPRECATED. IF PARTICLE_PATH IS NOT GIVEN, MAKE PARTICLE_SIMULATION. OTHERWISE GO STRAIGHT TO RASTER. 
        context = self.context 
        
        if termvel is None:
            termvel = self.termvel
         
        if decay_rate is None:
            decay_rate = self.decay_rate
        
        if self.particle_path is None:
            #logger.debug('Run starting')
                
            self.pnum = pnum
            # this gives the option of doing e.g. monthly runs to avoid crashing when seeding a lot of particles. 
            #time = [t0.strftime('%Y-%m-%d'), t1.strftime('%Y-%m-%d')]
            #iter_dt = t1-t0
            particle_list = {}

            # instead, split into smaller batches of particles (rather than time segments).
            # repeat run gives the number of times that exact same run has to be repeated. this coefficient is saved in 'origin_marker', which now appears in particle_path. 
            
            ### di default, ogni run è shiftata di 28 giorni rispetto alla precedente (scelta empirica)
            tshift = tshift
            
            for n in range(0, reps):
                print(f'----- STARTING RUN #{n+1} -----')
                if n==0:
                    t0 = datetime.strptime(start_time, '%Y-%m-%d')
                else:
                    t0 = datetime.strptime(start_time, '%Y-%m-%d')+tshift

                start_time = t0.strftime("%Y-%m-%d")
                
                self.origin_marker = n 
                
                self.particle_simulation(pnum=pnum, start_time=start_time, season=season, duration_days=duration_days, crs=crs, s_bounds=s_bounds, z=z, tstep=tstep, hdiff=hdiff, termvel=termvel, loglevel=loglevel)    
                particle_list[n] = self.particle_path
                print(f'----- DONE WITH RUN #{n+1} -----')

            
            if len(particle_list) == 1:
                self.particle_path == list(particle_list.values())[0]
            else:
                self.particle_path = particle_list

        # compute raster
        if raster == True:
            if type(use_path) == str:
                r = self.particle_raster(res=res, crs=crs, tinterp=tinterp, r_bounds=r_bounds, use_path=use_path, decay_rate=decay_rate,  aggregate=aggregate, depth_layer=depth_layer, z_bounds=z_bounds, particle_status=particle_status)
                self.raster = r
            elif type(use_path) == list:
                for idx, use in enumerate(use_path):
                    r = self.particle_raster(res=res, crs=crs, tinterp=tinterp, r_bounds=r_bounds, use_path=use, decay_rate=decay_rate,  aggregate=aggregate, depth_layer=depth_layer, z_bounds=z_bounds, particle_status=particle_status)
                    if idx == 0:
                        self.raster = r
                    else:
                        self.raster = xr.merge([self.raster, r.rename({'r0': f'r{idx}', 'lon_bin': f'lon_bin_{idx}', 'lat_bin': f'lat_bin_{idx}'})], compat='broadcast_equals', join='outer', fill_value=0, combine_attrs='no_conflicts')
            
            else:
                raise ValueError('"use_path" not recognised. "use_path" must be either a string or list of strings containing path(s) to use layers.')
        
            # save final raster(s) and delete temporary files
            if save_to is None:
                Path(self.basedir / 'rasters').mkdir(parents=True, exist_ok=True)
                outputdir = self.basedir / 'rasters'
            else:
                Path(save_to).mkdir(parents=True, exist_ok=True)
                outputdir = Path(save_to)

            for i, r in enumerate(self.raster.data_vars): # save all available rasters in output directory
                    print(f'creating thumbnail #{i}...')

                    self.raster[f'{r}'].rio.to_raster(outputdir / f'raster_{i}.tif')
                    # save corresponding thumbnails in save output directory
                    self.plot(r=self.raster[f'{r}'], save_fig=f'{str(outputdir)}/thumbnail_raster_{i}.png')
                    
            self.outputdir = outputdir
        
        
        return self
    
    
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
        poly_path = f'polygon-crs_epsg:{crs}-lon_{lon[0]}_{lon[1]}-lat—{lat[0]}_{lat[1]}.shp'
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
    

    
    def particle_simulation(self, pnum, start_time='2019-01-01', season=None, duration_days=30, s_bounds=None, z=-0.5, tstep=timedelta(hours=6), hdiff=10, termvel=None, crs='4326', loglevel=40):
        """
        Method to start a trajectory simulation, after initial configuration, using OpenDrift by MET Norway.
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
        
        
        # raise error if unsupported context is requested
        avail_contexts = ['bridge-bs', 'med-cmems', 'bs-cmems']
        if context not in avail_contexts:
            raise ValueError(f"Unsupported context given. Context variable must be one of {avail_contexts}")
        self.context = context
        
        Path(self.basedir / 'particles').mkdir(parents=True, exist_ok=True)
        
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

        if self.tseed is None:
            self.tseed = timedelta(days=int(duration_days * 20 / 100)) #tseed is 20% of total duration
            
        tseed = self.tseed
        
        duration = timedelta(days=duration_days)+tseed-timedelta(days=1) # true duration of the run. the tseed time period is then deleted.
        
        self.tstep = tstep
        
        # polygon used for seeding of particles. by default, it is the whole basin. but if lon and lat are given, a new polygon is created using those bounds. 
        if s_bounds is not None:
            lon = s_bounds[0::2]
            lat = s_bounds[1::2]
            self.make_poly(lon, lat, crs=crs)
        else:
            if self.poly_path is None:
                if 'med' in self.context:
                    self.poly_path = f'{DATA_DIR}/polygon-med-full-basin.shp'
                elif 'bs' in self.context:
                    self.poly_path = f'{DATA_DIR}/polygon-bs-full-basin.shp'
                else:
                    raise ValueError('No polygon could be identified for seeding.')
        
        # path to write particle simulation file. also used for our 'cache'        
        bds = np.round(gpd.read_file(self.poly_path).total_bounds) # only used in particle_path
        t0 = start_time.strftime('%Y-%m-%d')
        t1 = end_time.strftime('%Y-%m-%d')
        particle_path = f'{str(self.context)}-lon_{int(bds[0])}-{int(bds[2])}_lat_{int(bds[1])}-{int(bds[3])}-time_{t0}_to_{t1}-pnum_{pnum}-tstep_{int(tstep.total_seconds())}s-tseed_{int(self.tseed.total_seconds())}s-hdiff_{hdiff}-termvel_{termvel}-depth_{str(self.depth)}-marker_{str(self.origin_marker)}.nc'
        q =  self.basedir / 'particles' / particle_path
        
        
        # initialise OpenDrift object
        if self.pressure == 'oil':
            self.o = OpenOil(loglevel=loglevel)
        else:
            self.o = OceanDrift(loglevel=loglevel) 
        

        # if a file with that name already exists, simply import it  
        if q.exists() == True:
            #self.o.io_import_file(str(q)) # this is sometimes too heavy
            ps = xr.open_mfdataset(q)
            print('NOTE: File with these configurations already exists within basedir and has been imported. Please delete the existing file to produce a new simulation.')

        # otherwise, run requested simulation
        elif q.exists() == False:
            print('adding landmask...')
            # landmask from cartopy (from "use shapefile as landmask" example on OpenDrift documentation)
            shpfilename = shpreader.natural_earth(resolution='10m',
                                    category='cultural',
                                    name='admin_0_countries')
            reader_landmask = reader_shape.Reader.from_shpfiles(shpfilename)
            
            # opendrift landmask
            #from opendrift.readers import reader_global_landmask
            #reader_landmask = reader_global_landmask.Reader()
            #           extent=[2, 59, 8, 63]) #can also specify extent if needed
            
            self.o.add_reader([reader_landmask])
            self.o.set_config('general:use_auto_landmask', False)  # Disabling the automatic GSHHG landmask
            
            # import relevant readers based on context
            if context == 'bridge-bs': # local WP2 data
                
                if self.localdatadir is None:
                    raise ValueError('bridge-bs data not found. Please provide absolute path to directory containing oceanographic / atmospheric data from bridge-bs, i.e. localdatadir')
                else:    
                    bridge_dir = Path(self.localdatadir)
                
                dates = pd.date_range(start_time.strftime("%Y-%m"),(start_time+duration).strftime("%Y-%m"),freq='MS').strftime("%Y-%m").tolist()
                
                uvpaths={}
                mldpaths={}
                windpaths = {}

                for idx, d in enumerate(dates):
                    date = d.replace('-', '')
                    uvpaths[idx] = str(bridge_dir / f'BS_1d_{date}*UV.nc')
                    mldpaths[idx] = str(bridge_dir / f'BS_1d_{date}*MLD.nc') 
                
                uv_path = list(uvpaths.values())
                mld_path = list(mldpaths.values())
                
                for i in range(start_time.year, end_time.year+1):
                    windpaths[i] = f'/home/sbosi/data/BRIDGE-WP2/era5_y{i}.nc'
                
                wind_path = list(windpaths.values())
                bathy_path = '/home/sbosi/data/BRIDGE-WP2/bs_bathymetry.nc'
                
            elif context == 'bs-cmems': # copernicus Black Sea data (stream)
                DATASET_ID = 'cmems_mod_blk_phy-cur_my_2.5km_P1D-m'
                userinfo = self.get_userinfo('my.cmems-du.eu')
                uv_path = f'https://{userinfo}my.cmems-du.eu/thredds/dodsC/{DATASET_ID}'
                
                WIND_ID = 'cmems_obs-wind_glo_phy_my_l4_P1M' 
                wind_path = f'https://{userinfo}my.cmems-du.eu/thredds/dodsC/{WIND_ID}'
                
                mld_ID = 'cmems_mod_blk_phy-mld_my_2.5km_P1D-m'#'bs-cmcc-mld-rean-d'
                mld_path = f'https://{userinfo}my.cmems-du.eu/thredds/dodsC/{mld_ID}'        
                
                bathy_path = '/home/sbosi/data/input/bathymetry_gebco_2022_n46.8018_s29.1797_w-6.5918_e43.8574.nc'
            
            elif context == 'med-cmems': # copernicus Med Sea data (stream)
                DATASET_ID = 'med-cmcc-cur-rean-d'
                userinfo = self.get_userinfo('med-cmcc-cur-rean-d')
                uv_path = f'https://{userinfo}my.cmems-du.eu/thredds/dodsC/{DATASET_ID}'                
                
                WIND_ID = 'cmems_obs-wind_glo_phy_my_l4_P1M'              
                wind_path = f'https://{userinfo}my.cmems-du.eu/thredds/dodsC/{WIND_ID}'
            
                mld_ID = 'med-cmcc-mld-rean-d'
                mld_path = f'https://{userinfo}my.cmems-du.eu/thredds/dodsC/{mld_ID}'      
                
                bathy_path = '/home/sbosi/data/input/bathymetry_gebco_2022_n46.8018_s29.1797_w-6.5918_e43.8574.nc'

            else:
                raise ValueError("Unsupported context. Please choose one of 'bridge-bs', 'bs-cmems' or 'med-cmems'.")

            print('adding ocean readers...')
            self.o.add_readers_from_list(uv_path, lazy=True) 
            print('adding wind readers...')
            self.o.add_readers_from_list(wind_path, lazy=True) # add reader from local file
            
            if self.depth == True:
                print('adding mixed layer readers...')
                self.o.add_readers_from_list(mld_path, lazy=True) # add reader from local file
                print('adding bathymetry readers...')
                bathy_reader = reader_netCDF_CF_generic.Reader(bathy_path)
                self.o.add_reader(bathy_reader)

            # some OpenDrift configurations
            self.o.set_config('general:coastline_action', 'stranding') # behaviour at coastline. 'stranding' means beaching of particles is allowed
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
                self.o.seed_from_shapefile(shapefile=str(self.poly_path), number=pnum, time=[start_time, start_time+self.tseed], 
                                           terminal_velocity=termvel, z=z, origin_marker=self.origin_marker, radius=1e4)
                #self.o.set_config('vertical_mixing:diffusivitymodel', 'windspeed_Large1994')
                self.o.set_config('general:seafloor_action', 'deactivate')
                self.o.set_config('drift:vertical_mixing', True)
            
            # if simulation is 2D, simply seed particles over polygon
            else:
                self.o.seed_from_shapefile(shapefile=str(self.poly_path), number=pnum, time=[start_time, start_time+self.tseed],
                                           origin_marker=self.origin_marker, radius=1e4)
            
            # run simulation and write to temporary file
            temp_outfile = str(self.basedir)+'/temp_particle_file.nc'
            
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")

            #print(f'starting OpenDrift run at {current_time}...')
            
            self.o.run(duration=duration, #end_time=end_time, 
                       time_step=time_step, #time_step_output=timedelta(hours=24), 
                       outfile=temp_outfile, export_variables=['lon', 'lat', 'z', 'status', 'sea_floor_depth_below_sea_level', 'age_seconds', 'origin_marker', 'x_sea_water_velocity', 'y_sea_water_velocity', 'x_wind', 'y_wind', 'ocean_mixed_layer_thickness'])
            
            elapsed = (T.time() - t_0)
            print("total simulation runtime %s" % timedelta(minutes=elapsed/60)) 

            #### A BIT OF POST-PROCESSING ####
            print('writing to netcdf...')
            
            _ps = xr.open_dataset(temp_outfile) # open temporary file
            
            # keep 'inactive' particles visible (i.e. particles that have beached or gotten stuck on seafloor)
            ps = _ps.where(_ps.status>=0).ffill('time') 
            
            # align trajectories by particles' age
            shift_by = -ps.age_seconds.argmin('time') 
            if self.tseed.days!=0: 

                def shift_by_age(da, shift_by):
                    newda = xr.apply_ufunc(np.roll, da, shift_by, input_core_dims=[['time'], []], output_core_dims=[['time']], vectorize=True, dask='parallelized', keep_attrs='drop_conflicts')
                    return newda

                ps=ps.apply(shift_by_age, shift_by=shift_by) 

                # remove tail of nan values
                _idx = ps.age_seconds.argmin('time', skipna=False)
                idx = _idx.where(_idx!=0).min() # time index of first nan value across all trajectories
                ps = ps.isel(time=slice(None, int(idx)))
            
            # write useful time attributes
            ps = ps.assign_attrs({'tseed': self.tseed.total_seconds(), 'tstep': tstep.total_seconds(), 'poly_path': str(self.poly_path), 'opendrift_log': str(self.o)})
            #self.tstep = tstep
            #self.tseed = tseed # should already be there
            
            ps.to_netcdf(str(q)) 
        
        self.ds = ps
        
        self.particle_path = str(q) #particle_path
        
        print('done.')
        
        pass     
    

    def particle_raster(self, res=3000, crs='4326', tinterp=None, r_bounds=None, use_path='even', decay_rate=None, aggregate='mean', depth_layer='full_depth', z_bounds=[1,-10], particle_status='active_only'):
        """
        Method to compute a 2D horizontal histogram of particle concentration using the xhistogram.xarray package. 
        
        If a use_path is given, particle 'weight' is the value of the given human use in the particle's initial position. 
        Default weight is 1 for all particles.
        
        
        Parameters
        ----------
        res : float, optional
            Spatial resolution for raster in [m]. Default is 3km
        crs : str, optional
            EPSG string for raster. Default is 4326
        tinterp : float, optional
            Timestep (in hours) used for interpolation of trajectories in particle_raster method. Default is None
            ds
        r_bounds : bounds, optional
            Spatial bounds for computation of raster written as bounds=[x1,y1,x2,y2]. Default is None (bounds are taken from self.poly_path)
        use_path : str, optional
            Path to file representing density of human use, used for 'weights' of particles in histogram calculation. 
            If no use_path is given, a weight of 1 is given to all particles by default ('even').
        decay_rate : float, optional
            Decay rate of particles in [days-1]. Default is None, meaning decay_rate is defined in __init__
        aggregate : 
            String indicating whether trajectories should be aggregated by calculating their maximum ('max') or 'mean' over time. Default is 'mean'
        depth_layer : str, optional
            Depth layer chosen for computing histogram. One of 'full_depth', 'water_column', 'surface' or 'seafloor'. Default is 'full_depth'
        z_bounds : list, optional
            Two parameters, given as z_bounds=[z_surface, z_bottom], determining the depth layers' thickness in [m]. The first represents vertical distance from the ocean surface (z=0), whhile the second represents vertical distance from the ocean bottom, given by the bathymetry. Default is z_bounds=[10,100].
        particle_status : str, optional
            Parameter controlling which particles to include in raster, based on their status at the end of the run. Options are ['all', 'stranded', 'seafloor', 'active_only'], Default is 'active_only' 
        """
    
        t_0 = T.time()
         
        if decay_rate is None:
            decay_rate = self.decay_rate
        
        ### copied this section to __init__, if particle_path is not None
        # load opendrift dataset 
        #if self.ds is None:
        if type(self.particle_path) == dict: 
            for i in self.particle_path:
                path_list = self.particle_path
                path_list[i] = str(path_list[i])

            paths = path_list.values()
        else:
            paths = str(self.particle_path)
                
        _ds = xr.open_mfdataset(paths, concat_dim='trajectory', combine='nested', join='override', parallel=True, chunks={'trajectory': 10000, 'time':1000})

        _ds['trajectory'] = np.arange(0, len(_ds.trajectory)) # give trajectories unique ID            
        
        cond_stranded = {
                'stranded': _ds.status==1, 
                'seafloor': _ds.status==2, 
                'active_only': _ds.status==0,
               }

        if particle_status in cond_stranded.keys():
            _ds = _ds.where(cond_stranded[particle_status])
            
        self.ds = _ds
            
            
        ### if there is no poly_path to extract bounds from, take the one from the correct basin (which can be extracted from filename)
        if self.poly_path is None:
            if self.ds.poly_path is not None:
                self.poly_path = str(self.ds.poly_path)
            elif 'med' in self.context:
                self.poly_path = f'{DATA_DIR}/polygon-med-full-basin.shp'
            elif 'bs' in self.context:
                self.poly_path = f'{DATA_DIR}/polygon-bs-full-basin.shp'
            else:
                raise ValueError("No polygon could be extracted from particle_path. Please provide 'self.poly_path' manually.")
        else:
            pass


        ### TIME INTERPOLATION ###
        if tinterp is not None:
            new_time = np.arange(pd.to_datetime(_ds.time[0].values), pd.to_datetime(_ds.time[-1].values),timedelta(hours=tinterp)) #new time variables used for interpolation
            ds = _ds.interp(time=new_time, method='slinear') # interpolate dataset 
            ds['tinterp'] = tinterp
        else:
            ds = _ds
            

        ### BINS ### 
        
        # this polygon is only used to extract bounds for construction of bins in the case of an 'even' use distribution.
        if r_bounds is not None:
            poly = self.make_poly(lon=[r_bounds[0], r_bounds[2]], lat=[r_bounds[1], r_bounds[3]], write=False)
        else:    
            poly = gpd.read_file(self.poly_path)
            r_bounds = poly.total_bounds
           
        # if no use path is given, compute bins from resolution and bounds of polygon
        if use_path == 'even':
            weight = np.ones((ds.lon.shape))             # all particles have weight 1
            ds = ds.assign({'weight': (('trajectory', 'time'), weight)}, )
            xbin = np.arange(r_bounds[0],r_bounds[2]+res/1e5,res/1e5) # factor 1e5 is to approximately convert m to latlon deg
            ybin = np.arange(r_bounds[1],r_bounds[3]+res/1e5,res/1e5)

        else: 
            _use = rxr.open_rasterio(use_path).sortby('x').sortby('y').isel(band=0).drop('band')
            spatial_ref = _use.spatial_ref.crs_wkt
            bds_reproj = poly.to_crs(spatial_ref).total_bounds 
            #create `weight` variable from value of `use` at starting positions of particles
            
            use = _use.sel(x=slice(bds_reproj[0], bds_reproj[2]), y=slice(bds_reproj[1], bds_reproj[3])).rio.reproject(spatial_ref, resolution=res, nodata=0).rio.reproject('epsg:4326', nodata=0).sortby('x').sortby('y')
            
            _weight = use.sel(x=ds.isel(time=0).lon, y=ds.isel(time=0).lat, method='nearest')/30/24/60/60*self.tstep.total_seconds()   # matching timestep of simulation ### NEED TO GENERALISE
            
            ds['weight'] = _weight

            #use grid from use file as `bins` to compute histogram. (but need to shift from center to get same coordinates)
            #xbin = np.append(use.x - res/2, use.x[-1]+res/2)
            #ybin = np.append(use.y - res/2, use.y[-1]+res/2) 
            # res is now given in m
            xbin = np.append(use.x - np.diff(use.x).mean()/2, use.x[-1]+np.diff(use.x).mean()/2)
            ybin = np.append(use.y - np.diff(use.y).mean()/2, use.y[-1]+np.diff(use.y).mean()/2) 
            
        ### DECAY RATE ####
        k = decay_rate #decay coefficient given by user
        y = np.exp(-k*(ds.time-ds.time.min()).astype(int)/60/60/1e9/24) #decay function 
        ds['decay'] = y  
        ds['weight'] = ds.weight*ds.decay

        #### HISTOGRAMS. 2 CASES: 2D or 3D ####
        if self.depth is None:
            if len(np.unique(ds.z)) > 2:
                self.depth = True
            else:
                self.depth = False
        else:
            pass
        
        # FILTER OUT PARTICLES WHERE WEIGHT IS 0 TO FREE UP MEMORY.
        # NB: filter out entire TRAJECTORIES of particles which are released in cells where use layer is 0. this is done by extracting their particle ID. 
        p_id = ds.isel(time=0).where(ds.weight.isel(time=0)!=0).trajectory.data
        ds = ds.sel(trajectory=p_id).dropna('trajectory', 'all')
            
        # 3D
        if self.depth == True:
            ds = ds.assign(depth=-ds.z)
            # condition to filter out ds based on depth_layer
            cond = {'surface': ds.depth<=z_bounds[0], 'seafloor': ds.depth>(ds.sea_floor_depth_below_sea_level+z_bounds[1]), 'water_column': np.logical_and(ds.depth>z_bounds[0], ds.depth<-z_bounds[1]), 'mixed_layer': ds.depth <= ds.ocean_mixed_layer_thickness}
            
            if depth_layer in cond.keys():
                ds = ds.where(cond[depth_layer])
                
        elif self.depth == False:
            pass
        else:
            raise ValueError('cannot detect whether 2D or 3D')
        
        self.ds = ds

        
        ### chunking + aggregation method (sum, max, etc)
        step = 100 # this is completely arbitrary for now
        slices = int(len(ds.time)/step) # numer of slices / chunks
        
        ### need to rewrite this and make it cleaner (maybe use .exec()?), but for now:
        #### AGGREGATION METHOD ####
        qtemp = str('tempfiles'+"{:05d}".format(random.randint(0,99999)))
        Path(self.basedir / qtemp).mkdir(parents=True, exist_ok=True)
        
        if aggregate == 'mean':
            for i in range(0,slices+1):
                d = ds.isel(time=slice(step*i,step+step*i))
                hh = histogram(d.lon, d.lat, bins=[xbin, ybin], dim=['trajectory'], weights=d.weight, block_size=len(d.trajectory)).chunk({'lon_bin':10, 'lat_bin': 10}).mean('time')
                hh.to_netcdf(f'{self.basedir}/{qtemp}/temphist_{i}.nc')
                del hh, d

            _h = xr.open_mfdataset(f'{self.basedir}/{qtemp}/temphist*.nc', concat_dim='time', combine='nested').sum('time').histogram_lon_lat
            
        elif aggregate == 'max': 
            for i in range(0,slices+1):
                d = ds.isel(time=slice(step*i,step+step*i))
                hh = histogram(d.lon, d.lat, bins=[xbin, ybin], dim=['trajectory'], weights=d.weight, block_size=len(d.trajectory)).chunk({'lon_bin':10, 'lat_bin': 10}).max('time')
                hh.to_netcdf(f'{self.basedir}/{qtemp}/temphist_{i}.nc')
                del hh, d

            _h = xr.open_mfdataset(f'{self.basedir}/{qtemp}/temphist*.nc', concat_dim='time', combine='nested').max('time').histogram_lon_lat
        else:
            raise ValueError("'aggregate' must be one of 'mean' or 'max'.")
        
        h = _h.transpose()
                
        # write geo information to xarray and save as geotiff
        r = (
            xr.DataArray(h) # need to transpose it because xhistogram does that for some reason
            .rio.write_nodata(-1)
            .rio.write_crs('epsg:'+str(crs))
            .rio.write_coordinate_system())
        
        r=r.assign_attrs({'use_path': use_path}).to_dataset().rename({'histogram_lon_lat': 'r0'})
        
        ####### Landmask ######
        shpfilename = shpreader.natural_earth(resolution='10m',
                                    category='physical',
                                    name='land')
                    
        mask = gpd.read_file(shpfilename)

        if particle_status in ['active_only', 'seafloor']:
            ShapeMask = rasterio.features.geometry_mask(mask.to_crs(r.rio.crs).geometry,
                                                  out_shape=(len(r.lat_bin), len(r.lon_bin)),
                                                  transform=r.rio.transform(),
                                                  invert=True, all_touched=True)
            ShapeMask = xr.DataArray(ShapeMask , dims=("lat_bin", "lon_bin"))
            
            r = r.where(ShapeMask==0)
            
        elif particle_status == 'stranded':
            r = r.where(r!=0)
            
        else: 
            newm = mask.buffer(distance=-0.1) # add buffer to include both stranded and active particles
            ShapeMask = rasterio.features.geometry_mask(newm.to_crs(r.rio.crs).geometry,
                                                  out_shape=(len(r.lat_bin), len(r.lon_bin)),
                                                  transform=r.rio.transform(),
                                                  invert=True, all_touched=True)
            ShapeMask = xr.DataArray(ShapeMask , dims=("lat_bin", "lon_bin"))
            r = r.where(ShapeMask==0)
        
        # remove extra nan values on grid (land)
        r = r.dropna('lat_bin', 'all').dropna('lon_bin', 'all')
        
        # remove temporary files and folder
        for p in Path(self.basedir / qtemp).glob("temphist*.nc"):
            p.unlink()    
        
        # rm qtemp        
        os.rmdir(self.basedir / qtemp)

        elapsed = (T.time() - t_0)
        print("--- RASTER CREATED IN %s seconds ---" % timedelta(minutes=elapsed/60))
        
        #self.raster = r
        
        
        
        return r
    
    def plot(self, r=None, xlim=None, ylim=None, cmap=spectral_r, shading='flat', vmin=None, vmax=None, norm=None, coastres='10m', proj=ccrs.PlateCarree(), dpi=120, figsize=[10,7], rivers=False, title=None, save_fig=True):
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
        
        #for i, r in enumerate(self.raster.data_vars): # plot all available rasters
        else:
            fig = plt.figure(figsize=figsize)
            ax = plt.axes(projection=proj)
            ax.coastlines(coastres, zorder=11, color='k', linewidth=.5)
            ax.add_feature(cartopy.feature.LAND, facecolor='0.9', zorder=1) #'#FFE9B5'
            ax.add_feature(cartopy.feature.BORDERS, zorder=10, linewidth=.5, linestyle=':')
            if rivers is True:
                ax.add_feature(cartopy.feature.RIVERS, zorder=12)
            gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, linewidth=.5, color='gray', linestyle='--')
            gl.top_labels = False
            gl.right_labels = False    

            #im = self.raster[f'{r}'].where(self.raster[f'{r}']!=0).plot(vmin=vmin, vmax=vmax, norm=norm, shading=shading, cmap=cmap, add_colorbar=False)
            im = r.where(r!=0).plot(vmin=vmin, vmax=vmax, norm=norm, shading=shading, cmap=cmap, add_colorbar=False)
            cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
            cbar = plt.colorbar(im, cax=cax, extend='max')
            cbar.set_label('particles/gridcell', rotation=90)

            if title is not None:
                ax.set_title(title+'\n use_path: '+r.use_path, fontsize=12)
            else:
                ax.set_title('use_path: '+r.use_path, fontsize=8)

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
             #   Path(self.basedir / 'thumbnails').mkdir(parents=True, exist_ok=True)
                plt.savefig(str(save_fig), dpi=160, bbox_inches='tight')

            return fig, ax

    
    def scatter(self, t=None, xlim=None, ylim=None, s=1, c='age', cmap='rainbow', coastres='10m', proj=ccrs.PlateCarree(), dpi=120, figsize=[10,7], rivers=False, save_to=None):
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

        im = ax.scatter(ds.lon, ds.lat, s=s, c=c, cmap=cmap)
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

