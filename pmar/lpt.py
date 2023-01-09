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
    plot_map()
        plots output of particle_raster()
    animate()
        WIP
    
    """
    
    
    def __init__(self, basin, basedir='lpt_output', uv_path='cmems', wind_path='cmems', mld_path='cmems', bathy_path=None, particle_path=None, depth=False):
        """
        Parameters
        ----------
        basin : str
            Either one of 'med' for Mediterranean Sea or 'bs' for Black Sea. 
        basedir : str, optional
            path to the base directory where all output will be stored. Default is to create a directory called 'lpt output' in the current directory.
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
        # raise error if unsupported basin is requested
        if basin != 'med' and basin != 'bs':
            raise ValueError("Method is currently optimised for Med Sea and Black Sea only, thus 'basin' must be one of 'med' or 'bs'.")
        
        if depth == True and bathy_path == None:
                raise ValueError("Simulation is 3D but no bathymetry file was given. Please specify a bathy_path.")
        
        self.basin = basin
        Path(basedir).mkdir(parents=True, exist_ok=True)
        self.uv_path = uv_path
        self.wind_path = wind_path
        self.mld_path = mld_path # so that one can also specify a local path to data file. default is 'cmems' meaning we stream the data from copernicus
        self.bathy_path = bathy_path 
        self.basedir = Path(basedir)
        self.particle_path = particle_path
        
        if self.particle_path is not None: #take depth value from filename
            self.depth = bool(self.particle_path.split('depth_')[1].split('.')[0])                   
        elif self.bathy_path is not None:
            self.depth = True
        else:
            self.depth = depth    
        
        self.o = None
        self.poly_path = None # could give bath to full baisn as default
        self.raster = None
        self.ds = None
        self.origin_marker = 0
        pass
    
    def run(self, pnum, time, repeat_run=1, res=0.04, crs='4326', lon=None, lat=None, z=-0.5, tstep=6, hdiff=10, termvel=1e-3, bounds=None, use_path='even_dist', decay_rate=0, depth_layer='water_column', z_bounds=[10,100], loglevel=20, save_to=None):         
        """
        Launches methods particle_simulation and particle_raster. 
        
        
        Parameters 
        ----------
        pnum : int 
            The number of particles to be seeded
        time : list 
            List of strings giving the time bounds (start time and end time) for the particle simulation
        repeat_run = int, optional
            Number of desired iterations for the particle simulation. Default is 1
        res : float, optional
            Spatial resolution for raster in [deg]. Default is 0.04, which is ~4km
        crs : str, optional
            EPSG string for raster. Default is 4326
        lon : list, optional
            List giving two longitudinal bounds for particle seeding. Default is None (full basin)
        lat : list, optional
            List giving two latitudinal bounds for particle seeding. Default is None (full basin)
        z : float, optional
            Depth at which to seed particles in [m]. Default is -0.5m. 
        tstep : int, optional
            Time step used for OpenDrift simulation, in hours. Default is 6
        hdiff : float, optional
            Horizontal diffusivity particles. Default is 10m2/s
        termvel : float, optional
            Terminal velocity representing buoyancy of particles. Default is 0.001m/s
        bounds : list, optional
            Spatial bounds for computation of histogram (raster) written as bounds=[x1,y1,x2,y2]. Default is None (full basin)
        use_path : str, optional
            Path to .tif file representing density of human use of marine environment, used for 'weights' of particles in histogram calculation. 
            If no use_path is given, a weight of 1 is given to all particles ('even_dist' for even distribution).
        depth_layer : str, optional
            Depth layer chosen for computing histogram. One of 'full_depth', 'water_column', 'surface' or 'seafloor'. Default is 'full_depth'
        decay_rate : float, optional
            Decay rate of particles in [days-1]. Default is 0
        loglevel : int, optional
            OpenDrift loglevel. Set to 0 (default) to retrieve all debug information.
            Provide a higher value (e.g. 20) to receive less output.
        save_to : str, optional
            Filename to write raster figure to within the 'basedir' directory. 
        
        """
        # raise error if particle_path is already given
        if self.particle_path is not None:
            raise ValueError("Cannot use 'run' method if a 'particle_path' is already given. In this case, launch method 'particle_raster' directly.")
        

        logger.debug('Run starting')

        
        # this gives the option of doing e.g. monthly runs to avoid crashing when seeding a lot of particles. 
        t0 = datetime.strptime(time[0], '%Y-%m-%d')
        t1 = datetime.strptime(time[1], '%Y-%m-%d')
        iter_dt = t1-t0
        particle_list = {}
        
        for i in range(0, repeat_run):

            self.origin_marker = i 

            self.particle_simulation(pnum=pnum, time=[t0.strftime('%Y-%m-%d'), t1.strftime('%Y-%m-%d')], 
                                     res=res, crs=crs, lon=lon, lat=lat, z=z, tstep=tstep, hdiff=hdiff, termvel=termvel, loglevel=loglevel)    
            particle_list[i] = self.particle_path
            
            t0 = t1
            t1 = t0+iter_dt
                        
        if len(particle_list) == 1:
            self.particle_path == list(particle_list.values())[0]
        else:
            self.particle_path = particle_list
                
        # compute raster
        self.particle_raster(res=res, crs=crs, tstep=tstep, bounds=bounds, use_path=use_path, decay_rate=decay_rate, depth_layer=depth_layer, z_bounds=z_bounds, save_to=save_to)
        
        
        return self
    
    
    def make_poly(self, lon, lat, crs='4326'):
        """
        Creates shapely.Polygon where particles will be released homogeneously and writes it to a shapefile. 
        
        Parameters
        ----------
        lon : list or array
            Either a list giving 2 bounds or an array of lon coordinates.
        lat : list or array
            Either a list giving 2 bounds or an array of lat coordinates.
        crs : str
            EPSG string for Polygon crs. Default is '4326'
        """
        
        
        Path(self.basedir / 'polygons').mkdir(parents=True, exist_ok=True)
        poly_path = f'polygon-crs_epsg:{crs}.shp'
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
        
        poly.to_file(str(q), driver='ESRI Shapefile')
        self.poly_path = q
        pass
    

    
    def particle_simulation(self, pnum, time, res=0.04, crs='4326', lon=None, lat=None, z=-0.5, tstep=6, hdiff=10, termvel=1e-3, loglevel=20):
        """
        Method to start a trajectory simulation, after initial configuration, using OpenDrift by MET Norway.
        Uses OceanDrift module. 
                
        This method is currently optimised for use over the Mediterranean and Black Sea only. 
        Forcing data is streamed from Copernicus (ocean currents, wind, mixed layer depth), while GEBCO bathymetry is stored locally. 
        
        Particles are seeded homogeneously over polygon created using make_poly() method. 
        
        
        Parameters
        ----------
        pnum : int 
            The number of particles to be seeded
        time : list 
            List of strings giving the time bounds (start time and end time) for the particle simulation
        res : float, optional
            Spatial resolution for raster in [deg]. Default is 0.04, which is ~4km
        crs : str, optional
            EPSG string for raster. Default is 4326
        lon : list, optional
            List giving two longitudinal bounds for particle seeding. Default is None (full basin)
        lat : list, optional
            List giving two latitudinal bounds for particle seeding. Default is None (full basin)
        z : float, optional
            Depth at which to seed particles in [m]. Default is -0.5m. 
        tstep : int, optional
            Time step used for OpenDrift simulation, in hours. Default is 6
        hdiff : float, optional
            Horizontal diffusivity particles. Default is 10m2/s
        termvel : float, optional
            Terminal velocity representing buoyancy of particles. Default is 0.001m/s
        loglevel : int, optional
            OpenDrift loglevel. Set to 0 (default) to retrieve all debug information.
            Provide a higher value (e.g. 20) to receive less output.
        """
        
        
        Path(self.basedir / 'particles').mkdir(parents=True, exist_ok=True)
        
        self.o = OceanDrift(loglevel=loglevel) # initialise OpenDrift object
        
        
        # polygon used for seeding of particles. by default, it is the whole basin. but if lon and lat are given, a new polygon is created using those bounds. 
        if lon is not None and lat is not None:
            self.make_poly(lon, lat, crs=crs)
        else:
            if self.poly_path is None:
                self.poly_path = f'{DATA_DIR}/polygon-{str(self.basin)}-full-basin.shp'
        
        bds = np.round(gpd.read_file(self.poly_path).total_bounds) # only used in particle_path
       
        # stream data from Copernicus CMEMS (currents, wind)
        if self.uv_path == 'cmems':
            if self.basin == 'med':
                DATASET_ID = 'med-cmcc-cur-rean-d' #ocean currents med
            elif self.basin == 'bs':
                DATASET_ID = 'cmems_mod_blk_phy-cur_my_2.5km_P1D-m' # ocean currents bs
            else:
                raise ValueError("basin not recognised. Must be one of 'med' or 'bs'")
            self.o.add_readers_from_list([f'https://my.cmems-du.eu/thredds/dodsC/{DATASET_ID}'])
        elif self.uv_path != 'cmems': 
            #uv_reader = reader_netCDF_CF_generic.Reader(self.uv_path)
            self.o.add_readers_from_list(self.uv_path) # add reader from local file
        else: 
            raise ValueError('path to uv data file not recognised')
            
            
        if self.wind_path == 'cmems':
            WIND_ID = 'cmems_obs-wind_glo_phy_my_l4_P1M' #wind
            self.o.add_readers_from_list([f'https://my.cmems-du.eu/thredds/dodsC/{WIND_ID}'])
        elif self.wind_path != 'cmems':
            wind_reader = reader_netCDF_CF_generic.Reader(self.wind_path)
            self.o.add_readers_from_list(self.wind_path) # add reader from local file
        else: 
            raise ValueError('path to wind data file not recognised')            
        
        # bathymetry, mixed layer depth (3D case)
        if self.depth == True:
            # bathymetry
            bathy_reader = reader_netCDF_CF_generic.Reader(self.bathy_path)
            self.o.add_reader(bathy_reader)
            
            if self.mld_path == 'cmems':
                # mixed layer depth (cmems)
                mld_ID = f'{str(self.basin)}-cmcc-mld-rean-d'
                self.o.add_readers_from_list([f'https://my.cmems-du.eu/thredds/dodsC/{mld_ID}'])
            elif self.mld_path != 'cmems':
                mld_reader = reader_netCDF_CF_generic.Reader(self.mld_path)
                self.o.add_reader(mld_reader) # add reader from local file
            else: 
                raise ValueError('path to mld data file not recognised') 
        
        # time step and time range for simulation (user-defined)
        time_step = timedelta(hours=tstep)
        start_time = datetime.strptime(time[0], '%Y-%m-%d') 
        end_time = datetime.strptime(time[1], '%Y-%m-%d') 
        
        # path to write particle simulation file. used for our 'cache'
        particle_path = f'{str(self.basin)}-lon_{int(bds[0])}-{int(bds[2])}_lat_{int(bds[1])}-{int(bds[3])}-pnum_{pnum}-time_{time[0]}_to_{time[1]}-tstep_{tstep}hrs-hdiff_{hdiff}-termvel_{termvel}-depth_{str(self.depth)}.nc'
        q =  self.basedir / 'particles' / particle_path
            
        # if a file with that name already exists, simply import it using OpenDrift import method. 
        if q.exists() == True:
            self.o.io_import_file(str(q))
            ps = xr.open_mfdataset(q)
            print('NOTE: File with these configurations already exists within basedir and has been imported. Please delete the existing file to produce a new simulation.')
        # otherwise, run requested simulation
        elif q.exists() == False:
            
            # landmask from cartopy (from "use shapefile as landmask" example on OpenDrift documentation)
            shpfilename = shpreader.natural_earth(resolution='10m',
                                    category='cultural',
                                    name='admin_0_countries')
            reader_natural = reader_shape.Reader.from_shpfiles(shpfilename)

            self.o.add_reader([reader_natural])
            self.o.set_config('general:use_auto_landmask', False)  # Disabling the automatic GSHHG landmask
            self.o.set_config('general:coastline_action', 'stranding')
            
            # horizontal diffusivity
            self.o.set_config('drift:horizontal_diffusivity', hdiff)    
            
            if self.depth == True:
                # if simulation is 3D, set 3D parameters (terminal velocity, vertical mixing, seafloor action) and seed particles over polygon
                self.o.set_config('seed:terminal_velocity', termvel)
                self.o.seed_from_shapefile(shapefile=str(self.poly_path), number=pnum, time=start_time, terminal_velocity=termvel, z=z, origin_marker=self.origin_marker)
                #self.o.set_config('vertical_mixing:diffusivitymodel', 'windspeed_Large1994')
                self.o.set_config('general:seafloor_action', 'deactivate')
                self.o.set_config('drift:vertical_mixing', True)
            
            else:
                logger.debug(f'2D seeding from shapefile {self.poly_path}')
                # if simulation is 2D, simply seed particles over polygon
                self.o.seed_from_shapefile(shapefile=str(self.poly_path), number=pnum, time=start_time, origin_marker=self.origin_marker)
            
            # run simulation and write to temporary file
            temp_outfile = str(self.basedir)+'/temp_particle_file.nc'
            self.o.run(end_time=end_time, time_step=time_step, outfile=temp_outfile)

            # open temporary file to handle 'inactive' particles (i.e. particles that have beached or gotten stuck on seafloor). 
            # inactive particles are kept visible rather than deactivated, so they contribute to final count in particle_raster method. 
            # new file is written with final filename given by particle_path.  
            _ps = xr.open_dataset(temp_outfile)
            ps = _ps.where(_ps.status==0).ffill('time') 
            ps['dt'] = tstep #adding timestep to ds, will be useful later 
            ps.to_netcdf(str(q)) 
        
        self.ds = ps
        
        self.particle_path = str(q) #particle_path

        pass     
    

    def particle_raster(self, res=0.04, crs='4326', tstep=None, bounds=None, use_path='even_dist', decay_rate=0, depth_layer='full_depth', z_bounds=[10,100], save_to=None):
        """
        Method to compute a 2D horizontal histogram of particle concentration using the xhistogram.xarray package. 
        
        If a use_path is given, particle 'weight' is the value of the given human use in the particle's initial position. 
        Default weight is 1 for all particles.
        
        
        Parameters
        ----------
        res : float, optional
            Spatial resolution for raster in [deg]. Default is 0.04, which is ~4km
        crs : str, optional
            EPSG string for raster. Default is 4326
        tstep : int, optional
            New timestep used for interpolation. 
        bounds : bounds, optional
            Spatial bounds for computation of histogram (raster) written as bounds=[x1,y1,x2,y2]. Default is None (full basin)
        use_path : str, optional
            Path to file representing density of human use, used for 'weights' of particles in histogram calculation. 
            If no use_path is given, a weight of 1 is given to all particles. Default is None
        decay_rate : float, optional
            Decay rate of particles in [days-1]. Default is 0
        depth_layer : str, optional
            Depth layer chosen for computing histogram. One of 'full_depth', 'water_column', 'surface' or 'seafloor'. Default is 'full_depth'
        z_bounds : list, optional
            Two parameters, given as z_bounds=[z_surface, z_bottom], determining the depth layers' thickness in [m]. The first represents vertical distance from the ocean surface (z=0), whhile the second represents vertical distance from the ocean bottom, given by the bathymetry. Default is z_bounds=[10,100].
        save_to : str, optional
            Filename to write raster figure to within the 'basedir' directory. 
        """
    
        t0 = T.time()
        Path(self.basedir / 'rasters').mkdir(parents=True, exist_ok=True)

        
        ### if there is no poly_path to extract bounds from, take the one from the correct basin (which can be extracted from filename)
        if self.poly_path is None:
            if type(self.particle_path) != str:
                particle_path = self.particle_path[1]
            else:
                particle_path = self.particle_path

            #basin = particle_path.split('-',1)[0]
            self.poly_path = self.poly_path = f'/{DATA_DIR}/polygon-{str(self.basin)}-full-basin.shp'
        else:
            pass

        if type(self.particle_path) == dict: 
            for i in self.particle_path:
                path_list = self.particle_path
                path_list[i] = str(path_list[i])
            
            _ds = xr.open_mfdataset(path_list.values(), concat_dim='trajectory', combine='nested', join='outer').isel(time=slice(1,None)).chunk({'trajectory': 10000, 'time':1000}) # removing time 0 
        else:
            #q = self.basedir / list(self.particle_path)[0]
            _ds = xr.open_mfdataset(str(self.particle_path), concat_dim='trajectory', combine='nested', join='outer').isel(time=slice(1,None)).chunk({'trajectory': 10000, 'time':1000}) # removing time 0

        
        _ds['trajectory'] = np.arange(0, len(_ds.trajectory)) # give trajectories unique ID


        # if new timestep is given in raster request, interpolate dataset to fill in gaps
        if tstep is not None:
            new_time = np.arange(pd.to_datetime(_ds.time[0].values), pd.to_datetime(_ds.time[-1].values),timedelta(hours=tstep)) #new time variables used for interpolation
            ds = _ds.interp(time=new_time) # interpolate dataset 
            ds['dt'] = tstep
        else:
            ds = _ds
            

        ### BINS ### 
        if bounds is None:
            poly = gpd.read_file(self.poly_path)
            bounds = poly.total_bounds
        else:
            pass

        # if no use path is given, take resolution and polybounds (works even if it is full basin) and calculate bins this way. 
        if use_path == 'even_dist':
            # all particles have weight 1
            weight = np.ones((ds.lon.shape))
            ds = ds.assign({'weight': (('trajectory', 'time'), weight)}, )
            # bins are calculated from given res and bounds
            xbin = np.arange(bounds[0],bounds[2]+res,res)
            ybin = np.arange(bounds[1],bounds[3]+res,res)
        else: 
        #create `weight` variable from value of `use` at starting positions of particles
            use = rxr.open_rasterio(use_path).isel(band=0).rio.reproject('epsg:'+str(crs), resolution=res, nodata=0).drop('band').squeeze().sortby('x').sortby('y').sel(x=slice(bounds[0], bounds[2]), y=slice(bounds[1], bounds[3]))
            
            _weight = use.sel(x=ds.isel(time=0).lon, y=ds.isel(time=0).lat, method='nearest')/30/24*ds.dt # matching timestep of simulation
            #weight = xr.broadcast(_weight, ds.time)[0].fillna(0)
            
            ds['weight'] = _weight

            #use grid from use file as `bins` to compute histogram. (but need to shift from center to get same coordinates)
            xbin = np.append(use.x - res/2, use.x[-1]+res/2)
            ybin = np.append(use.y - res/2, use.y[-1]+res/2) 
            
        #print('weight no decay', ds.weight)
        
        ### DECAY RATE ####
#        if decay_rate is not None:
        k = decay_rate #decay coefficient given by user
        y = np.exp(-k*(ds.time-ds.time.min()).astype(int)/60/60/1e9/24) #decay function 
        decay = xr.broadcast(y, ds.lon, exclude=['time'])[0]
        ds['decay'] = y  
        ds['weight'] = ds.weight*ds.decay
 #       else:
  #          pass
        
        #print('weight after decay', ds.weight)
        
        #### HISTOGRAMS. 2 CASES: 2D or 3D ####
        if self.depth is None:
            if len(np.unique(ds.z)) > 2:
                self.depth = True
            else:
                self.depth = False
        else:
            pass
        
        # FILTER OUT PARTICLES WHERE WEIGHT IS 0 TO FREE UP MEMORY
        ds = ds.where(ds.weight!=0)
        
        self.ds = ds
        
        
        # calculate histogram
        if self.depth == False:
            # compute histogram for each timestep
            _h = histogram(ds.lon, ds.lat, bins=[xbin, ybin], dim=['trajectory', 'time'], weights=ds.weight)#, density=True)
            
        # 3D
        elif self.depth == True:
            ds = ds.assign(depth=-ds.z)
            
            # in 3D case, in addition to merge alg we also have options for vertical layers
            if depth_layer == 'surface':
                surface = ds.where(ds.depth<=z_bounds[0])
                _h = histogram(surface.lon, surface.lat, bins=[xbin, ybin], dim=['trajectory', 'time'], weights=surface.weight)
                
            elif depth_layer == 'seafloor':
                seafloor = ds.where(ds.depth>(ds.sea_floor_depth_below_sea_level-z_bounds[1]))
                _h = histogram(seafloor.lon, seafloor.lat, bins=[xbin, ybin], dim=['trajectory', 'time'], weights=seafloor.weight)
            
            elif depth_layer == 'water_column':
                column = ds.where(np.logical_and(ds.depth>z_bounds[0], ds.depth<z_bounds[1]))
                _h = histogram(column.lon, column.lat, bins=[xbin, ybin], dim=['trajectory', 'time'], weights=column.weight)            
            
            elif depth_layer == 'full_depth':
                _h = histogram(ds.lon, ds.lat, bins=[xbin, ybin], dim=['trajectory', 'time'], weights=ds.weight)
            
            else:
                raise ValueError('"depth_layer" must be one of "full_depth", "surface", "seafloor" or "water_column"')    
        
        else:
            raise ValueError('cannot detect whether 2D or 3D')
            
        h = _h.transpose()

            
        # write geo information to xarray and save as geotiff
        r = (
            xr.DataArray(h) # need to transpose it because xhistogram does that for some reason
            .rio.write_nodata(-1)
            .rio.write_crs('epsg:'+str(crs))
            .rio.write_coordinate_system())

        self.raster = r

        if save_to is not None:
            #rpath = list(self.particle_path.values())[0][:-3]+'_RASTER_'+use_path.split('/')[-1].split('.')[0]+'.tif'
            r.rio.to_raster(self.basedir / 'rasters' / save_to)
        #elif save_to == 'netcdf':
         #   rpath = list(self.particle_path.values())[0][:-3]+'_RASTER_'+use_path.split('/')[-1].split('.')[0]+'.nc'
          #  r.to_netcdf(self.basedir / 'rasters' / rpath)
        else:
            pass
        #else:
         #   raise ValueError("'save_to', when specified, has to be one of 'tiff' ot 'netcdf'")

        elapsed = (T.time() - t0)
        print("--- RASTER CREATED IN %s seconds ---" % timedelta(minutes=elapsed/60))    

        pass
    
    
    def plot_map(self, xlim=None, ylim=None, cmap=spectral_r, shading='flat', vmin=None, vmax=None, norm=None, coastres='10m', proj=ccrs.PlateCarree(), dpi=120, figsize=[10,7]):
        """
        
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
        """
        
        
        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 16
        
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=proj)
        ax.coastlines(coastres, zorder=10, color='k', linewidth=2)
        ax.add_feature(cartopy.feature.LAND, facecolor='0.9', zorder=10) #'#FFE9B5'
        ax.add_feature(cartopy.feature.BORDERS, zorder=10, linewidth=.5, linestyle=':')

        gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, linewidth=.5, color='gray', linestyle='--')
        gl.top_labels = False
        gl.right_labels = False    
        
        r = self.raster#.where(self.raster>0, 1e-30)
        im = r.plot(vmin=vmin, vmax=vmax, norm=norm, shading=shading, cmap=cmap, add_colorbar=False)
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        plt.colorbar(im, cax=cax, extend='max')
        
        ax.set_title('mean particle distribution')
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
                

    def animate(self):
        """
        WIP. Histogram animation using xmovie
        
        """
        pass

