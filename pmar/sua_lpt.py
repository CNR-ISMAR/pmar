import logging
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import time as T
import warnings
from pmar.lpt import LagrangianDispersion
from SALib.sample import saltelli
from SALib.analyze import sobol
from joblib import Parallel, delayed
from copy import deepcopy
import rioxarray as rxr
import matplotlib.pyplot as plt
import time 
import geopandas as gpd

class PMARCaseStudy(object):
    """
    Developed at CNR ISMAR in Venice.
    Class to create a case study object under the PMAR module. 

    ...

    Attributes
    ----------
    context : str
        String defining the context of the simulation i.e., the ocean model output to be used for particle forcing. Options are 'med-cmems', 'bs-cmems' and 'bridge-bs'. 
    pnum : int, optional
        The number of particles to be seeded. Default is 20000 
    duration : int, optional
        Integer defining the length of the particle simulation, in days. Default is 30 (for test runs)    
    tstep : timedelta, optional
        Time step of the OpenDrift simulation i.e., how often data is registered along the trajectory. Default is 6 hours
    particle_status : str, optional
        Filter particles based on their status at the end of the run i.e., whether they are still active or have beached or sunk. Options are ['all', 'stranded', 'seafloor', 'active'], Default is 'all' 
    runtypelevel : int, optional
        Default is 3
    poly_path : str, optional
        Path to shapefile containing polygon used for particle seeding
        
    Methods
    -------
    get_main_output()
        Method returning main output of the lpt run
    get_SUA_target()
        Method returning lpt output to run Sensitivity Analysis on
    run()
        Method running lpt simulation with selected parameters

    """
    def __init__(self, context, ptot=1000, particle_status='all', poly_path=None, s_bounds = None, basedir='lpt_output', netrc_path=None, loglevel=40):
        """
        Parameters
        ----------   
        context : str
            String defining the context of the simulation i.e., the ocean model output to be used for particle forcing. Options are 'med-cmems', 'bs-cmems' and 'bridge-bs'. 
        ptot : int, optional
            Total number of particles forming raster, i.e. sum of pnum in each rep. Default is 100 (for test runs)
        duration : int, optional
            Integer defining the length of the particle simulation, in days. Default is 30 (for test runs)    
        tstep : timedelta, optional
            Time step of the OpenDrift simulation i.e., how often data is registered along the trajectory. Default is 6 hours
        particle_status : str, optional
            Filter particles based on their status at the end of the run i.e., whether they are still active or have beached or sunk. Options are ['all', 'stranded', 'seafloor', 'active'], Default is 'all' 
        poly_path : str, optional
            Path to shapefile containing polygon used for particle seeding        
        """
        self.lpt = None
        #self.pnum = pnum
        self.duration = None
        self.tstep = None
        self.context = context
        self.particle_status = particle_status
        self.runtypelevel = None
        self.poly_path = poly_path
        self.basedir = basedir
        self.netrc_path = netrc_path
        self.tshift = None
        self.ptot = ptot
        self.reps = None # NUMBER OF REPS DEPENDS ON HOW MANY WE CAN FIT IN A YEAR WITH THAT TSHIFT
        self.loglevel = loglevel
        self.hdiff = 10
        self.decay_rate = 0
        #self.start_time_delay = 0
        self.start_time = '2019-01-01'
        self.s_bounds = s_bounds
        pass

    def get_main_output(self):
        """
        Method returning main output of the lpt run
        """
        #self.lpt.raster.r0.rio.to_raster(self.lpt.particle_path)
        return self.lpt.raster.r0
    
    def get_SUA_target(self):
        """
        Method returning lpt output to run Sensitivity Analysis on
        """
        return self.lpt.raster.r0.sum()
    
    def run(self, runtypelevel=3):
        """
        Method running lpt simulation with selected parameters
        
        Parameters
        ----------
        runtypelevel : int, optional
            Default is 3
        """
        self.lpt = LagrangianDispersion(context=self.context, poly_path=self.poly_path, basedir=self.basedir, netrc_path=self.netrc_path)        
        self.runtypelevel=runtypelevel
        
        self.lpt.repeat_run(reps=int(np.round(365/self.tshift)), start_time=self.start_time, tshift=self.tshift, ptot=self.ptot, s_bounds=self.s_bounds, duration_days=self.duration, tstep=self.tstep, particle_status=self.particle_status, hdiff=self.hdiff, decay_rate=self.decay_rate, loglevel=self.loglevel) # here would have to put start_time = self.start_time if removing starttime delay
        pass


class RunningStats2D:
    """
    This is a class for online computation of mean, variance and convergence arrays.
    """
    def __init__(self, percentiles=None):
        """
        :param percentiles: list-like (list, tuple, ecc) of percentile threshold values. E.g. [25, 50, 75].
        """
        self.n = 0
        self.old_m = None
        self.new_m = None
        self.old_s = None
        self.new_s = None
        self.initialized = False

        self.percentiles = percentiles
        self._convergence_arrays = []

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        
        
        # mean and variances
        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = np.zeros_like(x)
            print(f'######################### old_m {self.old_m.shape}, new_m {self.new_m.shape}, x {x.shape}')

            if self.percentiles is not None:
                for p in self.percentiles:
                    self._convergence_arrays.append(np.zeros_like(x))
        else:
            print(f'######################### old_m {self.old_m.shape}, new_m {self.new_m.shape}, x {x.shape}')
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

        if self.percentiles is not None:
            pvals = np.percentile(np.ndarray.flatten(x[~x.mask]),
                                  self.percentiles)
            logger.debug('min={} max={} pvals={}'.format(x.min(), x.max(), pvals))
            for i, pval in enumerate(pvals):
                r = x.copy()
                r[x<pval] = 0
                r[x>=pval] = 1
                self._convergence_arrays[i] += r

    @property
    def mean(self):
        return self.new_m if self.n else 0.0

    @property
    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self):
        return np.sqrt(self.variance)

    def convergence_arrays(self):
        return [m / self.n for m in self._convergence_arrays]
    
    
class CaseStudySUA(object):
    """
    This is a base class for Sensitivity and Uncertainty Analysis.
    Child classes have to implement the "set_problem" and "set_params" methods.
    """
    def __init__(self, module_cs_params=None, nparams=40, bygroup=True, calc_second_order=False,
                 kwargs_run={}):
        
        #if module_cs_params is not None: 
        self.module_cs_params = module_cs_params
          #  module_cs = PMARCaseStudy(**self.module_cs_params)
        #self.module_cs = module_cs

        self.kwargs_run = kwargs_run
        # check runtypelevel
        #if self.module_cs.runtypelevel is None or self.module_cs.runtypelevel < 3:
         #   raise Exception("Case Study doesn't have a sufficient runlevel")
          #  pass
        # TODO: casestudy should store info on input parameters (e.g. uses,
        #  pres, envs) in order to allow multiple runs
        self.problem = {
            'num_vars': 0,
            'names': [],  # ['mscf'],  # , 'rfunc_b', 'rfuncb_m'],
            'bounds': [],  # [[0, 1]],  # , [1, 30], [0.3, 0.7]],
            'dists': [],  # ['unif'],  # 'unif', 'unif'],
            # 'groups': [],  # ['mscf'],  # 'rfunc', 'rfunc']
        }

        if bygroup:
            self.problem['groups'] = []

        self.var_index = []
        self.model_output_stats = None
        self.target_values = None
        self.nparams = nparams
        self.bygroup = bygroup
        self.calc_second_order = calc_second_order
        #self.module_cs_params = module_cs_params
    
    def set_problem(self):
        """This is override by child classes"""
        pass

    def set_params(self, params):
        """This is override by child classes"""
        pass

    def add_problem_var(self, var_idx, bound, dist, group=None):
        name = ' '.join(var_idx)
        self.problem['num_vars'] += 1
        self.problem['names'].append(name)
        self.problem['bounds'].append(bound)
        self.problem['dists'].append(dist)
        if group is not None and self.bygroup:
            self.problem['groups'].append(group)
        elif group is None and self.bygroup:
            self.problem['groups'].append(name)
        else:
            pass

        self.var_index.append(var_idx)
        #logger.debug('add problem var {} {} {} {}'.format(name, bound, dist, group))
        
    def sample(self, nruns, calc_second_order=False):
        return saltelli.sample(self.problem, nruns,
                               calc_second_order=calc_second_order)     
        
    
    def analyze(self, calc_second_order=False):
        sa_results = sobol.analyze(self.problem,
                                   self.target_values,
                                   calc_second_order=calc_second_order)
        return sa_results

    @property
    def mean(self):
        return self.model_output_stats.mean

    @property
    def std(self):
        return self.model_output_stats.std

    @property
    def cv(self):
        mean = self.mean
        std = self.std
        return std/mean

class PMARCaseStudySUA(CaseStudySUA):
    """
    Child class of CaseStudySUA for Sensitivity and Uncertainty Analysis on outputs of the PMAR module. 
    The class defines the two methods 'set_problem' and 'set_params' specifically for PMAR Sensitivity and Uncertainty Analysis (SUA). 

    ...

    Methods
    -------
    set_problem()
        Method used to set the problem by adding the variables to be analysed in SUA.
    set_params()
        Method used to correctly feed the sampled values of each variables to the lpt run. 
    """
    def __init__(self, module_cs_params=None, calc_second_order=False,
                 kwargs_run={}):
        super().__init__(module_cs_params, calc_second_order=calc_second_order) # this calls the init from the parent class, and adds the following line
        if module_cs_params is not None:
            self.module_cs = PMARCaseStudy(**self.module_cs_params)
    
        #self.decay_shift = 0.01 # because we are using logunif dist for the decay rate but we want values from 0 to 1
        
    def set_problem(self):
        """
        Define the problem by giving a set of variables, their names, bounds and probability distributions. 
        
        """
        
        #nparams = self.nparams # ?
        #module_cs = self.module_cs # ? 
        #self.normalize_distance = None # ?
        
        # the problem is defined by a number of vairables, names of the variables, and their bounds.
        # D = 2 # number of variables
        
        self.add_problem_var(['time_var', 'duration', 'duration'], # length of the simulation in days # ['duration_days', 'duration', label], "label" is for tools4msp like ENV, USEPRE ecc
                     [30, 90], # bounds # tra 1 mese e 3 mesi (cioè una stagione)
                     'unif', # distribution
                     #group_label if self.bygroup else None
                     )

        self.add_problem_var(['time_var', 'tstep', 'tstep'], # length of timestep in hours
                     [1, 6], # bounds # 4 is the max at the 90th percentile of particle velocity for 4km resolution of raster
                     'unif', # distribution
                     #group_label if self.bygroup else None
                     )
        
        #self.add_problem_var(['hdiff', 'hdiff', 'hdiff'], # horizontal diffusivity
         #            [0, 15], # bounds # from baudena et al., streaming of plastic in the mediterranean sea
          #           'unif', # distribution
                     #group_label if self.bygroup else None
                    # )
        
        #self.add_problem_var(['termvel', 'termvel', 'termvel'], # 
         #            [1e-3, 1e-2], # bounds
          #           'unif', # distribution
           #          #group_label if self.bygroup else None
            #         )        
        
        #self.add_problem_var(['decay_rate', 'decay_rate', 'decay_rate'], # decay rate
                    # [0+self.decay_shift, 1+self.decay_shift], # bounds
                     #'logunif', # distribution
                     #group_label if self.bygroup else None
                    # )
        
        self.add_problem_var(['time_var', 'tshift', 'tshift'], # time shift between each rep
                     [15, 30], # bounds # 90 dà 4 stagioni diverse, 15 due volte al mese
                     'unif', # distribution
                     #group_label if self.bygroup else None
                     )    

        #self.add_problem_var(['time_var', 'start_time_delay', 'start_time_delay'], # start time of first rep of each run is 2019-01-01 + start_time_delay
         #            [0, 365], 
          #           'unif', # distribution
                     #group_label if self.bygroup else None
           #          )    

        self.add_problem_var(['time_var', 'start_time', 'start_time'], # start time of first rep of each run 
                     [1420070400.0, 1609372800.0], # the range is given by what is available on CMEMS black sea physics reanalysis. e.g. from 2015 to 2020 we have both wind and currents and we 
                     'unif', # distribution
                     #group_label if self.bygroup else None
                     )   
    
    def set_params(self, params, module_cs=None):
        """
        Method used to correctly feed the sampled values of each variables to the lpt run. 
        
        Parameters
        ----------
        params : np.array
            Numpy array containing output of Saltelli sample for the variables defined in 'set_problem'
        module_cs : 
            Tools4MSP case study object. In this case, PMARCaseStudy
        
        """
        
        if module_cs is None:
            module_cs = self.module_cs
            
        module_cs.duration = params[0] 
        module_cs.tstep = timedelta(hours=params[1])
        module_cs.tshift = params[2]
        #module_cs.start_time_delay = timedelta(days=params[3])
        module_cs.start_time = time.strftime('%Y-%m-%d', time.localtime(params[3])) 
        if params[1] == 2.5234375 and params[0] == 73.59375: ### this is to avoid a bug in OpenDrift which has been addressed in Slack by Sofia on 4/06/24
            print('########## ROUNDING TSTEP !!!!! ######')
            module_cs.tstep = timedelta(hours=np.round(params[1],1))
            print('#####################', module_cs.tstep) 
            
    def runall(self, nruns, calc_second_order=False, njobs=1):
        self.set_problem()
        samples = self.sample(nruns, calc_second_order)
        #logger.debug('Samples={} calc_second_order={}'.format(len(samples), calc_second_order))
        model_output_stats = RunningStats2D(percentiles=None) # initialise objects of class runningstats2d
        #print('model_output_stats #1', model_output_stats) # debugging 6/03/24
        # TODO: add parallelization
        
        def _single_run(i, params):
            print("########", i, params)    
                  
            module_cs = deepcopy(self.module_cs) 
                
            self.set_params(params, module_cs=module_cs)
            module_cs.run(runtypelevel=0, **self.kwargs_run)
            model_output = module_cs.get_main_output()
            print(model_output.max(), model_output.min())
            model_output_stats.push(model_output) # as input for method push (x) give model output, which in this case is r0
            target_value = module_cs.get_SUA_target()

            return target_value
            
        with Parallel(n_jobs=njobs, backend='threading', require='sharedmem') as parallel:
            target_values = parallel(delayed(_single_run)(i, params) for i, params in enumerate(samples))

        # for i, params in enumerate(samples):
        #     self.set_params(params)
        #     self.module_cs.run(runtypelevel=0, **self.kwargs_run)
        #     model_output = self.module_cs.get_main_output()
        #     model_output_stats.push(model_output)
        #     target_value = self.module_cs.get_SUA_target()
        #     target_values.append(target_value)
        #     logger.debug('run {} target={}'.format(i, target_value))
        self.model_output_stats = model_output_stats
        self.target_values = np.array(target_values)
        #print(self.model_output_stats, self.target_values)
        

    #### addition by sofbo
    def readall(self, path_to_output, njobs=1, clip=None):
        """ 
        Read previously made outputs. Feeds model_output directly.
        
        Parameters
        ----------
        path_to_output : str
            String containing the path to the output files to be analysed.
        clip : str
            Path to shapefile with polygon used to clip raster
        """
        self.set_problem()
        #samples = self.sample(nruns, calc_second_order)
        #logger.debug('Samples={} calc_second_order={}'.format(len(samples), calc_second_order))
        model_output_stats = RunningStats2D(percentiles=None)
        
        # TODO: add parallelization
        
        def _single_run(i, outputs):
            # module_cs = deepcopy(self.module_cs)
            #module_cs = PMARCaseStudy(**self.module_cs_params) ### COMMENTING TO TEST, 08/01/24
            #self.module_cs = module_cs ### COMMENTING TO TEST, 08/01/24
            #self.set_params(params, module_cs=module_cs)
            #module_cs.run(runtypelevel=0, **self.kwargs_run)
            print(outputs) # printing file names
            model_output = rxr.open_rasterio(outputs).isel(band=0).drop('band')

            # clip the raster to focus analysis on a specific region
            if clip is not None:
                poly = gpd.read_file(str(clip))
                model_output = model_output.rio.clip(poly.geometry.values, poly.crs, drop=False)
                
            model_output_stats.push(model_output.data)
            target_value = model_output.sum()
            #logger.debug('run {} target={}'.format(i, target_value))
            return target_value
            
        with Parallel(n_jobs=njobs, backend='threading', require='sharedmem') as parallel:
            target_values = parallel(delayed(_single_run)(i, outputs) for i, outputs in enumerate(path_to_output))
                
        #for i, outputs in enumerate(path_to_output):
         #   target_values = _single_run(i, outputs)
            
        self.model_output_stats = model_output_stats
        self.target_values = np.array(target_values)   