# fix these by using plot_map
# add plot_origin, plot_destination, plot_network, plot_CM
import cartopy
import numpy as np
from matplotlib.ticker import LogFormatter, PercentFormatter
from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib.colors as col
import seaborn as sns
from cmap import Colormap
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import matplotlib as mpl

tol_inc = Colormap('tol:incandescent').to_mpl()
tol_muted = Colormap('tol:muted').to_mpl()
rainbow = Colormap('tol:rainbow_PuRd').to_mpl()
tol_div = Colormap('tol:BuRd').to_mpl()

def plot_map(subplot_rows=1, subplot_cols=1, proj=cartopy.crs.PlateCarree(), extent=None, coastres='10m', figsize=[8,6], title='', filename=None):
    fig, axs = plt.subplots(subplot_rows, subplot_cols, figsize=figsize, subplot_kw={'projection':proj})
    for ax in fig.axes:
        ax.coastlines(coastres, zorder=12, color='k', linewidth=.7)
        ax.add_feature(cartopy.feature.LAND, facecolor='0.9', zorder=2, alpha=0.5)
        ax.add_feature(cartopy.feature.BORDERS, zorder=11, linewidth=.5, linestyle=':')
        if extent is not None:
            ax.set_extent(extent, crs=proj) 
        
        gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, linewidth=0, color='gray', linestyle='--')
        #levels=np.array([1,10,100,1000])
        gl.top_labels = False
        gl.right_labels = False
        
        ax.set_title(title)

    if filename is not None:
        plt.savefig(str(filename), dpi=160, bbox_inches='tight')
    
    return fig, axs

def plot_array(array, tick_labels=None, figsize=[8,6], annot=False, linewidths=0, vmax=None, norm=None, cmap=tol_inc, ax=None):
    # use cm_grid for tick_labels
    
    #CM = self.CM.toarray() # my CM is now by default a sparse array. to plot it, must convert it to a dense array 
    
    if len(array) > 30:
        annot = False
        
    #fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(array, annot=annot, linewidths=linewidths, square=True, ax=ax, vmax=vmax, norm=norm, cmap=cmap)
    ax.set_ylabel('FROM')
    ax.set_xlabel('TO')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 

    if tick_labels is not None:
        ax.set_yticks(ticks=np.arange(0.5,len(tick_labels)), labels=tick_labels, rotation=0);
        ax.set_xticks(ticks=np.arange(0.5,len(tick_labels)), labels=tick_labels, rotation=90);

    #return fig, ax 


def _plot_network(G, node_shapes=None, show_map=False, width=2, rescale_edge=None, figsize=[7,7], extent=None, cmap=tol_inc):

    if show_map == True:
        fig, ax = plot_map(figsize=figsize, extent=extent)
    else:
        fig, ax = plt.subplots(figsize=figsize)

    if node_shapes is not None:
    
        coordinates = np.column_stack((node_shapes.geometry.centroid.x, node_shapes.geometry.centroid.y))

        positions = dict(zip(G.nodes, coordinates))
        node_shapes.boundary.plot(ax=ax, color='grey', linewidth=.5)
        node_size = 0
    else:
        node_size = 5
        positions = nx.spring_layout(G)

    #if nx.get_node_attributes(G, 'node_name'):
     #   labels = nx.get_node_attributes(G, 'node_name')
    #else:
     #   labels = None

    if rescale_edge is not None:
        width = np.array(list(nx.get_edge_attributes(G,'weight').values()))*50*rescale_edge
    else:
        width = width

    if nx.get_node_attributes(G, 'node_name'):
        nx.draw_networkx_labels(G, 
                                positions, 
                                labels=nx.get_node_attributes(G, 'node_name'),
                                font_size=8)#, font_weight="bold")

    nodes = nx.draw_networkx_nodes(G, 
                                   ax=ax,
                                   pos=positions, 
                                   node_size=node_size, 
                                   )
    edges = nx.draw_networkx_edges(G,
                                   ax=ax,
                                   pos=positions,
                                   node_size=node_size,
                                   arrowstyle="->",
                                   connectionstyle='arc3, rad = 0.1',
                                   edge_color=nx.get_edge_attributes(G, 'weight').values(),
                                   edge_cmap=cmap,
                                    width=width,
                                ) 

    pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    
    ax = plt.gca()
    ax.set_axis_off()
    plt.colorbar(pc, ax=ax)

    return fig, ax 
    


def custom_plotfunc(ds, fig, tt, *args, **kwargs):
    '''
    For pmar.animate. WIP.
    '''
    
    ax = fig.subplots(subplot_kw={'projection':cartopy.crs.PlateCarree()})
    ax.coastlines('10m', zorder=12, color='k', linewidth=.7)
    ax.add_feature(cartopy.feature.LAND, facecolor='0.9', zorder=2)
    ax.add_feature(cartopy.feature.BORDERS, zorder=11, linewidth=.5, linestyle=':')
    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, linewidth=0, color='gray', linestyle='--')
    #levels=np.array([1,10,100,1000])
    gl.top_labels = False
    gl.right_labels = False

    # Colorlimits need to be fixed or your video is going to cause seizures.
    ds.isel(time=tt).plot(ax=ax, cmap=spectral_r, vmin=0, vmax=ds.max())
    ax.set_title('');
    return None, None

def plot(self, raster, title=None, xlim=None, ylim=None, cmap=None, shading='flat', vmin=None, vmax=None, norm=None, coastres='10m', proj=cartopy.crs.epsg(3857), transform=cartopy.crs.PlateCarree(), dpi=120, figsize=[10,7], rivers=False, savepng=None):
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
    if cmap is None:
        import seaborn as sns
        cmap = sns.color_palette("Spectral_r", as_cmap=True)
    
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
        #Path(basedir / 'thumbnails').mkdir(parents=True, exist_ok=True) # useful for geoplatform to keep them in same dir as raster
        plt.savefig(str(savepng), dpi=160, bbox_inches='tight')

    return fig, ax


def plot_scatter(ds, t=None, xlim=None, ylim=None, s=1, alpha=1, c='age', cmap='rainbow', coastres='10m', proj=cartopy.crs.PlateCarree(), transform=None, dpi=120, figsize=[10,7], rivers=False, save_to=None):
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
