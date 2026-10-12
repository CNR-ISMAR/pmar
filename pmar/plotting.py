# fix these by using plot_map
# add plot_origin, plot_destination, plot_network, plot_cm
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

def plot_map(subplot_rows=1, subplot_cols=1, sharex=False, sharey=False, proj=cartopy.crs.PlateCarree(), extent=None, coastres='10m', figsize=[8,6], title='', filename=None, top_labels=False, right_labels=False, left_labels=True, bottom_labels=True, dpi=160):
    """
    Create a base map (or grid of subplots) with cartopy coastlines, land, borders and
    gridlines, ready to plot data onto.

    Parameters
    ----------
    subplot_rows : int, optional
        Number of subplot rows. Default is 1.
    subplot_cols : int, optional
        Number of subplot columns. Default is 1.
    sharex, sharey : bool, optional
        Whether subplots share the x/y axis. Default is False.
    proj : cartopy.crs.Projection, optional
        Map projection used for all subplots. Default is cartopy.crs.PlateCarree().
    extent : array-like, optional
        Map extent as [x1, x2, y1, y2], in proj's coordinates. Default is None, i.e. the
        full projection extent.
    coastres : str, optional
        Named resolution to use from the Natural Earth coastline dataset ('110m', '50m' or
        '10m'). Default is '10m'.
    figsize : list, optional
        Figure size [width, height] in inches. Default is [8, 6].
    title : str, optional
        Title applied to every subplot's axes. Default is ''.
    filename : str or Path, optional
        If given, save the figure as a PNG at this path. Default is None, i.e. the figure
        is not saved.
    top_labels, right_labels, left_labels, bottom_labels : bool, optional
        Whether to draw gridline tick labels on each side of the subplots. Defaults are
        False, False, True and True respectively.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : matplotlib.axes.Axes or array of Axes
        The created subplot axes, each already set up with coastlines, land, borders and
        gridlines.
    """
    fig, axs = plt.subplots(subplot_rows, subplot_cols, sharex=sharex, sharey=sharey, figsize=figsize, subplot_kw={'projection':proj})
    for ax in fig.axes:
        ax.coastlines(coastres, zorder=12, color='k', linewidth=.7)
        ax.add_feature(cartopy.feature.LAND, facecolor='0.9', zorder=2, alpha=0.5)
        ax.add_feature(cartopy.feature.BORDERS, zorder=11, linewidth=.5, linestyle=':')
        if extent is not None:
            ax.set_extent(extent, crs=proj) 
        
        gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, linewidth=0, color='gray', linestyle='--')
        gl.top_labels = top_labels
        gl.right_labels = right_labels
        gl.left_labels = left_labels
        gl.bottom_labels = bottom_labels
        
        ax.set_title(title)

    if filename is not None:
        plt.savefig(str(filename), dpi=dpi, bbox_inches='tight')
    
    return fig, axs

def plot_matrix(array, tick_labels=None, figsize=[8,6], annot=False, linewidths=0, vmin=0, vmax=None, norm=None, cmap=tol_inc, ax=None, mask=None):
    """
    Plot a 2D array as a heatmap, with rows labelled 'FROM' and columns labelled 'TO'.

    Parameters
    ----------
    array : array-like
        2D matrix to plot.
    tick_labels : array-like, optional
        Labels applied to both axes (rows and columns share the same labels, e.g. cell ids
        from cm_grid). Default is None, i.e. default numeric tick labels are kept.
    figsize : list, optional
        Currently unused by this function; ax must be supplied. Default is [8, 6].
    annot : bool, optional
        Whether to annotate each cell with its value. Automatically disabled if array has
        more than 30 rows, to avoid overcrowding. Default is False.
    linewidths : float, optional
        Width of the lines separating cells. Default is 0.
    vmin, vmax : float, optional
        Values to anchor the colormap. Default is vmin=0, vmax=None (inferred from data).
    norm : matplotlib.colors.Normalize, optional
        Normalization for the colormap. Default is None.
    cmap : matplotlib colormap, optional
        Colormap for the heatmap. Default is tol_inc.
    ax : matplotlib.axes.Axes
        Axes to plot onto. Required in practice: the axes is used directly after the
        heatmap call, so a value must be passed despite the None default.
    mask : array-like, optional
        Boolean mask; cells where True are left blank. Default is None.
    """
    # use cm_grid for tick_labels

    if len(array) > 30:
        annot = False
        
    sns.heatmap(array, annot=annot, linewidths=linewidths, square=True, ax=ax, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, mask=mask)
    ax.set_ylabel('FROM')
    ax.set_xlabel('TO')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 

    if tick_labels is not None:
        ax.set_yticks(ticks=np.arange(0.5,len(tick_labels)), labels=tick_labels, rotation=0);
        ax.set_xticks(ticks=np.arange(0.5,len(tick_labels)), labels=tick_labels, rotation=90);


def _plot_network(G, node_shapes=None, node_size=5, show_map=False, node_color='tab:blue', edge_color='p', width=2, rescale_edge=None, figsize=[7,7], extent=None, cmap=tol_inc, show_clusters=False, node_vmin=None, node_vmax=None, edge_vmin=None, edge_vmax=None):
    """
    Draw a networkx graph (e.g. a PMAR connectivity network) on an optional basemap.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Graph to plot (e.g. from PMAR.get_network()).
    node_shapes : geopandas.GeoDataFrame, optional
        Polygon grid whose centroids give node positions (and whose boundaries are drawn as
        context). Default is None, i.e. a spring layout is used instead.
    node_size : float, optional
        Size of drawn nodes. Default is 5.
    show_map : bool, optional
        Whether to draw a basemap behind the network via plot_map(). Default is False.
    node_color : str or array-like, optional
        Node color, or values to color nodes by. Default is 'tab:blue'.
    edge_color : str or array-like, optional
        Edge color; if 'p', edges are colored by their 'p' (transition probability)
        attribute. Default is 'p'.
    width : float, optional
        Base edge width, overridden by `rescale_edge` if given. Default is 2.
    rescale_edge : float, optional
        If given, edge widths are set to edge weight * 50 * rescale_edge instead of a
        constant `width`. Default is None.
    figsize : list, optional
        Figure size [width, height]. Default is [7, 7].
    extent : list, optional
        Map extent, passed to plot_map() when show_map is True. Default is None.
    cmap : matplotlib colormap, optional
        Colormap for edges/nodes. Default is tol_inc.
    show_clusters : bool, optional
        If True, color nodes by their strongly-connected-component id instead of
        `node_color`. Default is False.
    node_vmin, node_vmax, edge_vmin, edge_vmax : float, optional
        Color scale bounds for nodes/edges. Default is None.
    """
    if show_map == True:
        fig, ax = plot_map(figsize=figsize, extent=extent)
    else:
        fig, ax = plt.subplots(figsize=figsize)

    if node_shapes is not None:
    
        coordinates = np.column_stack((node_shapes.geometry.centroid.x, node_shapes.geometry.centroid.y))

        positions = dict(zip(G.nodes, coordinates))
        node_shapes.boundary.plot(ax=ax, color='grey', linewidth=.5)
    else:
        positions = nx.spring_layout(G)

    if nx.get_node_attributes(G, 'node_name'):
        labels = nx.get_node_attributes(G, 'node_name')
    else:
        labels = None

    if rescale_edge is not None:
        width = np.array(list(nx.get_edge_attributes(G,'p').values()))*50*rescale_edge
    else:
        width = width

    if show_clusters == True:
        clusters = list(nx.strongly_connected_components(G))
        cluster_id = {}
        for id, comp in enumerate(clusters):
            for node in comp:
                cluster_id[node] = id
        
        # add as node attribute
        nx.set_node_attributes(G, cluster_id, name="cluster")
        node_color = nx.get_node_attributes(G,'cluster').values()

    if edge_color == 'p':
        _edge_color = nx.get_edge_attributes(G, 'p').values()
    else:
        _edge_color = edge_color

    if nx.get_node_attributes(G, 'node_name'):
        nx.draw_networkx_labels(G, 
                                positions, 
                                labels=nx.get_node_attributes(G, 'node_name'),
                                font_size=8)#, font_weight="bold")

    nodes = nx.draw_networkx_nodes(G, 
                                   ax=ax,
                                   pos=positions, 
                                   node_size=node_size, 
                                   node_color=node_color,
                                   linewidths=.5,
                                   edgecolors='k',
                                   cmap=cmap, 
                                   vmin=node_vmin, 
                                   vmax=node_vmax,
                                   label=labels
                                   )

            
    edges = nx.draw_networkx_edges(G,
                                   ax=ax,
                                   pos=positions,
                                   arrowstyle="->",
                                   connectionstyle='arc3, rad = 0.1',
                                   edge_color=_edge_color,
                                   edge_cmap=cmap,
                                   width=width,
                                   edge_vmin=edge_vmin, 
                                   edge_vmax=edge_vmax,
                                ) 



    if edge_color == 'p':
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        if edge_vmin is None:
            edge_vmin = min(_edge_color)
        if edge_vmax is None:
            edge_vmax = max(_edge_color)
        patches = ScalarMappable(norm= Normalize(vmin=edge_vmin,vmax=edge_vmax), cmap=cmap)
        patches.set_array([])

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        # this should add a colorbar without shrinking the figure axes, so that figsize is conserved
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        
        fig.colorbar(patches, cax=cax)
    
    
    elif type(node_color) != str:
        fig.colorbar(nodes, ax=ax)
    elif show_clusters == True:
        fig.colorbar(clusters, ax=ax)

    


    return fig, ax 

