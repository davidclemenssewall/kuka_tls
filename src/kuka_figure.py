#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kuka_figure.py

Script for creating radial figure showing surface change around kuka radar.

Created on Fri Aug 27 11:40:58 2021

@author: David Clemens-Sewall
"""

import os
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# Data Parameters
data_path = os.path.join('..', 'data')
project_names = [
                 'mosaic_01_110119.RiSCAN',
                 'mosaic_01_110819.RiSCAN',
                 'mosaic_01_111519.RiSCAN',
                 ]

# %% Load data

data_dict = {}
for project_name in project_names:
    data_dict[project_name] = np.loadtxt(os.path.join(data_path, 
                                                      project_name))

# %% Useful functions for binning and plotting

# Note, the cython version of this function is much faster, fortunately these
# pointclouds are small...
def create_counts_means_M2_cy(nbin_0, nbin_1, Points, xy):
    """
    Return the binwise number of points, mean, and M2 estimate of the sum of
    squared deviations (using Welford's algorithm)
                        
    from: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    
    Parameters
    ----------
    nbin_0 : long
        Number of bins along the zeroth axis
    nbin_1 : long
        Number of bins along the first axis
    Points : float[:, :]
        Pointcloud, Nx3 array of type np.float32
    xy : long[:]
        Bin index for each point, must be same as numbe rof points.

    Returns:
    --------
    counts : float[:]
        Array with the counts for each bin. Length nbin_0*nbin_1
    means : float[:]
        Array with the mean of z values for each bin. Length nbin_0*nbin_1
    m2s : float[:]
        Array with the M2 estimate of the sum of squared deviations
        Length nbin_0*nbin_1
    """

    # Chose to make this float for division purposes but could have unexpected
    # results, check if so
    counts = np.zeros(nbin_0 * nbin_1, dtype=np.float32)
    
    means = np.zeros(nbin_0 * nbin_1, dtype=np.float32)

    m2s = np.zeros(nbin_0 * nbin_1, dtype=np.float32)

    for i in range(len(xy)):
        counts[xy[i]] += 1
        delta = Points[i, 2] - means[xy[i]]
        means[xy[i]] += delta / counts[xy[i]]
        delta2 = Points[i, 2] - means[xy[i]]
        m2s += delta*delta2
    
    return counts, means, m2s

def gridded_counts_means_vars(points, edges):
    """
    Grids a point could in x and y and returns the cellwise counts, means and
    variances.

    Parameters
    ----------
    points : float[:, :]
        Pointcloud, Nx3 array of type np.float32
    edges : list
        2 item list containing edges for gridding

    Returns:
    --------
    counts : long[:, :]
        Gridded array with the counts for each bin
    means : float[:, :]
        Gridded array with the mean z value for each bin.
    vars : float[:, :]
        Gridded array with the variance in z values for each bin.

    """

    Ncount = tuple(np.searchsorted(edges[i], points[:,i], 
                               side='right') for i in range(2))

    nbin = np.empty(2, np.int_)
    nbin[0] = len(edges[0]) + 1
    nbin[1] = len(edges[1]) + 1

    xy = np.ravel_multi_index(Ncount, nbin)

    # Compute gridded mins and counts
    counts, means, m2s = create_counts_means_M2_cy(nbin[0], nbin[1],
                                                     points,
                                                     np.int_(xy))
    counts = counts.reshape(nbin)
    means = means.reshape(nbin)
    m2s = m2s.reshape(nbin)
    core = 2*(slice(1, -1),)
    counts = counts[core]
    means = means[core]
    m2s = m2s[core]
    means[counts==0] = np.nan
    m2s[counts==0] = np.nan
    var = m2s/counts

    return counts, means, var

# %% Bin on polar grid

# Set z_max so that radar itself is excluded
z_max = -1.2

# Adjust r and theta spacing here
r_edges = np.linspace(0.25, 5.0, num=20)
t_edges = np.linspace(-np.pi, np.pi, num=37)

pol_mean = {}
pol_cts = {}
pol_var = {}

for project_name in data_dict:
    # Get the points
    pts = data_dict[project_name]
    pol_pts = np.vstack((np.sqrt(np.square(pts[:,:2]).sum(axis=1)),
                         np.arctan2(pts[:,1], pts[:,0]),
                         pts[:,2])).T
    pol_pts = pol_pts[pol_pts[:,2]<=z_max]
    (pol_cts[project_name], pol_mean[project_name], 
     pol_var[project_name]) = gridded_counts_means_vars(
         pol_pts, (r_edges, t_edges))
         
# %% Plot

hmin = -1.6
hmax = -1.3

cmap_div = copy.copy(cm.get_cmap('RdBu_r'))
cmap_div.set_bad(color='black')
cmap_seq = copy.copy(cm.get_cmap('rainbow'))
cmap_seq.set_bad(color='black')

f, axs = plt.subplots(1, 3, figsize=(15,5), 
                      subplot_kw=dict(projection='polar'))

h = axs[0].pcolormesh(t_edges, r_edges, 
              pol_mean['mosaic_01_110119.RiSCAN'],
              cmap=cmap_seq, vmin=hmin, vmax=hmax)

f.colorbar(h, ax=axs[0], shrink=0.8, label='Height (m)')
axs[0].set_xlim([-np.pi/3, np.pi/3])
axs[0].set_title('Topography Nov. 1')
label_position=-60
axs[0].text(np.radians(label_position-30),axs[0].get_rmax()/2.,'Distance (m)',
        rotation=label_position,ha='center',va='center')
h = axs[1].pcolormesh(t_edges, r_edges, 
              pol_mean['mosaic_01_111519.RiSCAN'],
              cmap=cmap_seq, vmin=hmin, vmax=hmax)

f.colorbar(h, ax=axs[1], shrink=0.8, label='Height (m)')
axs[1].set_xlim([-np.pi/3, np.pi/3])
axs[1].set_title('Topography Nov. 15')
axs[1].text(np.radians(label_position-30),axs[1].get_rmax()/2.,'Distance (m)',
        rotation=label_position,ha='center',va='center')

h = axs[2].pcolormesh(t_edges, r_edges, 
              pol_mean['mosaic_01_111519.RiSCAN']
              -pol_mean['mosaic_01_110119.RiSCAN'],
              vmin=-.2, vmax=.2, cmap=cmap_div)

f.colorbar(h, ax=axs[2], shrink=0.8, label='Change (m)')
axs[2].set_xlim([-np.pi/3, np.pi/3])
axs[2].set_title('Difference')
axs[2].text(np.radians(label_position-30),axs[2].get_rmax()/2.,'Distance (m)',
        rotation=label_position,ha='center',va='center')

f.savefig(os.path.join('..', 'figures', 'kuka_figure.png'), transparent=True)
