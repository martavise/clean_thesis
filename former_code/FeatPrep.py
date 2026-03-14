# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 14:10:30 2025

Feature Preparation Pipeline

This pipeline is as follows:
1. Read timeseries list from file;
2. Calculate connectomic features (functional connectivity vectors, fAlff values) 
   on both ROI- and network-level;
3. Convert FC matrices into distance matrices and then calculate topological
   descriptors (subject-level)

Things inside hash:
1. Persistent diagram normalisation on subject level;
2. (Group level) Topological descriptors calculation using gtda package 

@author: Ruan
"""

import pandas as pd
import numpy as np
import os
import pickle
from gtda.homology import VietorisRipsPersistence
from gtda import diagrams
from nilearn import connectome
from scipy.fft import fft, fftfreq

# %%

# Essential funtions
def fALFF(timeseries, TR, low_freq=(0.01, 0.1), fractional=True):
    """
    timeseries : numpy.ndarray
        Array of timeseries, with shape of (n_volumes, n_rois).
    TR : float
        Repition time (seconds) of the timeseries.
    low_freq : tuple of float, optional
        Range of low frequencies. Default=(0.01, 0.1).
    fractional : Boolean, optional
        If True, calculate and return the fractional value of ALFF. Default=True.
    
    Returns
    -------
    (f)alff : float
        (fractional) Amplitude of low frequency fluctuations of the given timeseries.
    """
    n_volumes, n_rois = timeseries.shape
    Freqs = fftfreq(n_volumes, d=TR)
    FFT = fft(timeseries, axis=0)
    PowerSpec = np.abs(FFT) ** 2
    FreqBand = (Freqs >= low_freq[0]) & (Freqs <= low_freq[1])
    alff = np.sqrt(np.sum(PowerSpec[FreqBand, :], axis=0))
    if fractional is True:
        total_power = np.sqrt(np.sum(PowerSpec, axis=0))
        falff = alff / total_power
        return falff
    else:
        return alff
    
def betti_curve(diagram, num_samples=100):
    """
    diagram: numpy.ndarray of shape (N, 3)
        The persistent diagram data, where each row is [birth, death, dim].
    num_samples: int, optional
        Number of points to sample along the scale axis (default=100).
    
    Returns:
    -------
    betti_curves: numpy.ndarray
        the Betti number at each t-value
    """
    betti_list = []
    for dim in np.unique(diagram[:, 2]):
        dim_mask = (diagram[:, 2] == dim)
        diag_dim = diagram[dim_mask]
        
        births = diag_dim[:, 0]
        deaths = diag_dim[:, 1]
        
        t_min = births.min()
        t_max = deaths.max()
        ts = np.linspace(t_min, t_max, num_samples)
        
        betti = np.array([np.sum((births <= t) & (deaths > t)) for t in ts])
        betti_list.append(betti)
    betti_curves = np.vstack(betti_list)
    return betti_curves

def persistent_landscape(diagram, num_samples=100):
    """
    diagram : numpy.ndarray of shape (N, 3)
        The persistent diagram data, where each row is [birth, death, dim].
    num_samples : int, optional
        Number of points to sample along the scale axis (default=100).
    
    Returns:
        -------
    landscape : numpy.ndarray
        The first-layer landscape values at each t in `ts` (shape = (num_samples,)).
    """
    
    for dim in np.unique(diagram[:, 2]):
        diag_dim = diagram[diagram[:, 2] == dim]
        
        births = diag_dim[:, 0]
        deaths = diag_dim[:, 1]
        
        t_min = births.min()
        t_max = deaths.max()
        ts = np.linspace(t_min, t_max, num_samples)
        
        landscape = []
        for t in ts:
            max_tent_value = 0.0
            for (b, d, _) in diag_dim:
                if b <= t <= d:
                    mid = 0.5 * (b + d)
                    if t <= mid:
                        tent_val = t - b
                    else:
                        tent_val = d - t
                    if tent_val > max_tent_value:
                        max_tent_value = tent_val
            landscape.append(max_tent_value)
        landscape = np.array(landscape).T
    landscapes = np.vstack(landscape)
    return landscapes

def persistent_silhouette(diagram, num_samples=100, power=1):
    """
    diagram : numpy.ndarray of shape (N, 3)
        Array of persistence intervals, each given by [birth, death, dimension].
    num_samples : int, optional
        Number of points to sample along the filtration axis per dimension.
    power : Real, optional
        Exponent to use in weighting the intervals (default 1). The weight for an interval
        is computed as (d - b)^power.
    
    Returns
    -------
    silhouettes : numpy.ndarray of shape (n_dimensions, num_samples)
        Each row contains the silhouette values (weighted average of tent functions)
        for one homology dimension.
    """
    for dim in np.unique(diagram[:, 2]):
        ts_list = []
        silhouettes_list = []
        
        dim_mask = (diagram[:, 2] == dim)
        diag_dim = diagram[dim_mask]
        
        births = diag_dim[:, 0]
        deaths = diag_dim[:, 1]
        
        t_min = births.min()
        t_max = deaths.max()
        ts = np.linspace(t_min, t_max, num_samples)
            
        silhouette = np.zeros_like(ts)
        total_weight = 0.0
            
        for (b, de, _) in diag_dim:
            persistence = de - b
            weight = persistence ** power
            total_weight += weight
            mid = 0.5 * (b + de)
            
            tent = np.zeros_like(ts)
            in_interval = (ts >= b) & (ts <= de)
            left_mask = in_interval & (ts <= mid)
            tent[left_mask] = ts[left_mask] - b
            right_mask = in_interval & (ts > mid)
            tent[right_mask] = de - ts[right_mask]
            silhouette += weight * tent
        
        if total_weight > 0:
            silhouette /= total_weight
        
        ts_list.append(ts)
        silhouettes_list.append(silhouette)
        
    silhouettes = np.vstack(silhouettes_list)
    return silhouettes

def persistent_image(diagram, resolution=(100, 100), sigma=1.0, weight_power=1):
    """
    diagram : numpy.ndarray of shape (N, 3)
        Array of persistence intervals, each given by [birth, death, dimension].
    resolution : tuple of int, optional
        The number of pixels (n_birth, n_persistence) in the output image grid.
    sigma : float, optional
        Standard deviation for the Gaussian kernel.
    weight_power : Real, optional
        Exponent for the weight function. The weight for an interval is computed as (d - b)^weight_power.
    
    Returns
    -------
    images : numpy.ndarray of shape (n_dimensions, resolution[1], resolution[0])
        A stack of persistent images, one per homology dimension.
        (The first axis indexes the homology dimension.)
    """
    images_list = []
    
    for dim in np.unique(diagram[:, 2]):
        dim_mask = (diagram[:, 2] == dim)
        diag_dim = diagram[dim_mask]
        
        births = diag_dim[:, 0]
        deaths = diag_dim[:, 1]
        persistences = deaths - births
        
        x_min, x_max = births.min(), births.max()
        y_min, y_max = 0, persistences.max()
        x_vals = np.linspace(x_min, x_max, resolution[0])
        y_vals = np.linspace(y_min, y_max, resolution[1])
        X, Y = np.meshgrid(x_vals, y_vals)
        
        img = np.zeros_like(X)
        for (b, de, _) in diag_dim:
            p_val = de - b
            weight = p_val ** weight_power
            sq_dist = (X - b) ** 2 + (Y - p_val) ** 2
            bump = weight * np.exp(-sq_dist / (2 * sigma ** 2))
            img += bump
        images_list.append(img)
    images = np.stack(images_list, axis=0)
    return images

# %%

MainDir = "/home/marta/Downloads"
# Load the time series list
with open('/home/marta/Downloads/TSCom.pkl', 'rb') as f:
    TS_Org = pickle.load(f)

# Exclude 7 subjects for mismatching volumes of subcortical and cortical ROIs 
Excluded = [1556, 1557, 1582, 1583, 1584, 1587, 1589]

DemTable = pd.read_excel('/home/marta/Documents/Bachelor-Thesis/DemTable.xlsx')
ExcludedInd = DemTable.index[DemTable.iloc[:, 0].isin(Excluded)].tolist()
TS = [arr for idx, arr in enumerate(TS_Org) if idx not in ExcludedInd]
DemTable_filtered = DemTable[~DemTable.iloc[:, 0].isin(Excluded)].reset_index(drop=True)

ROIInd = pd.read_csv("/home/marta/Documents/Bachelor-Thesis/included_regions_Schaefer.csv")

# %%

# FC metrics
Conn = connectome.ConnectivityMeasure(kind='correlation')
FCMatrices = Conn.fit_transform(TS)

ConnVec = connectome.ConnectivityMeasure(kind='correlation', vectorize=True)
FCVectors = ConnVec.fit_transform(TS)

#  fAlff
fALFF_All = []

for i in range(len(TS)):
    TS_Sub = TS[i]
    fAlff_Sub = fALFF(TS_Sub, DemTable_filtered['TR'].iloc[i])
    fALFF_All.append(fAlff_Sub)
fALFF_All = np.array(fALFF_All)

# Network-level FC
Networks = {'VisCent', 'VisPeri', 'SomMotA', 'SomMotB', 'DorsAttnA', 'DorsAttnB', 'SalVentAttnA',
           'SalVentAttnB', 'LimbicA', 'ContA', 'ContB', 'ContC', 'DefaultA', 'DefaultB', 'DefaultC',
           'TempPar', 'Sub-Cortical', 'Cerebellum'}
TS_Net = []
fALFF_Net  =[]
for i in range(len(TS)):
    Net_Sub = []
    TS_Sub = TS[i]
    for Network in Networks:
        idx = np.where(ROIInd['Net'] == Network)[0]
        Net = TS_Sub[:, idx]
        NetTS = np.mean(Net, axis=1)
        Net_Sub.append(NetTS)
    Net_Sub = np.array(Net_Sub).T
    TS_Net.append(Net_Sub)
    fALFF_Net_Sub = fALFF(Net_Sub, DemTable_filtered['TR'].iloc[i])
    fALFF_Net.append(fALFF_Net_Sub)
    
FCVectors_Net = ConnVec.fit_transform(TS_Net)
fALFF_Net = np.array(fALFF_Net)

Output = os.path.join(MainDir, 'FCMetrics')
np.savez(Output, FCVectors = FCVectors, fALFF = fALFF_All, FCVectors_Net = FCVectors_Net, fALFF_Net = fALFF_Net)
print("FC Metrics saved.")

# %%

# Topological descriptors

# Tranform FC matrices into distance matrices
DisMatrices = np.empty_like(FCMatrices) # Empty distance matrices
for i in range(FCMatrices.shape[0]):
    FC_Sub = FCMatrices[i, :, :]
    
    # Here use 10% for thresholding FC matrix (Can be changed)
    threshold = np.percentile(FCVectors[i, :], 90) 
    FCFiltered = np.where(FC_Sub >= threshold, FC_Sub, 0)
    
    # Distance matrix conversion (Can be changed)
    Dis_Sub = np.square(1 - FCFiltered**2)
    DisMatrices[i, :, :] = Dis_Sub

VR = VietorisRipsPersistence(metric='precomputed', homology_dimensions=[0,1])
PD = VR.fit_transform(DisMatrices)

# Apply a filtration to persistent diagram, excluding pointa whose d-b<0.01
VR_filt = diagrams.Filtering()
PD_filt = VR_filt.fit_transform(PD)

# # Normalise persistent diagrams (subject level)
# PD_Norm = []
# for sub in range(PD_filt.shape[0]):
#     PD_Sub = PD_filt[sub, :, :]
#     dims = np.unique(PD_sub[:, 2])
#     PD_Sub_Norm = np.zeros_like(PD_Sub, dtype=float)
#     PD_Sub_Norm[:, 2] = PD_Sub[:, 2]
#     for dim in dims:
#         mask = PD_Sub[:,2] == dim
#         BDP = PD_Sub[mask, :2]
#         BDP_min = BDP.min()
#         BDP_max = BDP.max()
#         BDP_scaled = (BDP - BDP_min) / (BDP_max - BDP_min)
#         PD_Sub_Norm[mask, :2] = BDP_scaled
#     PD_Norm.append(PD_Sub_Norm)
# PD_Norm = np.array(PD_Norm)  

BC = []
PL = []
PS = []
PI = []
for i in range(np.size(PD_filt, axis=0)):
    diagram = PD_filt[i, :, :]
    betti = betti_curve(diagram)
    BC.append(betti)
    landscapes = persistent_landscape(diagram)
    PL.append(landscapes)
    silhouettes = persistent_silhouette(diagram)
    PS.append(silhouettes)
    images = persistent_image(diagram)
    PI.append(images)
BC = np.array(BC)
PL = np.array(PL)
PS = np.array(PS)
PI = np.array(PI)

# Save all descriptors
Output = os.path.join(MainDir, 'TDADes_400_precomputed_Sparse')
np.savez(Output, PD=PD_filt, PL=PL, PI=PI, PS=PS, BC=BC)

# %%

# Calculation of descriptors by the gtda package (group level)
# VR_PL = diagrams.PersistenceLandscape(n_layers=1)
# PL_norm = VR_PL.fit_transform(PD_norm)

# VR_PI = diagrams.PersistenceImage()
# PI_norm = VR_PI.fit_transform(PD_norm).reshape(PD.shape[0],2,10000)

# VR_PS = diagrams.Silhouette()
# PS_norm = VR_PS.fit_transform(PD_norm)

# VR_BC = diagrams.BettiCurve()
# BC_norm = VR_BC.fit_transform(PD_norm)



