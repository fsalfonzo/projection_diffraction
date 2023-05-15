
import sys
sys.path.insert(1, '/home/gh464/Documents/GitHub/SyMBac//') # Not needed if you installed SyMBac using pip
sys.path.insert(1, '/home/georgeos/Documents/GitHub/SyMBac//') # Not needed if you installed SyMBac using pip

import numpy as np
from glob import glob
from PIL import Image
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
from numba import njit, jit
from natsort import natsorted
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import pandas as pd
import seaborn as sns
from skimage.morphology import convex_hull_image
from skimage.measure import regionprops, label
from skimage.transform import rescale, resize
from skimage import future
from skimage.morphology import dilation
from skimage import graph
import sys
import skimage
sys.path.insert(0, "../") 
import tifffile
from SyMBac.PSF import PSF_generator
from SyMBac.renderer import convolve_rescale
from skimage.util import random_noise
from scipy.ndimage import gaussian_filter

from skimage.exposure import rescale_intensity
from skimage import color, data, restoration
import sys
sys.path.insert(0, "../") 

import numpy as np
from glob import glob
from PIL import Image
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
from numba import njit, jit
from natsort import natsorted
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import pandas as pd
import seaborn as sns
from skimage.morphology import convex_hull_image, remove_small_objects
from skimage.measure import regionprops, label, perimeter
import tifffile

@njit()
def get_intensities(img, mask, img_2 = None):
    mask_labels = np.unique(mask)[1:]
    total_intensities = np.zeros(len(mask_labels))
    for l, mask_label in enumerate(mask_labels):
        idxs = np.where(mask == mask_label)
        if len(idxs[0]) > 100:
            pixel_intensities = np.zeros(len(idxs[0]))
            for k, (i, j) in enumerate(zip(idxs[0], idxs[1])):
                pixel_intensities[k] = img[i,j]
            total_intensity = np.sum(pixel_intensities)
            total_intensities[l] = total_intensity
    return total_intensities

#@njit
#def get_CV(arr):
#    return arr.std()/arr.mean()

def load_img_mask_pair(img_dir, mask_dir, normalise = False):
    suffix = img_dir.split(".")[-1]
    if "png" in suffix:
        img = np.array(Image.open(img_dir))
    elif "tif" in suffix:
        img = tifffile.imread(img_dir).sum(axis=0)
    if normalise:
        img = rescale_intensity(img, out_range=(0,1))
    mask = np.array(Image.open(mask_dir)) #Masks are always PNG
    return img, mask

#def get_CV_from_img_mask(img, mask):
#    mean_intensities, total_intensities = get_intensities(img,mask)
#    return get_CV(mean_intensities), get_CV(total_intensities)

def nearest_nonzero_idx(a,x,y):
    idx = np.argwhere(a)

    # If (x,y) itself is also non-zero, we want to avoid those, so delete that
    # But, if we are sure that (x,y) won't be non-zero, skip the next step
    idx = idx[~(idx == [x,y]).all(1)]

    return idx[((idx - [x,y])**2).sum(1).argmin()]


    
def get_central_cell_intensity(img, mask):
    colony_centroid_y, colony_centroid_x = get_colony_centroid(mask)
    cell_y, cell_x = nearest_nonzero_idx(mask, colony_centroid_y, colony_centroid_x)
    central_cell_label = mask[cell_y, cell_x]

    intensity = img[np.where(mask==central_cell_label)]
    return np.mean(intensity)

def perc_diff(obs, act):
    if (act == 0) and (obs == 0):
        return 0
    elif act == 0:
        return np.nan
    return (obs-act)/act * 100

def clean_up_mask(mask):
    return remove_small_objects(label(mask))

def get_circularity(mask):
    chull = convex_hull_image(mask > 0)
    colony_perimeter = perimeter(chull)
    colony_area = np.sum(chull)
    colony_circularity = colony_perimeter**2 / (4*np.pi * colony_area)
    return colony_circularity


def calculate_params(img_dir, mask_dir, return_colony_props = True):
    img, mask = load_img_mask_pair(img_dir, mask_dir)
    CV_mean_intensity, CV_total_intensity = get_CV_from_img_mask(img, mask)
    n_cells = len(np.unique(mask))-1
    
    if return_colony_props:
        colony_circularity = get_circularity(mask)
        colony_solidity = regionprops((mask>0)*1)[0].solidity
        return CV_mean_intensity, CV_total_intensity, n_cells, img_dir, colony_circularity, colony_solidity
    
    return CV_mean_intensity, CV_total_intensity, n_cells, img_dir

def deconv_calculate_params(img_dir, mask_dir, psf, return_colony_props = False):
    img, mask = load_img_mask_pair(img_dir, mask_dir)
    img = np.pad(img,100)
    img = restoration.wiener(img, psf, 0.00001, clip=False)
    CV_mean_intensity, CV_total_intensity = get_CV_from_img_mask(img, mask)
    n_cells = len(np.unique(mask))-1
    
    if return_colony_props:
        colony_circularity = get_circularity(mask)
        colony_solidity = regionprops((mask>0)*1)[0].solidity
        return CV_mean_intensity, CV_total_intensity, n_cells, img_dir, colony_circularity, colony_solidity
    
    return CV_mean_intensity, CV_total_intensity, n_cells, img_dir

def per_cell_intensity_error(img_dir, mask_dir, synth_img_dir, synth_mask_dir, resize_amount=1, normalise=False):
    
    img, mask = load_img_mask_pair(img_dir, mask_dir, normalise)
    #img = img/img.max()
    #img, mask = img[:,1:], mask[:,1:]

    synth_img, synth_mask = load_img_mask_pair(synth_img_dir, synth_mask_dir, normalise)
    synth_img = synth_img/synth_img.max()
    if resize_amount != 1:
        rescaled_synth_img = resize(synth_img, img.shape, anti_aliasing=False, order=0, preserve_range=True) #rescale(synth_img, resize_amount, anti_aliasing=False, order=0, preserve_range=True)[1:,:]
    else:
        rescaled_synth_img = synth_img
        
    img_regionprops = regionprops(mask, intensity_image=img)
    synth_regionprops = regionprops(mask, intensity_image=rescaled_synth_img)
    
    cell_mean_intensity_errors = []
    cell_mean_intensity_mags = []
    cell_total_intensity_errors = []
    cell_total_intensity_mags = []
    for img_regionprop, synth_regionprop in (zip(img_regionprops, synth_regionprops)):
        
        actual_mean_intensity = img_regionprop.mean_intensity
        observed_mean_intensity = synth_regionprop.mean_intensity
        cell_mean_intensity_errors.append(perc_diff(observed_mean_intensity, actual_mean_intensity))
        cell_mean_intensity_mags.append(observed_mean_intensity/actual_mean_intensity)
        
        actual_total_intensity = np.sum(img_regionprop.intensity_image)
        observed_total_intensity = np.sum(synth_regionprop.intensity_image)
        cell_total_intensity_errors.append(perc_diff(observed_total_intensity, actual_total_intensity))
        cell_total_intensity_mags.append(observed_total_intensity/actual_total_intensity)

    
    mean_cell_mean_intensity_errors = np.nanmean(cell_mean_intensity_errors)
    mean_cell_mean_intensity_mags = np.nanmean(cell_mean_intensity_mags)
    
    mean_cell_total_intensity_errors = np.nanmean(cell_total_intensity_errors)
    mean_cell_total_intensity_mags = np.nanmean(cell_total_intensity_mags)
    

    
    return mean_cell_mean_intensity_errors, mean_cell_mean_intensity_mags, mean_cell_total_intensity_errors, mean_cell_total_intensity_mags

def all_cell_intensity_error(img_dir, mask_dir, synth_img_dir, synth_mask_dir, resize_amount=1, normalise=False):
    
    img, mask = load_img_mask_pair(img_dir, mask_dir, normalise)
    #img /= img.max()
    #img, mask = img[:,1:], mask[:,1:]

    synth_img, synth_mask = load_img_mask_pair(synth_img_dir, synth_mask_dir, normalise)
    #synth_img /= synth_img.max()
    if resize_amount != 1:
        rescaled_synth_img = resize(synth_img, img.shape, anti_aliasing=False, order=0, preserve_range=True) #rescale(synth_img, resize_amount, anti_aliasing=False, order=0, preserve_range=True)[1:,:]
    else:
        rescaled_synth_img = synth_img
    
    img_regionprops = regionprops(mask, intensity_image=img)
    synth_regionprops = regionprops(mask, intensity_image=rescaled_synth_img)
    
    actual_mean_intensities = []
    observed_mean_intensities = []

    actual_total_intensities = []
    observed_total_intensities = []
    
    dists_from_colony_centre = []
    colony_centroid_y, colony_centroid_x, feret_diameter_max = get_colony_centroid(mask, return_size = True)
    
    radius = 5
    dilated = dilation(mask, skimage.morphology.disk(radius))
    rag = graph.RAG(dilated)
    rag.remove_node(0)

    n_neighbours = []
    
    for img_regionprop, synth_regionprop in (zip(img_regionprops, synth_regionprops)):
        
        actual_mean_intensities.append(img_regionprop.mean_intensity)
        observed_mean_intensities.append(synth_regionprop.mean_intensity)
        
        actual_total_intensities.append(np.sum(img_regionprop.intensity_image))
        observed_total_intensities.append(np.sum(synth_regionprop.intensity_image))
        
        cell_y, cell_x = img_regionprop.centroid
        dist_from_colony_centre = np.sqrt( (cell_y - colony_centroid_y)**2 + (cell_x - colony_centroid_x)**2 )
        dists_from_colony_centre.append(dist_from_colony_centre)
        
        i = img_regionprop.label
        n_neighbours.append(len(list(rag.neighbors(i))))
        
    return actual_mean_intensities, observed_mean_intensities, actual_total_intensities, observed_total_intensities, dists_from_colony_centre, [img_dir]*len(dists_from_colony_centre), [feret_diameter_max]*len(dists_from_colony_centre), n_neighbours

def flatten(l):
    return [item for sublist in l for item in sublist]

