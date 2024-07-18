#!/usr/bin/env python
""" Radial profile calculation for DRG images.
    Process all images in a folder. Configuration of the process is based
    on a config file, containing various parameters.
    A list of options see in the source code at the default definition of config[].

    is a background correction, then the identified pixels are
    enumerated and a radial distribution is created
    it is also normalized to the circumference of that radius

    Newest /ersion contains three different estimations of distances:
    - based on a center of mass and polar coordinates
    - based on angular segments and the distance from the mask within using
        again polar coordinates
    - based on distance tarnsform, calculating the distance from the binary mask

    Author: Tomio
    License: Creative Commons v. 4.0
    Warranty: None
    Date: 2017 - 2024
"""
# you may want to go sure no extra ouput is made:
#from matplotlib import use
#use('Agg')

config = {'dir': './', 'outdir':'dir', 'fmask':'*.tif', 'bgmask':'',
        'dpi':150, 'ext':'.png', 'threshold': -1.0,
        'CutEdge': -1,
        'Nroll':-1, 'MinSize': 1000, 'r_threshold': 0.01,
        'hist_threshold': 0.05,
        'dilate': 0, 'erode': 0, 'erode2': 0,
        'RGaussBg': -1, 'WGaussBg': 0, 'RGaussSm':0, 'WGaussSm':0,
        'gamma': 1, 'Nbins': -1,
        'scaler': 1.0, 'scaler_file': '',
        'MaskBefore': False,
        'Nangle': 100, 'DistanceTransform': True}

# if InvertMask is defined, maskimg gets inverted
# erode goes first, then dilate, then erode 2
# This way we can do dilate/erode sets or the opposite, according to image quality
# if AreaNorm is set, it will take over.
# r_threshold is the limit which defines the noise level for the outgrowth
#
# Alternative distance methods to be calculated and recorded also as histograms:
# Nangle: use this many bins to calculate the distances from the mask such,
#    that the maximal radius of the mask in this angular segment is subtracted from
#    each radal distance (from the center of the mask)
# DistanceTransform: if True, calculate the distances from the mask using a distance
#    transform as well.
#
# If scaler_file is provided, then every row is a file name and a scaler
# separated with a comma
# If a file name is missing, scaler is used instead (default value)

from numpy import sqrt, floor, pi, linspace, sin, cos, median, ones, arctan2
from ImageP import *
from BatchAnalyzer import *
from glob import glob
import os,sys

def dist_rel_distance(hst:dict, threshold:float) -> float:
    """ Take a histogram, find the maximum, then the distance
        where it first falls under maximum * threshold, and
        return this distance.

        parameters:
        hst:        a dict containing a histogram,
                    with keys: 'midpoints', 'dist'
        threshold:  a floating point number between 0 and 1

        return:
        the distance estimated
    """
    if threshold > 1 or threshold < 0:
        print('Invalid threshold value:', threshold)
        return 0

    rs = hst['midpoints']
    Is = hst['dist']
    max_Is = Is.max()
    I_threshold = max_Is * threshold

    i_r = (Is == max_Is).nonzero()[0][0]
    i_rp = (Is[i_r:] <= I_threshold).nonzero()[0]

    if len(i_rp) > 0:
        i_rad = i_r + i_rp[0]
    else:
        i_rad = len(rs) -1

    print('Relative radius calculation')
    print('threshold:', threshold, 'limit:', I_threshold)
    print('max distance', rs.max(), 'distance', rs[i_rad])
    print('')
    return rs[i_rad]
# end of dist_rel_distance

#####################################################################
#####################################################################
configfile = "config.txt"
if __name__ == '__main__':
    args = sys.argv[1:]

    if len(args) > 0:
        if os.path.isfile(args[0] ):
            configfile = args[0]
            print("Received command line config file: %s" %configfile)
        else:
            print("Please provide config file name with command");
#end if  main program
config = ReadConf( configfile, config)

indir = os.path.abspath(config['dir'][-1])
outdir = indir if config['outdir'][-1] == 'dir' else config['outdir'][-1]

outdir = os.path.abspath(outdir)

if  not os.path.isdir(outdir):
    os.mkdir(outdir)

fmask = config['fmask'][-1]
ext = config['ext'][-1] if config['ext'][-1][0] == '.' else '.%s' %config['ext'][-1]
dpi = int( config['dpi'][-1])

scaler = config['scaler'][-1]

scaler_dict = {}
if os.path.isfile(os.path.join(indir, config['scaler_file'][-1])):

    with open(os.path.join(indir, config['scaler_file'][-1])) as fp:
        txtlines = fp.readlines()
        if txtlines and ',' in txtlines[0]:
            linelist = [i.split(',') for i in txtlines if not i.startswith('#')]
            scaler_dict = {i[0]:float(i[1]) for i in linelist if len(i) > 1}
            del(linelist, txtlines)
# end getting scaler_file

#Rolling ball background filter
Nroll = int( config['Nroll'][-1] )
MinSize = int( config['MinSize'][-1])
threshold = config['threshold'][-1]
r_threshold = config['r_threshold'][-1]
hist_threshold = config['hist_threshold'][-1]
N_erode = int(config['erode'][-1])
N_dilate = int(config['dilate'][-1])
N_erode2 = int(config['erode2'][-1])

RGaussBg = config['RGaussBg'][-1]
WGaussBg = config['WGaussBg'][-1]
RGaussSm = config['RGaussSm'][-1]
WGaussSm = config['WGaussSm'][-1]

gamma = config['gamma'][-1]

MaskBefore = bool(config['MaskBefore'][-1])
Nbins = int(config['Nbins'][-1])

Nangle = int(config['Nangle'][-1])
DistanceTransform= bool(config['DistanceTransform'][-1])

# cut off a frame with this much pixels
CutEdge = int(config['CutEdge'][-1])

#file type we are reading, e.g. .jpg, .tif, .png etc.
fmask = config['fmask'][-1]

maskmask = config['bgmask'][-1]

lst = glob(os.path.join(indir, '%s' %fmask))
lst.sort()

# set a limiting N
N = int(config['N'][-1]) if 'N' in config else -1
if N > 0 and N < len(lst):
    lst = lst[:N]
else:
    N = len(lst)
# limiting list length

############# Do some reporting
rep = Report(outdir, add_time= False, header="Analyzing radial distances in images v 2.0")

rep.write('Opening session with parameters:')
rep.write('Input dir:', indir)
rep.write('Output dir:', outdir)
rep.write('File mask:', fmask)

if maskmask != '':
    rep.write('Use binary mask images with prefix:', maskmask)
# end if printing mask

rep.write('Background correction parameters')
if Nroll > 0:
    rep.write('Rolling ball filter is set to radius:', Nroll)
elif RGaussBg > 0:
    rep.write('Gauss blurr background with window radius:', RGaussBg, 'width:', WGaussBg)
    rep.write('Gauss smoothening with window radius:', RGaussSm, 'width:', WGaussSm)
else:
    rep.write('No background correction is requested')
# end if background

if config['scaler_file'][-1]:
    if scaler_dict:
        rep.write('List of scalers is loaded', len(scaler_dict))
    else:
        rep.write('List of scalers could not be loaded', config['scaler_file'][-1], color='red')

if scaler != 1:
    rep.write('Global scaler is set to', scaler, 'micron/pixels')

if gamma > 0 and gamma != 1.0:
    rep.write('Power law dynamic compression / expansion is applied with power:', gamma)
#end if gamma

if CutEdge > 0:
    rep.write('Cut off an edge of', CutEdge, 'pixels')

if MaskBefore:
    rep.write('Mask area is nulled before threshold calculation')

if Nbins < 0:
    rep.write('Use 1 step radial distribution')
else:
    rep.write('Number of bins set to:', Nbins, 'for the histogram')
#end if Nbins

if N_erode > 0 or N_dilate >0 or N_erode2 > 0:
    rep.write('Erosion:dilation:erosion to define center', N_erode, N_dilate, N_erode2)
#end if there is erosion/dilation preformed

rep.write('Noise level in normalized distribution is set to:', r_threshold)
rep.write('Noise level for normal histograms:', hist_threshold)

if Nangle > 0:
    rep.write('Calculate segmentwise distance from the explant in', Nangle, 'segments')

if DistanceTransform:
    rep.write('Calculate distance from explant based on the distance transform')

rep.write('\nStart processing:')
rep.write('Found', N, 'images')

results = []
# start the main loop over the images:
for fn in lst:
    # define for any case, so if not found, it is still valid
    # and deletes the previous one
    maskimg = None

    img = read_img(fn)
    ffn = os.path.split(fn)[-1]
    rep.write('Processing:', ffn,  color='cyan')

    # do we have the image?
    if img is None:
        rep.write("image:", fn,"not found")
        continue
    #end if

    # collect results for this image
    row_res= {}

    # set curent scaler
    fnn = os.path.splitext(ffn)[0]
    row_res['file'] = fnn

    if scaler_dict and fnn in scaler_dict:
        scaler = scaler_dict[fnn]
        rep.write('scaler is set to:', scaler, color='cyan')
    else:
        scaler = float(config['scaler'][-1])

    row_res['scaler'] = scaler

    # trim off the frame
    if CutEdge >0 and CutEdge < min(img.shape):
        img = img[CutEdge:-CutEdge, CutEdge:-CutEdge]

    # do we need binary mask images?
    if maskmask != '':
        maskfn = os.path.join(indir, '%s%s' %(maskmask, ffn))

        if os.path.isfile(maskfn):
            maskimg = read_img(maskfn)
        if not maskimg is None:
            rep.write('Loaded mask image:', os.path.split(maskfn)[-1])
        else:
            rep.write('Mask file not found!')
        #end if mask was loaded

        # if we have a mask, it is the same size as the image
        if maskimg is not None\
            and CutEdge > 0\
            and CutEdge < min(maskimg.shape):
            maskimg = maskimg[CutEdge:-CutEdge, CutEdge:-CutEdge]

        if 'InvertMask' in config:
            rep.write('Inverting mask image')
            maskimg = maskimg.max() - maskimg

    if Nroll > 0:
        #rolling ball requires the images being in size a multiple of 'reduce'
        #because we hardcoded reduce, we use only Nroll,
        #which should be a multiple of reduce=4.
        rep.write('Rolling ball background correction')
        x0 = sx%4
        y0 = sy%4
        if x0 != 0 or y0 != 0:
            #truncate the image:
            Nx_new = int(sx/4)*4
            Ny_new = int(sy/4)*4
            rep.write("Resized image for rolling ball to:", \
                                    Nx_new, Ny_new,"from", sx, sy);

            x02 = int(x0/2)
            y02 = int(y0/2)
            # now resize all images to have the same shape:
            img = img[x02:(x02+Nx_new), y02:(y02+Ny_new)]

            # resize the mask too
            if maskimg is not None:
                maskimg = maskimg[x02:(x02+Nx_new), y02:(y02+Ny_new)]

        #end if
        #print("Rolling ball background with radius: %d" %Nroll)
        img = img - RollingBall(img, Nroll, reduce=4)
        img[img < 0 ] = 0

    elif RGaussBg > 0:
        rep.write('Gaussian correction')
        img = GaussDeblurr(img, RGaussBg, WGaussBg, RGaussSm, WGaussSm, KillZero= True)
    else:
        img = img - img.min()

    if gamma != 1 and gamma != 0:
        rep.write('applying power', gamma, 'to the image')
        #img = img**gamma
        img = Compress(img, gamma, rel=True)

    if MaskBefore and maskimg is not None:
        if  maskimg.shape[0] == img.shape[0] and\
            maskimg.shape[1] == img.shape[1]:
            img[maskimg > 0] = 0
        else:
            rep.write('Maskimg vs. image shap mismatch!', color='red')
            maskimg = None

    th = graythresh(img) if threshold < 0 else threshold
    rep.write("Threshold is:", th)

    bimg = img > th*img.max()
    bwimg = bwlabel(bimg, MinSize= MinSize) > 0

    if bwimg.sum() < 1:
        print(colortext("Empty image! Skip further processing", 'red'))
        continue

    # apply mask:
    if maskimg is not None:
        mi, mj = (maskimg >0).nonzero()
        print(colortext('Applying mask!', 'green'))
        #bwimg[mi, mj] = 0
        bwimg[maskimg >0] = 0

        # get the mask center as center:
        i0 = mi.mean()
        j0 = mj.mean()

        # for the plotting:
        bwimg2 = bwimg

    else:
        bwimg2 = bwimg.copy()
        rep.write('Performing erosion/dilation/erosion cycles')

        bwimg2 = SimpleErode(bwimg2, N_erode)
        bwimg2 = SimpleDilate(bwimg2, N_dilate)
        bwimg2 = SimpleErode(bwimg2, N_erode2)

        ind2i, ind2j = bwimg2.nonzero()
        # Alternatively we could go for a center of mass,
        # but in many cases this is very noisy and uneven...
        # So, we go for the geometric venter, so dilation can
        # complensate for uneven illumination
        i0 = ind2i.mean()
        j0 = ind2j.mean()

        del(ind2i, ind2j)
    #end finding the center

    rep.write('New center indices:', i0, j0)

    ind_i, ind_j = bwimg.nonzero()
    # apply the center:
    ind_i = ind_i - i0
    ind_j = ind_j - j0

    # here is the point to convert to microns
    r_ij = sqrt(ind_i**2 + ind_j**2 )*scaler
    # the real angle is arctan2(y,x), but j -> y and i -> x!
    alpha_ij = arctan2(ind_i, ind_j)

    r_max = floor(r_ij.max())+1

    if Nbins < 0:
        h = hist(r_ij, range(int(r_max)))
        rel_dist = h['dist']/(2*pi*h['midpoints'])
    else:
        h = hist(r_ij, Nbins)
        # the rings are not 1 pixel wide anymore, so calculate their area
        print("Area normalization!")
        rel_dist = h['dist']/ (pi*(h['pockets'][1:]**2 - h['pockets'][:-1]**2))

    h['rel_dist'] = rel_dist
    i_r = (rel_dist == rel_dist.max()).nonzero()[0][0]
    i_rp = (rel_dist[i_r:] <= r_threshold).nonzero()[0]

    if len(i_rp) > 0:
        i_rad = i_r + i_rp[0]
    else:
        i_rad = len(rel_dist) -1

    # the radius is the last point before the distribution fell below the threshold
    radius = h['midpoints'][i_rad - 1]
    # we also need this in pixels:
    iradius = radius / scaler
    rep.write('Outgrowth radius:', radius, color='red')
    row_res['circ_normalized_thresholded_radius'] = radius
    row_res['histogram_max_distance'] = h['midpoints'].max()
    row_res['rel_abs_radius'] = dist_rel_distance(h, hist_threshold)

    fn_out = os.path.splitext(os.path.split(fn)[-1])[0]

    # Plot( h['pockets'][1:], rel_dist, xlabel='radius, $\mu$m', ylabel='relative number density',
    Plot( h['midpoints'], rel_dist, xlabel= r'radius, $\mu$m', ylabel='relative number density',
            title='radial number density')
    Plot([h['midpoints'][i_rad]], [rel_dist[i_rad]], fmt='ro', newplot= False,
            dpi=dpi, ext=ext, filename= "%s-radial-plot" %fn_out,
            outpath = outdir)

    alpha = linspace(0, 2*pi, 8*int(iradius))
    display( img*bwimg, dpi = dpi)

    Plot( j0 + iradius*cos(alpha), i0 + iradius*sin(alpha), fmt='r-', newplot= False)
    Plot([j0], [i0], fmt='go', newplot= False,
            dpi=dpi, ext=ext, filename = "%s-summary-composite" %fn_out,
            outpath = outdir)

    fnn_out = os.path.join(outdir, fn_out)

    SaveData(["radius", "N of pixels", "sum #", "rel. distribution"],
            zip(h['pockets'][1:],h['dist'],h['integ'],h['rel_dist']),
            "%s-table.txt" %fnn_out,\
            "#Radial distribution of pixels over threshold\n#Radius: %d pixels" %radius)
    display(composite(bwimg2, bwimg2, bwimg), dpi=dpi, fname="%s-eroded%s" %(fnn_out, ext))
    del(alpha)

    ########## 2021-10-26: dump the r_ij, alpha_ij table!

    # now, have a look at the alternatives:
    if maskimg is not None and Nangle > 0:
        rep.write('Run angle segment distances from the mask for', Nangle, 'slots')
        alpha_bins = linspace(-pi, pi, Nangle+1)
        # create a radial profile for the mask image too
        # this way we get the boundaries for each segment
        mi = mi - i0
        mj = mj - j0
        mr_ij = sqrt(mi**2 + mj**2) * scaler
        malpha_ij = arctan2(mj, mi)

        for i in range(Nangle):
            # index of this segment in the data:
            indx = (alpha_ij >= alpha_bins[i]) & (alpha_ij < alpha_bins[i+1])
            # where is this segment in the mask:
            mindx = (malpha_ij >= alpha_bins[i]) & (malpha_ij < alpha_bins[i+1])

            # the edge of the mask here:
            r0 = mr_ij[mindx].max()
            rep.write('For segment', i,'mask radius max is', r0, 'microns')
            # distances from the mask in this segment are:
            r_ij[indx] = r_ij[indx] - r0
        #end for each alpha segment

        SaveData(['Segmented distance'], zip(r_ij), '%s-segmented-distance-table.txt' %fnn_out,
                '# distance from mask edge in radial segments')

        # requested number of bind defined?
        if Nbins < 0:
            h = hist(r_ij, range(int(r_max)))
        else:
            h = hist(r_ij, Nbins)

        SaveData(["radius", "N of pixels", "sum #"],
                zip(h['pockets'][1:],h['dist'],h['integ']),
                "%s-segmented-histogram-table.txt" %fnn_out,\
                "# Distance distribution in segments from the explant")

        Plot(h['midpoints'], h['dist'],
                xlabel= r'distance, $\mu$m', ylabel='Count',
            title='radial slotted distance',
            dpi= dpi, ext= ext, outpath= outdir,
            filename= '%s-radial-segmented-histogram' %fn_out)

        rep.write('Maximal distance for segmenst is', h['midpoints'].max(), 'microns',
                color='green')
        row_res['Maximal_segment_distance'] = h['midpoints'].max()
        row_res['rel_angular_avg_radius'] = dist_rel_distance(h, hist_threshold)

    else:
        row_res['Maximal_segment_distance'] = -1
        row_res['rel_angular_avg_radius'] = -1
    #end if radial distribution is asked for: Nangle > 0

    if maskimg is not None and DistanceTransform:
        rep.write('Calculating distance transform for the mask')
        # distance filter returns the squared Eucledian distances:
        dimg = DistanceFilter(1 - maskimg)

        r_ij = sqrt(dimg[bwimg.nonzero()])* scaler

        SaveData(['Distance'], zip(r_ij), '%s--distance-from-mask-table.txt' %fnn_out,
                '# distance from mask edge using distance transform, in micrometers')
        # plot the histogram:

        if Nbins < 0:
            h = hist(r_ij, range(int(r_max)))
        else:
            h = hist(r_ij, Nbins)
            SaveData(["distance (micron)", "N of pixels", "sum #"],
                zip(h['pockets'][1:],h['dist'],h['integ']),
                "%s-distance-filter-histogram-table.txt" %fnn_out,\
                "# Distance distribution from the edges of the explant")

        Plot( h['pockets'][1:], h['dist'],
                xlabel= r'distance, $\mu$m', ylabel='Count',
            title='distance distribution from the explant',
            dpi= dpi, ext= ext, outpath= outdir,
            filename= '%s-distance-from-mask-histogram' %fn_out)

        rep.write('Maximal distance from explant is', h['midpoints'].max(), 'microns',
                color='green')
        row_res['distance_transform_max'] = h['midpoints'].max()
        row_res['rel_dist_transform_radius'] = dist_rel_distance(h, hist_threshold)

    else:
        row_res['distance_transform_max'] = -1
        row_res['rel_dist_transform_radius'] = -1

    results.append(list(row_res.values()))
    rep.write('... done')
#end for

SaveData(list(row_res.keys()), results,
         os.path.join(outdir, 'Summary-table.txt'),
         'Radial profile results of stack',
         report= rep)

rep.close()
