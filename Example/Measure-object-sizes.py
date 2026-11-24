#!/usr/bin/env python
""" Measure the length of fiber like structures in images.
    The idea is to take an external perspective and try identifying a skeleton
    that defines the structure.
    Author: tomio
    Licence: CC(4)-BY
    Date: 2022 - 2026-
    Warranty: None
"""

from ImageP import *
from numpy import sqrt, quantile
from matplotlib import pyplot as pl
from BatchAnalyzer import SaveData, Report

import os
import sys
import yaml
from glob import glob

colors = ['red','green','blue','cyan','magenta','yellow','white']
NC = len(colors)

config_defaults = {'dir': './', 'outdir': 'dir', 'fmask': '*.tif',
        'dpi': 150, 'ext': '.png',
        'Step Height': 4, 'Nt': 100,
        'Rbg': 400, 'Wbg': 100, 'Rsmooth': 8, 'Wsmooth': 2,
        'gamma': 1.0,
        'threshold': 'auto', 'q': 0.5,  'MinSize': 500, 'MaxSize': 0,
        'Closing N': 0, 'Opening N': 0
        }
configfile= 'config.txt'

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) > 0:
        if os.path.isfile(args[0]):
            configfile= args[0]
            print(f'Received command line config file name: {configfile}')
        else:
            print('Invalid config file name')
            print('Usage: command config_file name')
        #end if
    # end args
# end start up

with open(configfile, 'rt') as fp:
    config = yaml.safe_load(fp)

# set the default configuration
for k,v in config_defaults.items():
    if k not in config:
        config[k] = v
# end filling up defaults

indir = os.path.abspath(os.path.expanduser(config['dir']))
outdir = os.path.abspath(os.path.expanduser(config['outdir'])) if config['outdir'] != 'dir'\
        else indir

# if the output directory is not created, make one
if not os.path.isdir(outdir):
    print('Creating ouptut directory:', outdir)
    os.mkdir(outdir)
# end if no outdir

rep = Report(outdir, 'report.rep', add_time= False, header='Object size analyzer script')

rep.write('Listing configuration parameters:')
for i, v in config.items():
    rep.write(i,':', v)
rep.write('---***---')


# this is related to the width of the objects
# if too small, many small skeletal lines will be found
# if too high, the object will be missed.
StHeight= int(config['Step Height'])# step height in pixels for skeleton

dpi = int(config['dpi'])

#filelist = glob('./*.bmp')
filelist = glob(os.path.join(indir, config['fmask']))
filelist.sort()

# number of theta values / points in a cirle
Nt = int(config['Nt'])

res = []
for fn in filelist:
    img = read_img(fn)
    if img is None:
        rep.write('\nCould not load', fn, color='red')
        continue
    # end if no image was found
    fnn = os.path.splitext(os.path.split(fn)[-1])[0]
    rep.write('\nLoaded:', fnn, color='green')
    resrow = {}

    if 'invert' in config and config['invert']:
        rep.write('Inverting image')
        img = img.max() - img

    # img2 = GaussDeblurr(img.max() - img, 400, 100, 8, 2, True)
    img2 = GaussDeblurr(
            img.max() - img,
            config['Rbg'],
            config['Wbg'],
            config['Rsmooth'],
            config['Wsmooth'],
            True
            )

    if config['gamma'] > 0 and config['gamma'] != 1.0:
        img2 = Compress(img2, config['gamma'])
        rep.write('Applying dynamic compression with gamma:', config['gamma'])

    # Otsu's method for threshold:
    mx = img2.max()
    if config['threshold'] == 'auto':
        th = graythresh(img2) * mx
        rep.write('Threshold is:', th, img2.min(),':', mx,'/', th*mx, color='cyan')

    elif config['threshold'] == 'quantile':
        th = quantile(img2, config['q'])
        rep.write('Thresholding with quantile:', config['q'], th, color='cyan')
    else :
        th= mx*config['threshold']
        rep.write('Threshold:', config['threshold'], th, color='cyan')

    if 'bf image' in config and config['bf image']:
        rep.write('inverting the binary image')
        bimg = img2 < th
    else :
        bimg = img2 > th

    if config['Closing N'] > 0:
        rep.write('Dilating / eroding the binary mask', config['Closing N'], 'times')
        bimg = SimpleDilate(bimg, config['Closing N'])
        bimg = SimpleErode(bimg, config['Closing N'])

    elif config['Opening N'] > 0:
        rep.write('Eroding / dilating the binary mask', config['Closing N'], 'times')
        bimg = SimpleErode(bimg, config['Closing N'])
        bimg = SimpleDilate(bimg, config['Closing N'])


    bimg = bwlabel(bimg,
                   MinSize= config['MinSize'],
                   MaxSize= config['MaxSize'])


    # try filling up the individual detected shapes:
    if 'fill' in config and config['fill']:
        for j in range(1, bimg.max()+1):
            segment = bimg == j
            segment = SimpleContour(segment, fill= True)
            bimg[segment] = j

    if bimg.max() == 0:
        rep.write('Nothing is found', color='red')
        continue

    bwimg = bimg > 0
    pl.clf()
    pl.imshow(bwimg)

    # extend the structure identification

    # form a skeleton based on the distance transform and local maxima:
    d = DistanceFilter(bwimg)
    t = FindLocalMaxima(d, height= StHeight)
    # direct erosion based skeletonization:
    t2 = Skel(bwimg)

    # display(composite(img, t, img),
    display(composite(bimg >0, t, t2),
            fname= os.path.join(outdir, f'{fnn}-distances{config["ext"]}'),
            dpi= dpi
            )
    pltlist = {'X':[], 'Y':[], 'size':[]}

    for j in range(1, bimg.max()+1):
        rep.write('Analyzing structure', j)
        # i -> y, j -> x

        segment = bimg == j
        y,x = segment.nonzero()
        ell = bwanalyze(bimg, j, 'e')
        e = ell['Eigenvector1']
        # now analyze the result
        x0 = x.mean()
        y0 = y.mean()

        # the coordinates are in index, thus y and x are swapped
        # the vector of e is o.k.
        profile = ExtractProfile(segment,
                                 y0, x0,
                                 e[0], e[1], verbose= True)
        # this profiles is interpolated, but it should be nonzero only
        # in the object. An estimate of size is the number of nonzero
        # points. Intensity values are in the profile[3] or profile[-1]
        # array.

        rep.write('center', x0,y0)
        # resrow.update([['file', fnn], ['X0', x0], ['Y0', y0]])
        # equivalent, easier to read:
        resrow.update({'file':fnn, 'X0':x0, 'Y0': y0})
        pltlist['X'].append(x0)
        pltlist['Y'].append(y0)
        pltlist['size'].append(len(x))

        # How many pixels we have in 't' for this patch?
        pix = t[y, x]
        pix_cl = t2[y, x]
        # the distance transform spits oud the quare of the distance to the
        # nearest edge. On the ridges (t > 0) or (t2 > 0)
        # we can then calculate the width as
        widths = 2.0*sqrt(d[y,x])[pix >0]
        width_mean = widths.mean()
        width_error = widths.std()

        # the number of pixels is a measure of length
        # with all hairs, false branches and
        # without taking into accound diagonal moves...
        pix_len = (pix > 0).sum()
        pix_cl_len = (pix_cl >0).sum()

        resrow.update([['ID', j],
                    ['area_pix', len(x)],
                    # ['radial_length', length],
                    # ['compact', compact],
                    ['profile_length', (profile[3] ==1).sum()],
                    ['pix_distance_length', pix_len],
                    ['pix_classic_skel_length', pix_cl_len],
                    ['pix_distance_width_mean', width_mean],
                    ['pix_distance_width_sd', width_error]
                    ])
        res.append(list(resrow.values()))
    #end for in the image

    # make an overplot indicating the enumerated particles
    pl.clf()
    OverPlot(img, pltlist, NewFig= False)
    for i in range(len(pltlist['X'])):
        txt = '%d' %i
        pl.text(pltlist['X'][i]- sqrt(pltlist['size'][i]),
                pltlist['Y'][i],
                txt,
                color= 'white',
                horizontalalignment='left'
                )

    pl.savefig(
            os.path.join(outdir, f'{fnn}-overplot{config["ext"]}'),
            dpi= dpi,
            bbox_inches='tight',
            pad_inches=0
            )
#end for

if len(res) > 0:
    SaveData(
        list(resrow.keys()),
        res,
        os.path.join(outdir, 'Summary-table.txt'),
        '# microplastic length estimation'
        )
