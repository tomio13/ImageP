#!/usr/bin/env python
""" fit a circle to ring shapes extracted from a fluorescence image
    author: T. Haraszti
    Licence: CC(4)-BY
    Warranty: None
    Date: 2020 - 2024-
    version: 4.3
"""

import os
import sys

from matplotlib import pyplot as pl
import numpy as nu

from BatchAnalyzer import (Report, ReadConf, ReadTable)
from ImageP import (read_img, bwlabel, bwanalyze, Circle_fit,
                    display, graythresh, SimpleDilate, SimpleErode,
                    SimpleContour, PerimeterImage, GaussDeblur,
                    EdgeDetect)

configfile = 'config.txt'

config = {'lst': 'lst.txt', 'N': -1, 'indir':'./', 'outdir':'./Results',\
          'RGaussBg':30, 'WGaussBg':10, 'RGaussSm':10, 'WGaussSm': 3,\
          'gradient': False, 'quantile': -1, 'threshold': -1,\
          'MinSize': 500, 'MaxSize': 150000, 'gap': 0,
          'Ndilate': 5, 'thickness':12}
# 'invert': true is an option ot invert an image before processing

#analysis function


def Get_Circles(img,
                th= -1,
                n_dilate= 5,
                thk= 5,
                MinSize= 500,
                MaxSize= 50000,
                gap=0,
                verbose= True):
    """ find circles in fluorescence images and fit them

        Take the image, threshold to get the structures. Then use the invert to
        get the patches surrounded by the (hopefully circular) tructures.
        Expand the patches to include the lines, then extract the coordinates
        of the pixels in the structures (lines).
        Fit them to a circle (Circle_fit()).

        Parameters:
        img:        a 2D image to be processed
        th:         relative intensity threshold to find the circles
        n_dilate:   dilate and erode the structures this many times to remove gaps
        MinSize, MaxSize:
                    minimum and maximal number of pixels in patches to be considered

        gap:        gap to jump over between patches

        Return:
            a list containing
        R:          radii of fitted circles
        x0, y0      center of circles fitted
    """

    if th <= 0:
        th = graythresh(img)
        print('set relative threshold to:', th)

    if th < 1.:
        th = th * img.max()
    print('applied thrshold intensity:', th)

    # end sorting out threshold
    # to detect the lines, the structure we need
    b = img > th

    # use dilate/erode to fuse gaps
    if n_dilate > 0:
        # from ImageP 0.355
        b = SimpleDilate(b, times= n_dilate)
        b = SimpleErode(b, times= n_dilate)

    #now, invert the structure image and detect the spots between the lines:
    c = bwlabel(1-b, MinSize= MinSize, MaxSize= MaxSize, gap=gap)

    if verbose:
        display(c)
        print('Detected', c.max(), 'areas to analyze')


    # collect the results as list of lists
    # every element is the set of one object
    res = []
    for i in range(1, c.max()+1):
        if verbose:
            print('analyzing patch:',i)

        rowData= {}
        # get the object
        d = c==i
        # it is possible that we have a non-circular area,
        # and not every filled in completely

        # fill up the area, in a relatively brute force way
        # this removes the internal membrane pieces, so we can
        # use the perifery
        # and it provides information about the geometry enclosed
        # in this patch
        d_filled = SimpleContour(d, 0, True)

        # now, get some stats about this area:
        # this is not dilated, because we analyse the internal area
        # of this patch, but filled
        ell = bwanalyze(d_filled, 1, 'ae')
        rowData['MaxSize'] = ell['MajorAxis']
        rowData['MinSize'] = ell['MinorAxis']
        rowData['Area'] = ell['PixArea']
        rowData['Sqrt-Size'] = nu.sqrt(float(ell['PixArea']))
        rowData['equivalent_diameter'] = 2.0*nu.sqrt(float(ell['PixArea'])/nu.pi)

        # the lines have a thickness, and the patches are this much
        # away from the actual lines
        #
        # first we take the external contour of the image only
        d = PerimeterImage(d_filled)
        # get min and max diameter (2*radius):
        ell_p = bwanalyze(d,1, 'p')
        rowData['min_diameter'] = 2.0*ell_p['r'].min()
        rowData['max_diameter'] = 2.0*ell_p['r'].max()

        # then we fatten that to cover the image points
        d = SimpleDilate(d, thk)
        #d = ((d*img) > th*img.max())

        #the common part between the original structure and the fat patch
        #is at least a piece of the edge:
        d = d*b
        #now we should have a circle:
        x,y = d.nonzero()
        fit = Circle_fit(x,y,True)
        # Circle_fit returns keys: ['R', 'x0', 'y0', 'xfit', 'yfit', 'err2',
        #                               'chi2', 'relError']
        # load some fit parameters to rowData
        translate = [['R', 'R'], ['x0','x0'],
                     ['y0', 'y0'], ['xfit', 'xfit'],
                     ['yfit', 'yfit']]
        for klist in translate:
            rowData[klist[1]] = fit[klist[0]]


        # comment 2022-09-16:
        # include the number of points as:
        rowData['N'] = len(x)
        # to remove non-circular objects, maybe try a mean relative squared error
        # relative to the radius
        rowData['meanerr'] = (fit['err2']/fit['R']**2).mean() if len(x) > 1 else 1

        res.append(list(rowData.values()))
    #end for

    if not res:
        return {}

    keys = list(rowData.keys())
    out = {}
    print("N rows:", len(res))
    print("N columns:", len(res[0]))

    for i,j in enumerate(zip(*res)):
        out[keys[i]] = list(j)

    # build the result
    return out
#end Get_Circles


########################################################
if __name__=='__main__':
    args = sys.argv[1:]
    if len(args) > 0:
        if os.path.isfile( args[0] ):
            configfile = args[0]
            print("Received command line config file name:", configfile)
        else:
            print('Invalid config file name')
            print('Usage: program config.txt')
            sys.exit(0)

config = ReadConf( configfile, config)

indir = config['indir'][-1]
outdir = config['outdir'][-1] if indir != 'dir' else indir

indir = os.path.abspath(os.path.expanduser(indir))
outdir = os.path.abspath(os.path.expanduser(outdir))

if not os.path.isdir(outdir):
    os.makedirs(outdir)
# end if outdir does not exit


lst = ReadTable(os.path.join(indir, config['lst'][-1] ), sep='  ', keys= ['file','scaler'])

N = int(config['N'][-1])
if not lst:
    print('No data found!')
    sys.exit(0)

if N > 0 and N < len(lst['file']):
    # truncate the lines
    lst['file'] = lst['file'][:N]
    lst['scaler'] = lst['scaler'][:N]
else:
    N = len(lst['file'])

#fmask = '*.png'

if not os.path.isdir(outdir):
    os.mkdir(outdir)

Rb = config['RGaussBg'][-1]
Wb = config['WGaussBg'][-1]
Rg = config['RGaussSm'][-1]
Wg = config['WGaussSm'][-1]

gradient = bool( config['gradient'][-1] )
#use quantile for threshold:
q = config['quantile'][-1]

th = config['threshold'][-1] #if q is not defined
MinSize = int( config['MinSize'][-1] )
MaxSize = int( config['MaxSize'][-1] )
n_dilate = int( config['Ndilate'][-1] ) #edge width + 1 or so
thk = int(config['thickness'][-1]) #thickness to expect

gap = int(config['gap'][-1]) # gap to jump over finding patches
########################################################
# Report parameters:
rep = Report(outdir, header='Circle analysis for membrane images version 4.1')
#we detect patches in the inverse image to separate
#areas of interest
#Fuse the lines first using dialte  then erode Ndilate times:r,
#   header='Circle analysis of fluorescence images', add_time= False)

rep.write('File path', indir)
rep.write('Results folder:', outdir)
rep.write('File list and calibration from:', config['lst'][-1])

if Rb >0 :
    rep.write('Gaussian deblur parameters')
    rep.write('Background window radius:', Rb)
    rep.write('Background width:', Wb)
    rep.write('Smoothing window radius:', Rg)
    rep.write('Smoothing width:', Wg)
else:
    rep.write('Do not correct background')

if gradient:
    rep.write('Use gradient!')
    rep.write('Gradient Gaussian window', Rg)
    rep.write('Gradient Gaussian width', Wg)

if q > 0:
    rep.write('Threshold at quantile', q)
elif th < 1:
    rep.write('Relative threshold:', th)
else:#we detect patches in the inverse image to separate
#areas of interest
#Fuse the lines first using dialte  then erode Ndilate times:
    rep.write('Threshold intensities above', th)

rep.write('Detect (inverse) patches larger than:', MinSize, 'and smaller than', MaxSize, 'in area')

if gap > 0:
    rep.write('Patch identification accepts gaps of', gap, 'pixels')
rep.write('Dilate / errode fuse pixels', n_dilate, 'times in patches')
rep.write('Line thickness seeked', thk)


rep.write('*************************************************************')
rep.write('Processing images')
########################################################

for i in range(N):
    fn = lst['file'][i]

    fnn = os.path.splitext(os.path.split(fn)[-1])[0]
    a = read_img(os.path.join(indir,fn))
    scaler = lst['scaler'][i]

    if a is None:
        rep.write("image", fnn, "not found", color='red')
        continue

    rep.write('Loaded:', fnn, color='cyan')
    rep.write('Scaler:', scaler, color='green')
    #end if no image was loaded

    if 'invert' in config and config['invert']:
        rep.write('Inverting image')
        a = a.max() - a

    if Rb > 0:
        b = GaussDeblur(a, Rb, Wb, Rg, Wg, True)
    else:
        b = a - a.min()

    #we can detect edges on smoothened images, but
    #often this is a bad idea
    if gradient:
        b = EdgeDetect(b, Rg, Wg)


    #set threshold to quantile:
    if q > 0 and q < 1:
        th = nu.quantile(b, q)

    #now, get the circles detected / fitted / collected; see above
    ft = Get_Circles(b, th, n_dilate, thk, MinSize, MaxSize)
    if not ft:
        rep.write('Nothing found!', color='red')
        continue

    fout = f'{fnn}-patches.png'
    pl.savefig(os.path.join(outdir, fout), dpi=200, bbox_inches='tight', pad_inches=0)

    #generate the results:
    display(b)
    print('Summary:')
    keylist = list(ft.keys())
    #keylist.remove('x')
    #keylist.remove('y')

    headline= '\t'.join(['index'] + keylist)

    fout = f'{fnn}-results-table.txt'
    with open(os.path.join(outdir, fout),
              'wt',
              encoding='UTF-8') as fp:
        fp.write(f'#{headline}\n')

        Np = len(ft['R'])
        rep.write('found', Np, 'circles')
        rep.write('Writing results table to', fout)
        print(headline)

        # put the result in the plot and on screen:
        for i in range(Np):
            pl.plot(ft['yfit'][i],ft['xfit'][i], 'r-')
            pl.text(ft['y0'][i], ft['x0'][i],
                    f'{(i+1):d}',
                    color='red',
                    horizontalalignment='center')

            line = [i+1] + [ft[j][i] * scaler for j in keylist]
            indx = keylist.index('Area') +1 #  compensate for adding the index to the line
            line[indx] *= scaler # needs scaler**2 for area!
            indx = keylist.index('N') +1 # index should have no scaler
            line[indx] /= scaler

            txt = '\t'.join([str(j) for j in line]) + '\n'
            print(txt, end='')

            #to the summary table:
            fp.write(txt)
        #end for
    # end fp open

    #dump the figure to a file:
    fout = f'{fnn}-summary.png'
    pl.savefig(os.path.join( outdir, fout), dpi=200, bbox_inches='tight', pad_inches=0)
    rep.write('Dumped summary image to:', fout)


#end for file list
