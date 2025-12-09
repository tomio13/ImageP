#!/usr/bin/env python
""" Orientation analysis based on an oriented Gauss: Mexican hat filter

    Author:     T. Haraszti
    Email:      haraszti@dwi.rwth-aachen.de
    Date:       2016-06- 2021
    Licence:    CC(4)-BY

"""
from ImageP import *
from numpy import linspace, zeros, sqrt, nanmax, append, pi, arange, asarray, abs
from numpy import savez_compressed, quantile
from matplotlib import pyplot as pl
#do not even bother plotting:
pl.ioff()

from glob import glob
import os, sys
from BatchAnalyzer import SaveData, ReadConf, Report
from time import time
######################## Default config ####################
configfile="config.txt"
config = {'dir': './', 'outdir': './Results', 'ext':'.png', 'dpi': 150, \
        'fmask': '*.tif', 'Nroll': -64, 'reduce': 4, 'Nangles': 20, \
        'Qwidth': 0.5,\
        'RGaussBg': -250, 'WGaussBg': -100,\
        'RGaussSm': -10, "WGaussSm": -0.75, \
        # enable a size filter if non-negative
        'MinSize': -1, 'MaxSize': -1,
        # gamma is a dynamic range compressor applied after background
        # subtraction and edge detection
        'gamma': -1,
        'deriv': 0, 'N':-1,\
        'Langle': 15, 'Wangle': 0.75, 'Rangle':25 ,\
        'masked':False, 'mask':"mask ", "invertmask": True, 'maskerode':10,\
        'maskth': -1,\
        'MinBlob': 3.0, 'MaxBlob': 6.0, 'dump': False,
        'UseStructure': False, 'StructureScaler': 5,
        'CutEdge': False, 'EdgeWidth': -1}
#alternative switches:
# CutNegative cut the negative pixels in edge detection
# for background correction, negative pixels are always cut
# Blob removal: RemoveBlob, MinBlob and MaxBlob: k/ra > 3*pi and k/ra < 6*pi
# Save arrays in binary numpy format compressed: dump = True
# SoftThreshold: use the minimum of relative mean or Otsu's threshold
# invert: if the keyword is present, invert the images

######################## FUNCTIONS #########################

def FWHM(x, y = None):
    """ calculate the full width at half maximum for a dataset
        Parameters:
        x,y : one dimensional data arrays of the same length

        Return:
            a single floating point number
    """
    if y is None:
        x = y.copy()
        y = arange(x.shape)
    elif x.size != y.size:
        print("The two arrays must have the same size!")
        raise ValueError

    ymax = max(y)
    ymax2 = ymax/2.0

    indx = arange(len(y))[y >= ymax2]
    #print(indx)

    #first position:
    i1 = indx.min()
    print("FWHM start at i1: %d, x: %f" %(i1, x[i1]))

    if i1 > 0:
        i0 = i1 - 1
        if x[i0] == x[i1] :
            x_start = x[i0]
        else:
            A = (y[i1]-y[i0])/(x[i1]-x[i0])
            B = y[i0] - A*x[i0]
            x_start = (ymax2 - B)/A
    else:
        x_start = x[0]
    ##############################
    #now the other end:
    i0 = indx.max()

    #print("end at i1: %d, x: %f" %(i0, x[i0]))
    if i0 == (len(x)-1):
        x_end = x[i0]
    else:
        i1 = i0+1
        if x[i0] == x[i1]:
            x_end = x[i1]
        else:
            A = (y[i1]-y[i0])/(x[i1]-x[i0])
            B = y[i0] - A*x[i0]
            x_end = (ymax2 - B)/A
    #############
    #print("result: x-start: %f, x-end:for i in range(r1): %f" %(x_start, x_end))

    #pl.figure(5);
    #pl.plot(x,y,'b+-')
    #pl.plot((x_start, x_end),(ymax2,ymax2),'g-o')

    return (x_end - x_start)
#end of FWHM

##################################### Program space

if __name__ == '__main__':
    args = sys.argv[1:]

    if len(args) > 0:
        #we have command line arguments...
        if os.path.isfile( args[0] ):
            configfile = args[0]
            print("Received command line config file name: %s" %configfile)
        else:
            print("Invalid config file name")
            print("Usage: program config.txt")
        #end if
    #end if args
#end if main...

#process the config file, and use defaults from config:
config = ReadConf(configfile, config)

#manage file paths:
#print("Config dir", config['dir'])
indir = os.path.abspath( config['dir'][-1] )
outdir = os.path.abspath( config['outdir'][-1] ) \
                if config['outdir'][-1] != 'dir' else indir

ext = config['ext'][-1]
dpi = int( config['dpi'][-1])

if not os.path.isdir(outdir):
    # os.mkdir(outdir)
    os.makedirs(outdir)
#end create output dir?

N = int(config['N'][-1])
#fmask defines what files we are looking for:
lst = glob(os.path.join(indir, config['fmask'][-1]))
lst.sort()
#sanitize N:
Nlst = len(lst)
if N < 0 or N > Nlst:
    N = Nlst

#background filtering: turn them off by setting them to < 0!
Nroll = int(config['Nroll'][-1]) #zoom factor of the rolling ball
Nreduce = int( config['reduce'][-1]) #reduction factor for rolling ball, normally 2, 4, 8, etc.
RGaussBg = int(config['RGaussBg'][-1]) #radius for Gauss deblurr
WGaussBg = float(config['WGaussBg'][-1]) #sigma for Gauss deblurr
deriv = int(config['deriv'][-1])
gamma = config['gamma'][-1] #use gamma correction if it is not negative

#sanitize user input
if deriv > 2 or deriv < 0:
    deriv = 0
#end check deriv

RGaussSm = int(config['RGaussSm'][-1]) #Radius for Gauss deblurr smoothing
WGaussSm = float(config['WGaussSm'][-1]) #sigma for Gauss deblurr smoothing part
Invert = True if 'invert' in config else False

Rangle = int(config['Rangle'][-1])  #radius of the rotating angle Gauss filter
Langle = float(config['Langle'][-1]) #length of the narrow part (sigma long)
Wangle = float( config['Wangle'][-1] ) #the width of the narrow part (sigma short)

Nangles = int(config['Nangles'][-1]) #how many angles to calculate

SoftThreshold = True if "SoftThreshold" in config else False

#what part of the population we want to use for quantile width?
Qwidth = config['Qwidth'][-1]
if Qwidth >= 1.0 <= 0.0:
    print('Invalid Qwidth, falling back to 0.5')
    Qwidth = 0.5

#allow using a mask image with the name: "%s-%s" %(mask, fn)
masked = config['masked'][-1]
fnmask = config['mask'][-1]
invertmask = config['invertmask'][-1]
maskerode = int( config['maskerode'][-1])
maskth = config['maskth'][-1]

#for blob removal:
MinBlob = config['MinBlob'][-1]
MaxBlob = config['MaxBlob'][-1]

UseStructure = bool( config['UseStructure'][-1] ) if 'UseStructure' in config else False
StructureScaler = config['StructureScaler'][-1]

#should we dump arrays for numpy?
dump = bool(config['dump'][-1])


# drop this much of the data at the edge of the orientation image
# if < 0 then use the Wangle for this
CutEdge = bool(config['CutEdge'][-1])
EdgeWidth = int(config['EdgeWidth'][-1])
if EdgeWidth < 0:
    EdgeWidth = max(int(2*Wangle), 1)

########################################################################################
########## Summary output first in report:
rep = Report(outdir, header="Orientation analysis based on Oriented Mexican hat kernels",\
        add_time= False)

rep.write("File path", indir)
rep.write("Generating results into:", outdir)
rep.write("Analyzing extension: ", config['fmask'][-1])
rep.write('Loading', N, 'images')

if Nroll > 0:
    rep.write("Rolling ball background correction with radius of", Nroll, "pixels")
    rep.write('Current reduction factor is set to', Nreduce)

if RGaussBg >0 and WGaussBg > 0:
    rep.write("Gauss deblurr background R:", RGaussBg, "W:", WGaussBg, \
            "smoothening window radius:", RGaussSm, "width:", WGaussSm)

elif RGaussSm > 0 and WGaussSm > 0:
    rep.write("Smoothening with a Gaussian. Window radius:",\
                    RGaussSm, "width:",WGaussSm)
else:
    rep.write("No Gaussian correction is made\n")

if config['MinSize'][-1] > 1 or config['MaxSize'][-1] > 0:
    rep.write('Apply size filter with minimum',
              config['MinSize'][-1],
              'and maximum', config['MaxSize'][-1], 'pixels')

if gamma > 0:
    rep.write('Apply a power conversion after background correction and smoothing, with power', gamma)


if deriv > 0:
    rep.write("Smoothing filter calculates derivative. Order", deriv)

if Invert:
    rep.write("Image intensities will be inverted\n")

rep.write("Number of angles calculated:", Nangles)
rep.write("Radius of the rotating Gauss filter window", Rangle, "pixels")
rep.write("Length  sigma of the rotating Gauss filter", Langle, "pixels")
rep.write("Width sigma of the rotating Gauss filter", Wangle, "pixels")

if SoftThreshold:
    rep.write("Softer thresholding is activated (min of mean or Otsu's)")

if "CutNegative" in config:
    rep.write("Negative values are ignored in ridge detection")

if masked:
    rep.write("Mask the obtained data using an image mask")
    rep.write("Mask file name is prefixed with:", fnmask)
    if maskth < 0:
        rep.write('use autometic threshold on masks')
    else:
        rep.write('Use a threshold of', maskth, 'on masks')

    if invertmask:
        rep.write("Negative mask is used (mask == mask.min())")
    rep.write('erode the binary mask image', maskerode, 'times')
#end report on masking

if "RemoveBlob" in config:
    rep.write("Try removing blobs from the image")

    if UseStructure:
        rep.write('Use a structure tensor with', StructureScaler,\
                'increasing the Gaussian smoother')
    else:
        rep.write("Minimal perimeter / size ratio", MinBlob, "*pi")
        rep.write("Maximal perimeter / size ratio", MaxBlob, "*pi")
        MinBlob = MinBlob*pi
        MaxBlob = MaxBlob*pi
#end reporting blob removal

#Quantile width analysis
rep.write('Part of population used by quantile width analysis', Qwidth)

if CutEdge:
    rep.write('Drop the edge of the image with a width of', EdgeWidth, 'pixels')

########################################################################################
# end of report header, start working here...
#
#exclute the right end point: enpoint=False
alpha_min = 0.0
alpha_max = 180.0
dalpha = alpha_max - alpha_min #\Delta alpha, range of angles
alpha = linspace(alpha_min, alpha_max, Nangles, endpoint=False)

# results table
res = []
# the table header
res_keys = []

for fn in lst[0:N]:
    # row of results:
    res_row= {}

    # get the image as a gray scale intensity image (I16 by default)
    img = read_img(fn)
    fn = os.path.split(fn)[1]
    ffn = os.path.splitext(fn)[0]

    if img is None:
        rep.write('Image', fn, 'not found!', color='red')
        continue

    rep.write("Read image from file:", fn, color='cyan')

    res_row['file'] = ffn

    # Should we invert the image around its average background?
    # we talk about bright-field images here
    if Invert:
        img = img.max() - img
    else:
        # simplest background correction:
        img = img - img.min()

    if masked:
        mfn = f"{fnmask}{fn}"
        # not checking here intentionally, to crash if mask is not
        # provided properly...
        rep.write("reading mask:", mfn)

        maskimg = read_img(os.path.join(indir, mfn))

        # is the mask image available?
        # is it also the right size?
        # if not, make a dummy:
        if maskimg is None or \
            (maskimg.shape[0] != img.shape[0] and\
            maskimg.shape[1] != img.shape[1]):

            rep.write('Error with mask image, ignoring mask!')
            maskimg = zeros(img.shape)
        else:
            maskimg = maskimg - maskimg.min()

        # the used masl has to be a bool, we take the nonzero values:
        if maskth < 0:
            usemaskth = graythresh(maskimg)
        else:
            usemaskth = maskth

        # our mask is 1 or True where we keep the pixels!
        if invertmask:
            maskimg = maskimg > usemaskth*maskimg.max()
        else:
            maskimg = maskimg < usemaskth*maskimg.max()

        # first kill the image pixels to something of moderate value:
        if maskimg.sum() > 0:
            img[maskimg == 0] = img.mean()
            # grow the mask to kill the kernel effects:
            # erode maskerode times
            # it shall be used in the max-intensity image
            maskimg = SimpleErode(maskimg, maskerode)
        else:
            rep.write('Empty mask image, ignoring mask')
    # end preparing mask


    ################## Background subtratction and filters:
    sx,sy = img.shape
    if Nroll > 0:
        #x0 = sx%Nroll
        #y0 = sy%Nroll
        #caused by the reduce in the rolling ball, not the Nroll
        x0 = sx%Nreduce
        y0 = sy%Nreduce
        if x0 != 0 or y0 != 0:
            #truncate the image:
            Nx_new = int(sx/Nreduce)*Nreduce
            Ny_new = int(sy/Nreduce)*Nreduce
            rep.write("Resized image for rolling ball to:", \
                                    Nx_new, Ny_new,"from", sx, sy);

            x02 = int(x0/2)
            y02 = int(y0/2)
            img = img[x02:(x02+Nx_new), y02:(y02+Ny_new)]

            #if we have a mask, we have to cut to the same size:
            if masked:
                maskimg = maskimg[x02:(x02+Nx_new), y02:(y02+Ny_new)]

        #end if
        #print("Rolling ball background with radius: %d" %Nroll)
        img = img - RollingBall(img, Nroll, reduce= Nreduce)
        img[img <0 ] = 0
    # end rolling ball background

    # If we want to remove small size noise, it is a good place here

    # we have three situations here:
    #   1. Gauss deblur
    #   2. no Gauss deblurr, but smoothing
    #   3. with or without Gauss deblurr we need derivative filter

    if RGaussBg > 0 and WGaussBg > 0:
        # GaussDeblurr does nothing if the background filter is invalid
        img = GaussDeblurr(img, RGaussBg, WGaussBg, RGaussSm, WGaussSm, KillZero=True)

    # we can smoothen the image to enhance features:
    # this can be ignored, then the filter picks up all possible noises
    if RGaussSm > 0 and WGaussSm > 0:
        #print("Smoothening with a Gaussian, window: %d, width: %.3f" %(RGaussSm, WGaussSm))
        if deriv == 1:
            rep.write("Calculating derivatives");
            img= EdgeDetect(img, R= RGaussSm, w= WGaussSm, \
                    KillZero= ('CutNegative' in config))

        # the case where RGaussSm and WGaussSm are defined and no gradient is
        # needed, we have in GaussDeblur already
        # This would be a double smoothing...
        elif RGaussBg <= 0 or WGaussBg <= 0:
            # smoothing without background correction
            gk = GaussKernel(RGaussSm, WGaussSm, norm= True, OneD= True)
            img = ConvFilter1D(img, gk)
        #end sorting out which filter
    #end if smoothing

    if CutEdge:
        print('Cutting the edges')
        img = img[EdgeWidth:-EdgeWidth, EdgeWidth:-EdgeWidth]

        if masked:
            maskimg = maskimg[EdgeWidth:-EdgeWidth, EdgeWidth:-EdgeWidth]
    #end cutting the edges

    # use gamma for any case
    c = zeros([Nangles, img.shape[0],img.shape[1]])

    # run through the angle list:
    for i in range(Nangles):
        print("calculating: %d alpha: %.1f" %(i, alpha[i]))
        gkr = -RotatingGaussKernel(Langle,Wangle, alpha[i], \
                                        shape=[2*Rangle+1,2*Rangle+1],deriv=2)

        # this is the slow part, a 2D convolution filter
        c[i,:,:] = ConvFilter(img, gkr)
    # end for

    # analyze the resulted stack:
    # Easy to find the maximum but avoid undefined pixels ...
    b = nanmax(c, axis=0)
    # erase negative pixels...
    # do inplace erasure, we need sparing some memory
    b[b < 0] = 0
    if masked and maskimg.sum() > 0:
        b[maskimg == 0] = 0

    # first we find the maximum angles, and free up 'c'!
    print("Generating index image")
    # generate angle image:
    aimg = zeros(b.shape)
    # take the coordinates, where the 3D c-image is equal to its maximum b
    # only for nonzero c values (the ones kept after filtering)
    indx = ((c == b) * (c > 0) ).nonzero()

    # the first dimension holds the angle index value, 1,2 are the X,Y projection:
    aimg[indx[1],indx[2]] = indx[0]*dalpha/Nangles + alpha_min

    # clean some memory:
    del(c)

    # generate threshold
    print("Gray thresholding the orientation intensities")
    tt = time()
    if b[b>0].sum() == 0:
        # this image is empty!
        rep.write('Empty result image!:', fn)
        continue
    #end if empty intensity image

    #remove blobs from the angle image, based on the original
    #this way we improve the threshold later, if the blobs are bright (for
    #fluorescence often they are)
    if "RemoveBlob" in config:
        if UseStructure:
            d = StructureTensor(
                    img,
                    radius= StructureScaler*abs(RGaussSm),
                    width= StructureScaler*abs(WGaussSm)
                    )
            mask = d['indx']&(d['c'] < 0.5)

            if CutEdge:
                mask = mask[EdgeWidth:-EdgeWidth, EdgeWidth:-EdgeWidth]
            #end cutting the edges


            #kill pixels, not the mask:
            b[mask] = 0
            del(d)
        else:
            #we want to find roundish blobs...
            bimg = bwlabel(img > graythresh(img)*img.max())
            for bi in range(1, bimg.max()+1):
                #how many pixels are at the perimeter of this patch?
                pbi = PerimeterImage(bimg == bi).sum()
                #what is the 'size' of this patch?
                rabi = sqrt(float(bwanalyze(bimg, bi, 'a')['PixArea'])/pi)
                ratbi = pbi/rabi

                #MinBlob and MaxBlob were multiplied with pi at writing the report...
                if ratbi > MinBlob and ratbi < MaxBlob:
                    #kill pixels, not the mask:
                    b[bimg == bi] = 0
            #end for bi in patches
            del(pbi, rabi, ratbi)
        #end blob removal

    # the soft threshold uses the mean,
    # the conservative one Otsu's method:
    th = min(graythresh(b),\
                b.mean()/b.max()) if SoftThreshold else graythresh(b)
    #end generate threshold

    # now, we want to filter the results for noise / other problems
    #
    # here we filter along the alpha axis, so it does not matter what
    # we get as a result in the masked areas...
    # filter with b (mask is already imposed)
    bindx = b > th*b.max()

    # do we have size filtering activated?
    if config['MinSize'][-1] > 1 or config['MaxSize'][-1] > 0:
        rep.write('calling size filter')
        bindx = bwlabel(bindx,
                        # MinSize is at least 1 (default)
                        MinSize= max(int(config['MinSize'][-1]), 1),
                        # default MaxSize is -1, we need 0 for being not set
                        MaxSize= max(int(config['MaxSize'][-1]), 0)
                        ) > 0

    # convert the image to a mask
    # otherwise python3 makes some funny stuff...
    bindx = bindx.astype(bool)

    print("it took: %.3f seconds" %(time()-tt))
    rep.write("Selected intensity threshold is", th, color='green')

    # Output filtered angle and intensity images
    # pl.figure(1)

    pl.clf();
    pl.gray();
    # img is still the background corrected image after compression
    pl.imshow(img)
    pl.axis('off')
    pl.title("Filtered image %s" %fn)
    fout = os.path.join(outdir, f"{ffn}-filtered-image{ext}")
    pl.savefig(fout, dpi= dpi, bbox_inches="tight", pad_inches=0)


    # filter the angle image to the meaningful amplitudes only:
    aimg = aimg*bindx
    # because of this point, the masking is incorporated in the meaningful
    # alpha values, thus no worries...

    ####################
    # More output:
    #pl.figure(3);
    pl.jet()
    pl.imshow(aimg)
    pl.axis('off')
    pl.title('alpha image')
    fout = os.path.join(outdir, f"{ffn}-alpha-image.png")
    pl.savefig(fout, dpi= dpi, bbox_inches="tight", pad_inches=0)

    #pl.figure(2)
    pl.clf();
    if masked:
        pl.imshow(b)
    else:
        pl.imshow(b)
    pl.gray()
    pl.axis('off')
    pl.title('Max. image')
    fout = os.path.join(outdir, f"{ffn}-max-image.png")
    pl.savefig(fout, dpi= dpi, bbox_inches="tight", pad_inches=0)

    # try a composite
    fout = os.path.join(outdir, f'{ffn}-max-filtered-composite.png')
    pl.clf()
    pl.imshow(composite(b, img, b))
    pl.axis('off')
    pl.title('max-filtered-composite')
    pl.savefig(fout, dpi= dpi, bbox_inches="tight", pad_inches=0)

    dalpha2 = (alpha[1] - alpha[0])/2.0
    #we shift the pocket boundaries down a bit to accomodate
    #the real alpha values to the midpoints
    bins = append(alpha, alpha_max) - dalpha2
    ais = aimg[bindx]   #the rest is set to 0 and meaningless.

    #store some information 1:
    res_row['Average angle (deg.)'] = ais.mean()
    #here the histogram can be over broad, because of wrapping around
    #angles... std() we derive after reorientation


    h = hist(ais, bins = bins)

    pl.figure(4)
    pl.clf()
    pl.axes(polar=True)
    pl.plot(h['midpoints']*pi/180,h['dist'],'bo-')
    da = 180/Nangles
    #pl.thetagrids(arange(-180+da,180+da,da),labels=None)
    pl.thetagrids(arange(0,360,da),labels=None)
    pl.title('angle histogram')
    fout = os.path.join(outdir, f"{ffn}-angle-histogram.png")
    pl.savefig(fout, dpi= dpi)
    #dump the original angles:
    fout = os.path.join(outdir, f"{ffn}-angles.txt")
    SaveData(['angle'],zip(ais), fout, \
            "Angle values obtained")

    #rotate to the major peak:
    central =  h['midpoints'][ h['dist'] == h['dist'].max()][0]
    ais = ais - central
    #all those falling to the 3rd quarter get back to the top
    ais[ ais < -90] += 180.0
    #all those in the 2nd quarter back to the bottom:
    ais[ ais > 90 ] -= 180.0

    #recalculate the histogram:
    #but first recalculate the bind: central is one of the
    #actual alpha values...
    alpha2 = alpha - central
    alpha2[ alpha2 > 90] -= 180
    alpha2[ alpha2 < -90] += 180
    alpha2.sort()
    #we do not know where alpha2 now ended, but we add a bin wall
    #above it. And again, shift that all values are in the middle
    bins2 = append(alpha2, alpha2.max() + 2.0*dalpha2) - dalpha2

    h2 = hist(ais, bins=bins2)
    # alternatively, use the maximum:
    ind_max = (h2['dist'] == h2['dist'].max()).nonzero()[0][0]
    i0 = max(ind_max-2, 0)
    i1 = min(ind_max+2, len(h['midpoints']))
    # this is a refined 'average'
    # since we have the maximum around 0 and the distribution recentered, this sum
    # should work all right
    ais_max = (h2['midpoints'][i0:i1]*h2['dist'][i0:i1]).sum()/h2['dist'][i0:i1].sum() + central
    # rotate it back to the original direction
    # if somehow it gets out of range, correct it:
    if ais_max > 90:
        ais_max -= 180

    elif ais_max < -90:
        ais_max += 180

    res_row['Angle max (deg.)'] = ais_max

#    h['midpoints'] = h['midpoints'] - central
#    h['pockets'] = h['pockets'] - central

    pl.figure(4)
    pl.clf()
    pl.axes(polar=True)
    pl.plot(h2['midpoints']*pi/180,h2['dist'],'bo-')
    da = 180/Nangles
    #pl.thetagrids(arange(-180+da,180+da,da),labels=None)
    pl.thetagrids(arange(0,360,da),labels=None)
    pl.title('angle histogram')
    fout = os.path.join(outdir, f"{ffn}-rotated-angle-histogram.png")
    pl.savefig(fout, dpi= dpi)

    fout = os.path.join(outdir, f"{ffn}-rotated-angles.txt")
    SaveData(['angle'],zip(ais), fout, \
            "Angle values relative to their maximum (rotated)")

    fout = os.path.join(outdir, f"{ffn}-histogram-data.txt")

    SaveData(h.keys(), zip(*h.values()), fout, \
            "Angle histogram")

    #add normalized to the histogram:
    h2['norm_dist'] = h2['dist']/float( h2['dist'].sum() )

    fout = os.path.join(outdir, f"{ffn}-rotated-histogram-data.txt")
    SaveData(h2.keys(), zip(*h2.values()), fout, \
            "Rotated angle histogram")

    #append the radius of gyration = standard deviation:
    #ais got recentered since...
    aistd = ais.std()
    #distribution width:
    res_row['Standard dev. (deg.)'] = aistd

    #store the FWHM of the histogram:
    aisfwhm = FWHM(h2['midpoints'],h2['dist'])
    res_row['FWHM (deg.)'] = aisfwhm
    #we can also use quantiles to analyze width
    quants = quantile( ais, asarray((0.5- Qwidth/2, 0.5+Qwidth/2)))
    res_row['Quantile (deg.)'] = quants[1]-quants[0]

    rep.write("Distribution widths are std:", aistd, "\tFWHM:", aisfwhm, '\tQuant:', quants[1]-quants[0])

    #dump numpy objects?
    if dump:
        fout = os.path.join(outdir, f"{ffn}-numpy-dump.npz")
        savez_compressed(fout, alpha= aimg, maximg= b, indx= bindx )

    if not res_keys:
        res_keys= list(res_row.keys())

    # res.append(list(res_row.values()))

    SaveData(res_keys,
         [list(res_row.values())],
         os.path.join(outdir, "Angle-widths.txt"),
         "Direction and standard deviation of angle histograms",
         append = True)
#end for...

#dump summary about all histograms processed here:
#Rg = asarray(Rg)
#avg_angle = asarray(avg_angle)

rep.write("Done, results are all saved")
rep.close()

