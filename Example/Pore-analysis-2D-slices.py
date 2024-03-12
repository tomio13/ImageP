#!/usr/bin/env python
""" Analyze the space between objects in 3D stacks, taking each slice and summarizing their
    distributions together into one.
    The method is pseudo 3D, because the Z-axis (stack axis) is not considered.
    Based loosely  on the publication of:
    Donald E. Griffin ... Tatiana Segura, Nature Materials vol. 14: 737 - 744 (2015)
    and the published Matlab code in their supplementary information.

    Author: Tamás Haraszti
    Email: haraszti@dwi.rwth-aachen.de
    Date: 2020-11-01
    Licence: CC(4)
    Waranty: None

    Dev. notes:
        2022-03-24: shift the edge removal after the skeletonization to minimize artificial new
                skeleton lines

    2023-08-25: add dilation to background removal
"""

from matplotlib.pyplot import ion, ioff
from ImageP import *
from BatchAnalyzer import *
from numpy import zeros, sqrt, savetxt, quantile
from glob import glob
import sys
import os

# turn off interactive plotting
ioff()
configfile= 'config.txt'
conf = {'dir': './', 'outdir': 'dir', 'fmask':'*.tif', 'ext':'.png', 'dpi': 150,
        #'MinSize': 250, 'MaxSize': 1E5,
        'ColorImage': True,
        'RGaussBg': -1, 'WGaussBg': -1, 'RGaussSm': -1, 'WGaussSm': -1, 'Ndilate': 0,
        'quantile': -1, 'threshold': -1, 'N': -1,
        'bg_dilate': 0, 'MinBgSize': 5000,
        'scaler': 1.0,
        'UseSkel': False, 'StHeight': 16,
        'CutEdges': 0, 'FilterPeriphery': True
        }


if __name__ !="__main__":
    print ("this script should be run self standing!")

else:
    args = sys.argv[1:]
    if len(args) > 0:
        configfile = args[0]
        print("Received config file name:", configfile)
#end if main probram
if not os.path.isfile(configfile):
    print('Unable to open config file:', configfile)
    print('Usage: program configfile')
    sys.exit(0)
#end of checking config file
config = ReadConf(configfile, conf)

###############################################################################
# Manage configuration options
indir = config['dir'][-1]
indir = os.path.abspath(indir)

outdir = config['outdir'][-1] if config['outdir'][-1] != 'dir'\
                            else '%s-Results' %indir

outdir = os.path.abspath(outdir)

if not os.path.isdir(outdir):
    os.mkdir(outdir)
# end make sure of outdir

fmask = config['fmask'][-1]
dpi = config['dpi'][-1]
ext= config['ext'][-1]

ColorImage = bool(config['ColorImage'][-1])
Rb = config['RGaussBg'][-1]
Wb = config['WGaussBg'][-1]
Rg = config['RGaussSm'][-1]
Wg = config['WGaussSm'][-1]

#use quantile for threshold:
q = config['quantile'][-1]

th = config['threshold'][-1] #if q is not defined
# MinSize = int( config['MinSize'][-1] )
# MaxSize = int( config['MaxSize'][-1] )
n_dilate = int(config['Ndilate'][-1]) #edge width + 1 or so
bg_dilate= int(config['bg_dilate'][-1]) # for large background removal
MinBGSize = int(config['MinBGSize'][-1]) # for large background removal, minimum size

UseSkel = bool(config['UseSkel'][-1])
StHeight= config['StHeight'][-1]

scaler = config['scaler'][-1]

CutEdges = int(config['CutEdges'][-1])
FilterPeriphery = bool(config['FilterPeriphery'][-1])

N = int(config['N'][-1])

# Get the file list:
lst = glob(os.path.join(indir, fmask))
lst.sort()
Nlst = len(lst)

if Nlst < 1:
    print('No data found error')
    sys.exit(0)

if N > 0 and N < Nlst:
    lst = lst[:N]
else:
    N = Nlst
########################################################
# Report parameters:
rep = Report(outdir, header='Pore analysis of fluorescence images 1.2', add_time= False)

rep.write('File path', indir)
rep.write('Results folder:', outdir)
rep.write(N, 'files to read', color='green')

rep.write('Image scale is set to', scaler, 'microns/pixel')

if ColorImage:
    rep.write('Handle color images with independent channels')
else:
    rep.write('Images are converted to grayscale at loading')

if Rb >0 :
    rep.write('Gaussian deblurr parameters')
    rep.write('Background window radius:', Rb)
    rep.write('Background width:', Wb)
    rep.write('Smoothing window radius:', Rg)
    rep.write('Smoothing width:', Wg)
else:
    rep.write('Do not correct background')

if q > 0:
    rep.write('Threshold at quantile', q)
elif th < 1:
    rep.write('Relative threshold:', th)
else:
    rep.write('Threshold intensities above', th)

#rep.write('Detect patches larger than:', MinSize,
#        'pixels and smaller than', MaxSize, 'pixels in area')

rep.write('Dilate / errode fuse pixels', n_dilate, 'times in patches')

if UseSkel:
    rep.write('Use skeletonisation to find pore sizes')
else:
    rep.write('Use a local maxima search to find pore sizes')
    rep.write('Hehght step to detect is set to', StHeight)

if FilterPeriphery:
    rep.write('Cutting peripheral areas outside the object extremes is set')
    rep.write('using', bg_dilate,'steps to fuse the structure')
    rep.write('BG is a structure containing more than', MinBGSize,'pixels')
else:
    rep.write('Edges are not filtered')

rep.write('*************************************************************')
rep.write('Processing images', color='cyan')

###########################################################
# Main work starts here

dsts = []
Nvoid_stack= 0
Nall_stack = 0
voidlist = []
for fn in lst:
    img = read_img(fn, asis= ColorImage)

    fnn = os.path.splitext(os.path.split(fn)[-1])[0]

    if img is None:
        rep.write('Image', fn, 'not found', color='red')
        continue
    else:
        rep.write('Processing', fn, color='green')

    if ColorImage:
        # we have a color image, where the first two channels are
        # two independent recording
        # we scale them to the same maximum before adding them up
        maxes = img.max(axis=(0,1))
        maxmax = maxes.max()
        for m in range(3):
            if maxes[m] > 0:
                img[:,:,m] = maxmax* img[:,:,m]/maxes[m]

        # now, we add the channels together
        img = img.sum(axis=2)

    if CutEdges > 0:
        img = img[CutEdges:-CutEdges, CutEdges:-CutEdges]

    img = img - img.min()

    if Rb > 0 and Wb > 0:
        rep.write('Applying Gaussian deblurring')
        img = GaussDeblurr(img, Rb, Wb, Rg, Wg, True)
    elif Rg > 0 and Wg > 0:
        rep.write('Smoothening the image')
        img = ConvFilter1D(img, GaussKernel(Rg, Wg, OneD= True))
    # end of applying filters

    # now, check for threshold definition:
    if q >= 0 and q <= 1.0:
        curr_th = quantile(img, q)
    elif th < 0:
        curr_th = graythresh(img)*img.max()
    else:
        curr_th = th*img.max()
    #end  if
    rep.write('Applying threshold with intensity:', curr_th)
    # we highlight the empty part:
    bimg = img > curr_th

    # dilate the structures
    bimg = SimpleDilate(bimg, n_dilate)
    bimg = SimpleErode(bimg, n_dilate)

    # turn the pores to the structure:
    bimg = 1 - bimg
    # bwimg = bwlabel(bimg, MinSize= MinSize, MaxSize= MaxSize)
    # rep.write('Found', bwimg.max(), 'background structures')

    # one measure is how much of this image is background:
    Nvoid = bimg.sum()
    Nimg = img.size
    rep.write('Void pixels:', Nvoid, 'all pixels:', Nimg)
    rep.write('Image porosity:', float(Nvoid)/Nimg, color='cyan')

    print('start distance filter')
    dstimg = DistanceFilter(bimg.astype(int))
    print('end distance filter')

    print('Finding local maxima')
    if UseSkel:
        t = Skel(dstimg)
    else:
        # height is a parameter to use to ignore too small steps in local height
        # we use the quadratic image, so StHeight may have to be increased quadratically
        t = FindLocalMaxima(dstimg, height= StHeight)

    # reject the periphery, make a negative of the perihperal areas,
    # that we can use as a binary mask:
    if FilterPeriphery:
        rep.write('Removing all outside the object (edges), in distance and skeleton images',
                  color='green')
        perimg = SimpleErode(SimpleDilate(1 - bimg, bg_dilate), bg_dilate) \
                    if bg_dilate > 0 else 1 - bimg
        perimg = SimpleContour(bwlabel(perimg, MinSize= MinBGSize) >0, 0, True)
        dstimg = dstimg * perimg
        Nimg = perimg.sum()
        Nvoid = (bimg*perimg).sum()
        t = t*perimg
        rep.write('Correcting void area for detected object area', color='cyan')
        rep.write('Void pixels:', Nvoid, 'all pixels:', Nimg)
        rep.write('Image porosity (override):', float(Nvoid)/Nimg, color='green')

    # store the void and valid pixels, but here so
    # correction could also take place if needed!
    Nvoid_stack += Nvoid
    Nall_stack += Nimg
    voidlist.append([Nvoid, Nimg, float(Nvoid)/Nimg])

    fout = '%s-detected%s' %(fnn, ext)
    display(composite(t, 255*img/img.max(), 1-bimg),
            fname= os.path.join(outdir, fout), dpi= dpi)

    fout = '%s-detected-clearcut%s' %(fnn, ext)
    display(composite(t, t, bimg),
            fname= os.path.join(outdir, fout), dpi= dpi)

    # a pore size is 2x the minimal distance to its 'middle'
    # we assume the middle is what we found with the local distance maxima or skeleton
    thks = 2.0*sqrt(dstimg[(t > 0) & (dstimg > 0)])*scaler

    # turn to 1D array:
    thks = thks.ravel()
    if thks.size > 10:
        h_local = hist(thks, min(50, int(thks.size/3)))

        fout= '%s-pore-histogram-table.txt' %fnn
        SaveData(h_local.keys(), zip(*h_local.values()),
            os.path.join(outdir, fout),
            'Pore sizes in microns for the slice')
        # plot it out as well
        Plot(h_local['midpoints'], h_local['dist'],
                fmt='-',
                title= fnn,
                xlabel= 'radius, $\mu$m',
                ylabel= 'counts',
                ext= ext,
                filename= f'{fout}-histogram',
                outpath= outdir,
                # draw style passed to pl.plot:
                ds = 'steps-mid'
                )


    else:
        rep.write('local point list is too short, not analyzed further')

    # dump the values independent of their length:
    fout = '%s-pore-size-list.txt' %fnn
    savetxt(os.path.join(outdir, fout),
        thks,
        fmt= '%.2e',
        header='Pore sizes in microns, estimated with maximal closest distances')

    # append the result to the big pool
    dsts.extend(thks.tolist())
# end for files in lst

rep.write('Stack void contains', Nvoid_stack, 'void voxels in',
    Nall_stack, 'voxels')
rep.write('Porosity is:', float(Nvoid_stack)/Nall_stack*100, '%', color= 'green')

h = hist(dsts, bins= min(50, int(len(dsts)/3)))

fout= 'Stack-pore-histogram-table.txt'
SaveData(h.keys(), zip(*h.values()), os.path.join(outdir, fout),
        'Pore sizes in microns for the stack')

SaveData(['Nvoid', 'Nimg', 'ratio'], voidlist,
        os.path.join(outdir, 'Porosity-stack-table.txt'),
        f'Porosity of images; Whole stack is: {float(Nvoid_stack)/Nall_stack*100:.3f} %')

# plot it out as well
Plot(h['midpoints'], h['dist'],
        fmt='-',
        title= fnn,
        xlabel= 'radius, $\mu$m',
        ylabel= 'counts',
        ext= ext,
        filename= f'Stack-pore-histogram',
        outpath= outdir,
        # draw style passed to pl.plot:
        ds = 'steps-mid'
        )
rep.write('Done')
rep.close()
# EOF
