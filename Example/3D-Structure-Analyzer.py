#!/usr/bin/env python
#test analysis for 3D image stacks
from BatchAnalyzer import SaveData, ReadConf, Report
from ImageP import *
from numpy import asarray, unique, zeros, minimum, inf, int16, int, sqrt, arange
from matplotlib import pyplot as pl
from glob import glob
from PIL import Image
import os, sys

#indir = '4000 Fragment 1.2'
#indir= "250 Fragment 1.3"
#indir= '2000 Fragment 1.3'
#indir= '4000 Fragment 1.3'

configfile = 'config.txt'
conf = {'indir': './', 'outdir':'dir', 'fmask': '*.tif', 'dpi': 150,\
        'ext': '.png', 'MinSize': 200, 'gap': 3,\
        'RGauss': -1, 'WGauss': 25, 'RsGauss': -1, 'WsGauss': 0,\
        'Rdistance': 5, 'Wdistance': 0.75,\
        'Rball': -1, 'Nfill': -1, 'threshold': -1, 'scale_xy': 1.0,\
        'scale_z': 1.0}

if __name__ =="__main__":
    args = sys.argv[1:]
    if len(args) > 0:
        configfile = args[0]
        print("Received config file name:", configfile)
#end if main probram

if not os.path.isfile(configfile):
    print('Invalid config file name:', configfile)
    print('Usage: program configfile')
    sys.exit(0)
#end of checking config file
conf = ReadConf(configfile, conf)

###############################################################################
# Manage configuration options
indir = conf['indir'][-1] if 'indir' in conf else conf['dir']
outdir = conf['outdir'][-1] if conf['outdir'][-1] != 'dir'\
                            else '%s-Results' %indir

lst = glob(os.path.join( os.path.abspath(indir), conf['fmask'][-1]))
lst.sort()
if len(lst) < 1:
    print('Images not found error')
    sys.exit(0)

#if the output directory does not exist, create it:
if not os.path.isdir( outdir ):
    os.mkdir(outdir)
#end if

#Background correction parameters:
WGauss = conf['WGauss'][-1]
RGauss = int(conf['RGauss'][-1]) if conf['RGauss'][-1] > 0 else 3*WGauss
WsGauss = conf['WsGauss'][-1]
RsGauss = int(conf['RsGauss'][-1]) if conf['RsGauss'][-1] > 0 else 3*WsGauss

Rball = conf['Rball'][-1]

MinSize= int(conf['MinSize'][-1])
gap = int(conf['gap'][-1])
Nfill = int(conf['Nfill'][-1])
#parameters of finding the maxima in the distance filtered image
#used by the FindLocalMaxima
rd = int(conf['Rdistance'][-1])
#rw = 1.0
wd = conf['Wdistance'][-1]
th = conf['threshold'][-1]

#scales in micron:
scale_xy = conf['scale_xy'][-1]
scale_z = conf['scale_z'][-1]

###############################################################################
#start reporting, summarize parameters
rep = Report(outdir, add_time= False, header='3D Skeleton analyzer 0.2')
rep.write('Configuration parameters set:')
for i in conf:
    rep.write(i,':', conf[i])
rep.write('*********************************')
rep.write('Input directory', indir)
rep.write('Found', len(lst), 'images')
rep.write('Background correction:')
if Rball > 0:
    rep.write('Rolling ball is set with radius', Rball)
else:
    rep.write('Gauss deblurr radius', RGauss,'sigma:', WGauss)
    rep.write('Smoothing radius in Gauss deblurr', RsGauss, 'sigma:', WsGauss)
#end if rolling ball is set
rep.write('Filling gaps with dilate and erode filters', Nfill, 'times')
rep.write('Patch detection uses', gap, 'pixels gap tolerance')
if th < 0:
    rep.write("Images are thresholded with Otsu's method")
else:
    rep.write('Thresnolding uses', th, 'times the image maximum')
#end if threshold is set

rep.write('Finding maxima in distance images usess a Gaussian peak detector')
rep.write('Radius:', rd, 'pixels, width:', wd, 'pixels')

rep.write('Images are scaled by', scale_xy, 'micron/pixel')
rep.write('Z-direction is scaled by', scale_z, 'micron/pixel')
rep.write('**************************************************')

###############################################################################
# Process data: the actual work starts here
#'a' stores the patch stack
#imgstack the intensity images
a = []
imgstack = []

patchcounter = 5
for fn in  lst:
    rep.write("Reading",fn, color='cyan')
    #crop off the scale bar
    img = read_img(fn)

    if Rball > 0:
        imgb = img - RollingBall(img, Rball)
        imgb[ imgb < 0] = 0
    elif WGauss > 0:
        imgb = GaussDeblurr(img, RGauss, WGauss, RsGauss, WsGauss,\
                KillZero= True)
    else:
        img = img - img.min()

    if th < 0:
        th = graythresh(imgb)
        rep.write('Otsu\'s threshold:', th)
    #end if th
    imgc = imgb > th*imgb.max()

    # stack up the images for 3D handling
    imgstack.append(imgb*imgc)

    #imgb = bwlabel(imgb, MinSize= MinSize, gap= gap, MaxSize = 0.75*imgb.size)
    imgb = bwlabel(imgc, MinSize= MinSize, gap= gap)
    if Nfill > 0:
        tmpimg = imgb > 0
        for i in range(Nfill): tmpimg = SimpleDilate(tmpimg)
        for i in range(Nfill):  tmpimg = SimpleErode(tmpimg)
        imgb = bwlabel(tmpimg, MinSize= MinSize, gap= gap)
    #end using Nfill
    # convert the patch numbers to a stack wide individual counter
    imgb[imgb > 0] = imgb[imgb >0 ] + patchcounter
    patchcounter = imgb.max()
    # the label images stacked
    a.append(imgb)
#end reading fn


# it is faster to handle the 3D stack as one ndarray object
a = asarray(a)

#start cleaning up, pulling confluent 3D patches together.

#use patchcounter 1 to higher to decouple the actual patch values
#from the continuous numbering we want through the stack
patchcounter = 0
for i in range(a.shape[0]-1):
    # what unique labels do we have in this array?
    jlist = unique(a[i,:,:][a[i,:,:] > 0])

    # we scan the patches and check if we have points above / below
    for j in jlist:
        #we may have a patch we updated from the previous image:
        if j > patchcounter:
            patchcounter += 1
            curr_patch = patchcounter
        else:
            curr_patch = j

        act_i = i
        next_i = i+1
        swapped = False
        # which points overlap or are boundary with this object?
        # we extend one pixel and get the overlap, this way
        # we get diagonal neighbors as well:
        patch = SimpleDilate( a[act_i,:,:] == j )
        hits = unique( a[next_i,:,:][ patch > 0 ] )
        # fill back with the consecutive patch number:
        a[act_i,:,:][a[act_i,:,:] == j] = curr_patch

        #remove the background value: 0
        hits = hits[ hits > 0 ] #no hits, go to next value
        while( hits.size > 0):
            print("at i:", i, "we have patch", j, "hits", hits.size)
            for k in  hits:
                #fill over those who belong:
                a[next_i,:,:][ a[next_i,:,:] == k ] = curr_patch
            #end of filling up next image
            #now, check back:
            patch = SimpleDilate( a[next_i, :,:] == curr_patch )
            hits = unique( a[act_i, :,:][ patch > 0])
            #remove circular reference
            hits = hits[ (hits > 0) & (hits != curr_patch) ]
            if swapped:
                swapped = False
                act_i = i
                next_i = i+1
            else:
                swapped = True
                act_i = i+1
                next_i = i
            #end if swapped
        #end filling while there are hits
    #end scanning image1
#end of go through image stack

#second round:
# we want to reiterate, so we shift all
# current labels over the value of patchcounter
# this way the new values below it will be distinct
a[a>0] = a[a>0] + patchcounter + 1

patchcounter = 0
for i in range(1, a.shape[0]):
    jlist = unique( a[-i,:,:][ a[-i,:,:] >0 ])
    for j in  jlist:
        # old values are above patchcounter
        # is this one not hit in this run?
        if j > patchcounter:
            # make a new value available
            patchcounter += 1
            curr_patch = patchcounter
        else:
            curr_patch = j
        #end if it is a new patch

        # we walk downwards:
        act_i = -i
        next_i = -i-1
        # again check for diagonal matches too:
        patch = SimpleDilate(a[act_i,:,:] == j)
        hits = unique(a[next_i, :,:][patch > 0])
        a[act_i,:,:][a[act_i,:,:] == j] = curr_patch

        #now, we have to overwrite one set only,
        #the previous sweep eliminated the local oscillations
        hits = hits[hits > 0]
        print("at i:", i, "we have patch", j, "hits", hits.size)
        for k in hits:
            a[next_i,:,:][ a[next_i,:,:] == k ] = curr_patch
        #end writing patches
    #end scanning the patchlist in a[-i,:,:]
#end scanning the stack backwards

fout = os.path.join(outdir, 'All-labeled-pixels-list.txt')
x,y,z = (a>0).nonzero()
SaveData(['z','x','y','i'], zip(*[x,y,z,a[x,y,z]]), fout,\
            '3D coordinate dataset with intensity')


outpath = os.path.join(outdir, 'labeled-stack')
if not os.path.isdir( outpath ):
    os.mkdir( outpath )

for i in range(a.shape[0]):
    im = Image.fromarray(a[i,:,:])
    im.save(os.path.join(outpath, "%d.png" %i))
#end dumping images

#3D distance transform:

def DistanceFilter3D(a, only_2d= False):
    """ Run the distance transform in 3D on a stack of images
        the first index is the z = stack index!
        Return the transformed 3D image
    """
    b1 = zeros(a.shape, dtype= a.dtype)

    print("L2 transform along x,y")
    for i in range(a.shape[0]):
        print("z:",i)
        #Distance filter will set all points which are > 0
        b1[i,:,:] = DistanceFilter(a[i,:,:])
    #end distance filter along y

    if only_2d:
        return(b1)

    #this should be the 3D distance filter
    b2 = zeros(b1.shape, dtype=b1.dtype)
    print("L1 transform along z")
    for i in range(a.shape[1]):
        print("x:",i)
        for j in range(a.shape[2]):
            b2[:,i,j] = DistanceFilter1D(b1[:,i,j])
    #end distance filter along z

    return(b2)
#End of DistanceFitler3D

# let us get a gradient:

def FindMaxima(b1, rd= rd, wd= wd, height= 1 ):
    """ Find the maxima location in the 3D stack, and mark them as 1
        Run the search only in 2D in the images!
        Parameters:
        b1: 3D image stack
        rd: radius of the Gauss derivative kernel
        wd: sigma of the Gauss derivative kernel
        Return a new 3D stack with these
    """
    c = zeros( b1.shape, dtype= int16)

    #we find the 2D maxima in each slice:
    print("Finding peaks")
    for i in range(b1.shape[0]):
        c[i,:,:] = FindLocalMaxima(b1[i,:,:], rd, wd, height)
    #end finding peaks
    return(c)
#end find Maxima

def GenerateField(a, seed, fmin= 2):
    """ Generate a distance like field, of the nonzero pixels based on
        a seed point list.
        This field does not touch the boundary pixels, assigns values
        to the nonzero pixels as minimum of their old value vs.
        if we come from a pixel with value n, then
        n+N, where N is
            fmin for one touching at the side
            fmin+1 for one touching at an edge
            fmin+3 for one touching at a corner

        The result is a kind of New York map like distance, which
        gives the number of steps needed to a specific point along
        straight lines (if fmin = 1).

        Parameters:
        a       a 3D image stack, z is first index
        seed    list of 3D points to seed the assignment
        fmin:   2 or 3 typically for the indexing values

        returns a 3D array of the new values
    """

    print("Generating fiels")
    b = zeros(a.shape)
    #set the valid pixels to practical Inf in int:
    b[ a > 0] = inf

    Ni,Nj,Nk = b.shape
    #the indexes of  neighbours for quick access:
    #        0,  1, 2,  3, 4,  5,  6, 7, 8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 29, 21, 22, 23, 24, 25
    # :6 --> side, 6:18 --> edge, 18: --> corner
    iindx = [1, -1, 0,  0, 0,  0,  1, 1, 0, -1,  0,  1, -1, -1,  0,  1,  0, -1,  1, -1,  1,  1,  1, -1, -1, -1]
    jindx = [0,  0, 1, -1, 0,  0,  1, 0, 1,  1, -1,  0, -1,  0, -1, -1,  1,  0,  1,  1, -1,  1, -1,  1, -1, -1]
    kindx = [0,  0, 0,  0, 1, -1,  0, 1, 1,  0,  1, -1,  0, -1, -1,  0, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1]

    iindx = asarray(iindx); jindx= asarray(jindx); kindx= asarray(kindx)

    searchlist = []
    for [i,j,k] in zip(*seed):
        print( "testing seed;", i,j,k)

        if a[i,j,k] == 0:
            continue
        #set the boundary point
        b[i,j,k] = 1;
        val = 0

        #first run, find neighbours:
        #side hits:
        si = i+iindx[:6]
        sj = j+jindx[:6]
        sk = k+kindx[:6]
        print(si,sj,sk)
        #Border patrol... do not walk off the stack:
        keepindx = (si >= 0) & (si < Ni) & (sj >= 0) & (sj < Nj) & (sk >= 0) & (sk < Nk)
        #we can use minimum() to compare two arrays and keep their shape
        #this should be faster than a for loop in python
        #for comparison: what were the new values?
        newval = zeros( keepindx.sum()) + fmin + val
        #what are the actual ones?
        currval = b[ si[keepindx], sj[keepindx], sk[keepindx] ]
        b[ si[keepindx], sj[keepindx], sk[keepindx] ] = minimum( currval, newval)
        #we changed the values, but we have not checked their neighbours:
        searchlist.extend([segment for segment in zip(*[si[keepindx], sj[keepindx], sk[keepindx]])])

        #edge hits:
        si = i+iindx[6:18]
        sj = j+jindx[6:18]
        sk = k+kindx[6:18]
        keepindx = (si >= 0) & (si < Ni) & (sj >= 0) & (sj < Nj) & (sk >= 0) & (sk < Nk)
        newval = zeros( keepindx.sum()) + val + fmin + 1
        currval = b[ si[keepindx], sj[keepindx], sk[keepindx] ]
        b[ si[keepindx], sj[keepindx], sk[keepindx] ] = minimum( currval, newval)
        searchlist.extend([segment for segment in zip(*[si[keepindx], sj[keepindx], sk[keepindx]])])

        #corner hits:
        si = i+iindx[18:]
        sj = j+jindx[18:]
        sk = k+kindx[18:]
        keepindx = (si >= 0) & (si < Ni) & (sj >= 0) & (sj < Nj) & (sk >= 0) & (sk < Nk)
        newval = zeros( keepindx.sum()) + val + fmin+2
        currval = b[ si[keepindx], sj[keepindx], sk[keepindx] ]
        b[ si[keepindx], sj[keepindx], sk[keepindx] ] = minimum( currval, newval)
        searchlist.extend([segment for segment in zip(*[si[keepindx], sj[keepindx], sk[keepindx]])])

    #end initializing seed points

    #now we have the starting point list
    print("walking the path")
    while( searchlist != []):
        #take the first element

        i0,j0,k0 = searchlist.pop(0)
        val = b[i0,j0,k0]
        #print('center:',i0,j0,k0, 'value:',val)

        #first run, find neighbours:
        #side hits:
        si = i0+iindx[:6]
        sj = j0+jindx[:6]
        sk = k0+kindx[:6]
        #These were not yet investigated:
        keepindx = (si >= 0) & (si < Ni) & (sj >= 0) & (sj < Nj) & (sk >= 0) & (sk < Nk)
        si = si[keepindx]
        sj = sj[keepindx]
        sk = sk[keepindx]
        keepindx = b[si,sj,sk] > 0
        visitindx = b[ si, sj, sk] == inf
        #and assign values to the nonzero ones only
        #but we assign our values to all neighbours
        newval = zeros( keepindx.sum()) + val + fmin
        currval = b[ si[keepindx], sj[keepindx], sk[keepindx] ]
        b[ si[keepindx], sj[keepindx], sk[keepindx] ] = minimum( currval, newval)
        searchlist.extend([segment for segment in zip(*[si[visitindx], sj[visitindx], sk[visitindx]])])

        #edge hits:
        si = i0+iindx[6:18]
        sj = j0+jindx[6:18]
        sk = k0+kindx[6:18]
        keepindx = (si >= 0) & (si < Ni) & (sj >= 0) & (sj < Nj) & (sk >= 0) & (sk < Nk)
        si = si[keepindx]
        sj = sj[keepindx]
        sk = sk[keepindx]
        keepindx = b[si,sj,sk] > 0
        visitindx = b[ si, sj, sk] == inf
        newval = zeros( keepindx.sum()) + val + fmin +1
        currval = b[ si[keepindx], sj[keepindx], sk[keepindx] ]
        b[ si[keepindx], sj[keepindx], sk[keepindx] ] = minimum( currval, newval)
        searchlist.extend([segment for segment in zip(*[si[visitindx], sj[visitindx], sk[visitindx]])])

        #corner hits:
        si = i0+iindx[18:]
        sj = j0+jindx[18:]
        sk = k0+kindx[18:]
        keepindx = (si >= 0) & (si < Ni) & (sj >= 0) & (sj < Nj) & (sk >= 0) & (sk < Nk)
        si = si[keepindx]
        sj = sj[keepindx]
        sk = sk[keepindx]
        keepindx = b[si,sj,sk] > 0
        visitindx = b[ si, sj, sk] == inf
        #keepindx = (si >= 0) & (si < Ni) & (sj >= 0) & (sj < Nj) & (sk >= 0) & (sk < Nk) & (b[si,sj,sk] > 0)
        #visitindx = (si >= 0) & (si < Ni) & (sj >= 0) & (sj < Nj) & (sk >= 0) & (sk < Nk) & (b[ si, sj, sk] == inf)
        newval = zeros( keepindx.sum()) + val + fmin + 2
        currval = b[ si[keepindx], sj[keepindx], sk[keepindx] ]
        b[ si[keepindx], sj[keepindx], sk[keepindx] ] = minimum( currval, newval)
        searchlist.extend([segment for segment in zip(*[si[visitindx], sj[visitindx], sk[visitindx]])])

    #end going through searchlist

    b[ b == inf ] = 0
    return b
#end GenerateField

def SearchField(a, seed):
    """ Search the distance-like field for a minimum neighbour trajectory
        return a stack with the highlighted pixels
    """
    i0,j0,k0 = seed

    if( a[i0,j0,k0] <= 0):
        print("I search from maximum backwards. 0 seed is found")
        return (zeros(d.shape), [0,0,0])
    #end if
    Ni,Nj,Nk = a.shape

    b = zeros( a.shape, dtype='int16')
    iindx = [-1, 1, 0,  0, 0,  0,  1, 1, 0, -1,  0,  1, -1, -1,  0,  1,  0, -1,  1, -1,  1,  1,  1, -1, -1, -1]
    jindx = [0,  0, 1, -1, 0,  0,  1, 0, 1,  1, -1,  0, -1,  0, -1, -1,  1,  0,  1,  1, -1,  1, -1,  1, -1, -1]
    kindx = [0,  0, 0,  0, 1, -1,  0, 1, 1,  0,  1, -1,  0, -1, -1,  0, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1]

    iindx = asarray(iindx); jindx= asarray(jindx); kindx= asarray(kindx)
    searchlist = []
    centers = []
    hits = 1
    while(hits >0):
        b[i0,j0,k0] = 1
        centers.append([i0,j0,k0, a[i0,j0,k0]])
        #the path has 1 values only at the seeding points. We reached one, we arrived
        #if we do not check, the code will go on walking following the least neighbours until
        #we walked the whole volume (almost)
        if a[i0,j0,k0] == 1:
            print("We arrived to", i0, j0, k0, a[i0,j0,k0])
            hits = 0
            continue

        print('center at:', i0,j0,k0, a[i0,j0,k0])

        #first run, find neighbours:
        #side hits:
        si = i0+iindx
        sj = j0+jindx
        sk = k0+kindx
        checkindx = (si >= 0) & (si < Ni) & (sj >= 0) & (sj < Nj) & (sk >= 0) & (sk < Nk)
        #we do not want to hit an already visited point:
        keepindx = (b[ si[checkindx], sj[checkindx], sk[checkindx]] == 0) & \
                    (a[ si[checkindx], sj[checkindx], sk[checkindx]] > 0)
        #now shrink our search path:
        #si, sj, sk are still 26 elements, we have to remove the ones killed by checkindx and killindx
        si = si[checkindx][keepindx]
        sj = sj[checkindx][keepindx]
        sk = sk[checkindx][keepindx]

        if len(si) > 0:
            candidates = a[si, sj, sk]
            #print('candidates:', candidates)
            indx = (candidates <= a[i0,j0,k0]) & (candidates == candidates.min())
        else:
            indx = zeros(0)

        if  indx.sum() == 0:
            hits = 0
        else:
            i0 = si[indx][0]
            j0 = sj[indx][0]
            k0 = sk[indx][0]
    #end walking the path
    return (b, centers)
#end of SearchField

def CountNeighbours(a, cut= 2):
    """ Count the neighbours in a binary image,
        but return only values > cut.
        In a straight line every pixel has 2 neighbors.
        In a branch, we have > 2.
    """
    b = zeros( a.shape, dtype='int16')
    Ni, Nj, Nk = a.shape

    # define indices of all 26 neigbors. (Read the lines vertically.)
    iindx = [-1, 1, 0,  0, 0, 0,  1, 1, 0, -1,  0,  1, -1, -1,  0,  1,  0, -1,  1, -1,  1,  1,  1, -1, -1, -1]
    jindx = [0,  0, 1, -1, 0, 0,  1, 0, 1,  1, -1,  0, -1,  0, -1, -1,  1,  0,  1,  1, -1,  1, -1,  1, -1, -1]
    kindx = [0,  0, 0,  0, 1,-1,  0, 1, 1,  0,  1, -1,  0, -1, -1,  0, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1]

    iindx = asarray(iindx); jindx= asarray(jindx); kindx= asarray(kindx)

    for [i0,j0,k0] in zip(*(a.nonzero())):
        si = i0+iindx
        sj = j0+jindx
        sk = k0+kindx
        checkindx = (si >= 0) & (si < Ni) & (sj >= 0) & (sj < Nj) & (sk >= 0) & (sk < Nk)
        si = si[checkindx]
        sj = sj[checkindx]
        sk = sk[checkindx]
        neigh = a[si, sj, sk].sum()
        b[i0,j0,k0] = neigh if neigh > cut else 0
    #end for
    return b
#end of CountNeighbours

###############################################################################
########## work starts here
# take our beautiful labeles stack, and turn it to a distance map
b1 = DistanceFilter3D(a)

outpath = os.path.join(outdir, 'distance')
if not os.path.isdir(outpath):
    os.mkdir(outpath)

#export b1:
for i in range(b1.shape[0]):
    im = Image.fromarray(b1[i,:,:])
    im.save( os.path.join( outpath,"%d.png" %i))

#Scan the blobs next:
rep.write('Scanning blobs')

results = []
for i in range(1, a.max()+1):
    blobspace = zeros(a.shape)

    rep.write("Processing blob", i)

    blobindx = (a == i)
    z,x,y = blobindx.nonzero()

    if len(z) == 0:
        continue
    #end if the strucutre is empty

    lowest = z.min()
    highest = z.max()

    #size estimate:
    z0 = z.mean()
    x0 = x.mean()
    y0 = y.mean()

    #the maximal size of the object is:
    blob_R = sqrt(((z-z0)**2 + (x-x0)**2 + (y-y0)**2).max())

    #we could go for geometry, or for intensity
    #we go for geometry for now:
    maximg = (a[lowest,:,:] == i)

    #we use here a 2D local peak finding:
    maximg = DistanceFilter(maximg)
    #alternatively we could take b1[lowest,:,:]...
    #but we are at the bottom, so probably it is heavily flattened
    #then go for the local maxima:
    bimg = bwlabel(maximg == maximg.max())
    bmx = bimg.max()
    if bmx > 5:
        print("We have plenty of hits!")
        rep.write("area discrimination")
        area = zeros(bmx)
        for j in  range(1, bmx+1):
            area[j-1] = (bimg == j).sum()
        #end for size array
        area.sort()
        #refactor for the largest patches
        #MinSize is <= N
        # bimg = bwlabel(bimg >0, MinSize = area[-5])
        # pick the largest:
        bimg = bwlabel(bimg >0, MinSize = area[-1])
    #end if too many hits

    outpath = os.path.join( outdir, 'blob-%d-stack' %i)
    if not os.path.isdir( outpath ):
        os.mkdir(outpath)
    #end if no path

    blob_result = []
    track_indx = 1
    blob_length = 0.0

    for j in range(1, bimg.max() +1):
        #scan up, then down: first make a distance set
        #next find a shortest path back
        x,y = (bimg == j).nonzero()
        Ni = len(x)
        xi = x[int(Ni/2)]
        yi = y[int(Ni/2)]

        d= GenerateField(blobindx, [[lowest],[xi],[yi]], 3)
        # for a single point it should be equivalent to calculating
        # R**2 +1 from that point
        # now get a trajectory back:
        di, dj, dk = (d  > 0).nonzero()
        di = di.max()
        if di in [lowest-1, lowest, lowest+1]:
            rep.write("Too thin blob! Ignoring! max:", di, "lowest:", lowest)
            continue
        #end if

        dindx = (d[di, :,:] == d[di,:,:].max())
        dbimg = bwlabel(dindx > 0)
        #if there are many small hits, the mean is closest to
        #the image center (to some extenet)
        dj, dk = (dbimg == int(dbimg.max()/2+0.5)).nonzero()

        dN = len(dj)
        dj = dj[int(dN/2)]
        dk = dk[int(dN/2)]

        d1, trace = SearchField( d, [di, dj, dk])
        trace = asarray(trace)

        if trace.all() == 0:
            rep.write("this did not work")
#        elif trace[-1,0] != di:
#            print("Target not reached")
#            continue
        else:
            blobspace[d1 >0] = 1
            SaveData(['z','x','y','d'], trace,\
                os.path.join(outpath, 'trace-blob-%d-%d.txt' %(i,track_indx)),\
                'Blob trace with distance intensities')

        xyz = trace[:,:3]
        #first dimension is Z:
        xyz[:,0] = scale_z*xyz[:,0]
        xyz[:,1] = scale_xy*xyz[:,1]
        xyz[:,2] = scale_xy*xyz[:,2]
        dxyz = xyz[1:,:] - xyz[:-1,:]
        path_length = sqrt( (dxyz*dxyz).sum() )
        blob_length += path_length

        blob_result.append([track_indx,lowest, di, path_length])

        track_indx += 1
        #end if
    #end for maxima in blob j
    SaveData(['track ID', 'start slice', 'end slice', 'full length (micron'],
            blob_result, os.path.join(outpath, "Blob-summary.txt"),\
                    "Summary data of an individual blob")

    surface = 0
    for j in range(blobspace.shape[0]):
        im = Image.fromarray(blobspace[j,:,:].astype('int16') )
        im.save(os.path.join(outpath, '%d.png' %j))

        pimg = PerimeterImage(blobspace[j,:,:])
        surface += pimg.sum()
    #end exporting images

    #size estimate of blobspace in 3D:

    count = CountNeighbours(blobspace, cut= 2)
    # for now, we count only the points that have > 2 neighbours
    # this is overcounting, crowded crossings will be counted double
    Ncount = (count > 0).sum()

    #results.append([i,bimg.max(), (blobspace >0).sum(), (count >2).sum(), lowest, highest])
    results.append([i, track_indx-1, blob_length, \
            Ncount, lowest, highest, blob_R,\
            surface*scale_xy*scale_z,\
            blobspace.sum()*scale_z*scale_xy**2])

    rep.write('Blob maximal size:', blob_R)
    #rep.write("Length of trajectories:", results[-1][2])
    rep.write("Length of trajectories:", blob_length)
    rep.write("Number of branching points:", Ncount)
#end for scanning blobs


rep.write('Saving results to', outdir,' in Summary-table\n', color='green')
SaveData(['indx','number of paths','length (micron)',\
        'branches along the blob', 'lowest', 'highest', 'size', 'surface (micron^2)',\
        'volume (micron^3)'],
        results,\
        os.path.join(outdir, "Summary-table.txt"),\
        "Summary numbers from path analysis")
