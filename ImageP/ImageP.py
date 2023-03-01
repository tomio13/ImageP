#!/usr/bin/env python
#######################################################################
""" ImageP: Image Processing routines collected together.
    The pack is based on:
        numpy, matplotlib, pylab and scipy

    Aim: provide basic image processing algoritms together, which are
    applicable to convolution filtering, FFT filtering, particle tracking
    and many other cases.

    Images should be converted to numpy arrays, functions return arrays or
    lists. This is done by read_img for TIF images.

    Author: Tamas Haraszti, Biophysical Chemistry group at the University of
        Heidelberg

    Copyright:  LGPL-3
    Warranty:   For any application, there is no warranty 8).
"""

#####################################################################
# Headers:
# To manipulate paths and variables: sys, os
# To manipulate and display we need: pylab
# Use numpy for arrays and pylab (matplotlib)
# numpy.linalg for linear algebra
# numpy.fft for Fourier transformation
# to plot figures

#import sys, os
#import hydro as h
#now the C routines:

try:
    from . Csource import *
except:
    print("Csource is not loadable, falling back to Python sources")
    from . Pysource import *
#end try

#functions in other sources:
from . Kernels import *
from . Contour import *
from . Convolve import *
from . Display import *
from . Comparison import *
from . StructureTensor import *

#from Csource import *

import matplotlib.pyplot as pl

from numpy import histogram, zeros, ones, arange, asarray, sqrt

from numpy import abs, diff, uint, int16, inf, log

from numpy import linalg, ndarray, concatenate, polyfit
from numpy import hstack, argsort, polyval, linspace
from numpy import pi, sin, cos, arctan2
#For read_img, to handle the image files:
try:
    import Image as im
except:
    from PIL import Image as im

####################################################################
__all__=["read_img",\
            "BoxcarKernel","CircMask","GaussKernel","RotatingGaussKernel",\
            "BinomKernel", "SoebelKernel", "BallKernel",\
            "ActiveContour", "Circle_fit", 'SimpleContour',\
            "BPass","ConvFilter","ConvFilter1D","MeanFilter", "GaussDeblurr", "GaussDeblur",\
            "GrayErode", "GrayDilate", "DownsizeImage", "UpsizeImage",\
            "RollingBall", "EdgeDetect", "FindLocalMaxima", \
            "SimpleFilter", "SimpleFilter1D","RankFilter",\
            "display","OverPlot", "composite",\
            "Manders", "Pearson",\
            "ParticleTrack","PeakFind","PeakPos",\
            "bwfloodlist","bwlabel","bwanalyze","despike","graythresh",\
            "hist", "poly_fill", "shift",\
            "ErodeImage", "PerimeterImage","SimpleErode","SimpleDilate",\
            "HitMiss",\
            "ExtractProfile", "StructureTensor", 'Fractal_Dimension',\
            "DistanceFilter1DL1", "DistanceFilter1D", "DistanceFilter",\
            "Thinning", "Skel", "Compress"]

#####################################################################

def read_img(filename, asis = False, verbose=False):
    """ Reads (TIF) images from a file and converts it to a numpy array
        Relies on the PIL. Uses PIL to open and convert images to
        gray scale if possible.

        If a multipage image is found, return a list of arrays (JPK images).

        Parameters:
            filename:   file to read
            asis:       if False, use image type I else take whatever
                        the file reports

            verbose:    if True, it displays the image with matplotlib

        returns:
            a 2D numpy array of the iamge
    """
    # rewrite this using with... we can skip the error message
    # use a = None by default, then assign if there is content
    # use the Nframes for a for loop reading the images
    # Tags can be imported for TIFF images, but it may be the
    # wrong place for that

    try:
        img = im.open(filename)

    except IOError :
        print("unable to open file")
        return None

    #prepare for a multipage image:
    a = list()
    i = 0
    # Nframes = im.n_frames
    # tags = im.tags.named()

    try:

        while 1:
            #kill the palette
            img.palette = None
            #Sometimes it is important: we need BW images!!!

            if not asis and "I" not in img.mode:
                img = img.convert(mode="I")
            #end if

            #print("Getting data")
            inp = asarray(img.getdata())

            #The size of the PIL image has to be
            # is it a color image?
            if len(inp.shape) == 2 and inp.shape[1] == 3:
                inp.shape = (img.size[1], img.size[0], 3)
            else:
                #inverted:
                inp.shape = img.size[::-1]

            #print("found %d pixels" %inp.size)

            if '16' in img.mode:
               # print("converting to U2")
                inp = inp.astype('u2')

            a.append( inp.astype('f') )
            #if all fine, go to the next:
            i = i + 1

            img.seek( img.tell() + 1 )

    except EOFError :
        #print("End reached at i=%d" %i)
        pass;
    #end of try

    if verbose :
        print("loaded %d images" %i)
        pl.cla()
        pl.imshow(a[0], origin='lower', interpolation='nearest')
        pl.draw()
    #end if

    if len(a) < 2 :
        return a[0]
    else :
        return a
#End of read_img


def hist(image, bins = 50, range= None, verbose = 0):
    """A little extension of numpy.histogram, which is now nice and fast.
       Calculates a cumulative sum and an integral histogram as well.
       Recently the pockets is the edges of the bins, providing N+1 points.
       So, a midpoints variable is added, addressing the middle of the bins.

       Input:
            image:      an array of points
            bins:       number of pockets or a set of float numbers
                        as a limit. Passed directly to numpy.bins.
                        (this means bins+1 wall values)
            range:      range of minimum and maximum bin edge to be used
                        passed to numpy.histogram

       Result is a dictionary containing:
       'pockets':       the boundaries between the values were counted
       'midpoints':     the middle values of the pockets
       'dist':          contains the numbers between
                   pocket[i] and pocket[i-1], with dist[0]=0
       'integ':        integrated histogram
       'integNorm':        the integral normalized by the number
                   of image points
    """
    #Start processing with the limiting points:

    if type(bins) in [float, int]:
        bins = int(bins)

    if range is not None:
        # for some reason the bin calculation of numpy is not
        # so nicely homogeneous as this
        bins = linspace(range[0], range[1], bins+1)
        h = histogram(image, bins, range= range)
    else:
        h = histogram(image, bins)


    dist = h[0]
    pockets = h[1]
    midpoints = 0.5*(pockets[1:] + pockets[:-1])
    integ = dist.cumsum()
    integnorm = integ.astype(float)/dist.sum()

    result={'pockets':pockets,\
        'midpoints': midpoints,\
        'dist':dist,\
        'integ':integ,\
        'integNorm':integnorm}

    if verbose :
        pl.cla()
        pl.plot(midpoints,dist,'r+-')
        pl.plot(midpoints,integ,'b+-')
        pl.draw()

    return result
#End of hist()

def graythresh(img, N=50, verbose=False):
    """     Finding an optimal treshold tolerance on a gray scaled image
        and returns this value.

        Based on the algorithm from Barre-Piquot written for Octave,
        which is based on the publication of Otsu:
        Reference N. Otsu, "A Threshold Selection Method from Gray-Level
        Histograms" IEEE Transactions on Systems, Man, and Cybernetics,
        vol. 9, no. 1, pp. 62-66, 1979.
        (under the GPL 2 or higher licence, see:
        "http://www.gnu.org/licenses/gpl-2.0.txt")

        original at:
        http://www.irit.fr/recherches/SAMOVA/MEMBERS/JOLY/Homepage_files/IRR05/
        Modified by Soren Hauberg
        http://www.koders.com/matlab/fid20E5AB02E4E9B2F04DACBAEF5BA19E8615E6A065.aspx
        -> unfortunately these links are now defunct.

        Adopted to Python by Tamas Haraszti

        Input parameters:
        img:        a data array
        N:          number of pockets for the histogram
        verbose:    (False/True) plot some graphs

        Return:
        floating point number [0,1]
    """

    img = img - img.min()

    if img.dtype.name == 'bool':
        print("invalid image type Bool!")
        return 
    elif 'int' in img.dtype.name: 
        if img.max() < 2 :
            print("Warning, the image has a too low dynamic range!")
            return

        if N > img.max():
            #this can raise some questions, but simply: we do not want
            #histograms with broad, but equal -and meaningless- values
            N = img.max()
    #end if

    #Calculate the normalized histogram:
    if N < 5 and verbose:
        print("Warning: This histogram may be too narrow")
    #end if
    Histogram = histogram(img, bins = N)
    x = Histogram[1]
    h = Histogram[0].astype(float)
    
    #Normalize the histogram, but not to the length,
    #as in the example code was, but
    #to the number of pixels instead
    #h = h / (len(h)+1)
    #this is a dirty trick to avoid
    #division by zero here and for w
    #h = h / (h.sum() + 1)
    hs = h.sum()
    if hs == 0.0 :
            print("invalid histogram (zero sum)")
            return 0.0
    #end if

    h = h / hs
    
    #Cumulative histogram:
    w = h.cumsum()
    #this declaration below is done automatically by numpy
    #mu = zeros(N, float)
    i = arange(N, dtype=float) + 1.0
    #calculate the weighed sum quick and dirty:
    mu = (h*i).cumsum()
    
    #Treshold calculations:
    w1 = 1.0 - w

    #at the edges, both mu and w are small, but when they
    #go to real zero, 0*0 may blow the system.
    #So, for safety:
    w[w == 0.0] = 1E-8
    w1[w1 == 0.0] = 1E-8

    mu0 = mu / w
    mu1 = (mu[-1] - mu)/w1

    s = w*w1*(mu1 - mu0)*(mu1 - mu0)

    #plot the optimum curve:
    if verbose:
        pl.clf()
        x = (x[1:] + x[:-1])/2
        pl.plot(x,s, '+-')
        pl.draw()

    #there are N s values, so the maximum on a 0-1 scale
    # is i where s == s.max(), then i/N (0 ... 1):
    i = (s == s.max()).nonzero()
    level = float((s == s.max()).nonzero()[0][0])/N

    return level
#end of graytresh


def despike(img, treshold=0.9, nlimit=10, radius=5, verbose=False ):
    """This function was inspired from X-ray microscopy, where
        sometimes we have to get rid of some spikes, which can 
        kill the whole statistics of an image.
        
        The algorithm checks for confluent objects with the size
        less than nlimit and replaces from the average of a window
        larger in all directions with radius.

        Relies on bwfloodlist(img, x, y, nhood=8), similar to the
        bwlabel.
        
        If so, then overwrites these pixels to a local average.

        Parameters:
            image:        image to test
        treshold:    relative tolerance around the maximum
        nlimit:        a minimum diameter of a feature
                (maximum size of a spike)
        radius:        number of pixels used below and above to
                average the new value replacing a spike

        return:    the new image
    """

    #Spikes involve only few points peaking out of the image,
    #rendering the rest unuseable:
    image = img.copy()

    ni, nj = image.shape

    imgtmp = image - image.min()
    imgtreshold = treshold * imgtmp.max()
    imgtmp = (imgtmp > imgtreshold)
    tmpc = imgtmp.nonzero()

    size = nlimit*nlimit

    if verbose :
        print("Max: %d" %(imgtmp.max()))

    #Find features:
    #used in  bwlabel as well:
    # for (x,y) in map(None,tmpc[0],tmpc[1]) :
    for (x,y) in zip(tmpc[0],tmpc[1]) :

        #those are nonzero which has not been checked yet
        if imgtmp[x,y] != 0 :
            (HitsI,HitsJ) = bwfloodlist(imgtmp,x,y,nhood = 8)
            HitsI = asarray(HitsI)
            HitsJ = asarray(HitsJ)

            #This is categorized as a spike:
            if len(HitsI) < size:

                i0 = max(HitsI.min() - radius, 0)
                i1 = min(HitsI.max() + radius, ni)
                j0 = max(HitsJ.min() - radius, 0)
                j1 = min(HitsJ.max() + radius, nj)

                #Construct an image from the original
                #with the pixels not found:
                oldvalue = image[HitsI,HitsJ].mean()
                image[HitsI, HitsJ] = 0

                tmpslice = image[i0:i1,j0:j1] 
                tmpindx = tmpslice.nonzero()
                tmpvalue = tmpslice[tmpindx[0],\
                        tmpindx[1]].mean()

                if verbose:
                    print("Found: %d" %(len(HitsJ)))
                    print("spot max: %.0f, treshold: %.0f" \
                        %(tmpslice.max(),imgtreshold))
                    #print("old: %.0f" %oldvalue, end=" ")
                    print("old: %.0f" %oldvalue)
                    print("new: %.0f" %tmpvalue)

                #Now, fill the spike:
                image[HitsI,HitsJ] = tmpvalue

            #End of if len(HitsI)

            #Remove the ones already found:
            imgtmp[HitsI, HitsJ] = 0
        #End of if
    #end of for

    return image
#end of despike()

def PeakPos(img, hits, size, circular=True, CutEdge=False, verbose=False):
    """Calculates the center of mass around the positions based
       on the list of hits from PeakFind(). A circular mask with
       size diameter is used to filter the image.
       R2 and I0 are sensitive to background correction and the
       size parameter as well.

       Parameters:
        img:            image
        hits:           a res{} dictionary of hits (provided by PeakFind)
        size:           Diameter of the window around (X,Y)
        circular:       True/False use the a circular mask? (True)
        CutEdge:        bool (False): should we drop positions which are at
                        the edge (within diameter/2 distance)

        verbose:        False/True: talk back a bit more

       Return value:
        a dictionary of float positions 'X', 'Y', 'I0', 'R2', 'E'
        I0:     is the sum of intensity
        X:      X position of the center
        Y:      Y position of the center
        R2:     is the square of the radius of gyration
        E:      the ellipticity parameter defined as:
     """

     #Start with a circular mask
    if size > 0:
        w = int(size/2)
        size = 2*w +1
        circ = CircMask(w)
    else :
        print("Error: size parameter missing\n")
        return None

    Ni,Nj = img.shape
    N = hits['X'].size

    X = []      # X position
    Y = []      # Y position
    I0 = []     # intensity
    R2 = []     # square of radius of gyration
    E = []      # Excentricity
    Mark = []             # is the patch too close to the edge?
    err = []              # positioning error (estimate)

    #we need imgbuff, that we can apply the circular mask
    #if img is at the edge, the mask would be off...
    imgbuf = zeros((size,size),dtype=img.dtype)

    for i in range(N) :

        #X <-> j, Y <-> i !!!
        #Playing with indices.
        #Does the ROI fit within the image?
        # the higher index is not reached -> add 1 more
        imgj0 = hits['X'][i]-w
        imgj1 = hits['X'][i]+w+1

        imgi0 = hits['Y'][i]-w
        imgi1 = hits['Y'][i]+w+1

        #Tailor the valid part within imgbuf then
        #if we run into a negative index in img, then the
        #buffer has to start that much higher. Thus:
        bufj0 = max( -imgj0, 0)
        bufi0 = max( -imgi0, 0)
        #Backwards we can index with negative numbers
        bufj1 = size - max( imgj1 - Nj, 0)
        bufi1 = size - max( imgi1 -Ni, 0)

        #Mark is an index if a particle is at the edge or not:
        #Particles partially out of the frame can not be
        #positioned precisely. Their statistics is ruined
        #by the missing part...
        if imgi0 < 0 or imgi1 >= Ni\
                     or imgj0 < 0 or imgj1 >= Nj :
            if verbose:
                print("%d -th particle too close to the edge" %i)

            #if CutEdge, then we step to the next particle here
            if CutEdge:
                continue
            else:
                #or we just label the particle being at the edge:
                Mark.append(1)
        else:
            Mark.append(0)
        #end if

        #Truncate the image segment to valid indices:
        imgi0 = max(imgi0, 0)
        imgj0 = max(imgj0, 0)
        imgi1 = min(imgi1, Ni)
        imgj1 = min(imgj1, Nj)

        if verbose:
            print("mask: [%d:%d,%d:%d]" %(imgi0,imgi1,imgj0,imgj1))

        #now pick up our image part into the buffer
        imgbuf[bufi0:bufi1, bufj0:bufj1] = img[imgi0:imgi1, imgj0:imgj1]

        #The masked piece of the image:
        if circular:
            if verbose:
                print("adding circular mask")
            imgbuf = imgbuf * circ
        #end if

        #We need the position of each nonzero points in the image:
        poss = imgbuf.nonzero()
        posx = poss[1]
        posy = poss[0]
        #nonzero intensities as a linear array:
        imgpart = imgbuf[poss]
        #Remove offset:
        #this ensures that all intensities >= 0!
        imgpartmin = imgpart.min()
        #if the imagepart is flat, we have bad
        #position data. But this is the concern of the user
        if imgpart.max() == imgpartmin:
            print("Warning: empty image segment or flat intensity profile!")
            imgpart = ones(imgpart.shape)
        else:
            imgpart = imgpart - imgpartmin
        #end if flat image

        intensity = imgpart.sum()

        #should we use intensity <= 1E-18?
        #if the intenisty is 0, then we have a problem:
        if verbose and intensity < 1E-16:
            print("Warning, very low intensity!")
        #end if

        # Center of mass :
        x = (posx * imgpart).sum()/intensity
        y = (posy * imgpart).sum()/intensity

        X.append(x + imgj0 )
        Y.append(y + imgi0 )
        I0.append( intensity )

        #the radius of gyration is the standard deviation:
        #Center the pos to the new center:
        posx = posx - x
        posy = posy - y

        #Generating the square displacements:
        ix = posx*posx
        iy = posy*posy

        #R^2 = X^2 + Y^2
        ix = ix + iy

        # I&R^2:
        R2.append( (ix * imgpart).sum()/intensity )

        #Excentricity: try something, but not well
        # implemented yet:

        #Calculate the distances:
        r2 = ix
        #This goes to the denomiator, so kill zeros:
        indx = (r2 == 0.0)
        indx = indx.nonzero()[0]
        r2[indx] = 1.0
        r = sqrt(r2)

        #Do it only with (sin theta *img)^2+(cos theta *img)^2
        ix = posx/r * imgpart
        iy = posy/r * imgpart
        ix = ix.sum()
        iy = iy.sum()

        E.append( sqrt((ix*ix+iy*iy)/intensity) )

        #error: 1/sqrt(N):
        err.append( 1.0/sqrt(float(posx.size)) )
    #End of for

    return {'X': asarray(X),\
            'Y': asarray(Y),\
            'I0':asarray(I0),\
            'R2': asarray(R2),\
            'E': asarray(E),\
            'Mark': asarray(Mark),\
            'Err': asarray(err)}
#End of PeakPos()

def ParticleTrack(LocalPos, maxstep, minlength=5, maxgap=0, verbose=False):
    """ Take a list of position dicts stored for images...
        More precisely: each image analysed produced a dict containing
        at least 'X' and 'Y' positions of features (particles).
        Now, take this list and find trajectories based on the shortest
        distance between the list elements. (Find, which is the closest
        position in the next dict.)

       LocalPos:    list of dictionaries of positions
                    each element is an dictionary containing information
                    about one image;
                    the elements of the dictionary are numpy arrays themselves
                    and the process will erase them!

       maxstep:     maximal step distance
       minlength:   minimum number of positions / track
       maxgap:      maximum how many images can be jumped over
                    if 0, we do not care.
                    (maxstep does not change with the gap!)

       verbose:     provide some feedback

       result:
            a list of tracks
            each track is a dictionary collecting positions
            and information of consecutive positions of the
            tracked object
    """
    if len(LocalPos) < 1 :
        print("Empty position list!")
        return None
    #end if

    #How many images we investigate (each contain a position set):
    N = len(LocalPos)
    #and the limiting radius:
    currlimit = maxstep*maxstep

    if verbose:
        print("number of images: %d\n" %N)
        print("start scan ... \n")
    #end if

    #the final result are stored in here:
    tracklist = []

    #Go through all images and all positions within
    #(a dual loop: i for images, j for positions within)
    #Naturally we need nothing from the last image,
    #we can skip it:
    for i in range(N-1):
        #Shortcut to the actual image, we pick a position
        #from this one:
        StartPosX = LocalPos[i]['X']
        StartPosY = LocalPos[i]['Y']

        #How many positions do we still have in this image?
        Ni = len( LocalPos[i]['X'] )

        #after a while, there may be many empty lists
        #so we check if it worths to go for the still
        #existing positions (are there any?)
        if Ni > 0:
            #Looping through the positions within the actual image
            #We arrive here with the position starting a new trajectory
            #the remaining trajectory is in the other images:
            for j in range(Ni):
                #A new track starts here:
                #'indx' records the image index:
                # i runs among the images, j within one image
                newtrack = {'indx':[i]}

                #to measure the gap we initialize an index
                #(the image index i is a kind of time stamp)
                #need to initiate here for each trajectory, because
                #it gets overwritten for each successfull hit...
                #if we get too far from this, then we have a problem:
                lastindx = i

                #Fill up with the first frame:
                frame = LocalPos[i]

                #Any keys are provided are recorded:
                for currkey,vallist in frame.items():
                    newtrack[currkey] = []
                    newtrack[currkey].append(vallist[j])
                #end for

                #Start coordinates:
                Xactual = StartPosX[j]
                Yactual = StartPosY[j]

                #Counters init: run through the rest of images
                #k = i+1

                #Now go through all images above i whether
                # they have any suitable position within:
                # k<N : we have images left
                # len(LocalPos[k]['X']) we have positions within
                # gap: the gap is not too high (continuous track)
                # while k<N :
                for k in range(i+1, N):
                    #position list from the next image:
                    CurrentPosX = LocalPos[k]['X']
                    CurrentPosY = LocalPos[k]['Y']

                    #is there anything in this image?
                    #if there is nothing, we have a gap...
                    #chkgap tells if we have to look for the maxgap
                    chkgap = True if maxgap > 0 else False

                    if len(CurrentPosX) > 0:
                        #Determine the distances from the origin:
                        dx = CurrentPosX - Xactual
                        dy = CurrentPosY - Yactual
                        r = dx*dx + dy*dy

                        #Look at the distance if there is something:
                        #if it is too far, then again a gap
                        if r.min() < currlimit:
                            #new hit -> actualize where we were:
                            # this is to check against maxgap...
                            lastindx = k
                            #we have a hit, no need to look for a gap...
                            chkgap= False
                            #now find the best hit(s):
                            indx = (r == r.min())
                            #if there are more than one, we need only one
                            #the others we should keep for further tracking
                            #the bool type is important here!
                            # KeepIndx = ones(len(r), dtype=bool)
                            KeepIndx = ~indx

                            #we need the actual index of the first hit:
                            indx = indx.nonzero()[0][0]
                            #We use the inverse to shorten the arrays:
                            #but we kill only one!
                            KeepIndx[indx] = False

                            #Update the new center:
                            Xactual = CurrentPosX[indx]
                            Yactual = CurrentPosY[indx]

                            #Store the tracked point:
                            frame = LocalPos[k]
                            #record the index of this hit:
                            newtrack['indx'].append(k)

                            #now the position values:
                            for currkey,vallist in frame.items():
                                # indx is a scalar, so this works:
                                newtrack[currkey].append(vallist[indx])
                                #Dump the position from the list:
                                #(avoiding multiple hits)
                                # now vallist is a list, indexing with numpy
                                # array throws an error!
                                # LocalPos[k][currkey] = vallist[KeepIndx]
                                LocalPos[k][currkey] = [j for i,j in enumerate(vallist) \
                                                        if  KeepIndx[i]]
                            #end filling up hits and kept list
                        #end of if: found a hit
                    #end of if: having a position...

                    #maxgap > 0 is checked for setting the chkgap
                    if chkgap and (k-lastindx) >= maxgap:
                        if verbose:
                            print("Maximal gap reached, cutting trajectory")
                            print("Started at %d, ended at %d, maxgap: %d" \
                                        %(i,k, maxgap))

                        #let us make sure that we stop this loop:
                        k = N+1
                        break
                    #end if maxgap is reached
                        #end if r.min() < currlimit - hit within reach
                    #end if len(CurrentPosX)

                    #next image please...
                #End of for k i+1 ... N(forward search)

                if len(newtrack['X']) > minlength :
                    tracklist.append(newtrack)
            #End of for j in Ni of LocalPos
        #end if Ni > 0
    #end of search

    return tracklist
#End of ParticleTrack()


def shift(img,x,y):
    """ Shift an image with x and y.
        The new image is shifted with (0,0) -> (x,y), the new
        pixels are set to zero. (x,y) are treated as integer indices.

        Parameters:
        img:        image to be shifted
        x,y        the new (x,y) indices of the (0,0) position
                thus the shift vector

        return:
            the shifted image
    """

    if img.ndim != 2 :
        print("2D image is expected")
        return

    #Rounding x,y properly:
    xx = int(x + 0.5)
    yy = int(y + 0.5)

    Ni,Nj = img.shape

    #window within the old image:
    i0 = max(xx,0)
    j0 = max(yy,0)
    i1 = min(Ni + xx, Ni)
    j1 = min(Nj + yy, Nj)

    #window within the new image:
    k0 = max(-xx,0)
    l0 = max(-yy,0)
    k1 = min(Ni - xx, Ni)
    l1 = min(Nj - yy, Nj)

    #preserve the image type!
    imgnew = zeros((Ni,Nj),dtype=img.dtype)
    imgnew[i0:i1,j0:j1] = img[k0:k1,l0:l1]

    return imgnew
#end of shift


def poly_fill( image, details=False ):
    """    Take a binary image containing only a perimeter.
        Completes the perimeter pixels in the angular space
        using a linear interpolation between the points.
        (Improved possibilities would include using scipy
        interpolation package or some more advanced snake
        spline fitting.)

        parameters:
        image:      a binary image
        details:    if True, return much more information

        return:
            a dict containing:
            'perimeter':    the original perimeter pixels
            'area':         the filled up binary image

        if details is set:
            'I','J':        position arrays
            'I0','J0':      center values used
            'angles',
            'distances':    polar coordinates of 'I','J' around
                            'I0','J0'
    """
    binimg = image.copy()
    #define the edge points along I:
    #kill all pixels between the edges
    #for i in range(image.shape[0]):
    #    if binimg[i,:].sum() > 1:
    #        jlist = binimg[i,:].nonzero()[0]
    #        jmin = jlist.min()
    #        jmax = jlist.max()
    #        binimg[i, jmin+1:jmax] = 0
    #end for i

    #now, we do have an image which is killed out in 1 direction
    I,J = binimg.nonzero()
    #convert to polar coordinates:
    I0 = I.mean()
    J0 = J.mean()
    #I have tried using angles and polar data, but the jumps
    #between neighboring points are terrible... Try using the I,J
    #directly, but after sorting them angle-wise.
    #we switched to polar coordinates:
    IC = I - I0
    JC = J - J0
    distances = sqrt(IC*IC + JC*JC)
    #arctan2(Y,X) not as in the help... test: arctan2(sqrt(3)/2,0.5): 60 deg.
    angles = arctan2(JC, IC)

    indx = argsort(angles)
    angles = angles[indx]
    distances = distances[indx]
    #we want the indices also sorted according to the angular
    #direction...
    #I = I[indx] -> we do it at the return if needed...
    #J = J[indx]
    #this is now a sorted polygon, we want to fill in...
    #x = concatenate((I[-1:],I,I[:1]))
    #y = concatenate((J[-1:],J,J[:1]))

    #we do a linear interpolation in the polar space, and fill up
    #the image then...
    pi2 = 2.0*pi
    x = concatenate((angles,angles[:1]+pi2))
    y = concatenate((distances,distances[:1]))
    polyms = list()

    binimg2 = zeros(binimg.shape,dtype='i')
    Ni = max(binimg.shape)
    #fill up:
    #a circle with radius R has 2piR points for 2pi radians
    dfi = 1/y.max() if y.max() > 0 else 1.0/float(Ni) #accuracy in radians
    Nx,Ny = binimg.shape

    for i in range(len(x)-1):
        x0 = x[i]
        x1 = x[i+1]
        y0 = y[i]
        y1 = y[i+1]

        a = (y1-y0)/(x1-x0) if x0 != x1 else 0
        b = y0
        polyms.append([a,b])

        Nsi = sqrt( ((x1-x0)/dfi)**2 + (y1-y0)**2) +1
        xs = linspace(x0,x1,Nsi) if x1 != x0 else zeros(Nsi)+x0
        ys = a*(xs-x0) + b if x0 != x1 else \
                                    (y1-y0)*linspace(0,1.0,Nsi)+b
        for j in range(len(xs)):
            ds = linspace(0.0, ys[j],Ni)
            xsi = ds*cos(xs[j]) + I0
            ysi = ds*sin(xs[j]) + J0
            xsi[ xsi > Nx-1] = Nx-1
            ysi[ ysi > Ny-1] = Ny-1
            xsi[ xsi < 0 ] = 0
            ysi[ ysi < 0] = 0
            binimg2[xsi.astype('i'),ysi.astype('i')] = 1
    #end for i

    if details:
        return {'area':binimg2, 'edge':binimg, \
            'angles':angles, 'distances':distances, \
            'parameters':polyms, 'I':I[indx], 'J':J[indx], 'I0':I0, 'J0':J0}
    else:
        return {'area':binimg2, 'edge':binimg}
#end of poly_fil

def bwanalyze(bwimg, key=1, feature="", WithMin= False):
    """ Analyze data obtained using the bwlabel method.
        The routine provides some characteristics upon request.

        Parameters:
        bwimg:      an image returned by bwlabel
                    or a grayscale image with the feature
                    to be analyzed.

        key:        an index of the structure to be analyzed:
                    pixels == key will be analyzed if key > 0
                    if key == 0, then pixels > 0 will be analyzed
                    if WithMin is set, then the minimum of the image
                    is subtracted first.
                    if key == -1, then graythresh() will define a
                    treshold and pixels > this treshold will be used
                    In all key < 1 cases the intensity values are used
                    as a weight. For key == -1 the minimum of the features
                    will be pulled to 0.

        feature:     a set of letters to request features
            a:       area; the bounding indices of the areas
                    returns:
                    'Area' = [[Imin,Jmin],[Imax,Jmax]]
                    'PixArea' the number of pixels in the patch
                    'SumI' if key < 1, the sum intensity

            c:  center: the position of the geometric center
                or the center of mass if key <= 0
                returns: 'Center' = [I,J]
                if key <= 0: 'Center' is the geometric (not weighted)
                            center [I,J]

            e:  Calculates the second central momentum tensor,
                its eigen values and eigenvectors,
                the major and minor ellipse half axis and the
                eccentricity value.

                'Eigenvalues' = [a,b]
                'Eigenvector1' = [e1,e2]
                'Eigenvector2' = [e1,e2] two unit wectors

                'MajorAxis' and 'MinorAxis' are the two
                            half axises for the ellipse,
                'Eccentricity' the eccentricity value

            r:  calculate the radius of gyration square
                'Rg2' = variance, or Rg^2

            p:  convert the coordinates of the x,y points to
                polar values, alpha, r and return them as:
                'r', 'alpha', and 'I' for the intensities.

        WithMin:    subtract the minimum from the grayscale image
                    actually, if min == 0 then try:
                    img[ img > 0].min()

        Refs: Jahne et al. Digital Image Processing
        Haralick and Shapiro, Computer and Robot Vision vol I,
        Addison-Wesley 1992, Appendix A.
    """
    results = {}

    if len(feature) == 0:
        return results

    #if key is positive, it is a structure
    #if 0, then the background
    #if -1, use automatic thresholding
    #if anything else, we have no idea what to do...
    if key > 0:
        if key > bwimg.max() or key < bwimg.min():
            print("invalid key value!")
            return results
        #end if
        indx = (bwimg == key)

    elif key == 0:
        #key is not meaningful, so we try figuring it out
        #get the background as minimum, and the rest separated:
        if WithMin:
            m = bwimg.min()
            if m == 0:
                indx = bwimg - bwimg[ bwimg > 0].min()
            else:
                indx = bwimg - m
        #end subtracting minimum
        else:
            #we use indx image from here on:
            indx = bwimg

    elif key == -1:
        #use graythresh
        bwimg = bwimg - bwimg.min()
        th = graythresh(bwimg)
        bwimg = bwimg*(bwimg > th*bwimg.max())
        indx = bwimg - (bwimg[bwimg > 0]).min()
    else:
        raise ValueError("Invalid key")
    #end sorting out key

    #do we have enough image points?
    if indx.size < 2 :
        print("error: empty patch or improper key %d" %key)
        return results
    #end if

    #meaningful pixels in the binary / gray image:
    indxindx = indx.nonzero()
    Yindx = indxindx[0]
    Xindx = indxindx[1]
    #we use weights: intensity for gray, or ones for binary = geometric parameters
    I = bwimg[Yindx, Xindx] if key <= 0 else ones(Xindx.shape, dtype=float)
    IN = float(I.sum())

    if IN == 0:
        print("We have an empty image!")
        raise ValueError("empty image")

    if 'a' in feature :
        #The min and max corner diagonally:
        results["Area"] = asarray( [[Yindx.min(),Xindx.min()],\
                     [Yindx.max(), Xindx.max()]])
        results['PixArea'] = len(I)
        if key < 1:
            results['SumI'] = IN
    #end if

    #center = asarray([Xindx.mean(), Yindx.mean()], dtype=float)
    center = asarray([(I*Xindx).sum()/IN, (I*Yindx).sum()/IN])

    if 'c' in feature :
        results["Center"] = center
        #if we have a center of mass, we have a geometric center as well:
        if key <= 0:
            results["GCenter"] = asarray([Xindx.mean(), Yindx.mean()], \
                                                        dtype=float)
    #end if

    Xindx = Xindx.astype(float) - center[0]
    Yindx = Yindx.astype(float) - center[1]

    if 'e' in feature :
        #for IN being len(I) this is the mean...
        xy = -(I*Xindx*Yindx).sum()/IN
        xx = (I*Xindx*Xindx).sum()/IN
        yy = (I*Yindx*Yindx).sum()/IN
        common2 = (xx - yy)**2 + 4*xy*xy
        common = sqrt(common2)

        theta = asarray([[yy,xy],[xy,xx]])
        thetaeig = linalg.eig(theta)
        thetaeigval = asarray(thetaeig[0])
        thetaeigvec = asarray(thetaeig[1])

        results['Eigenvalues'] = thetaeigval
        results['Eigenvector1'] = thetaeigvec[:,0]
        results['Eigenvector2']= thetaeigvec[:,1]
        #from Jahne: Digital Image Processing, chapter 19, eq: 19.6
        results['Eccentricity'] = common2/(xx+yy)**2
        results['MajorAxis'] = sqrt(2.0*(xx + yy + common))
        results['MinorAxis'] = sqrt(2.0*(xx + yy - common))

    if 'r' in feature:
        results['Rg2']= (I*(Xindx*Xindx + Yindx*Yindx)).sum()/IN

    if 'p' in feature:
        alpha = arctan2(Yindx, Xindx)
        aindx = argsort(alpha)
        results['alpha'] = alpha[aindx]

        results['r'] = sqrt(Xindx*Xindx + Yindx*Yindx)[aindx]
        results['I'] = I
    #end if

    return results
#to be continued
#end of bwanalyze


def ErodeImage(Img, Start=0, End=0, FindMax=True, NewCenter=False,\
        MinSize=1, verbose=False):
    """ Erode an image if there are multiple minima or maxima found.
        The method is based on a watershed algorithm, in the sense
        that it starts from the extrema and goes towards the average.

        The idea is, to find peaks -or patches- at Start, and grow them
        towards the End. Try to evaluate all new pixels in increasing or
        decreasing order of intensity whether:
            they belong to nothing
            they belong to the original image

            if they are separate, then add them to the nearest
            extremum

        Those having neighbours from at least two different patches
        are taken as border points and being set as a "wall".
        The list of this 'wall' points is returned.

        When NewCenter is True, any new patch appearing during the
        steps will be accepted as a new area, resulting a finer mesh.
        Patches having less then MinSize number of pixels shall be
        not taken as new patches into account.

        When FindMax is False, then we start with patches =< Start,
        if it is True, then start with patches >= Start.

        Parameters:
        Img:        The image to be eroded

        Start:        Start with points of img < or > Start

        End:        Leave out points where img > or < End

        FindMax:    find  maxima or minima (sets < or > above)
                    if True, find maxima and go downwards to the minima
                    if False, find minima and go upwards to maxima

        NewCenter:    accept new local peaks found or not
        verbose:    give graphical feedback

        Return value:
        shed list:        the list of erased indices

            use it as :
                for (x,y) in list:
                    img[x,y] = 0
                to erase them 8)
    """

    if Img.ndim != 2 :
        print("2D images are required!")
        return []

    img = Img.copy()
    #We have to define the constrains we are working at:

    if Start == 0:
        if FindMax :
            Start = img.max()
        else :
            Start = img.min()

    if End == 0:
        if FindMax:
            End = img.min()
        else:
            End = img.max()


    #Now, go through the image:
    if FindMax :
        #find the upper part patches to define seed centers:
        imgflood = bwlabel((img >= Start),\
                MinSize=MinSize,\
                verbose=False)
    else :
        #find local minima patches and define seed centers:
        imgflood = bwlabel((img <= Start),\
                MinSize=MinSize,\
                verbose=False)

    #The fast end of the game:
    if imgflood.max() < 2 and not NewCenter:
        print("Only %d peaks found, quit algorithm" %(imgflood.max()))
        return []

    if verbose:
        print("found %d points" %(imgflood.max()))
    #Collect the related centers here:
    MinList = []
    #Collect the deleted points; this is the result:
    ShedList = []
    #Keep an image to coordinate the game: SpreadImg
    #Each pixel is numbered by
    # index of its center,
    # -1 if blocked
    # 0 if not identified yet
    Ni, Nj = img.shape
#    SpreadImg = zeros((Ni,Nj))
    SpreadImg = imgflood.copy()


    #Identify the center areas, and some
    #center positions:
    for i in range(1, imgflood.max()+1):
        indx = (imgflood == i).nonzero()
        #Define the extremum as the geometric center
        #of the defined area:
        MinList.append([int(indx[0].mean()), int(indx[1].mean())])
    #    SpreadImg is already 0 where nothing was found and i at the
    #     patches. This is a property of the bwlabel().

    #Arange the rest of the points:
    #1., identify all points, which are between Start and End
    #2., keep those, which are not found yet
    #3., order them into a list with increasing/decreasing intensity
    #4., walk through and identify where they belong to

    #points unknown:
    indx = (SpreadImg == 0).nonzero()
    Xs = indx[0]
    Ys = indx[1]

    #Points out of threshold are rejected, but not defined
    #as a wall element.
    if FindMax:
        #going from maxima downwards until we reach End
        #if user specified End, then points less than that
        #are out of reach:
        #(this is a 1D array)
        NewIndx = (img[Xs,Ys] < End).nonzero()[0]
        #cut them off reach:
        SpreadImg[Xs[NewIndx],Ys[NewIndx]] = -1

    else :
        #we have minima in the centers, and we walk upwards
        #if End is specified points above that are out of reach:
        NewIndx = (img[Xs,Ys] > End).nonzero()[0]
        #cut them off reach:
        SpreadImg[Xs[NewIndx],Ys[NewIndx]] = -1

    #now show the number of those out of reach:
    print("disabled %d points" %(len(NewIndx)))

    #the rest of 0s we need to work on, but from  now
    #SpreadImg holds the interesting information:
    indx = (SpreadImg == 0).nonzero()
    Xs = indx[0]
    Ys = indx[1]
    #intensities:
    Is = img[Xs,Ys]
    Is = (Is - Is.min()) * 255/Is.max()
    Is = Is.astype(int)

    #for easier run, sort the intensities and the corresponding
    #positions:
    if FindMax :
        SortIndex = (Is.argsort(kind='mergesort'))[::-1]

    else :
        SortIndex = Is.argsort(kind='mergesort')
    #Is itself is not sorted yet!
    Is = Is[SortIndex]
    Xs = Xs[SortIndex]
    Ys = Ys[SortIndex]

    #run through the positions in order:
    # for (iX,iY) in map(None, Xs,Ys):
    for (iX,iY) in zip(Xs,Ys):
        #We have the candidate here:
        iI = SpreadImg[iX,iY]
        iList = []
        iList.append(img[iX,iY])

        #Is the pixel already forbidden?
        if iI < 0:
            continue
        if iI > 0:
            print(" Warning: have we been here before?")

        else :
            #Now, we have to decide:
            #Is it a new spot?
            #is it continuing an old one?
            #is it a boundary?

            indxs = ((iX+1, iY-1),\
                (iX+1, iY),\
                (iX+1,iY+1),\
                (iX,iY+1),\
                (iX-1,iY+1),\
                (iX-1,iY),\
                (iX-1,iY-1),\
                (iX,iY-1))

            Mark = 0
            for (x,y) in indxs:
                #out of the image?
                if x >= Ni or x < 0 or \
                   y >= Nj or y < 0:
                    #jump to the next pixel
                    #in the list
                    continue
                #end if

                #No, then check the value:
                Ixy = SpreadImg[x,y]
                iList.append(img[x,y])
                #no Mark yet, but it is a boundary,
                #set Mark to the patch value:
                if Mark== 0 and Ixy >0:
                    Mark = Ixy

                #is it a different patch?
                elif Ixy > 0 and Mark != Ixy :
                    #now, we have a bounray point
                    #block it and mark it in the results
                    SpreadImg[iX,iY] = -1
                    ShedList.append([iX,iY])
                    #we also set a special mark:
                    Mark = -1
                    #work done, pixel is a wall
                    break
                #end if
            #end for (x,y)

            #if Mark remained 0, then it was neither a boundary
            #point nor belonged to a patch
            if Mark == 0 :
                iList = asarray(iList)
                #we find  a new center in a new local
                #minimum or maximum only
                if NewCenter and iList.any() != img[iX,iY]:
                    #is it extremum?
                    if FindMax:
                        T = (img[iX,iY] == iList.max()) 
                    else:
                        T = (img[iX,iY] == iList.min())
                    #end if
                    if T :
                        MinList.append([iX,iY])
                        SpreadImg[iX,iY] = SpreadImg.max() + 1
                    #end if; else T is false anyway
                else:
                    T = False
                #end if
                #we have to attach the point to the nearest
                #patch if it is not a new center:
                if not T:
                    (x,y) = MinList[0]
                    dx = x-iX
                    dy = y-iY
                    d = dx*dx + dy*dy
                    i = SpreadImg[x,y]

                    for (x,y) in MinList[1:]:
                        dx = x-iX
                        dy = y-iY
                        dnew = dx*dx + dy*dy

                        if dnew < d :
                            d = dnew
                            i = SpreadImg[x,y]

                    SpreadImg[iX,iY] = i


            elif Mark > 0 :
                #now, [x,y] has neighbours only
                #belonging to one group:
                SpreadImg[iX,iY] = Mark
#            else :
#                print("Mark is %.1f" %Mark)

    if verbose:
        pl.clf()
        pl.imshow(SpreadImg)
        pl.draw()
    #end of for (iX,iY)

    print("Finished")
    return ShedList
#end of ErodeImage()

def ExtractProfile(img, i0,j0, ei0,ej0, N=0, verbose=False):
    """
    Extract a profile along a norm vector in the image from a given point.
    Use a quadrilinear interpolation between pixels (which is not exactly
    linear at the end).

    Parameters:
    img:        image to use
    x0, y0:        start coodinates (indices)
    ex0,ey0:    unit direction vector
    N:        number of pixels + and - direction
            N=0 goes to the edges of the image
            If N points out of the image, the routine truncates at the edge.

    verbose:    (False/True) increase verbosity

    return value:
    [l,x,y,val]:        numarray vectors of position (length)
                coordinates (x,y) and image values
    """
    if img.ndim != 2 :
        print("Error: 2D images are required!")
        return []

    ImgEnd = asarray(img.shape)
    if i0 < 0 or j0 < 0 or \
        i0 > ImgEnd[0] or j0 > ImgEnd[1]:

        print("Coordinates out of bound")
        return None
    #end if

    #What is the direction vector like?
    le = float(ei0*ei0 + ej0*ej0)

    if le == 0 :
        print("Error: zero direction vector!")
        return None

    #Make sure it has unit length:
    e = asarray((ei0,ej0))

    e = e/le if le != 1.0 else e
    ei0, ej0 = e

    #At some point vector math looks simpler:
    center = asarray((i0,j0))

    #vecStart shall be the starting point of the profile
    #i0 and j0 are the middle of it
    #now we have to find it:

    #Backward:
    #how many steps I have from i=0 to reach i0
    # and from j=0 to reach j0
    ni = abs(i0/ei0) if ei0 != 0 else img.size
    nj = abs(j0/ej0) if ej0 != 0 else img.size

    #we do not allow walking off the image:
    l1 = min(N,ni,nj) if N != 0 else min(ni,nj)
    StartVec = center - l1*e

    #now the end vector:
    ni = abs((ImgEnd[0]-1-i0)/ei0) if ei0 != 0 else 0.0
    nj = abs((ImgEnd[1]-1-j0)/ej0) if ej0 != 0 else 0.0

    l2 = min(N,ni,nj) if N!= 0 else min(ni,nj)
    #we could find the end point, but we do not need it
    #EndVec = center + l2*e

    #The profile contains Nl points:
    Nl = l1+l2
    l = arange( Nl )

    if verbose:
        print("Start point", StartVec)
        print("direction:", e)
        print("Number of points:", Nl)
    #vec contains the float(i,j) point pars which we
    #need to evaluate:
    vec = asarray( [i*e + StartVec for i in l])
    print(vec.shape)
    val = zeros( l.shape)
    k = 0

    for i,j in vec:
        ii0 = int(i)
        ij0 = int(j)
        ii1 = ii0 + 1
        ij1 = ij0 + 1

        aw = i - float(ii0)
        bw = j - float(ij0)
        #the interpolation: take the distance from the next
        #point as a weight. Thus, if it is closer to
        #the current point, it matters more...

        val[k] = (1.0-aw)*(1.0-bw)*img[ii0,ij0] +\
                aw*(1-bw)*img[ii1,ij0] +\
                (1.0-aw)*bw*img[ii0,ij1] +\
                aw*bw*img[ii1,ij1]

        k += 1
    #end for

    return [l, vec[:,0],vec[:,1],val]
#End of  ExtractProfile

def DistanceFilter(img, inf_th = 0, full=False):
    """ 2D version of the distance filter, based on the DistanceFilter1D.

        The algorithm is based on P. F. Felzenszwalb and D. Huttenlocher
        Theory of Computing 8:415-428 (2012)

        It is slow in python but fast in C, so use the C-sources of the
        one D parts.

        The DistanceFilter1D requires the object pixels being inf or
        its equivalent in intc.

        The transform is defined as:
        min( (x-x')**2 + (y-y')**2 + f(x',y')) for every x',y'
        This can be split into 2x 1D scans, one in X and one in Y direction.
        The first can be also done using the L1 transform.

        Parameters:
        image   any values > 0 are considered an object (integers)
        inf_th  if not None, img[ img > inf_th] = inf is set
        full    provide the 1D intermediate as well
    """
    Ni, Nj = img.shape
    djimg = zeros(img.shape)
    res = zeros(img.shape)
    img2 = img.copy()
    if inf_th is not None:
        #we need float because nu.inf is float
        img2 = (img > inf_th).astype(float)
        img2[img2 > 0] = inf

    #for binary lines this is 2x faster:
    for i in range(Ni):
        #this version scans the line twice filling in the distance
        #if square = 1 then squares the values
        djimg[i,:] = DistanceFilter1DL1(img2[i,:], squared=1)

    for j in range(Nj):
        res[:,j] = DistanceFilter1D(djimg[:,j])

    if full:
        return (res, djimg)

    return res
#end DistanceFilter

def Fractal_Dimension( image, points = 4, details= False ):
    """ Calculate the fractal dimension of a binary image using the box
        counting method.
        This algorithm uses a quick binning with averaging then detects
        the edge pixels in the binned image and assigns as the box count
        to the degree of binning.
        The slope of the log-log curve at the beginning is the resulted
        fractal dimension.

        On a non-binary image, this may fail in weird ways!

        Parameters:
        image:      a thresholded image, 0 background  >0 signal
        points:     try to use this many points for fitting
                    (< 3 will make no sense)
                    Consider that objects with sizes > 2**points are
                    needed to get enough data points in this calculation

        details:    Boolean: if True, return the scale and N arrays too
                    (False by default)

        Results:
        if details is False, then a number
        if details is True, then a dict containing:
        'D':        the fractal dimension
        'scale'     the length scale, 2**binning degree
        'N'         the number of edge boxes
    """

    l2 = log(2.0)
    Nmax = floor( log( min( image.shape )))/l2

    x = 2.0**arange(Nmax)
    y = zeros(x.shape)

    for i in range(int(Nmax)):
        bimg = DownsizeImage( image, int(x[i]))
        y[i] = PerimeterImage(bimg).sum()
    #end for filling up the array

    indx = y >0
    if indx.sum() < 3:
        print("Insufficient data was found!")

    x = x[indx]
    y = y[indx]

    nfit = min([len(x), points])
    pf = polyfit(log(x[:nfit])/l2, log(y[:nfit])/l2, 1)
    D = -pf[0]

    if details:
        return {'D': D, 'scale': x, 'N':y, 'fit':pf}
    else:
        return D
#end of fractal_dimension

def Skel(img, verbose= False):
    """ Erode a binary image to a skeleton.
        Based on the internet site:
        http://homepages.inf.ed.ac.uk/rbf/HIPR2/thin.htm

        Parameters: binary image
        Return: skeleton image
    """
    oldN = -1
    N = img.sum()
    img2 = img.copy()
    while( N > 0 and N != oldN):
        oldN = N
#        img2 = thinning(img, asarray([[0,0,0],[-1,1,-1],[1,1,1]]))
        img2 = Thinning(img2, asarray([[-1,0,0],[1,1,0],[-1,1,-1]]))
        img2 = Thinning(img2, asarray([[0,0,-1],[0,1,1],[-1,1,-1]]))
        img2 = Thinning(img2, asarray([[-1,1,-1],[0,1,1],[0,0,-1]]))
        img2 = Thinning(img2, asarray([[-1,1,-1],[1,1,0],[-1,0,0]]))
        img2 = Thinning(img2, asarray([[0,0,0],[-1,1,-1],[1,1,1]]))
        img2 = Thinning(img2, asarray([[1,1,1],[-1,1,-1],[0,0,0]]))
        img2 = Thinning(img2, asarray([[0,-1,1],[0,1,1],[0,-1,1]]))
        img2 = Thinning(img2, asarray([[1,-1,0],[1,1,0],[1,-1,0]]))
        N = img2.sum()
        if verbose: print('Current sum:',N)
    #end while
    return(img2)
#end Skel


def Compress(img, gamma= 1.0, rel= True, verbose= False):
    """ Use a simple power law transform on the image.
        If the exponent is < 1.0 it will compress the dynamic range
        of the image, if > 1.0 it will expand it.

        Parameters:
        img     a numpy array, with whatever dimensions
        gamma:  if different from 1 and 0, apply it as an exponent
                calculating img ** gamma
        rel     bool. If set, then use img.mean()*(img/img.mean())**gamma
                for the transformation, compressing / stretching the dynamic
                range around the mean of the image

        Return:
        a numpy array with the new values
    """
    if gamma == 1.0 or gamma == 0.0:
        return(img)

    if rel:
        m = img.mean()
        return(m*(img/m)**gamma)
    else:
        return(img**gamma)
#end Compress
