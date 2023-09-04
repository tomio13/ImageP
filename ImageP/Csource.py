#!/usr/bin/env python
# Python ctypes wrapper for the C algorithms:
from ctypes import *
from ctypes.util import find_library
import numpy as nu
from numpy import ctypeslib as ct

from matplotlib import pyplot as pl
#pl.ion()

import os, sys

#pth = map(lambda  x: os.path.join(x,"ImageP"), sys.path)
#in python3 this is an iterator, not a list anymore!
pth = [os.path.join(x,"ImageP") for x in sys.path]
pth.insert(0, "./")
lsoname="libCsources.dll" if "win" in sys.platform and not "darwin" in sys.platform else "libCsources.so"

for f in pth:
    #print("searching in: %s..." %f, end=" ")
    try:
        Flib = ct.load_library(lsoname, f)
    except:
        pass
        #print("failed")
    else:
        #print("library found in %s" %f)
        break
#end for

#if all went well, we define below the following functions:
__all__=["bwfloodlist", "bwlabel", \
        "PeakFind",\
        "SimpleFilter", "SimpleFilter1D", "HitMiss", "Thinning",\
        "RankFilter", "PerimeterImage", "SimpleErode", "SimpleDilate",\
        "DistanceFilter1DL1", "DistanceFilter1D"]

#image format declaration:
array_2d_double = ct.ndpointer( dtype=nu.double,\
                ndim=2,\
                flags='C_CONTIGUOUS')
array_1d_double = ct.ndpointer( dtype=nu.double,\
                ndim=1,\
                flags='C_CONTIGUOUS')
array_2d_int = ct.ndpointer( dtype=nu.intc,\
                ndim=2,\
                flags='C_CONTIGUOUS')
array_1d_int = ct.ndpointer( dtype=nu.intc,\
                ndim=1,\
                flags='C_CONTIGUOUS')
#########################################################
#Functions:
#MinMaxMeanVar:
#to estimate the three descriptors in one run
#parameters:    image, image.size
#           pointers to: min, max, mean, var
#returns 0 if fine, -1 on error
Flib.MinMaxMeanVar.argtypes = [array_2d_double, c_int,\
                    POINTER(c_double), POINTER(c_double),\
                    POINTER(c_double), POINTER(c_double)]
Flib.MinMaxMeanVar.restype = c_int

#SimpleFilter is a sum up 'convolution' for simple cases
#much faster than FFT for small kernels (3,5,7,9 wide and high)
#we may need to define the kernels as well somewhere
#Parameters: image, width, height, 
#           kernel, kernel_width,kernel_hight
#           return_image
Flib.SimpleFilter.argtypes = [array_2d_double, c_int, c_int,\
            array_2d_double, c_int, c_int,\
            array_2d_double]
Flib.SimpleFilter.restype = c_int

#bwfloodlist finds a confluent patch in the provided image,
#erases it and writes the same pixels with NewI value to the
# result image. Returns the size of the patch found.
#parameters: image, image.shape[0], image.shape[1]
# coordinate_i, coordinate_j, gap, result_image, 
# NewI (new value to write into result_image), eight? (1 or 0)
Flib.bwfloodlist.argtypes = [array_2d_int, c_int, c_int,\
                c_int, c_int, c_int, array_2d_int, c_int, c_int]
Flib.bwfloodlist.restype = c_int

#bwlabel (pure C version)
#Parameters:
# image, image.shape[0], image.shape[1]
# result image
# MinSize: the minimum patch size seeked (>=0)
# MaxSize: the maximal patch size seeked (>=0)
# gap:  tolerated gap size
# eight: if 8 neighbours should be taken
# Returns: number of patches found
Flib.bwlabel.argtypes = [array_2d_int, c_int, c_int, array_2d_int,\
                c_int, c_int, c_int, c_int]
Flib.bwlabel.restype = c_int

#A rank filter using min, max or median value of a +/-N surroundings
#Parameters:
#image, image.shape[0], image.shape[1]
#N,M for the kernel, result_image, 'i','x' or 'm' for min, max or median
Flib.RankFilter.argtypes = [array_2d_double, c_int, c_int, c_int, c_int,\
                    array_2d_double, c_char]
Flib.RankFilter.restype = c_int

#SimpleFilter1D is a sum up 'convolution' for simple cases
#much faster than FFT for small kernels (3,5,7,9 wide and high)
#we may need to define the kernels as well somewhere
#Parameters: image, width, height,
#           kernel, kernel_lenght,
#           return_image
Flib.SimpleFilter1D.argtypes = [array_2d_double, c_int, c_int,\
            array_1d_double, c_int, array_2d_double]
Flib.SimpleFilter1D.restype = c_int

#Perimeter: scan the image and mark every pixel which are at the
#edge of a structure (less than 8 neighbours)
#the perimeter pixels are set to 1 in res -> one can already have another
#set of pixels defined, or one can use res as a filter
#Parameters:
#img:   2D integer image
#Ni,Nj: img.shape[0], img.shape[1]
#res:   2D integer image -> result
#return value:  -1 on error, 0 on success
Flib.Perimeter.argtypes = [array_2d_int, c_int, c_int,\
                            array_2d_int]
Flib.Perimeter.restype = c_int

#Hit_Miss is a special filter with a kernel being 0, 1 or undefined
#it runs as a simple filter
#we may need to define the kernels as well somewhere
#Parameters: image, width, height,
#           kernel, kernel_width,kernel_hight
#           return_image
Flib.Hit_Miss.argtypes = [array_2d_int, c_int, c_int,\
            array_2d_int, c_int, c_int,\
            array_2d_int]
Flib.Hit_Miss.restype = c_int

#Thinning is Hit_Miss subtracted from the image.
#However, we can do it within the same loop, not minding repeating the code
Flib.Thinning.argtypes = [array_2d_int, c_int, c_int,\
            array_2d_int, c_int, c_int,\
            array_2d_int]
Flib.Thinning.restype = c_int

#SimpleErode: scan the image and copy all pixels with 8 nonzero
#               neigbours
#Parameters:
#img:   2D integer image
#Ni,Nj: img.shape[0], img.shape[1]
#bg:    background value
#res:   2D integer image -> result
#return value:  -1 on error, 0 on success
Flib.SimpleErode.argtypes = [array_2d_int, c_int, c_int, c_int,\
                            array_2d_int]
Flib.SimpleErode.restype = c_int


#SimpleDilate: scan the image and copy/set all pixels with at least
#               one nonzero neighbour
#Parameters:
#img:   2D integer image
#Ni,Nj: img.shape[0], img.shape[1]
#res:   2D integer image -> result
#bg:    background value
#return value:  -1 on error, 0 on success
Flib.SimpleDilate.argtypes = [array_2d_int, c_int, c_int, c_int,\
                            array_2d_int]
Flib.SimpleDilate.restype = c_int

#PeakFind:  Find local maxima in +/- width windows
# Parameters:
# img:          2D float image
# Ni,Nj:        img.shape
# threshold:    intensity above which investigate
# width:        window size (< img.shape/2.any)
# resI, resJ:   1D int arrays to receive the hits (allocate img.size)
# return value: 0 if o.k., or the number of hits

Flib.PeakFind.argtypes =[ array_2d_double, c_int, c_int,\
            c_double, c_int, array_1d_int, array_1d_int]
Flib.PeakFind.restype = c_int

# DistanceFilter1DL1:   calculate distance transform for a 1D stripe
#                      set each pixels to the number of pixels to
#                      the closest <= 0 pixel
#                      It overwrites the input array
#                      Only positive pixels are considered as set value
#  Parameters:
#  L:                  an array of pixels (int)
#  N:                  length of array
#  squared:            if > 0 set thes quare distance, else linear one
#
#  Return:             -1 on error, or 0 for success

Flib.DistanceFilter1DL1.argtypes = [array_1d_int, c_int, c_int]
Flib.DistanceFilter1DL1.restype = c_int


#DistanceFilter1D:        an Euclidean distance filter in 1D
#                          It uses the DistanceFilter1DL1 above for the first run
#
#                          Calculate the lower parabolic envelop for L, then
#                          a distance transform.
#                         The transform is:
#                          min( (x-x')**2 + f(x')) for every x'
#                          Assume background is 0, valid values are >0
#
#  Parameters:
#  L                       an array of 0 or >0 values; it would be overwritten
#  D                       an empty array for the return values, same length as L
#  N                       length of the array (int)
#  Return value:           0 if o.k., -1 if failer
#
Flib.DistanceFilter1D.argtypes = [array_1d_int, array_1d_int, c_int]
Flib.DistanceFilter1D.restype = c_int

##################################################################
# Python wrappers :
##################################################################

def bwlabel(img, nhood=8, MinSize=1, MaxSize=0, gap=0,\
            verbose=False, details=False):
    """Find connected regions in a binary image.
        Actually the image does not have to be binary, but any
        nonzero value is a valid hit.
        This is a wrapper to a C function.

        Uses bwfloodlist() for the individual patches.

        Parameters:
        img         a numpy array with two dimensions
        nhood:      number of neighbours to look at
                    valid values are 8 (default) and 4
        MinSize:    patches less than MinSize pixels will be rejected

        MaxSize:    patches larger than MaxSize pixels will be rejected
                    if == 0: image size (nothing is rejected)

        gap:        this many pixels are omitted as gap between
                    patches. For noisy images.
                    This parameter is passed to bwfloodlist.

        verbose:    show some information
                    in this version the size of each patch is
                    not directly available, we make it visible
                    by back checking

        details:    return details of the hits as well
                    returns [NewImage,hitsI,hitsJ]
                    where hitsI is a list of I indices
                    and hitsJ that of J indices of N-1 length
                    (showing patches with values 1 to N)

        Return:
        Newimage:    an image with confluent areas marked
                    with the same number
                    The maximum of this image is the number
                    of areas found
    """
    if img.ndim != 2 :
        print("2D arrays are required")
        return

    if img.dtype != nu.uintc and img.dtype != nu.intc:
        #print("Warning, integer image is expected")
        #by default astype does a copy
        imgtmp = nu.ascontiguousarray(img.astype(nu.intc))

    else:
        imgtmp = img.copy()
    #end if

    if MaxSize <= 0:
        MaxSize = img.size

    elif MaxSize <= MinSize:
        print("Invalid MaxSize: set MaxSize > MinSize or 0")
        return
    #end if

    nh = 1 if nhood == 8 else 0

    res = nu.zeros(imgtmp.shape, dtype=nu.intc)
    #just to make sure:
    MinSize = int(MinSize)
    MaxSize = int(MaxSize)
    gap = int(gap)

    N = Flib.bwlabel( imgtmp, imgtmp.shape[0], imgtmp.shape[1],\
                res, MinSize, MaxSize, gap, nh)

    if verbose :
        print("found: %d patches" %N)
        for i in range(1,N+1):
            print("%d: %d" %(i, (res==i).sum()))
        #end for
    #end if verbose

    if details:
        DetailListX = []
        DetailListY = []
        for i in range(1,N+1):
            indxlist = (res == i).nonzero()
            DetailListX.append(indxlist[0])
            DetailListY.append(indxlist[1])
        #end for
        return [res, DetailListX, DetailListY]
    else:
        return res
#end of bwlabel


def bwfloodlist(img, x,y, nhood=8, gap=0):
    """ take a binary (or integer) image, and mark a confluent
        patch starting at (x,y) if there is any.
        Returns an array of coordinates of the confluent area (x,y)
        is within.
        A wrapper to the underlying C routine

        Parameters:
        img :   image
        x,y:    i,j indices of the starting point
        nhood:  4 or 8 neighboors to check
        gap:    check pixels this much away, even if there are 
                zero pixels between

        Return: [Is, Js]
        """

    img2 = img.astype(nu.intc)
    img3 = nu.zeros(img.shape, dtype= nu.intc)

    nh = 1 if nhood == 8 else 0
    #to make sure:
    x = int(x)
    y = int(y)
    gap = int(gap)

    try:
        N = Flib.bwfloodlist(img2, img2.shape[0], img2.shape[1],\
                    x, y, gap, img3, 1, nh)
    except:
        print("Oops, something wrong")

    else:
        if N > 0:
            indx = (img3 > 0).nonzero()
            return [indx[0],indx[1]]
        else:
            return [[],[]]
        #end if
    #end try

#end of bwfloodlist

def SimpleFilter(img, kernel):
    """ A simple numerical convolution filter for small kernels
        A wrapper to the underlying C routine
    """

    if img.ndim != 2 or kernel.ndim != 2:
        print("this function is designed for 2D objects!")
        return None
    #end if
    res = img.astype(nu.double).copy()

    Flib.SimpleFilter( img.astype(nu.double), img.shape[0], img.shape[1],\
                    kernel.astype(nu.double), kernel.shape[0], kernel.shape[1],\
                    res)
    return res
#end SimpleFilter


def SimpleFilter1D(img, kernel):
    """ A simple numerical convolution filter for small kernels
        A wrapper to the underlying C routine
    """

    if img.ndim != 2 or kernel.ndim != 1:
        print("this function is designed for 2D image and 1D kernel!")
        return None
    #end if
    res = img.astype(nu.double).copy()

    Flib.SimpleFilter1D( img.astype(nu.double), img.shape[0], img.shape[1],\
                    kernel.astype(nu.double), kernel.shape[0],\
                    res)
    return res
#end SimpleFilter1D

def HitMiss(img, kernel):
    """ A special type of filter, where the kernel has three states:
        0, 1 or undefined, best set to -1
        A simple filter is run, and pixels where the image mathes the kernel
        both in 0s ant 1s are set, the rest is left 0.
        based on:
        http://homepages.inf.ed.ac.uk/rbf/HIPR2/thin.htm

        A wrapper to the underlying C-function
    """
    if img.ndim != 2 or kernel.ndim != 2:
        print("this function is designed for 2D objects!")
        return None
    #end if
    res = nu.zeros(img.shape, dtype='intc')
    err = Flib.Hit_Miss( img.astype(nu.intc), img.shape[0], img.shape[1],\
            kernel.astype(nu.intc), kernel.shape[0], kernel.shape[1],\
            res)
    if err != 0:
        raise ValueError('error running hit_,oss filter')
    return res
#end of hit_miss

def Thinning(img, kernel):
    """ A special type of filter, where the kernel has three states:
        0, 1 or undefined, best set to -1
        This removes the pixels found in the Hit_Miss operator.

        http://homepages.inf.ed.ac.uk/rbf/HIPR2/thin.htm

        A wrapper to the underlying C-function
    """
    if img.ndim != 2 or kernel.ndim != 2:
        print("this function is designed for 2D objects!")
        return None
    #end if
    res = nu.zeros(img.shape, dtype='intc')
    err = Flib.Thinning( img.astype(nu.intc), img.shape[0], img.shape[1],\
            kernel.astype(nu.intc), kernel.shape[0], kernel.shape[1],\
            res)
    if err != 0:
        raise ValueError('error running hit_,oss filter')
    return res
#end of Thinning

def RankFilter(img,N=1,M=1,t='m'):
    """ A rank filter takes +/- N points around a pixel (except the edges)
        and takes the min, max or median of them to substitute the pixel.
        All is done on a new image, of course, the result is a nonlinear
        smoothing.
        This is a wrapper to the underlying C function.

        Parameters:
        img:    2D numpy array
        N:      integer, number of pixels taken from a pixel in X
        N:      integer, number of pixels taken from a pixel in Y
        t:      type of filter: 'i' = min, 'x' = max, 'm'=median
    """
    if img.ndim != 2 :
        print("this function is designed for 2D objects!")
        return None
    #end if
    if N < 1 or M < 0:
        print("Invalid window: %d x %d" %(2*N+1, 2*M+1))
        return None
    #end if

    N = int(N)
    M = int(M)

    res = nu.zeros(img.shape, dtype=nu.double)
    Flib.RankFilter(img, img.shape[0], img.shape[1], N, M, res, t.encode('ascii'))

    return res
#end of RankFilter


def PerimeterImage(img, WithMin=False, verbose=False):
    """ ake an image and try to define the contouring line.
        Instead of using a convolution filter, to find gradients,
        this routine goes through the image and finds pixels with
        at least one background neighbour.

        Input parameters:
            img:    Image; background is either 0 or the
                        minimum of the image if WithMin is set.
                        The image is converted to int type for the
                        C routine, without scaling

            WithMin use img.min() as bkg. value
                    otherwise bkg=0

            verbose:    plot the resulted image

        return:
            a new image, where only the periferal pixels are left,
            the others are set to bkg value
    """
    if img.ndim != 2 :
        print("Error: 2D images are required!")
        return None
    #end if

    if WithMin :
        testimg = img - img.min()
        if verbose:
            print("Using bkg of: %d" %bkg)
    else:
        testimg = img
    #end if


    res = nu.zeros(img.shape, dtype= nu.intc)
    #one option is to use astype(nu.intc) as is, or to scale:
    # (255*testimg/testimg.max()).astype(nu.intc)
    # the problem may be with the background...
    #int image is required:
    if testimg.dtype != nu.intc or testimg.dtype != nu.uintc:
        testimg = testimg.astype(nu.intc)

    #end if image is int
    if Flib.Perimeter(testimg.astype(nu.intc), \
                testimg.shape[0],testimg.shape[1], res) == -1:
        print("Error calculating the perimeter")
        return None

    if verbose:
        pl.clf()
        pl.imshow(res, origin='lower', interpolation='nearest')
        pl.draw()
    #end if

    return res.astype( img.dtype)
#end of PerimeterImage

def SimpleErode(img, times= 1, WithMin=False, verbose=False):
    """ Take an image and erode pixels which do not have 8 nonzero neighbours.

        Input parameters:
            img:        Image (integer)
            times:      number of times to erode the image
            WithMin     use img.min() as background or work on nonzero pixels
            verbose:    plot the resulted image

        return:
            a new image, with the peripheral pixels removed
    """
    if img.ndim != 2 :
        print("Error: 2D images are required!")
        return None
    #end if

    if times < 1:
        return img

    if WithMin :
        bkg = img.min()
    else:
        bkg = 0
    #end if

    if img.dtype != nu.intc or img.dtype != nu.uintc:
        img = img.astype(nu.intc)

    for i in range(int(times)):
        res = nu.zeros(img.shape, dtype= nu.intc)

        if Flib.SimpleErode(img, img.shape[0], img.shape[1],bkg, res) == -1:
            print("Error eroding the image")
            return None
        #end if

        # spare some memory
        if i > 1:
            del img

        img = res
    # end for running times times

    if verbose:
        pl.clf()
        pl.imshow( res, origin='lower', interpolation='nearest')
        pl.draw()
    #end if

    return res
#end of SimpleErode

def SimpleDilate(img, times= 1, WithMin= False, verbose= False):
    """ Take an image and dilate the pixels which do not have 8 nonzero
        neighbours. (add missing neighbours)

        Input parameters:
            img:        Image (integer)
            times:      repeat the dilation this many times
            WithMin     use img.min() as background or work on nonzero pixels
            verbose:    plot the resulted image

        return:
            a new image, with the peripheral pixels expanded
    """
    if img.ndim != 2 :
        print("Error: 2D images are required!")
        return None
    #end if
    if times < 1:
        return img.copy()

    if WithMin :
        bkg = img.min()
    else:
        bkg = 0
    #end if
    if img.dtype != nu.intc or img.dtype != nu.uintc:
        img = img.astype(nu.intc)

    for i in range(int(times)):
        res = nu.zeros(img.shape, dtype= nu.intc)

        if Flib.SimpleDilate(img, img.shape[0], img.shape[1],bkg, res) == -1:
            print("Error dilating the image")
            return None
        #end if

        if i > 1:
            del img

        img = res

    # end for

    if verbose:
        pl.clf()
        pl.imshow( res, origin='lower', interpolation='nearest')
        pl.draw()
    #end if

    return res
#end of SimpleDilate

def PeakFind(img, threshold=0.667, width=10, verbose=False):
    """Find local maxima in an image within a window defined by width.
        A piece of note: on a gradient image it runs to the edge.

        Parameters:
        img:        2D image
        threshold:  rel. threshold above which intensities are taken
                    into account
        width:      the window is pixel +/- width in both directions
        verbose:    provide some information

        return:
        a dict of 'X', 'Y' for the x,y positions.
    """
    if img.ndim != 2 :
        print("Error: 2D images are required!")
        return None
    #end if

    b = img.copy()
    b = b - b.min()
    th = b.max() * threshold
    if verbose:
        print("Threshold: %.3f: %.1f" %(threshold,th))
    #end if

    resI = nu.zeros(b.size, dtype= nu.intc)
    resJ = nu.zeros(b.size, dtype= nu.intc)

    N = Flib.PeakFind(b, b.shape[0], b.shape[1], th,\
            width, resI, resJ)

    if N < 0:
        print("error processing image")
        return {'X':nu.asarray([]), 'Y':nu.asarray([])}
    #end if
    if verbose:
        print("%d hits found" %N)
    #end if
    resI = resI.copy()[:N]
    resJ = resJ.copy()[:N]
    return {'X':resJ, 'Y':resI}
#end PeakFind

def DistanceFilter1DL1(L, squared= False):
    """  The one dimensional distance filter for an array.
        Based on P. F. Felzenszwalb and D. Huttenlocher
        Theory of Computing 8:415-428 (2012)
        A wrapper to the C function.

        Paramters:
        L       1D array, larger than 0 for objects
        squared: if True then square the values

        Return:
        1D distance values
    """

    L1 = L.copy().astype(nu.intc)
    L1[L == nu.inf] = 2**31-1
    sq = 1 if squared else 0
    errtest = Flib.DistanceFilter1DL1(L1, len(L1), sq)

    if(errtest < 0):     raise ValueError('Invalid Input')
    return L1.astype(L.dtype)
#end DistanceFilter!DL1


def DistanceFilter1D(L, inf_th= -1 ):
    """ Implement an Euclidean distance filter in 1D.

        Based on P. F. Felzenszwalb and D. Huttenlocher
        Theory of Computing 8:415-428 (2012)

        First calcualte the lower parabolic envelop for L,
        then the distance transform.
        This algorithm allows for any gray scale images, not only
        binary ones.
        The transform is defined as:
        min( (x-x')**2 + f(x')) for every x'

        This version is a bit different from the published one,
        because that one did not work on structures at the edges.
        Here we consider < inf for measuring the distance from,
        and > bg to measure the distance for...

        Parameters:
        L           a data array (1D)
        inf_th:     if >= 0, then pixels > inf_th are set to inf

        Retrurn:
        the tranformed array

        An envelop to C source
    """
    if inf_th >= 0:
        L[L > inf_th] = nu.inf

    L1 = L.copy().astype(nu.intc)

    L1[L == nu.inf] = 2**31-1
    d = nu.zeros( L1.shape, dtype= nu.intc)
    errtest = Flib.DistanceFilter1D(L1, d, len(L1))
    if( errtest < 0):    raise ValueError('Invalid Input')

    return d.astype(L.dtype)
#end DistanceFilter1D

