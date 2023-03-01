#!/usr/bin/env python
#Python equivalents if the C ones are missing (the old ImageP lib)
import numpy as nu
from numpy import histogram, zeros, ones, arange, \
        asarray, resize, log, exp, indices, sqrt, inf

from numpy import sin, cos, floor, abs, diff, uint

#for testing only
#from matplotlib import use
#use('GTK')
#from matplotlib import pyplot as pl
#pl.ion()

# We define in this module:
__all__=["bwfloodlist", "bwlabel",\
        "PeakFind",\
        "SimpleFilter","SimpleFilter1D", "HitMiss", "Thinning",\
        "RankFilter", "PerimeterImage", "SimpleErode", "SimpleDilate",\
        "DistanceFilter1DL1", "DistanceFilter1D"]

##################################################################
# Python functions
##################################################################
def bwlabel(img, nhood=8, MinSize=1, MaxSize=0, gap=0, verbose=False, details=False):
    """ Find connected regions in a binary image.
        If the image is not binary, then convert it to one
        using img > img.min() as a criteria.

        Uses bwfloodlist() for the individual patches.

        Parameters:
        img:        a numpy array with two dimensions
        nhood:      number of neighbours to look at
                    valid values are 8 (default) and 4
        MinSize:    patches less than MinSize pixels will be rejected

        MaxSize:    patches larger than MaxSize pixels will be rejected
                    if == 0: image size (nothing is rejected)


        gap:    this many pixels are omitted as gap between
                patches. For noisy images.
                This parameter is passed to bwfloodlist.

        verbose:    show some information
        details:    return details of the hits as well
                    returns [NewImage,hitsI,hitsJ]
                    where hitsI is a list of I indices
                    and hitsJ that of J indices of length N-1
                    (patch values of 1 to N)

        Return:
        Newimage:   an image with confluent areas marked
                    with the same number
                    The maximum of this image is the number
                    of areas found
    """

    if img.ndim != 2 :
        print("2D arrays are required")
        return

    #If not a binary image, then convert to one:
    if img.max() != 1 and img.min() != 0:
        print("Not a binary image, converting to one:")
        imgtmp = (img > img.min())
    else :
        imgtmp = img.copy()

    if imgtmp.sum() < 2 :
        print("Empty image")
        return
    #end if

    if MaxSize <= 0:
        MaxSize = img.size

    elif MaxSize <= MinSize:
        print("Invalid MaxSize: set MaxSize > MinSize or 0")
        return
    #end if

    NewImg = zeros((imgtmp.shape),dtype=int)
    DetailListX = []
    DetailListY = []

    marker = 1

    #Fetch the points:
    tmpc = imgtmp.nonzero()

    for (x,y) in zip(tmpc[0],tmpc[1]) :

        if imgtmp[x,y] != 0 :
            (HitsX,HitsY) = bwfloodlist(imgtmp,x,y,nhood,gap)
            HitsX = asarray(HitsX)
            HitsY = asarray(HitsY)

            #positive logic: is n in [MinSize,MaxSize]?
            lH = len(HitsX)
            if lH >= MinSize and lH <= MaxSize:
                if verbose:
                    print("Found: %d" %(len(HitsX)))

                NewImg[HitsX,HitsY] = marker

                if details :
                    DetailListX.append(HitsX)
                    DetailListY.append(HitsY)
                marker = marker + 1

            #Remove the known ones:
            imgtmp[HitsX, HitsY] = 0

    if details :
        return (NewImg, DetailListX, DetailListY)

    else :
        return NewImg

# end of bwlabel

def bwfloodlist(img, x, y, nhood=8, gap=0):
    """ take a binary image, and if (x,y) is a nonzero pixel,
        return a list of coordinates of the confluent area this
        pixel is within.

        Based on the code from Eric S. Raymond posted to:
        http://mail.python.org/pipermail/image-sig/2005-September/003559.html

        Parameters:
        img :   a binary image
        x,y:    start coordinates
        nhood:  number of neighbours to take into accound
                8 (default) or 4
        gap:    gap number of empty pixels are tolerated, thus
                two patches this much away are still accepted as
                one. The 4 or all directionality still exists.

        Return:
          [I,J] a list of indices
    """

    if img.ndim != 2 :
        print("2D arrays are required")
        return

    #If not a binary image, then convert to one:
    if img.max() != 1 and img.min() != 0:
        print("Not a binary image, converting to one:")
        imgtmp = (img > img.min())
    else :
        imgtmp = img.copy()

    Ni,Nj = imgtmp.shape

    edge = [[x,y]]
    IList = [x]
    JList = [y]

    while edge:
        newedge = []

        #Check all pixels:
        for (x,y) in edge:

            if nhood == 4 :
                steplist = []

                for si in range(x-gap-1,x+gap+2):
                    steplist.append([si,y])
                #end for
                for sj in range(y-gap-1,y+gap+2):
                    steplist.append([x,sj])
                #end for
                steplist.remove([x,y])

            else :
                steplist = []
                for si in range(x-gap-1,x+gap+2):
                    for sj in range(y-gap-1,y+gap+2):
                        steplist.append([si,sj])
                    #end for sj
                #end for si
                steplist.remove([x,y])
            #end if

            for (i,j) in steplist :
                #protect for out of range indices:
                if i < 0 or j < 0 or i >= Ni or j >= Nj :
                    continue

                elif imgtmp[i,j] :
                    imgtmp[i,j] = 0
                    #store the point to further
                    #examination:
                    newedge.append((i,j))

                    #and it is also a hit:
                    IList.append(i)
                    JList.append(j)

            #End of marking close neighbors
        #End of going through this edges

        edge = newedge
    #End of check all points (while)

    return [IList,JList]

#end of bwfloodlist

def SimpleFilter(img, kernel):
    """ A simple filter to replace convolution for small kernels.
        Pure python version
    """
#    kernel = asarray([[0,1,0],[1,-4,1],[0,1,0]])

    if img.ndim != 2 or kernel.ndim != 2:
        print("this function is designed for 2D objects!")
        return None
    #end if

    img2 = img.copy()
    Ni, Nj = kernel.shape
    INi, INj = img.shape
    Ni2 = int(Ni/2)
    Nj2 = int(Nj/2)

    ni = INi-Ni+1
    nj = INj-Nj+1
    #print("ni,nj:", ni, nj)
    img2 = zeros((INi,INj))
    #img2  = img.copy()
    img2[Ni2:Ni2+ni, Nj2:Nj2+nj] = 0

    for i in range(Ni):
        for j in range(Nj):
            if kernel[i,j]:
                #ni = max(0, INi+i-Ni+1)

                img2[Ni2:Ni2+ni, Nj2:Nj2+nj]= img2[Ni2:Ni2+ni, Nj2:Nj2+nj] + \
                        kernel[i,j]*img[i:i+ni, j:j+nj]
            else:
                pass
            #end if
        #end for j
    #end for i
    return img2
#end SimpleFilter

def SimpleFilter1D(img, kernel):
    """ A simple filter to replace convolution for small kernels.
        This is for 1D kernels only.
        Pure python version
    """
#    kernel = asarray([[0,1,0],[1,-4,1],[0,1,0]])

    if img.ndim != 2 or kernel.ndim != 1:
        print("this function is designed for 2D images and 1D kernels!")
        return None
    #end if

    img2 = img.copy()
    Ni = kernel.shape
    INi, INj = img.shape

    ni = INi-Ni+1
    nj = INj-Ni+1
    #img2 = zeros((ni,nj))
    img2  = img.copy()
    Ni2= int(Ni/2)
    Nj2 = int(Nj/2)

    img2[Ni2: Ni2+ni, Nj2:Nj2+nj] = 0

    #First run, along j...
    for i in range(Ni):
        if kernel[i]:
            ni = max(0, INi+i-Ni+1)
            img2[Ni2: Ni2+ni, Nj2:Nj2+nj]= img2[Ni2:Ni2+ni,Nj2:Nj2+nj]\
                    + kernel[i]*img[i:INi+i-Ni+1, Nj2:Nj2+nj]
        else:
            pass
        #end if
    #end for i

    #Second run
    for i in range(Ni):
        if kernel[i]:
            ni = max(0, INi+i-Ni+1)
            img2[Ni2: Ni2+ni, Nj2:Nj2+nj]= img2[Ni2:Ni2+ni,Nj2:Nj2+nj]\
                    + kernel[i]*img[Ni2:Ni2+ni, i:INj+i-Nj+1]
        else:
            pass
        #end if
    #end for i
    return img2
#end SimpleFilter

def HitMiss( img, kernel):
    """ A special type of filter, where the kernel has three states:
        0, 1 or undefined, best set to -1
        A simple filter is run, and pixels where the image mathes the kernel
        both in 0s ant 1s are set, the rest is left 0.
        based on:
        http://homepages.inf.ed.ac.uk/rbf/HIPR2/thin.htm

        A wrapper to the underlying C-function
    """

    if kernel.shape[0] != 3 and kernel.shape[1] !=3:
        raise ValueError('Invalid kernel size')

    k1 = kernel == 1
    k2 = kernel == 0
    img2 = (SimpleFilter(img, k1) == k1.sum()) &\
            (SimpleFilter(img == 0, k2) ==  k2.sum())
    return img2
#end thinning


def RankFilter(img,N=1,M=1,t='m'):
    """ A rank filter takes +/- N points around a pixel (except the edges)
        and takes the min, max or median of them to substitute the pixel.
        All is done on a new image, of course, the result is a nonlinear
        smoothing.

        Parameters:
        img:    2D numpy array
        N:      integer, number of pixels taken from a pixel in X
        N:      integer, number of pixels taken from a pixel in Y
        t:      type of filter: 'i' = min, 'x' = max, 'm'=median
    """
    if img.ndim != 2 or kernel.ndim != 2:
        print("this function is designed for 2D objects!")
        return None
    #end if
    if N < 1 or M < 0:
        print("Invalid window: %d x %d" %(2*N+1, 2*M+1))
    #end if

    #this is terrible and uggly, running through all pixels!
    Ni,Nj = img.shape
    res = zeros(img.shape)

    for i in range(Ni):
        for j in range(Nj):
            i0 = max(0, i-N)
            i1 = min(i+N+1, Ni)
            j0 = max(0, j-N)
            j1 = min(j+N+1, Nj)
            
            slice = img[i0:i1, j0:j1]

            if( t == 'i' ):
                res[i,j] = slice.min()
            elif( t == 'j' ):
                res[i,j] = slice.max()
            else:
                res[i,j] = (slice.sort(kind='mergesort'))[N/2]
            #end if
        #end for j
    #end for i
#end of RankFilter

def PerimeterImage(img, WithMin=False, verbose=False):
    """	Take an image and try to define the contouring line.
        Instead of using a convolution filter, to find gradients,
        this routine goes through the image and finds pixels with
        at least one background neighbour.

        Input parameters:
            img:		Image
            WithMin		use img.min() as bkg. value
                    otherwise bkg=0
            verbose:	plot the resulted image

        return:
            a new image, where only the periferal pixels are left,
            the others are set to bkg value
    """

    if img.ndim != 2 :
        print("Error: 2D images are required!")
        return None
    #end if

    if WithMin :
        bkg = img.min()
    else:
        bkg = 0
    #end if

    if verbose :
        print("Using background: %.3f" %bkg)
    #end if

    #Create a list of pixels to be investigated:
    indx = (img != bkg).nonzero()

    Ni,Nj = img.shape
    imgresult = zeros(img.shape)
    
    for (x,y) in zip(indx[0],indx[1]) :

        #Edge pixels are not useable
        if x > 0 and x < Ni-1 and y > 0 and y < Nj-1:
        
            #generate indices for the neighbours:
            steplist0 = asarray([x-1,x,x+1,x,x-1,x+1,x-1,x+1])
            steplist1 =  asarray([y,y-1,y,y+1,y-1,y+1,y+1,y-1]) 

            #add pixels which has at least 1 bkg neighbor:
            #the same as the C code works.
            if (img[steplist0,steplist1] == bkg).any() :
                imgresult[x,y] = 1
            #end if
        #end if
        #this way peripheral pixels are left untouched
    #end for

    if verbose:
        pl.clf()
        pl.imshow(imgresult, origin='lower',\
            interpolation = "nearest")
        pl.draw()
    #end if

    return imgresult
#End of PerimeterImage

def SimpleErode(img, WithMin=False, verbose=False):
    """ Take an image and remove the edge pixels based on the 8 neighbours.
        Instead of using a convolution filter, to find gradients,
        this routine goes through the image and finds pixels with
        at least one background neighbour.

        Input parameters:
            img:        Image
            WithMin     use img.min() as bkg. value
                        otherwise bkg=0
            verbose:    plot the resulted image

        return:
            a new image, where only the periferal pixels are removed
    """

    if img.ndim != 2 :
        print("Error: 2D images are required!")
        return None
    #end if

    if WithMin :
        bkg = img.min()
    else:
        bkg = 0
    #end if

    if verbose :
        print("Using background: %.3f" %bkg)
    #end if

    #Create a list of pixels to be investigated:
    indx = (img != bkg).nonzero()

    Ni,Nj = img.shape
    imgresult = img.copy()

    for (x,y) in zip(indx[0],indx[1]) :

        #Edge pixels are not useable
        if x > 0 and x < Ni-1 and y > 0 and y < Nj-1:

            #generate indices for the neighbours:
            steplist0 = asarray([x-1,x,x+1,x,x-1,x+1,x-1,x+1])
            steplist1 =  asarray([y,y-1,y,y+1,y-1,y+1,y+1,y-1])

            #delete all pixels in the bulk:
            if (img[steplist0,steplist1] == bkg).any() :
                imgresult[x,y] = bkg
            #end if
        #end if
        #this way peripheral pixels are left untouched
    #end for

    if verbose:
        pl.clf()
        pl.imshow(imgresult, origin='lower',\
            interpolation = "nearest")
        pl.draw()
    #end if

    return imgresult
#End of SimpleErode

def SimpleDilate(img, WithMin=False, verbose=False):
    """ Take an image and expand the edges with mixing pixels.
        Instead of using a convolution filter,this routine goes
        through the image and finds pixels with at least one
        background neighbour, then fills up the blanks.

        Input parameters:
            img:        Image
            WithMin     use img.min() as bkg. value
                        otherwise bkg=0
            verbose:    plot the resulted image

        return:
            a new image, where the peripheral pixels are extended
    """

    if img.ndim != 2 :
        print("Error: 2D images are required!")
        return None
    #end if

    if WithMin :
        bkg = img.min()
    else:
        bkg = 0
    #end if

    if verbose :
        print("Using background: %.3f" %bkg)
    #end if

    #Create a list of pixels to be investigated:
    indx = (img != bkg).nonzero()

    Ni,Nj = img.shape
    imgresult = img.copy()

    for (x,y) in zip(indx[0],indx[1]) :

        #Edge pixels are not useable
        if x > 0 and x < Ni-1 and y > 0 and y < Nj-1:

            #generate indices for the neighbours:
            steplist0 = asarray([x-1,x,x+1,x,x-1,x+1,x-1,x+1])
            steplist1 =  asarray([y,y-1,y,y+1,y-1,y+1,y+1,y-1])

            #delete all pixels in the bulk:
            if (img[steplist0,steplist1] == bkg).any():
                indx2 = (img[steplist0, steplist1] == bkg).nonzero()[0]
                imgresult[steplist0[indx2],steplist1[indx2]] = \
                            (img[steplist0,steplist1]).max()
            #end if
        #end if
        #this way peripheral pixels are left untouched
    #end for

    if verbose:
        pl.clf()
        pl.imshow(imgresult, origin='lower',\
            interpolation = "nearest")
        pl.draw()
    #end if

    return imgresult
#End of SimpleDilate

def PeakFind(img, threshold=0.667, width=10, verbose=False):
    """Searching confluent spots, based on the bwlabel algorithm.

       img:             a 2D image
       threshold:       peaks above img.max()*threshold are seeked
       width:           width. The +/- width area around a pixel is used to
                        identify the local maximum.

        verbose:    provide some feedback

       Return value:
            a directory with 'X', 'Y' and 'Size values.
    """

    #We need this for correct comparison:
    img2 = img.copy()
    img2 = img2 - img2.min()
    minimum = img2.max() * threshold

    Ni,Nj = img2.shape

    indximg = (img2 > minimum)
    indx = indximg.nonzero()
    Is = indx[0]
    Js = indx[1]
    N = Is.size

    if verbose :
        print("Found %d hits\n" %N)

    #pythonish: go for lists
    hitsI = []
    hitsJ = []

    if N < 1:
        print("Zero hits")
        return {'X':asarray(hitsJ),'Y': asarray(hitsI)}
    #end if

    #Now go through the interesting points:
    i = 0
    for i in range(N):
        ni = Is[i]
        nj = Js[i]
        currpoint = img2[ni,nj]

        #examine only the remaining ones:
        if currpoint > 0:
            #now, let us see: is it a local maximum?
            #First we need the window around the pixel:

            i0 = max(0, ni-width)
            j0 = max(0, nj-width)
            i1 = min( Ni, ni+width+1)
            j1 = min( Nj, nj+width+1)

            maxint = img2[i0:i1,j0:j1].max()

            if maxint == currpoint:
                #we have a positive hit, go on:
                hitsI.append(ni)
                hitsJ.append(nj)
                #kill all others, that we do not have to check
                #everything even multiple times:
                img2[i0:i1, j0:j1] = 0

            #else: nothing to do, go on.
            #end if
        #end if
    #end for i
    hitsI = asarray(hitsI)
    hitsJ = asarray(hitsJ)

    if verbose:
        print("Found %d hits" %(hitsI.size))

    return {'X':hitsJ,'Y':hitsI}
#End of PeakFind()


def DistanceFilter1DL1(L, squared= False):
    """ The one dimensional distance filter for an array.
        Based on P. F. Felzenszwalb and D. Huttenlocher
        Theory of Computing 8:415-428 (2012)
    """
    #master = (L > 0)
    N = len(L)
    res = L.copy()

    #indx = master.nonzero()
    #we generate the results into res
    #res = zeros(N)
    #res[indx] = inf

    if (res == 0).all():
        return res

    ilist = res.nonzero()[0]
    uplist = ilist[1:] if ilist[0] == 0 else ilist
    downlist = ilist[:-1:-1] if ilist[-1] == 0 else ilist[::-1]

    if uplist.size > 0:
        #for i in range(1, N):
        for i in range(1, N):
            res[i] = min( res[i], res[i-1]+1)

    if downlist.size > 0:
        #for i in range(N-1)[::-1]:
        for i in range(N-1)[::-1]:
            res[i] = min(res[i], res[i+1]+1)

    if squared: res = res*res
    return res
#end DistanceFilter1DL1

def DistanceFilter1D(L, inf_th = -1):
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
    """
    N = len(L)
    if N < 2:
        raise(ValueError, "too short array!")
    #end if len
    res= zeros(N, order='C')
    #the parabola crossing is coming from:
    # s^2 - 2 si + i^2 + L[i] = s^2 - 2 sj + j^2 + L[j]
    #Issues arise when the line starts it inf values... In such a case the
    #algorithm wants to go backwards on an empty list making trouble
    #so, test this first:
    #indx = arange(N)[(L < inf)] -> using nonzero is 2x faster
    if( inf_th >= 0):
        L[ L > inf_th] = inf

    indx = (L < inf).nonzero()[0]

    if len(indx) <1 :
        #only inf array
        return L
    #end if inf array

    k = 0
    z = [indx[0]]
    v = [indx[0]]

    #if the image is not 0 at the edges, this algorithm goes silly...
    #we deviate from the original: only background and non  infinity
    #poits contribute to the hull. Infinite elements would pull the
    #crossing point to -inf or inf, making trouble
    #Thus a first run eliminates all unnecessary points in indx:
    #(it is an overhead to run through the array, but pays off...
    for i in indx[1:]:
        #the while should run free
        # while (True):
        while (k >=0 and k < len(v)):
            vk = v[k]
            ds = 2.0*(i-vk)
            s = (i*i - vk*vk + L[i] - L[vk])/ds
#            s = (i*i - vk*vk)/ds
            #inf - inf makes trouble in the calculation, so
            #we filter for it:
#            if L[i] == inf:
#                if L[vk] == inf:
#                    s = inf
#                #else leave s, quietly meaning inf-inf = 0
#            else:
#                s += (L[i] - L[vk])/ds
            #end if, sorted out s

            #the first one has to be added:
            if s <= z[k]:
                k -= 1
                v.pop()
                z.pop()
            else:
                break #stop this while loop, do something else
        #end while
        k += 1
        v.append(i)
        z.append(s)
    #end for

    #print("v", v)
    #print("z", z)
    Nz = len(z)
    k = 0
    # indx = (L > bg).nonzero()[0]
    indx = (L < inf).nonzero()[0]
    z = nu.asarray(z)

    for i in indx:
        #we want to use z[k] where z[k] < i, but z[k+1] > i
        #this we fix with the k < (N-1) limit:
        while (k < (Nz-1)  and z[k+1] < i):    k += 1
        #end while, k is set:
        #tried with (z[k+1:] > i).nonzero(), but it was even slower...

        res[i] = (i- v[k])**2 + L[v[k]]

    return res
#end DistanceFilter1D


def Thinning(img, kernel):
    """ Erosion based on the hit and miss operator
        The identified pixels are killed to 0
        Parameter:
        img     binary image
        kernel  three state kernel: 0, 1 and unset as -1
    """
    img2 = img.copy()
    #img2[ hit_miss(img, kernel) > 0] = 0
    img2[ HitMiss(img, kernel) > 0] = 0
    return img2
#end Thinning

