#!/usr/bin/env python
#######################################################################
""" ImageP.Convolve: some simple convolution based on numpy.FFT

    Author: Tamas Haraszti, Biophysical Chemistry group at the University of
            Heidelberg

    Copyright:  LGPL-3
    Warranty:   For any application, there is no warranty 8).
"""
try:
    from . Csource import bwlabel, Thinning, HitMiss
except:
    print("Csource is not loadable, falling back to Python sources")
    from . Pysource import bwlabel, Thinning, HitMiss
#end try


from . Kernels import *
from . Display import display, composite

from numpy import zeros, repeat, log, exp,  sqrt
from numpy import fft, concatenate, asarray

from time import time
####################################################################
__all__=["BPass","ConvFilter","ConvFilter1D",\
        "MeanFilter", "GaussDeblurr", "GaussDeblur" ,"GrayErode", "GrayDilate",\
        "DownsizeImage", "UpsizeImage", "RollingBall",\
        "EdgeDetect", "FindLocalMaxima",
        'get_nodes', 'Skel',
         ]


#####################################################################

def ConvFilter(image1, image2, MinPad=True, Smooth=False,\
        PadValue=0.0):
    """This is a Filter, to provide either smoothing or positioning
       information from images by doing convolution between image1
       and image2. It deviates a bit from the standard math when
       Smooth=False is used.

       The scipy.signal.convolve2d function is very slow,
       the fftconvolve is pretty good.
       This one was a middle step, with reasonable speed
       between the two. fftconvolve is much faster for large*large
       images, but for moderated size the two are about the same speed.

       Here we do the filtering in Fourier space.

        Then we take the F-transform, multiply, transform back and
        return the real part of the result. This is still faster than
        the built in convolution (in the scipy package).

        (The algorithm does not change the type of the data)
        Padding in 2D with the edge values will cause artefacts to form
        because of the 2D nature of the kernel, which artefacts are
        difficult to predict. We are safer to stay with the edge effect.

        Input parameters:
            image1, image2:     an image and a kernel
            MinPad:             If MinPad is True, then the two
                                images are padded to the
                                size of the sum of the dimensions
                                via zeros.
                                Else the larger size is doubled in
                                the padding (see Numerical Recipies
                                in C++, example).
        Smooth:     If True, then image2 is a kernel
                    else image2 is inverted, and
                    image2[::-1,::-1] is used.
                    This results peaks related to
                    how much image2 has to be shifted
                    to match the pattern in image1.
                    Let it be True for kernels, where
                    symmetry is not an issue.
        PadValue    use this to fill up the padding area, for the case
                    it is not 0.0

        return value:
        an image with the real part of the inverse FFT.
        If Smooth is set, then it supposed to be a smoothed image,
        else a real convolution.
    """

    r1,c1  = image1.shape
    r2,c2  = image2.shape

    #we need these FFTs:
    Fft = fft.fft2
    iFft = fft.ifft2

    #Padding to the end. Inserting to the middle would
    #break up the resulted image.
    #The window filled within:

    if MinPad:
        r_end = r1 + r2
        c_end = c1 + c2
    else :
        r_end = 2 * max(r1,r2)
        c_end = 2 * max(c1,c2)
    #end if MinPad


    #scale further up?
    lg2 = log(2)
    #round up to the next power of two:
    px2 = int(log(r_end )/lg2 + 1.0)
    py2 = int(log( c_end )/lg2 + 1.0)
    r_end = 2**px2
    c_end = 2**py2
    #now we have the final padded dimensions
    # to fill up properly, we need the pad windows around our image:

    Npad_c = int( (c_end - c1)/2 )
    Npad_r = int( (r_end - r1)/2 )
    #for the final cutting out:
    Ndiff_c = Npad_c + int( c2 / 2)
    Ndiff_r = Npad_r + int( r2 / 2)

    #build the padded image:
    pad_img = zeros( (r_end, c_end))
    #the rims: we can pad like this, but we add artefacts!
    #for i in range(r1):
    #    pad_img[i, :Npad_c] = image1[i,0]
    #    pad_img[i, Npad_c + c1:] = image1[i,-1]
    #for i in range(c1):
    #    pad_img[:Npad_r,  i] = image1[0, i]
    #    pad_img[Npad_r+r1:, i] = image1[-1, i]
    #end filling the edges
    #fill the middle:
    pad_img[ Npad_r: Npad_r+r1, Npad_c:Npad_c+c1] = image1

    #To have a good result, we have to pad up the images:
    #These FFT functions take care of the normalization as well!
    #The convolution has a rotation effect, which can be circumvented
    #by rotating the image. If used for feature identification, rotate the
    #second image! [::-1,::-1]
    if Smooth:
        fftimage = Fft(pad_img, s=(r_end,c_end)) * Fft(image2,s=(r_end,c_end))
    else :
        fftimage = Fft(pad_img, s=(r_end,c_end)) * \
                Fft(image2[::-1,::-1],s=(r_end,c_end))

    #Conjugation would turn it to correlation:
    #fftimage = Fft(image1,s=(x1,y1)) * \
    #Fft(image2, s=(x1,y1)).conjugate()*indx

    return ((iFft(fftimage))[Ndiff_r:Ndiff_r+r1, Ndiff_c:Ndiff_c+c1]).real
#end of ConvFilter

##############
def ConvFilter1D(image, kernel, MinPad=True, kernel_y=None):
    """This is a Filter, to provide smoothing with kernels, which
        can be generated from 1D arrays (Gaussian, binomial, boxcar,
        etc.).
       Here we do the filtering in Fourier space.

        Then we take the F-transform, multiply, transform back and
        return the real part of the result. This is still faster than
        the built in convolution (in the scipy package).

        Using different kernels for x- and y-direction, one can achieve
        quite complex analysis, e.g. generating gradients in one direction,
        etc.

        Run a line-by-line FFT and IFFT. Padding is done using the first
        or last point of the line (column) of the image to minimize windowing
        effects caused by the jump to 0 in the traditional zero padding.
        The resulted image is cut to its original size removing the padding.

        (The algorithm does not change the type of the data)
        The algorithm uses the real part of the inverse FFT, assuming a real
        image as a start.

        Input parameters:
        image, kernel:      an image and a kernel (this latter is 1D)
        MinPad:             If MinPad is True, then the two
                            images are padded to the
                            size of the sum of the dimensions
                            via zeros.
                            Else the larger size is doubled in
                            the padding (see Numerical Recipies
                            in C++, example).
        kernel_y:       if not None, use this kernel for the y-direction

        return value:
        an image with the real part of the inverse FFT.
    """

    r1,c1  = image.shape
    #columns are X-positions
    kc1 = kernel.shape[0]
    kr1 = kernel_y.shape[0] if kernel_y is not None else kc1

    if image.ndim != 2 :
        print("this filter is meant for 2D images")
        return None
    #end if

    #we need these FFTs:
    Fft = fft.fft
    iFft = fft.ifft

    #Padding to the end. Inserting to the middle would
    #break up the resulted image.
    #The window filled within:

    #y-direction is the i-index -> rows

    if MinPad:
        r_end_1 = r1 + kr1
        c_end_1 = c1 + kc1
    else :
        r_end_1 = 2 * max(r1, kr1)
        c_end_1 = 2 * max(c1, kc1)

    rOrig = r_end_1
    cOrig = c_end_1
    #scale further up

    lg2 = log(2)
    #round up to the next power of two:
    c_end_1= 2**int(log(c_end_1)/lg2 + 1.0)
    r_end_1 = 2**int(log(r_end_1)/lg2 + 1.0)

    #Now the job:
    resimg = zeros((r1, c1))

    #we need a symmetric padding on both ends up to the length of the FFT-d line!
    #this padding removes the jump to zero intrinsic in the fft:
    pad_N = int((c_end_1 - c1)/2)
    Ndiff = pad_N + int(kc1/2)

    #1. go along index 1.
    fftkernel = Fft(kernel,c_end_1)

    #we run through the image:
    for i in range( r1 ):
        #changes to improve padding 2019-05
        #lift the beginning and end up to the edge
        #fftline = Fft(image[i,:],y1) * fftkernel
        #resimg[i,:] = iFft(fftline)[:YOrig].real
        y = concatenate( (zeros(pad_N) + image[i,0],\
                            image[i,:],\
                            zeros(c_end_1- c1- pad_N)+ image[i,-1]) )

        #we add this to the middle of the padded image
        resimg[i , :] = iFft( Fft(y, c_end_1 ) * fftkernel )[Ndiff: Ndiff + c1].real
    #end for

    #now index 2:
    fftkernel = Fft(kernel, r_end_1) if kernel_y is None else Fft(kernel_y, r_end_1)

    pad_N = int((r_end_1 - r1)/2)
    Ndiff = pad_N + int(kr1/2)

    for i in range( c1 ):
        #fftline = Fft(resimg[:,j],x1) * fftkernel
        #resimg[:,j] = iFft(fftline)[:XOrig].real
        #we take the middle of the converted data and pad it up as before
        y = concatenate( (zeros(pad_N) + resimg[0, i ],\
                            resimg[ : , i ],\
                            zeros(r_end_1- r1- pad_N)+ resimg[ -1 ,i ]) )

        #we fill back the middle
        resimg[: , i ] = iFft( Fft(y, r_end_1 ) * fftkernel )[Ndiff :  Ndiff + r1].real

    #end for
    #It should be fine now, padding is already taken off

    return resimg
#end ConvFilter1D


def MeanFilter(image,r=5):
    """Do a simple averaging along the image in order
       to smooth out intensity fluctiations.

       It will also increase the size of the image.

       r: the range or radius of work to do.

       We do this by convolving the image with a Boxcar kernel of size r.
       To use it as a dilate algorithm, filter the image with a
       treshold, then apply the filter. Then use a 0.2-0.8 treshold
       again. The points which have enough neigbourhs will be turned to 1.
    """
    w = int(r/2)
    kernel = BoxcarKernel(w, norm=True, OneD=True)

    return ConvFilter1D(image,kernel)
#End of Mean function

def BPass(img, lobject, lnoise, AutoStretch=False):
    """This is an implementation of the bandpass routine of Grier at al.
       (and perhaps a variant as well 8) )

       The function takes an image, generates a Gaussian and a Boxcar
       mask, takes the convolcution of image with the two masks and
       returns the difference.

       The width of the Gaussian used in the original paper is different
       from the common standard deviation.
       Normal distribution has a form of exp(-x^2/(2*sigma^2)),
       the paper uses exp(-x^2/(4*width^2)), thus sigma = sqrt(2) width
       Based on the paper of Crocker et al. Journal of Colloid and Interface
       Science, vol. 179, page 298-310 (1996).

       The Gaussian should enhance + smooth features, while the Boxcar
       should surpress noise.

       Comment: 'noclip' is dropped, because the convolution filter manages
                the edges on its own.

       Comment 2:
        At the moment the code calls convolution for the two filters separately,
        which one may consider suboptimal.

       Parameters:
       lobject: length scale of the object
       lnoise:  length scale of the noise; related to the width of smoothing

       AutoStretch: (True or False(defaut) to stretch the size of the window
        (lobject) to keep it 2*lnoise size

       Return:
        the modified image
    """
    b = float(lnoise)
    w = float(lobject)

    #If the noise overruns the object size, than that scale
    #should dominate:

    if AutoStretch and w < (2*b):
        w = 2*b

    #Now the filtering:
    ## GaussKernel returns a 2*w+1 size matrix with the Gaussian filled in:
    # since our Gaussian is r^2/(2*width^2) and we want r^2/(4*b^2):
    s = sqrt(2)*b
    g = GaussKernel( int(3*s+1), width= b, norm=True, OneD=True)
    gi = ConvFilter1D(img,g)

    # BoxcarKernel returns a 2*w+1 size uniform array normalized to 1.
    # (filled up with 1/N)
    bb = BoxcarKernel(w, norm=True, OneD=True)
    bi = ConvFilter1D(img,bb)

    # the result is the difference between the smootened images
    result = gi - bi

    return result
#End of BPass()

def GaussDeblurr(img, r1=50, w1=25, r2=5, w2=2,\
        KillZero=False, verbose =False):
    """ a typo in the name, calls GaussDeblur()
    """
    return GaussDeblur(img, r1, w1, r2, w2, KillZero, verbose)
# end of GaussDeblurr

def GaussDeblur(img, r1=50, w1=25, r2=5, w2=2,\
        KillZero=False, verbose =False):
    """ Gaussian deblurr using Gauss(r1,w1)and smooth
        Such a deblurring filter is generally used in fluorescence
        microscopy to enhance contrast, as a quick and dirty approach
        instead of proper deconvolution.

        Make two Gaussian filters:
        one with a size of 2*r1+1, the other with size 2*r2+1.
        Their std is w1 and w2 (width parameter in GaussKernel)

        Both filters are normalized, and the result is clipped to its
        original size.

        Generate blurred image using kernel1, and subtract it from the
        original image.

        Convolve the result with the smoothing, secong kernel and return
        the result.
        (Warning: if there is a structure at the edges, they mey get
        distorted by the deblurring...)

        Parameters:
         r1, r2:        the size for the Gauss kernel matrix (2*r+1)
                        if r2 < 1: no smoothing is done
         w1, w2:        width parameter for the kernels
         KillZero:      if True, keep nonnegative values after the
                        background subtraction
         verbose:       if True, display image

    """

    #oneD convolution is faster:
    if r1 < 1:
        print("invalid kernel size")
        return None
    if r1 < 10 or w1 < 10:
        print("Warning: small kernel of %d, %.2f" %((2*r1+1), w1))

    k1 = GaussKernel(r1,w1, norm=True, OneD= True)

    t0 = time() if verbose else 0.0
    # we subtract the mean, we do not need to care about the minimum
    #img = img - img.min()
    imgmean = img.mean()
    img2 = ConvFilter1D(img -imgmean, k1) + imgmean

    if verbose:
        print("First convolution took %.3f seconds" %(time()-t0))

    #Subtracting blurr:
    img2 = img - img2

    if KillZero:
        img2[img2 <0] = 0
#    else:
#        img2 = img2 - img2.min()

    if r2 > 0 and w2 > 0:
        k2 = GaussKernel(r2,w2, norm=True, OneD= True)

        t0 = time() if verbose else 0.0
        #do in place to minimize memory usage (or shift it to deeper in python)
        imgmean = img2.mean()
        img2 = ConvFilter1D(img2 - imgmean, k2) + imgmean
        if verbose:
            print("Second convolution took %.3f seconds" %(time()-t0))

        #repeat killing zeros to remove small negative values
        if KillZero:
            img2[ img2 < 0] = 0

    else:
        if verbose:
            print("No smoothing (smoothing kernel is emtpy)")



    if verbose:
        display(img2, colorbar=False)

    return img2
#end of GaussDeblur


def GrayErode(image, kernel, cutZero= True):
    """ Do a grayscale erosion with a kernel. This is quite like
        the Minkowski sum for the image and the kernel.

        Result is calculated as:
        min(image(x-i, y-j) - kernel(i,j)) for all i,j at
        every x,y

        if cutZero is set, then ignore the points where the
        kernel is 0
    """
    kNi, kNj = kernel.shape

    #some kernels are there for a shape
    #zero values should be ignored
    kindx = kernel != 0

    kNi2 = int(kNi/2)
    kNj2 = int(kNj/2)

    Ni, Nj = image.shape
    result = image.copy()

    #we do using a single loop
    for ii in range(Ni*Nj):
        i = int(ii/Nj)
        j = ii %Nj
        #pixels under the kernel:

        i0 = max(0, i- kNi2)
        j0 = max(0, j- kNj2)
        i1 = min(Ni, i+ kNi2 +1)
        j1 = min(Nj, j+ kNj2 +1)
        imgpart = image[ i0:i1, j0:j1]

        #i-i0 = kNi2 or less. If less, we have to cut
        #from the ball image:
        bi0 = kNi2 - (i - i0)
        bj0 = kNj2 - (j - j0)
        bi1 = kNi - (kNi2 + 1 - (i1 - i))
        bj1 = kNj - (kNj2 + 1 - (j1 - j))
        kpart = kernel[bi0:bi1, bj0:bj1]

        #the difference:
        diff = imgpart - kpart
        result[i,j] =  (diff[kindx[bi0:bi1,bj0:bj1]]).min() if cutZero\
                        else diff.min()
    #end for ii

    return result
#end GrayErode


def GrayDilate(image, kernel, cutZero= True):
    """ Do a grayscale dilation with a kernel. This is quite like
        the Minkowski sum for the image and the kernel.

        Result is calculated as:
        max(image(x-i, y-j) + kernel(i,j)) for all i,j at
        every x,y

        if cutZero is set, then ignore the points where the
        kernel is 0
    """
    kNi, kNj = kernel.shape

    #some kernels are there for a shape
    #zero values should be ignored
    kindx = kernel != 0

    kNi2 = int(kNi/2)
    kNj2 = int(kNj/2)

    Ni, Nj = image.shape
    result = image.copy()

    #we do using a single loop
    for ii in range(Ni*Nj):
        i = int(ii/Nj)
        j = ii %Nj
        #pixels under the kernel:

        i0 = max(0, i- kNi2)
        j0 = max(0, j- kNj2)
        i1 = min(Ni, i+ kNi2 +1)
        j1 = min(Nj, j+ kNj2 +1)
        imgpart = image[ i0:i1, j0:j1]

        #i-i0 = kNi2 or less. If less, we have to cut
        #from the ball image:
        bi0 = kNi2 - (i - i0)
        bj0 = kNj2 - (j - j0)
        bi1 = kNi - (kNi2 + 1 - (i1 - i))
        bj1 = kNj - (kNj2 + 1 - (j1 - j))
        kpart = kernel[bi0:bi1, bj0:bj1]

        #the difference:
        diff = imgpart + kpart
        result[i,j] =  (diff[kindx[bi0:bi1,bj0:bj1]]).max() if cutZero\
                        else diff.max()
    #end for ii

    return result
#end GrayErode


def DownsizeImage(image, rI, rJ = 0):
    """ Reduce an image in size by averaging pixels.

        The method is a simple mean function by rearranging
        the image pixels. Thus, if the image size can not be
        divided by rI or rJ, we have a problem.

        The routine truncates the image for such cases, so be
        careful and check before calling.

        image:  the image to be reduced
        rI, rJ: reduction factors along i and j indices
                if rJ < 1, then use rI instead

        return: reduced image
    """
    if rJ < 1:
        rJ = rI

    Ni, Nj = image.shape
    Ni = Ni - Ni%rI
    Nj = Nj - Nj%rJ

    a = image[:Ni,:Nj].copy()
    if rI > 1:
        rNi = int(Ni/rI)
        a.shape = (rNi,rI,Nj)
        a = a.mean(axis=1)
    #end if rI

    if rJ > 1:
        rNj = int(Nj/rJ)
        #turn it around, so we can do the same as for Ni:
        a = a.transpose()
        a.shape = (rNj,rJ, rNi)
        #calculate and turn the result back:
        a = a.mean(axis=1).transpose()
    #end if rJ

    return a
#end DownsizeImage


def UpsizeImage(image, uI, uJ=0):
    """ Upsize the image using numpy.repeat.
        This will simply repeat the values of the image
        uI and uJ times.

        Parameters:
        image:      image to be upsized
        uI, uJ:     scaling factors to use
                    if uJ < 1 then uI = uJ

        return:
            upsized image
    """
    if uJ < 1:
        uJ = uI
    #end if
    Ni, Nj = image.shape

    if uI > 1:
        a = repeat(image, uI, axis=0)
    else:
        #we will need it for defining the end shape
        uI = 1
        a = image.copy()
    #end if uI

    if uJ > 1:
        a = repeat(a, uJ, axis=1)
    else:
        uJ = 1

    a.shape = (Ni*uI, Nj*uJ)
    return a
#end UpsizeImage

def RollingBall(image, r, reduce=4, verbose= False):
    """
        Rolling ball background calculation based on the java plugin
        written for ImageJ.

        Original source at:
        http://rsbweb.nih.gov/ij/developer/source/ij/plugin/filter/BackgroundSubtracter.java.html
        Original idea at:
        S.R. Sternberg, Biomedical Image Processing, 16:22-34 (1983)

        There is a variant at the github, which did not utilize the
        capabilities of numpy. This one is a complete rewrite, utilizing
        the numpy.ndarray features.

        An alternative version would be using GrayErode and GrayDilate
        also on a downsized image.

        image:      the image to be processed
        r:          radius of the ball to be used
        reduce:     scale both the image and the ball with this factor
                    (use DownsizeImage and UpsizeImage)
        verbose     show the kernel

        return:     background image
    """
    if r < 2:
        print("Required a ball radius >1 for processing")
        raise ValueError
    if r < 2*reduce:
        print("Radius %.2f too small to reduce with %d" %(r, reduce))
        raise ValueError

    if reduce > 2:
        reduce = int(reduce)
        #r = reduce/2
        img = DownsizeImage(image, reduce)
        r = r/reduce
    else:
        img = image.copy()
    #end if reduce

    ball = BallKernel(r)
    if verbose:
        display(ball,1)

    #image parameters:
    Ni, Nj = img.shape

    #ball parameters:

    ball_width = ball.shape[0]
    radius = int((ball_width-1) / 2)
    #kill zero pixels from the comparison
    bindx = (ball > 0)

    result = zeros(img.shape, dtype="f")

    #where it the ball standing?
    zcontrol = img[0,0]

    #the ball image hangs off +/- radius
    #we have the ball at each pixel of the image
    #running only one loop instead of 2:
    for ii in range(Ni*Nj):
        i = int(ii/Nj)
        j = ii %Nj
        #now there are pixels under the ball, either hitting
        #the ball or not
        i0 = max(0, i-radius)
        j0 = max(0, j-radius)
        i1 = min(Ni, i+radius+1)
        j1 = min(Nj, j+radius+1)
        imgpart = img[ i0:i1, j0:j1]

        #i-i0 = radius or less. If less, we have to cut
        #from the ball image:
        bi0 = radius - (i - i0)
        bj0 = radius - (j - j0)
        bi1 = ball_width - (radius + 1 - (i1 - i))
        bj1 = ball_width - (radius + 1 - (j1 - j))
        ballpart = zcontrol + ball[bi0:bi1, bj0:bj1]

        #print(ballpart)
        #print(imgpart)
        #the difference:
        diff = imgpart - ballpart
        min_z = (diff[bindx[bi0:bi1, bj0:bj1]]).min()
        if min_z < 0:
            i0 = i
            bi0 = radius

        #this is not what the paper said...
        #but a copy of what the ImageJ plugin does:
        #move the ball up or down to have it on the surface:
        zcontrol = zcontrol + min_z
        #now find and override those pixels where the ball
        #is over them:
        ballpart = zcontrol + ball[bi0:bi1, bj0:bj1]
        result[i,j] = ballpart[bindx[bi0:bi1, bj0:bj1]].max()
    #end for ii

    if reduce > 2:
        #now we have to build up our image again
        #simplest repeating scaling up:

        #result = repeat(repeat(result, reduce, axis=0),\
        #                reduce,axis=1)
        #result.shape = (img.shape[0]*reduce, img.shape[1]*reduce)
        result = UpsizeImage(result, reduce)
    return result
#RollingBall


def EdgeDetect(img, R= 5, w= 0.5, KillZero= True):
    """ Edge detector based on a Gauss kernel derivative filter
        Use the Gauss kernel in first derivative to get the X and Y
        differential image, then calcuate the square image.
        If KillZero is set, remove first the negative part of the derivative.

        Parameters:
        img     image to be analyzed
        R       2*R+1 is the window of the Gaussian
        w       sigma of the Gaussian
        KillZero    (True be default), remove the negative part to kill double
                    edges

        return
        the new image
    """
    if R <= 0 or w <= 0:
        raise ValueError("Invalid kernel parameters")

    gk = -GaussKernel(R, w,  OneD= 1, deriv= 1)
    gk2 = GaussKernel(R, w, OneD= 1)

    bx = ConvFilter1D(img, gk, kernel_y= gk2)
    by = ConvFilter1D(img, gk2, kernel_y= gk)

    if KillZero:
        return sqrt((bx > 0)*bx**2 + (by > 0)*by**2)
    else:
        return sqrt(bx**2 + by**2)
#end EdgeDetect

def FindLocalMaxima( img, R= 10, w= 1.0, height= 0.5, bg = None,\
                deriv= 2, verbose= False):
    """ Find local ridge / maxima in the image scanning in X and Y using the differential
        image through a Gaussian kernel

        Parameters:
        img:    a 2D image
        R       window size is 2R+1
        w       sigma of the Gaussian
        height  minimum change, > height is filtered
        bg      if specified a background minimum for the image
                All negative pixels are deleted after subtraction
        img > bg is used in the peak identification
        deriv   1 or 2 the derivative to use

        return
        an image highlighting the maxima
    """
    if R <= 0 or w <= 0:
        raise ValueError("Invalid kernel parameters")

    if deriv > 2 or deriv < 1:
        raise ValueError('Derivative must be 1 or 2')

    #use negative kernels. For the derivative the FFT makes it right
    #for ridges the logic for the gradient filter finds the negative areas
    #first, so it will pick the negative peaks, thus working just right
    #for both cases
    gk = GaussKernel( R, w,  OneD= 1, deriv= deriv)
    gk2 = GaussKernel( R, w, OneD= 1)

    if bg is not None:
        img2 = img - bg
        img2[ img2 < 0 ] = 0
    else:
        img2 = img

    #it seems that the FFT convolution inverts the kernel:
    bx = ConvFilter1D( img2, gk, kernel_y= gk2)
    #gk2 is symmetric, no problem there
    by = ConvFilter1D( img2, gk2, kernel_y= gk)
    if verbose:
        display(bx, fN=2)
        display(by, fN=3)
    w = int(w) if w >= 1 else 1
#    x_indx = (bx[1:,:] - bx[:-1,:]) > height
#    y_indx = (by[:,1:] - by[:,:-1]) > height
    #derivateive becomes negative right after the peak
    x_indx = (bx < -height).astype('int')
    y_indx = (by < -height).astype('int')
    #now we seek the change:
    x_i = (x_indx[:,w:] - x_indx[:,:-w]) >= 1
    y_i = (y_indx[w:,:] - y_indx[:-w,:]) >= 1

    #x_indx is 1 smaller in x, but not in y, so we shorten it
    #y_indx is 1 smaller in y, but not in x, so we shorten it
    x,y = (x_i[w:,:] | y_i[:,w:]).nonzero()
    c = zeros(img.shape)
    c[x+w,y+w] = 1
    return c
#end FindLocalMaxima


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


def get_nodes(img, kernels= cross_kernels):
    """ find nodes in a skeleton image using kernels and the
        HitMiss filter.

        parameters:
        img:        a 2D numpy array (image)
        kernels:    a list of the node defining kernels (3x3)

        return:
        a tuple, with the number of nodes found and a binary
        image indicating them
        (a node may be larger than one pixel)
    """
    res = zeros(img.shape)
    for k in kernels:
        res = res + HitMiss(img, k)

    nodes = bwlabel(res > 0)
    return (nodes.max(), nodes >0)
# end of get_nodes
