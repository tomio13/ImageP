#!/usr/bin/env python
from numpy import (pi, linspace, zeros, ones, sin, cos, sqrt, floor, arange,
                arctan2)
from matplotlib import pyplot as pl

from . Kernels import *
from . Convolve import *

def StructureTensor( img, keys="aetdrc", threshold= 0.01,\
        radius= 10, width= 0.5, SmoothFactor= 1.5, H= 0.04, verbose= False):
    """ Apply the structure tensor method on an image.
        This method uses local differentials to detect edges and corners.
        Possible output are the amplitude of edges, the angle derived,
        and an 'R' parameter calculated as:
        R = J11*J22-J12*J12 - 0.05*ampl**2

        Based on:
        Bernd Jähne: Digital Image Processing (5th Edition), Springer 2002
        ISBN: 3-540-67754-2
        Page: 344 - 354.

        and the appendix of:
        Phys. Rev. E 99-062401 (2019)

        Parameters:
        img:        image (2D)
        keys:       what to return? 'a' amplitude, 't' theta, angle in radians
                    'd' angle in degrees, 'r' R parameter
                    'e' egienvalues (e1, e2), 'c' for coherence

        threshold:  used to define index from the amplitude image (amp > th*amp.max())
        radius:     for the Gaussian, window is 2r+1
        width:      standard deviation of the Gaussian; increasing will give
                    larger smoothing in the image

        SmoothFactor:   increase width with this factor for the smoothing
                        operation (first calculate the derivatives, then
                        average them with a Gaussian kernel. This defines
                        the sigma for the smoothing kernel)
        H           Harris parameter for R = det(A) - H Tr^2(A),
                    where 'A" is the smoothed matrix.
        Return:
        a dict with the various output images generated according to key
    """
    ret = {}

    k = GaussKernel(radius, width, OneD=True, deriv=0);
    k2 = -GaussKernel(radius, width, OneD=True, deriv=1);
    #we use a bit broader Gaussian for smoothing
    ksmooth = GaussKernel(radius, SmoothFactor*width, OneD=True)

    #first get the two directions:
    J1 = ConvFilter1D(img, k, kernel_y=k2)
    J2 = ConvFilter1D(img, k2, kernel_y=k)

    #it is important that we smooth after the multiplication, because that is
    #not interchangeable! Thus, first gradients, then multiplication to get the
    #matrix elements, which are then smoothened to provide information about the
    #local neighborhood. Without this, they would be single point information,
    #resulting in a matrix with 0 determinant and 1 eigenvalue only!
    #the matrix elements for each pixels:
    J12 = ConvFilter1D( J1*J2, ksmooth)
    J11 = ConvFilter1D( J1*J1, ksmooth)
    J22 = ConvFilter1D( J2*J2, ksmooth)
    #J11+J22 is the absolute value square of the gradient (or second derivative)
    #IF it is close to 0, the whole thing is pointless...
    #ampl is the amplitude, also the trace of the matrix
    ampl = J11+J22
    if 'a' in keys or 'A' in keys:
        ret['a'] = ampl

    #diff: for calculating eigenvalues etc
    d = sqrt( (J11-J22)**2 + 4*J12*J12 )

    if 'e' in keys or 'E' in keys:
        l1 = 0.5*(ampl + d)
        l2 = 0.5*(ampl - d)
        ret['e1'] = l1
        ret['e2'] = l2
    #R to detect: R < 0 then edge, R >> 0 then corner, R == 0 flat:
    #0.04 an arbitrary Harris multiplier
    #det(J)- k*trace(J):
    #R = J11*J22-J12*J12 - 0.05*ampl**2
    #use det(J)/trace(J)
    # from: https://www.mathworks.com/help/visionhdl/examples/corner-detection.html
    if 'r' in keys or 'R' in keys:
        #R = (J11*J22-J12*J12)/ampl
        # J11, J22are squared before the convolution
        R = (J11*J22-J12*J12) - H*ampl**2
        ret['R'] = R
        #a corner is detected if R > a threshold

    #we filter for useful ones: enough signal both in convolution and in original image:
    #indx = (ampl > th)*(img > 0.1*img.max())
    if threshold < 0:
        threshold = 0.01

    indx = (ampl > threshold * ampl.max())
    ret['indx'] = indx

    #angle estimated from literature as:
    alpha = arctan2( 2*J12, J22-J11)/2.0
    alphaa = alpha*180/pi
    if 't' in keys or 'T' in keys:
        ret['t'] = alpha
    if 'd' in keys or 'D' in keys:
        ret['d'] = alphaa


    #coherence:
    if 'c' in keys or 'C' in keys:
        coh  = zeros(d.shape)
        coh[indx] = d[indx]/ampl[indx]
        ret['c'] = coh

    if verbose:
        pl.clf();
        pl.imshow(alphaa*indx);pl.title("alpha image");

        if 'c' in keys or 'C' in keys:
            pl.figure(2);
            pl.imshow(coh*indx);
            pl.title('coherence image');

    return ret
#end StructureTensor
