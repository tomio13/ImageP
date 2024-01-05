#!/usr/bin/env python
""" ImageP.Kernels: Kernel functions and definitions.
        There are several possibilities, depending on what one wants.

    Author: Tamas Haraszti, Biophysical Chemistry group at the University of
        Heidelberg

    Copyright:  LGPL-3
    Warranty:   For any application, there is no warranty 8).

"""

from numpy import (zeros, asarray, arange, indices, log, exp, sqrt,
                    abs, meshgrid, pi, sin, cos, floor)

#######################################################
# List of functions defined / exported here:
__all__=['cross_kernels', 'BoxcarKernel', 'BinomKernel', 'CircMask',\
        'GaussKernel', 'RotatingGaussKernel',\
        'SoebelKernel', 'BallKernel']

cross_kernels=[
        # left right top bottom
        asarray([[-1,-1,-1],[1,1,1],[-1, 1, -1]]),
        asarray([[-1,1,-1],[1,1,1],[-1, -1, -1]]),
        asarray([[-1,1,-1],[1,1,-1],[-1, 1, -1]]),
        asarray([[-1,1,-1],[-1,1,1],[-1, 1, -1]]),
        # diagonals
        asarray([[1,-1,1],[-1,1,-1],[1, -1, -1]]),
        asarray([[1,-1,-1],[-1,1,-1],[1, -1, 1]]),
        asarray([[-1,-1,1],[-1,1,-1],[1, -1, 1]]),
        asarray([[1,-1,1],[-1,1,-1],[-1, -1, 1]]),
        # Y-s
        asarray([[1,-1,1],[-1,1,-1],[-1, 1, -1]]),
        asarray([[-1,1,-1],[-1,1,-1],[1, -1, 1]]),
        asarray([[-1,-1,1],[1,1,-1],[-1, -1, 1]]),
        asarray([[1,-1,-1],[-1,1,1],[1, -1, -1]]),
        # diagonal Y-s
        asarray([[-1,1,-1],[1,1,-1],[-1, -1, 1]]),
        asarray([[-1,-1,1],[1,1,-1],[-1, 1, -1]]),
        asarray([[-1,1,-1],[-1,1,1],[1, -1, -1]]),
        asarray([[1,-1,-1],[-1,1,1],[-1, 1, -1]]),
        ]


def GaussKernel(r=10, width=0, FWHM=0, norm=True, OneD=False, deriv=0, direction='X'):
    """Generate a 1D or 2D matrix with 2r+1 size containing a
        normalized Gaussian function:

            exp(-4*ln(2)*(x-r)^2/FWHM^2)/sum
            or
            exp(- (x-r)^2/(2*width^2)) / sum

        The width is the standard deviation of the normal distribution.

        The FWHM is the full width (-x to +x) at the half maximum
        of the curve. If width is specified, it is the half distance
        of the two inflection points of the curve.
        FWHM = 2*sqrt(2 * ln(2))*width

        If width is not specified, use r/4 for it, so we have some reasonable
        run off of the Gaussian.

        norm:   is used to normalize to unit sum of the curve.
        OneD:     if True, return only a 1D array (direction becomes meaningless)
        deriv:  instead of the Gauss, give the derivative (1st or 2nd order)
                values: 0,1,2
        direction: 'X' or 'Y', which direction to be derived if OneD is False

        return:
            an image containing the Gaussian
    """

    #(FWHM/2)^2 = 2 log(2) sigma^2 ==> sigma = sqrt(FWHM / (8*log(2)))
    # width is sigma
    size = 2*r+1

    if width == 0 and FWHM == 0:
        #set width to r/2 for a decent Gaussian:
        width = r/4.0

    elif width == 0:
        # for FWHM:the Gaussian is exp(-x^2/(2 sigma^2) = 1/2 on both sides
        # thus FWHM = 2 x = 2 sqrt(2 ln(2)) sigma
        # width = sigma = FWHM /sqrt(8 ln(2))
        width = FWHM/sqrt(8.0*log(2.0))
        #or exp(-(x-r)^2/(2*sigma^2))
    #else: we have width defined
    #end if
    const = 1.0/(2.0*width**2)
    nfactor = 1 / ( sqrt(2*pi) * width)

    #do it in OneD:
    x = arange(size, dtype=float) - r
    kernel1 = exp(-const*(x**2))
    kernel2 = kernel1.copy()
    if norm:
        kernel2 = kernel2 * nfactor
        kernel1 = kernel1 * nfactor

    if deriv == 1:
            kernel1 = -2.0*const*x*kernel1

    elif deriv == 2:
            #const is already negative!
            kernel1 = 2.0*const*(2.0*const*x**2 - 1.0)*kernel1

    #now this gets a 2D array of k[i,j] = a[0,i]*b[j,0]
    if not OneD:
        kernel1.shape = (kernel1.shape[0],1)
        kernel2.shape = (1,kernel1.shape[0])

        kernel = kernel1*kernel2
        if direction == "Y":
            kernel = kernel.transpose()
    else:
        kernel = kernel1.copy()
    #Utilize the two index matrices generated by mgrid:
#    kernelcore = indices((size,size),dtype=float)

#    kernel = exp(const*((kernelcore[0]-r)**2 +(kernelcore[1]-r)**2))

    # Renormalization: if the kernel is too narrow, the standard
    # Gaussian normalization does not apply, because we miss
    # a good part of the edges.
    # Further, the kernel will have a nonzero offset because of
    # the same effect
    if norm and r < 3*width:
        kernel = kernel - kernel.min()
        kernel = kernel/kernel.sum()
    #end if too narraow

    return kernel
#End of GaussKernel

def RotatingGaussKernel(sx, sy, angle=0, shape=[-1,-1], deriv=0):
    """ Define an elliptic Gaussian kernel, with an orientation
        angle. If necessary, use the sum of the first (gradient)
        or the second derivative (Laplace filtered kernel).
        The main kernel is defined using the standard deviations as:

         1/(2 \pi s_x s_y) exp( - x^2/(2 s_x^2) - y^2/(2 s_y^2))

        Parameters:
            sx, sy: sigma in the i and j direction before rotation

            angle   angle of rotation in degrees
            shape   desired shape of the 2D array
                    if not specified, use 10sx+1 and 10sy+1
                    or at least 3 points
            deriv:  you can get some derivative instead of the
                    original Gaussian.
                    deriv 1 will give [df/dx, df/dy]
                    deriv 2 will give d^2f/dx^2 + d^2f/dy^2 (Laplace)

        Return
            a 2D array containing the Gauss function
    """

    if sx <= 0.0 or sy <= 0.0:
        raise ValueError("Invalid sigma value!")
    #end if

    if shape[0] < 3:
        print("X shape not defined, setting to 10sx + 1 or 3")
        shape[0] = max(int(10*sx+1), 3)
    if shape[1] < 3:
        print("Y shape not defined, setting to 10sy +1 or 3")
        shape[1] = max(int(10*sy+1), 3)
    #end if
    Nx2 = int(shape[0]/2)
    Ny2 = int(shape[1]/2)

    #define the x,y coordinates centered around the middle:
    x,y = meshgrid(arange(-Nx2,Nx2+1), arange(-Ny2,Ny2+1), indexing='xy')

    #The Gaussian is sx, sy in the rotated coordinate system,
    #which has an angle alpha vs. our current one.
    #We have to rotate BACK from there with -alpha
    #prepare some constants for the calculation:
    sx2 = 2.0*sx*sx
    sy2 = 2.0*sy*sy
    norm = 1.0/sqrt(2*pi*sx*sy)
    #use - angle:
    if angle > 360:
        angle = angle - floor(angle/360)*360

    rad = -angle*pi/180.0

    ca = cos(rad) if angle != 90 and angle != 270 else 0.0
    #actually sin is robust with 0 but has residues at 180
    sa = sin(rad) if angle != 0 and angle != 180 else 0.0

    #the rotation and the square operators together give:
    A = ca*ca/sx2 + sa*sa/sy2
    B = sin(2.0*rad)*(1/sy2 - 1/sx2)
    C = sa*sa/sx2 + ca*ca/sy2

    f = norm*exp(-(A*x*x + B*x*y + C*y*y))
    if deriv == 1 :
        #thanks to the Gauss, the original f can be used to calculate
        #the derivatives:
        f = asarray([-1.0*(2*A*x + B*y)*f, -1.0*(2*C*y + B*x)*f])
    elif deriv == 2:
        f = ((2*A*x + B*y)**2 -2.0*A + (2*C*y+B*x)**2 - 2*C)*f

    #reinforce normalization:
    f = f / sqrt((f*f).sum())
    return(f)
#end RotatingGaussKernel

def CircMask(r=10, norm=True):
    """ Create a circular mask with radius r
        Actually, the same as BoxcarKernel, but with a circular shape.

        if norm then normalize the sum to 1.0
        (There is no 1D version)
    """

    size = 2*r + 1

    core = indices((size,size),dtype=int)
    circle = ((core[0] - r)**2 + (core[1]-r)**2 ) <= (r*r)
    circle = circle.astype(float)

    if norm:
        return circle/circle.sum()

    else :
        return circle
#End of CircleMask()

def BoxcarKernel(r=10, norm=True, OneD=False):
    """A constant kernel for smoothing images.
       Parameters:
        r the radius of the structure. The image size is 2r+1
        norm Normalize to one or not? If True, divides the image
            by the sum of the image. False by default

        If OneD it is just a ones().
    """
    #We need an integer value:
    r = int(r)
    size = 2*r+ 1
    kernel1 = zeros(size,dtype=float) + 1.0

    if not OneD:
        kernel2 = kernel1.copy()
        kernel2.shape = (kernel1.shape[0],1)
        kernel = kernel1*kernel2
    else:
        kernel = kernel1

    if norm:
        #Normalize to 1:
        kernel = kernel/kernel.sum()

    return kernel
#End of BoxcarKernel()


def BinomKernel(N, norm= True, OneD=False):
    """generate a binomial kernel
        N = 1 -> [1]
        N = 2 -> [1,1] -> [[1,1],[1,1]]
        N = 3 -> [1,2,1] -> [[ 1.,  2.,  1.],
                            [ 2.,  4.,  2.],
                            [ 1.,  2.,  1.]]
        etc.

        if norm is True, normalize the sum to 1.0
        if OneD is True, return a 1D array only

        Return value: the kernel matrix.
    """

    if N<1 :
        print("positive integer is expected (N>0)")
        return None
    #end if
    a = asarray([])
    for i in range(1,N+1):
        b = zeros(i)
        b[0] = b[-1] = 1

        if a.size > 0 :
            b[1:-1] = a[1:] + a[:-1]
        #end if
        a = b.copy()
    #end for
    if not OneD:
        c = b.copy()
        c.shape=(b.shape[0],1)
        res = b*c
    else:
        res = b
    #end if

    if norm:
        return res/res.sum()
    else:
        return res
#end Binomial


def SoebelKernel(N=3, X=True):
    """ Define a Soebel kernel.
        This is far from ideal at the moment, a better implementation is needed.
        For the 3x3 version it uses the optimized version from Scharr
        (Digital Image Processing by Jahne et al.).

        The others are just a simple variant, which may be far from optimal.

        Since this kernel is either X or Y direction,
        one can specify which to use.
        N:  3, 5, 7, or 9
        X:  Bool (True) for the X direction filter, False for Y direction

        return:
        a matrix containing the kernel
    """
    if N > 9 or N < 3 :
        print(" 3 <= N <= 9 are implemented. Falling back to N=3")
        N = 3
    #end if

    if N == 3 :
        res = asarray([[3,10,3],[0,0,0],[-3,-10,-3]])
    elif N == 5:
        res = asarray([[1,2,3,2,1],\
                    [2,3,4,3,2],\
                    [0,0,0,0,0],\
                    [-2,-3,-4,-3,-2],\
                    [-1,-2,-3,-2,-1 ]])
    elif N == 7 :
        res = asarray([[1,2,3,4,3,2,1],\
                    [2,3,4,5,4,3,2],\
                    [3,4,5,6,5,4,3],\
                    [0,0,0,0,0,0,0],\
                    [-3,-4,-5,-6,-5,-4,-3],\
                    [-2,-3,-4,-5,-4,-3,-2],\
                    [-1,-2,-3,-4,-3,-2,-1 ]])
    elif N == 9:
        res = asarray([[1,2,3,4,5,4,3,2,1],\
                    [2,3,4,5,6,5,4,3,2],\
                    [3,4,5,6,7,6,5,4,3],\
                    [4,5,6,7,8,7,6,5,4],\
                    [0,0,0,0,0,0,0,0,0],\
                    [-4,-5,-6,-7,-8,-7,-6,-5,-4],\
                    [-3,-4,-5,-6,-7,-6,-5,-4,-3],\
                    [-2,-3,-4,-5,-6,-5,-4,-3,-2],\
                    [-1,-2,-3,-4,-5,-4,-3,-2,-1 ]])
    else:
        print("Invalid N, please use one from: 3,5,7,9")
        return None

    #normalize:
    res = res.astype('f')/(abs(res)).sum()

    if X:
        return res.transpose()
    else :
        return res
#end of SoebelKernel

def BallKernel(r1, r2=0, norm=False, OneD=False):
    """ Make a ball kernel with window 2r1+1 and
        radius r2.
        If r2 not specified, use r1.
    """
    if r2 <= 0.0:
           r2 = r1
    #end if
    if r1 <= 0.0:
        raise ValueError()

    rsquare = r2*r2
    width = int( 2*r1 + 1.5) #we use +0.5 for rounding
    x = arange(width, dtype= 'f')

    #The center is the middle of width, defined by r1
    x = x - r1
    #make it quadratic:
    x = x*x

    if OneD:
        kernel = rsquare - x
        kernel[ kernel < 0.0 ] = 0.0
        k = sqrt(kernel)
        if norm:
            k = k / k.sum()
        return k


    #now make a col vector:
    y = x.copy()
    y.shape = (x.shape[0],1)

    #x+y will generate a matrix with the quadratic values
    xy = x + y
    kernel = rsquare - xy
    kernel[ kernel < 0.0 ] = 0.0

    if norm:
        k = sqrt(kernel)
        return k/k.sum()
    else:
        return sqrt(kernel)
#end of BallKernel
