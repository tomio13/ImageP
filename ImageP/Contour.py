#!/usr/bin/env python
from numpy import (pi, linspace, zeros, ones, sin, cos, sqrt, floor, arange,
                matrix, array, abs, append, asarray, arctan2, argsort)
from numpy.linalg import inv, pinv, svd
from matplotlib import pyplot as pl

#from ImageP import *
from . Kernels import *
from . Convolve import *

def ActiveContour(img, X = [], Y= [],\
        alpha = 0.5, beta= 0.75, gamma= 10.0,\
        coupled = False,
        N=50, max_iter = 100, eps = 0.01, \
        width = 2,
        full=False, verbose=True):
    """ Fit an active contour to an image, using the intensity
        values as negative potential values.
        The basic idea is to employ an "elasticity" term on contgrolling
        the length and bending of the curve, described by two factors,
        alpha and beta. Simultaneously, add a numerical gradient walk
        to find the contour points, controlled by a parameter gamma.
        In the original equations, gamma is coupled to the elastic terms,
        resulting in paramters, where high alpha and beta required
        low gamma and vice versa.
        In this algorithm, we decoupled the terms. Thus, alpha and beta
        go into an elastic matrix P, which gets inverted and the inverse
        used in each step to control the curve. Gamma multiplies the
        gradients Fx and Fy to interate the curve.
        Something like: s = xj + yi
        ds/dt = alpha s'' - beta s'''' + Fgrad(s(t))

        The elastic response matrix P also contains the circular coupling,
        thus the end points are related to the derivatives of the other end.
        This ensures that the cure remains smooth at the ends.
        For inversing P we use the numpy.linalg.inv, but if it fails we use
        the generalised Penrose inversion pinv (it is much slower though).

        The diagonal in P is 1 + ..., meaning high alpha and beta will override
        1, but very low alpha and beta will make the off diagonals dominate
        the inverse of P.

        The gradients are generated from the image using the 5x5 Soebel
        filter. (This may get more flexible in the future.)

        Based on the matlab code from
        http://www.cb.uu.se/~cris/blog/index.php/archives/217
        and
        http://www.cb.uu.se/~cris/blog/index.php/archives/258

        Parameters:
        img             image
        X,Y             starting list of points (Y=i, X=j).
                        If left empty, the algorithm stars with a circle
                        of min(Ni,Nj)/2 - 2 in radius.

        alpha           weight of shrinking
                        Higher alpha can accelerate the iteration, but may
                        pull out from local potentials.
        beta            weight of stretching (prevents curvature)
        gamma           weight of iteration
                        A high gamma will speed up the process, but also
                        cause oscillations due to image noise.

        coupled         (False) couple gamma to alpha and beta, as in
                        the references. In this case, if you 
                        increasing gamma, decrease alpha and
                        beta with about the same factor.
                        In such a case, a parameter set like:
                        alpha = 0.03, beta = 0.1, gamma = 10.0
                        is more appropriate

        N               the maximal number of points in the contour
        max_iter        maximal number of iterations
        eps             error to achive
        width           for the derivative filter
        full            return back P and the inverse P matrices
        verbose         generate information plots of the gradients,
                        the potential, showing the contour (evolving)
                        and the actual step size during iteration

        Return:
        a dict containing:
        'X'     : array of x-coordinates
        'Y'     : array of y-coordinates
    """
    Ni, Nj = img.shape
    if X != [] and Y!= [] and len(X) == len(Y):
        x = asarray(X)
        y = asarray(Y)
        Nx = len(X)

        if verbose:
            print("Received initial point arrays X,Y with length %d" %Nx)
    else:
        R = min(Ni, Nj)/2.0 - 2.0
        print("Radius: %.1f" %R)

        #we initiate an image large circle for the start
        Nx = min( int(2*pi*R), N)
        phi = linspace(0, 2*pi, Nx)
        x = R*cos(phi) + Nj/2
        y = R*sin(phi) + Ni/2
    #end if X,Y provided

    V = img
    #combining gamma into the snake variables will allow
    #to get it normalized out through the inversion and
    #the iteration (one is having then 1/gamma, the other gamma
    if coupled:
        #the original forms from the references:
        a = 1.0 + gamma*(2.0*alpha + 6.0*beta)
        b = -1.0*gamma*(alpha + 4.0*beta)
        c = gamma*beta
    else:
        a = 1.0 + (2.0*alpha + 6.0*beta)
        b = -1.0*(alpha + 4.0*beta)
        c = beta

    #create the matrix to use:
    P = matrix(zeros( (Nx,Nx)))
    indx = arange(Nx)
    #diag is not really working as in matlab (below numpy 1.10)
    #set the diagonal to 1.0
    P[indx,indx] = a

    #1 above and 1 below the diagonal we set b
    P[indx[1:], indx[:-1]] = b
    P[indx[:-1], indx[1:]] = b
    #2 above and 2 below we set c:
    P[indx[2:], indx[:-2]] = c
    P[indx[:-2],indx[2:]] = c
    print("p matrix is filled up")
    print("Shape of P is: %d x %d" %P.shape)

    #the last and first elements of the diagonals determine how the
    #points will behave at the end
    #We need some corner elements to account for closing the curve:
    P[-2,0] = P[1,-1] = P[0, -2] = P[-1, 1] = c
    P[-1,0] = P[0, -1] = b

    try:
        Pinv = inv(P)
    except:
        #about 10x slower, but works for all kind of matrices
        print("Falling back in inversion")
        Pinv = pinv(P)
    print("inversion is done")

    #P is inverted....
    #now we build up the force vectors and the iteration begins:
    Fx = zeros(x.shape); Fy = zeros(y.shape)

    #we want to generate a gradient image with smoothening:
#    Gkx, Gky = RotatingGaussKernel(10,10,0, [51,51], deriv=1)
    Gkx = SoebelKernel(5,True)
    Gky = SoebelKernel(5,False)
    print("Starting convolutions")
#    GimX = ConvFilter(V, Gkx)[25:-26]
#    GimY = ConvFilter(V, Gky)[25:-26]
    Rk = 2*width
    gk = GaussKernel(Rk, width, OneD= True)
    gkd = -GaussKernel(Rk, width, OneD= True, deriv= 1)
#    GimX = ConvFilter(V, Gkx)[2:-3]
#    GimY = ConvFilter(V, Gky)[2:-3]
    GimX = ConvFilter1D(V, gkd, kernel_y = gk)
    GimY = ConvFilter1D(V, gk, kernel_y = gkd)
    GimX = 2*GimX/(GimX.max() - GimX.min())
    GimY = 2*GimY/(GimY.max() - GimY.min())
    GimX[ abs(GimX) < 1E-8 ] =0
    GimY[ abs(GimY) < 1E-8 ] =0
    print("done")

    if verbose:
        pl.figure(1)
        pl.clf()
        pl.jet()
        pl.imshow(GimX)
        pl.title('X grad')
        pl.draw()

        pl.figure(2)
        pl.clf()
        pl.imshow(GimY)
        pl.title("Y grad")
        pl.draw()

        pl.figure(3)
        pl.clf()
        pl.imshow(V)
        pl.gray()
        pl.plot(x,y,'r-');
#        raw_input("PRESS ENTER")

    steps = []
    for i in range(max_iter):
        #x and y values are floats, in between the pixels
        #we can estimate the local gradients
        i0 = floor(y); j0 = floor(x)
        i1 = i0 +1;    j1 = j0 + 1
        wi = y - i0 #the fraction to the previous pixel in i
        wj = x - j0
        i0 = i0.astype('i')
        i1 = i1.astype('i')
        j0 = j0.astype('i')
        j1 = j1.astype('i')

        #the interpolated "forces" from the gradient images:
        Fx = (1-wj)*(1-wi)*GimX[i0,j0] + wj*(1-wi)*GimX[i0,j1] + \
                (1-wj)*wi*GimX[i1,j0] + wj*wi*GimX[i1,j1]

        Fy = (1-wj)*(1-wi)*GimY[i0,j0] + wj*(1-wi)*GimY[i0,j1] + \
                (1-wj)*wi*GimY[i1,j0] + wj*wi*GimY[i1,j1]

        #store the actual points before the next iteration:
        xold = array(x.copy()); yold = array(y.copy())

        x = array( Pinv * matrix(x + gamma*Fx).transpose() )[:,0]
        y = array( Pinv * matrix(y + gamma*Fy).transpose() )[:,0]
        #x = array( Pinv * matrix(x + Fx).transpose() )[:,0]
        #y = array( Pinv * matrix(y + Fy).transpose() )[:,0]

        stp = (sqrt((array(x)-xold)**2 + (array(y)-yold)**2)).mean()
        steps.append(stp)

        if stp < eps:
            if verbose:
                print ("Iteration reached its limit")
                print("Average iteration step size in iteration %d was %f" \
                        %(i, stp))
            break
        #end if iteration ends

        #some control:
        #these are the valid points:
        #because we have a rounding down and up, we confine 1 pixel from
        #the real boundaries
        indx = (x > 0) & (y > 0) & (x < (Nj-1)) & (y < (Ni-1))

        #define the valid limit:
        xmin = x[indx].min(); ymin = y[indx].min()
        xmax = x[indx].max(); ymax = y[indx].max()

        #pack those back, who got out of the image:
        x[x < 0] = xmin; x[x > (Nj-1)] = xmax
        y[y < 0] = ymin; y[y > (Ni-1)] = ymax

        if verbose:
            pl.figure(3)
            pl.clf();
            pl.imshow(V)
            pl.axis('equal')
            pl.plot(x,y,'r-', alpha=1.)
            pl.draw()

            pl.figure(4)
            pl.clf()
            #pl.plot(Fx,'r-')
            #pl.plot(Fy, 'b-')
            pl.plot(steps,'ro-')
            pl.title("Iteration steps")
            pl.draw()
    #end for ii
    if verbose and i >= max_iter-1:
        print ("Exceeded maximal number of iterations %d" %max_iter)
    #end if

    #collect the return values:
    res = {'x':x, 'y':y, 'N_iter': i, 'step': stp}
    #if the user wants, we can provide some more information
    if full:
        res['P'] = P
        res['Pinv'] = Pinv
        res['V'] = V

    return res
#end Active_contour


def Circle_fit(x,y, is_sort= False):
    """ fit a circle at the data

        Based on the Matlab code published in:
        https://people.cas.uab.edu/~mosya/cl/TaubinSVD.m

        Note: this is a version optimized for stability, not for speed
        from Matlab central

        See also:
        G. Taubin, "Estimation Of Planar Curves, Surfaces And Nonplanar
                    Space Curves Defined By Implicit Equations, With
                    Applications To Edge And Range Image Segmentation",
                    IEEE Trans. PAMI, Vol. 13, pages 1115-1138, (1991)

        A. Al-Sharadqah, N. Chernov, Error analysis for circle fitting
                algorithms. E. J. Statistics Vol. 3, pages 886-911, (2009)


        Parameters:
            x,y         set of points
            is_sort     if set, sort the results according
                        to the angles

        Return:
            a dict containing
            'R'         radius of fitted circle
            'x0','y0'   coordinates of the center
            'xfit', 'yfit'  the fitted points
            'err2'      radial error of the points
            'chi2'      sum error of the radii
            'relError'  mean of relative error normalized to radius
    """
    if len(x) < 1:
        raise ValueError("Empty array received")

    xm = x.mean()
    ym = y.mean()

    xx = x - xm; yy = y - ym;
    #we take the squared distance from the center:
    z = xx*xx + yy*yy

    zm = z.mean()
    z0 = (z-zm) / (2.0*sqrt(zm)) #zm > 0 should be...
    #make a matrix with rows for z0, xx, yy,
    #thus Nx3
    zmat = matrix([z0, xx, yy])

    #numpy.linalg.svd
    u, s, v = svd(zmat) #check if this works
    #A = v[:,2] was in the matlab code, here it is:
    #the third column of u is taken: (u is 3x3)
    A = u[:,2]

    A[0] = A[0]/(2.0*sqrt(zm))
    A = append(asarray(A), -zm*A[0])
    a0, b0 = -A[1:3] / A[0]/2.0
    #shift back to the xm, ym coordinage system:
    a = a0 + xm
    b = b0 + ym
    R = sqrt(A[1]*A[1]+A[2]*A[2]-4*A[0]*A[3])/abs(A[0])/2;

    #calculate error, etc...
    x1 = x - a
    y1 = y - b
    alpha = arctan2(y1, x1)
    #x,y are sorted according to x, so the drawing is a zig-zag
    #to have improved drawing, the user can request sorting
    #with the angle (alpha)
    if is_sort:
        sindx = argsort(alpha)
        x1 = x1[sindx]
        y1 = y1[sindx]
        alpha = alpha[sindx]
    #end if

    xfit = R*cos(alpha) + a
    yfit = R*sin(alpha) + b
    err2= (xfit - x)**2 + (yfit - y)**2
    relErr = err2/R**2

    return {'x0': a, 'y0':b, 'R':R, 'xfit':xfit, 'yfit':yfit, 'err2':err2, \
            'chi2':err2.sum(), 'relError': relErr.mean()}
#end Circle_fit


def SimpleContour(img, bg=0, fill= True):
    """ Run through the image, and define the left-right extremes.
        Do a scan along both X and Y and connect the edges to get
        the contour.

        Parameters
        img     a 2D image
        bg      use img > bg as object pixels
        fill    if True, fill up the edges

        return
        if fill is true, return an image with the filled up outside region
        if fill is false, return a (2, img.shape[1]) array, of the i and j indices
        of the external contour
    """
    if len(img.shape) != 2:
        raise ValueError('2 dimensional image is expected!')
    # fill up the image between left/right, top/bottom extremes
    bimg = zeros(img.shape, 'bool')

    # we scan the image along every dimension
    indexarray = []
    for i in range(img.shape[0]):
        line = img[i,:] > bg
        li = line.nonzero()[0]
        if len(li) < 2:
            continue
        i0 = li.min()
        i1 = li.max()
        # print(i, ':', i0, i1)
        if fill:
            bimg[i,i0:i1] = True
        else:
            bimg[i, i0] = True
            bimg[i, i1] = True

        indexarray.append(asarray([i, i0]))
        indexarray.append(asarray([i, i1]))
    #end scanning the image
    # now the other direction
    for i in range(img.shape[1]):
        line = img[:,i] > bg
        li = line.nonzero()[0]
        if len(li) < 2:
            continue
        i0 = li.min()
        i1 = li.max()
        # print(i, ':', i0, i1)
        if fill:
            bimg[i0:i1, i] = True
        else:
            bimg[i0, i] = True
            bimg[i1, i] = True

        indexarray.append(asarray([i, i0]))
        indexarray.append(asarray([i, i1]))
    #end scanning the image

    if fill == True:
        return bimg
    else:
        return indxarray
# end of SimpleContour
