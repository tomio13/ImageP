#!/usr/bin/env python

""" ImageP.Display: display functions to simplify image displaying with
    matplotlib.

    Author: Tamas Haraszti, Biophysical Chemistry group at the University of
            Heidelberg

    Copyright   LGPL-3
    Warranty:   For any application, there is no warranty 8).
"""

from matplotlib import pyplot as pl

from numpy import zeros, sin, cos, arange, sqrt

#for the export:
__all__ = ['composite', 'OverPlot', 'display']

########################################################################
#Display functions:
def composite(img1, img2= None, img3= None, norm=True):
    """ compose an RGB image from the three inputs
        Convert each image to 0 - 255 uint and combine
        to a single 3 channel matrix

        Parameters:
        three images with the same size, single channel

        if img2 or img3 are not specified, zeros are used.

        norm: bool, if True, scale the images to 0-255

        Return:
        an RGB image matrix
    """
    if img2 is None:
        img2 = zeros(img1.shape, dtype='u1')
    if img3 is None:
        img3 = zeros(img1.shape, dtype='u1')
    #end if

    if img1.shape != img2.shape or img2.shape != img3.shape:
        print("the images do not match in shape")
        return None

    Ni, Nj = img1.shape
    comp = zeros( (Ni,Nj,3), dtype='u1')
    m = img1.max()
    comp[:,:,0] = (255.0*img1/m).astype('u1') if norm and m!= 0.0 else img1.astype('u1')
    m = img2.max()
    comp[:,:,1] = (255.0*img2/m).astype('u1') if norm and m != 0.0 else img2.astype('u1')
    m = img3.max()
    comp[:,:,2] = (255.0*img3/m).astype('u1') if norm and m!= 0.0 else img3.astype('u1')

    return comp
#end of composite

#simplify life:
def display(image, fN = 1, colorbar=False, col='jet',\
        title="",\
        imgmode=True, GUI=False, NewFig=False, fname="", dpi=150,
        vmin= None, vmax= None, alpha= None):

    """ Simplified envelop to matplotlib.pyplot.imshow with
        some spicing up for more than one channel images.

        If a single 2D image is provided then the specified color table
        is used to present the image.

        If a 3D set is provided, then a composite is made from 3 slices
        using composite() above.

        If a 3D set with more than 3 channels provided, then the plot
        scrolls through the images as a movie.

        Parameters:
            image:  what to plot (2 or 3D)
            fN:     figure number (one can choose)
            colorbar:   add a colorbar?

        Compatibility parmeters: these are provided because of the previous
        version of Image
            imgmode:    use 0,0 in the left bottom corner (bool, True)
            GUI:        when done, call the pl.show() GUI loop. This should
                        be the last thing in a program (see matplotlib)
            NewFig:     override fN, and create a new figure

            fname:      spit out the image to a file
            dpi:        dot per inch value passed to savefig.
            vmin, vmax: if not None, use them as limiting in the image.
            alpha       alpha channel information to pass to imshow

        Return: nothing
    """

    fN = fN if fN > 1 else 1
    # origin = 'lower' if imgmode else 'upper'
    # as of 2021 upper provides proper image orientation
    origin = 'upper'

    if NewFig:
        fig = pl.figure()
    else:
        fig = pl.figure(fN)


    fig.clf()
    plt = fig.add_subplot(111)

    nd = image.ndim

    if col=='jet':
        cl=pl.cm.jet
    elif col == 'copper':
        cl = pl.cm.copper
    else :
        try:
            cl = eval("pl.cm.%s" %col)
        except:
            cl = pl.cm.gray
    #end if

    if nd == 1:
        plt.plot(image,'-o');

    elif nd== 2 or (nd == 3 and image.shape[2]==3):
        Img = plt.imshow(image, origin= origin, \
                cmap = cl,\
                interpolation=None,
                vmin= vmin, vmax= vmax,
                alpha= alpha)

        #plt.axis('image')
        plt.axis('off')
        plt.axis('equal')

        if nd== 2 and colorbar:
            fig.colorbar(Img)


    elif nd == 3:

        plt.axis('off')
        plt.axis('equal')
        print("playing movie:")

        for i in range(image.shape[2]):
            plt.cla()
            Img = plt.imshow(image, origin= origin,\
                    cmap= cl, interpolation=None,
                    vmin= vmin, vmax= vmax,
                    alpha= alpha)
            pl.draw()
    else:
        print("Can not handle %d dimensional data" %nd)
        return

    pl.draw()
    if title != "":
        pl.title(title)

    if fname != "":
        transp = True if alpha is not None else False
        if not colorbar:
            #improved for tight box:
            pl.savefig(fname, bbox_inches="tight", pad_inches=0, dpi=dpi, transparent= transp)
        else:
            pl.savefig(fname, dpi=dpi, transparent= transp)

    if GUI:
        pl.show()

    return
#end display


def OverPlot(img, res, size=0, outline=True, NewFig=True):
    """ Overplots circles with size or sizes from res on the image.
       Parameters:
        img:        image to display
        res:        a directory with 'X', 'Y' and 'size' arrays
        size:       if nonzero, then the default size
        outline:    plot circles with size diameter?
        newFig:     True Should it generate a new figure? Use
                    pylab.close() to get rid of it later.

       Return value: none
    """

    if len(res['X']) != len(res['Y']) :
        print("X and Y length mismatch\n")
        return

    if NewFig:
        fig = pl.figure()
    else:
        fig = pl.figure(1)

    t = arange(-3.14,3.14,0.03)
    ym = len(img)

    plt = fig.add_subplot(111)
    plt.cla()

    # plt.imshow(img, origin='lower', interpolation='nearest')
    # for some reason, as of 2021 November, this produces the
    # proper image orientation:
    plt.imshow(img, origin='upper', interpolation='nearest')
    plt.axis('image')

    plt.plot(res['X'],res['Y'],'r+')

    if outline:

        for i in range(0, len(res['X'])):
            x = res['X'][i]
            y = res['Y'][i]

            if size == 0:
                r = sqrt(float(res['size'][i]))/2 if 'size' in res else 1.0
            else :
                r = size/2.0

            plt.plot((x+r*cos(t)),(y+r*sin(t)),'r-')

    pl.draw()
    return
#End of OverPlot()

