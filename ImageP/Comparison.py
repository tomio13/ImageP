#!/usr/bin/env python

""" ImageP.Comparison:  comparison functions, mostly used for fluorescence
                    microscopy images.

    Pearson and Manders are for colocalization analysis of various
    fluorescence channels. It may be useful to align the images before
    doing this analysis.

    Author: Tamas Haraszti, Biophysical Chemistry group at the University of
        Heidelberg

    Copyright:  LGPL-3
    Warranty:   For any application, there is no warranty 8).
"""

from matplotlib import pyplot as pl

from numpy import sqrt, zeros

__all__=["Pearson","Manders"]

##################################################################
def Pearson(img1, img2, rejectNull= False, verbose=True):
    """ Calculate the Pearson intensity correlation of two images.
        The Pearson correlation coefficient is in the range of [-1,1],
        showing how the intensities are distributed around their mean
        value within two images. If there is no overlap the result is 0.

        For each pixel it is (Img1*Img2)/sqrt(sum(Img1^2)*sum(Img2^2))
        where img1 and Img2 are:
                Img1=img1 - img1.mean();
                Img2 = img2-img2.mean()

        Parameters:
        img1, img2:     the images to be calculated
        rejectNull:     if True, then average pixels that are > 0 only
        verbose:        provide some feedback

        Return:
        a tupple of (correlation coefficient, image of coefficients)

        Ref: Manders et al. Journal of Cell Science vol. 103 857-862 (1992)
    """

    if img1.size != img2.size:
        print("The two images do not match in size!")
        return (None,None)
    #end if

    #this method is insensitive to a constant scaling:
    #Img1 = img1 / img1.max()
    #Img2 = img2 / img2.max()
    #but this also does not help the precision...

    if rejectNull:
        Img1 = img1 - img1[img1 > 0].mean()
        Img2 = img2 - img2[img2 > 0].mean()
    else:
        Img1 = img1 - img1.mean()
        Img2 = img2 - img2.mean()

    # the denomiator is the multiplication of the stdevs,
    # thus sqrt((Img1*Img1).sum()) * sqrt((Img2*Img2).sum())
    # which we can merge as:
    DN = sqrt((Img1*Img1).sum() * (Img2*Img2).sum())

#    if DN < 1E-12:
#        print("Denomiator is < 1E-12!!!")
    #end if

    #ret = zeros(img1.shape)
    #If any of the images are empty, then this is
    #a zero correlation...
    ret = Img1*Img2 / DN if DN != 0.0 else zeros(img1.shape)

    if verbose:
        print("Pearson correlation:")
        print("P: %f, denomiator: %g\n" %(ret.sum(), DN))

    return (ret.sum(), ret)
#end Pearson

def Manders( img1, img2, bkg=True, verbose=True):
    """ Calculate the Manders's coefficients of colocalization.
        This correlation is based on absolute intensity values,
        not taking any mean off. There are various results generated.
        If any of the images are zero images or their sum is zero,
        the corresponding coefficient is set to 0.

        Parameters:
        img1, img2:     two images to compare (monochrome)
        bkg:            if True, remove the minimum of both images
        verbose:        provide some feedback

        return:
        a set of parameters in a tupple
        m1,m2:  sum(colocalized pixels)/sum(all pixels) in images 1,2
        k1,k2:  overlap coefficients:
                sum(img1*img2)/sum(img1^2)
                sum(img1*img2)/sum(img2^2)
        R:      the Manders coefficient: (similar to Pearson's coefficient
                but without the averages removed)
                sum(img1*img2)/sqrt(sum(img1^2)*sum(img2^2))
        Rimage: the same as R, but without summing up the pixels in the
                numerator.

        Ref: Acta Histochem. Cytochem. vol. 40: 101 - 111 (2007)
    """

    if img1.size != img2.size:
        print("Size of the two images do not match!")
        return (0,0,0,0,0)
    #end if

    Img1 = img1.copy()
    Img2 = img2.copy()
    if bkg:
        Img1 = Img1 - Img1.min()
        Img2 = Img2 - Img2.min()
    #end if

    indx1 = Img1 > 0
    indx2 = Img2 > 0

    #rescaling: m1,m2 and Mr are insensitive,
    #but k1 and k2 are sensitive to scaling...
    #except if the images are scaled with the same number
    s1 = img1.sum()
    s2 = img2.sum()

    m1 = img1[indx2].sum()/s1 if s1 != 0 else 0.0
    m2 = img2[indx1].sum()/s2 if s2 != 0 else 0.0

    s1 = (img1*img1).sum()
    s2 = (img2*img2).sum()

    mix = img1*img2
    mixs = mix.sum()
    #the denomiator is sqrt(sum(imge^2)*sum(img2^2)) just as in Pearson's
    ss = s1*s2

    k1 = mixs/s1 if s1 != 0 else 0.0
    k2 = mixs/s2 if s2 != 0 else 0.0

    R = mixs / sqrt(ss) if ss != 0 else 0.0
    rimg = mix/sqrt(ss) if ss != 0 else zeros(img1.shape)

    if verbose:
        print("Manders parameters:")
        print("m1, m2: %f, %f" %(m1,m2))
        print("k1, k2: %f, %f" %(k1,k2))
        print("Denomiator^2: %g" %ss)
    #end if

    return (m1,m2,k1,k2,R, rimg)
#end Manders

