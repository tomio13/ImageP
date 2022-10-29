#include <stdio.h>
/* C helper functions for image manipulation in python  */
/*  Author: Tamas Haraszti                              */
/*  Date: 2010                                          */
/*  License: LGPL 3                                     */
/*                                                      */
/*  numpy images are double * arrays, whith a linear    */
/*  address space. The indexing then:                   */
/*  image[i,j] = *(image + i*Nj+j)                      */
/*  image.shape = (Ni,Nj)                               */
/* To comply:                                           */
/* gcc -shared -Wl,-soname,Laplace3.so -fPIC -o Laplace3.so, Laplace3.c */


/* PeakFind: find a local maxima within an image
 *              maxima is searched around each pixels which are brighter
 *              than a 'threshold' within an area +/- width around the pixel
 * Parameters:
 * img:         the image
 * Ni,Nj        image size
 * threshold:   threshold
 * width:       window 'radius'
 * resI, resJ:  array receiving the i,j index of the maxima
 *
 * Return:      -1 on error, or the number of hits
 */
int PeakFind( double* img, int Ni, int Nj,\
            double threshold, int width, \
            int* resI, int* resJ)
{
    int i,ii,jj, tmpindx;
    int Nij, ni, nj;
    double maximg;
    int ires, maxi;
    int jump = 0;

    Nij = Ni*Nj;

    if( img == NULL || Ni < 0 || Nj < 0 || width < 1 ||\
            width > Ni/2 || width > Nj/2 )
    {
        printf("Invalid incoming parameters\n");
        return -1;
    }/*end if*/

    ires = 0;
    i = 0;

    while (i < Nij )
    {
        if( *(img + i) > threshold)
        {
            /* Now we need the 2D point to define a frame:*/
            ni = (int)(i/Nj);
            nj = i - ni*Nj;
            maximg = *(img + i);
            maxi = i;
            jump = 0; /* the max is intact, no changes */

            /* This frame is investigated: it may go off of course */
            for( ii= ni-width; ii < ni+width +1; ii++)
            {
                for(jj=nj-width; jj< nj+width+1; jj++)
                {
                    /* first: let us stay within the image */
                    if( ii > 0 &&  ii<Ni && jj > 0 && jj < Nj)
                    {
                        /* temporary index of the current pixel: */
                        tmpindx = ii*Nj +jj;
                        /*if the new pixel
                         * is a new maximum: set it -> this is the new center
                         * smaller then max -> delete it */
                        if( maximg < *(img + tmpindx))
                        {
                            maxi = tmpindx ;
                            maximg = *(img + maxi);
                            /* This is a new hit -> jump
                             * If this is a different line (ii != ni)
                             *  or backwards (jj <= nj) -> ni, nj can not be max. anymore
                             *  set maxi to the edge of ni,nj and jump
                             *
                             *else: jump to this max (maxi is already set)
                             */
                            if( ii != ni || jj <= nj )
                            {
                                jj = nj + width + 1;
                                /* maxi stores the next start position */
                                jj = (jj >= Nj) ? Nj-1 : jj;
                                maxi = ni*Nj + jj;
                            }
                            jump =1 ;
                            break ;
                        }/* if  new max found... else: */
                        else
                        {
                            /* kill the useless pixel, so we do not use it
                             * anymore */
                            *(img + tmpindx ) = threshold - 1.0;
                        }/* if maximg < ...*/

                    }/* end if ii, jj in frame */

                }/*end of for jj; scanning within a line of the window */

                /* we have a new hit -> stop processing, go to the next pixel */
                if( jump != 0)  break;

            }/* end for ii; scanning btw. the window lines*/


            /* Did the maximum move? If not, then it is a local maximum, store it*/
            if( i == maxi)
            {
                *(resI + ires) = ni;
                *(resJ + ires) = i - ni*Nj;
                    /*printf("Max found: %d, %d\n", *(resI+ires), *(resJ+ires)); */
                ires ++;
                /* This window is done, we can go to its edge in the line */
                nj += width + 1;
                nj = (nj < Nj) ? nj : Nj;
                /* recalculate the linear position: */
                i = ni * Nj + nj;

                    /*printf("Set new start to i: %d of %d\n", i, Nij); */
            }
            else
            {
                i = maxi;
                /* printf("Jump to: %d on line %d\n", i, ni); */
            }
        }
        else
        {
            i++;
        }/* end if: do the job ; else just walk (i++)*/

    }/*end while i*/

    return ires;
}
