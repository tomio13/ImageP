#include <stdio.h>
#include <stdlib.h>
/* C helper functions for image manipulation in python  */
/*  Author: Tamas Haraszti                              */
/*  Date: 2010                                          */
/*  License: LGPL 3                                     */
/*                                                      */
/*  numpy images are double * arrays, whith a linear    */
/*  address space. The indexing then:                   */
/*  image[i,j] = *(image + i*Nj+j)                      */
/*  image.shape = (Ni,Nj)                               */

/* SimpleDilate: add perimeter pixels (I == bg)
 * Use a second empty image and fill it up with existing pixels,
 * but also set neighbors not set yet on the result and on the
 * image.
 * It is important to use a second image for the filling, so
 * we do not iteratively fill up the whole frame redefining
 * the edge on the fly.
 */
/* Parameters:
 * img:         image; 2D array of integers
 * Ni,Nj        size of the image
 * bg           background value -> used to compare the pixels to
 * res          resulted image - the perimeter pixels are set to 1
 *
 * Return value:    -1 on error, 0 on success
 */
int SimpleDilate(int *img, int Ni, int Nj, int bg, int* res)
{
    int i, j, ii, imgi;

    if(img == NULL || res == NULL || Ni < 0 || Nj < 0)
    {
        printf("parameter error\n");
        return -1;
    }

    for(i=1; i<Ni-1; i++)
    {
        for(j=1; j<Nj-1; j++)
        {
            ii = Nj*i+j;
            imgi = *(img + ii);

            if(imgi != bg)
            {
                /* copy the pixel: */
                *(res+ii) = imgi;

                /* set all neighbours which are background
                 * and not yet set in the result image
                 */
                if( (*(img+ii +Nj) == bg) && (*(res+ii+Nj) < imgi) )
                    *(res + ii + Nj) = imgi;

                if( (*(img+ii -Nj)== bg) && (*(res+ii-Nj) < imgi) )
                    *(res+ii - Nj) = imgi;

                if( (*(img+ii +1) == bg) && (*(res+ii +1) < imgi ))
                    *(res+ii +1) = imgi;

                if( (*(img+ii -1) == bg)&& (*(res+ii -1) < imgi ))
                    *(res+ii -1) = imgi;

                if( (*(img+ii +Nj +1) == bg) && (*(res+ii +Nj +1) < imgi))
                    *(res+ii + Nj +1) = imgi;

                if( (*(img+ii -Nj +1) == bg) && (*(res+ii -Nj +1) < imgi))
                    *(res+ii - Nj +1) = imgi;

                if( (*(img+ii +Nj -1) == bg) && (*(res+ii +Nj -1) < imgi))
                    *(res+ii + Nj -1) = imgi;

                if( (*(img+ii -Nj -1) == bg) && (*(res+ii -Nj -1) < imgi) )
                    *(res+ii -Nj -1) = imgi;

            }/*end if*/
        }/*for j*/
    }/*for i*/

    return 0;
}
