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
    int i, j, ii, imgi, NN;

    if(img == NULL || res == NULL || Ni < 0 || Nj < 0)
    {
        printf("parameter error\n");
        return -1;
    }
    /* the maximum index is: Ni*Nj */
    NN = Ni*Nj;

    for(i=0; i< Ni; i++)
    {
        for(j=0; j< Nj; j++)
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
                if( (ii+Nj) < NN && (*(img+ii +Nj) == bg) && (*(res+ii+Nj) < imgi) )
                    *(res + ii + Nj) = imgi;

                if( (ii-Nj) > 0 && (*(img+ii -Nj)== bg) && (*(res+ii-Nj) < imgi) )
                    *(res+ii - Nj) = imgi;

                /* if the row is at the end, do not use it
                 * (ii+1) < NN would go to the next line
                 */
                if( (j+1) < Nj  &&  (*(img+ii +1) == bg) && (*(res+ii +1) < imgi ))
                    *(res+ii +1) = imgi;

                /* again, if left goes off...
                 * (ii-1) >0 would flow back to the previous row
                 */
                if( (j - 1) > 0 && (*(img+ii -1) == bg)&& (*(res+ii -1) < imgi ))
                    *(res+ii -1) = imgi;

                /* diagonals are tricky */
                if( (i+1)<Ni && (j+1)<Nj &&
                        (ii+Nj+1) < NN && (*(img+ii +Nj +1) == bg) && (*(res+ii +Nj +1) < imgi))
                    *(res+ii + Nj +1) = imgi;

                if( (i-1) > 0 && (j+1) < Nj &&
                        (*(img+ii -Nj +1) == bg) && (*(res+ii -Nj +1) < imgi))
                    *(res+ii - Nj +1) = imgi;

                if( (i+1) < Ni && (j-1) > 0 &&
                        (*(img+ii +Nj -1) == bg) && (*(res+ii +Nj -1) < imgi))
                    *(res+ii + Nj -1) = imgi;

                if( (i-1) > 0 && (j-1)>0 &&
                        (*(img+ii -Nj -1) == bg) && (*(res+ii -Nj -1) < imgi) )
                    *(res+ii -Nj -1) = imgi;

            }/*end if*/
        }/*for j*/
    }/*for i*/

    return 0;
}
