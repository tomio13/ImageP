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
/* To comply:                                           */
/* gcc -shared -Wl,-soname,Laplace3.so -fPIC -o Laplace3.so, Laplace3.c */

/* SimpleErode: remove perimeter pixels (I > 0) */
/* Parameters:
 * img:         image; 2D array of integers
 * Ni,Nj        size of the image
 * bg           background value -> used to compare the pixels to
 * res          resulted image
 *
 * Return value:    -1 on error, 0 on success
 */
int SimpleErode(int *img, int Ni, int Nj, int bg, int* res)
{
    int i, j, ii;

    if( img == NULL || res == NULL || Ni < 0 || Nj < 0)
    {
        printf("parameter error\n");
        return -1;
    }

    for(i=0; i<Ni; i++)
    {
        for(j=0; j<Nj; j++)
        {
            ii = Nj*i+j;

            if( *(img + ii) != bg)
            {
                /* check the neighbors if any of them is 0, stop,
                 * else set the pixel in res
                 */
                if ((i-1) > 0 && *(img+ii-1) == bg) continue;
                if ((j-1) > 0 && *(img+ii-Nj) == bg) continue;
                if ((i+1) < Ni && *(img+ii+1) == bg) continue;
                if ((j+1) < Nj && *(img+ii+Nj) == bg) continue;
                if ((j-1) > 0 && (i-1) > 0 && *(img+ii-Nj-1) == bg) continue;
                if ((j+1) < Nj && (i-1) > 0 && *(img+ii+Nj-1) == bg) continue;
                if ((j-1) > 0 && (i+1) < Ni && *(img+ii-Nj+1) == bg) continue;
                if ((j+1) < Nj && (i+1) < Ni && *(img+ii+Nj+1) == bg) continue;

                    *(res+ii) = *(img+ii);
            }/*end if*/
        }/*for j*/
    }/*for i*/

    return 0;
}
