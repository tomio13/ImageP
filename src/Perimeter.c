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

/* Perimeter: find the perimeter of structures with > 0 intensity */
/* Parameters:
 * img:         image; 2D array of integers
 * Ni,Nj        size of the image
 * res          resulted image - the perimeter pixels are set to 1
 *
 * Return value:    -1 on error, 0 on success
 */
int Perimeter(int *img, int Ni, int Nj, int* res)
{
    int i, j, ii;

    if( img == NULL || res == NULL || Ni < 0 || Nj < 0)
    {
        printf("parameter error\n");
        return -1;
    }

    for(i=1; i<Ni-1; i++)
    {
        for(j=1; j<Nj-1; j++)
        {
            ii = Nj*i+j;

            if( *(img + ii) >0)
            {
                if( (*(img + ii+Nj  )==0) ||\
                        (*(img + ii-Nj)==0) ||\
                        (*(img + ii +1) ==0) ||\
                        (*(img + ii -1) ==0) ||\
                        (*(img + ii + Nj +1) ==0) ||\
                        (*(img + ii -Nj + 1) ==0) ||\
                        (*(img + ii + Nj -1) ==0) ||\
                        (*(img + ii -Nj -1) ==0))
                {
                    *(res+ii) = 1;
                }/*end if *img...*/
            }/*end if*/
        }/*for j*/
    }/*for i*/

    return 0;
}
