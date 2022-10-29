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


/* Apply a 1D kernel in X then Y direction on an image
 * this should speed up the job a bit */
/* Parameters:
 * img:     the image to be processed
 * Ni,Nj    size of the image
 * kernel   the kernel array
 * Nii      size of the kernel
 * res      resulted image  */
int SimpleFilter1D(double *img, int Ni, int Nj,\
        double *kernel, int Nii,\
        double *res)
{
    int i,j,ii;
    int nj;
    int Nii2=Nii/2;
    double resp=0.0;

   if (img == NULL || kernel == NULL || res == NULL)
   {
       printf("An image, kernel and return image required\n");
       return(1);
    }

    if(Ni < Nii || Nj < Nii)
    {
        printf("The kernel is larger than the image!\n");
            return(1);
    }

    /*Go along the i index: */
    for( i = 0; i< Ni; i++)
    {
        /*Process each line with the kernel*/
        /* for(j=Nii2; j<Nj-Nii2; j++) -> the edges are distorted */
        for(j=0; j<Nj; j++)
        {
            resp = 0.0;
            for(ii = 0; ii<Nii; ii++)
            {
                /* the addressing is linear, not array of arrays as usual
                 * then the index is: i*Nj+j, but now we have to transform
                 * i -> i+ii-Nii2
                 * j -> j+jj-Njj2 */
                /* this way the edge takes the outermost pixels multiple times
                 * into account, and we lose some time. But the algorithm is more
                 * general than taking only odd size kernels */
                nj = j + ii - Nii2;
                if( nj > 0 && nj <Nj )
                    resp += *(kernel + ii) * \
                        (*(img+ i*Nj + nj));
            }/*end of for jj */
        }/*end of for ii */
        *(res+ i*Nj +j) = resp;
    }/*end of for i*/

    /* second run: along j*/
    for(j=0; j<Nj; j++)
    {
        /*process each i with the kernel*/
        /* for(i=Nii2; i<Ni-Nii2; i++) -> edges are distorted */
        for(i=0; i<Ni; i++)
        {
            resp = 0.0;
            for(ii=0; ii<Nii; ii++)
            {
                /*Watch out, nj is an i index!*/
                nj = i + ii - Nii2;
                if( nj > 0 && nj < Ni)
                    resp += *(kernel +ii) * \
                            (*(img + nj*Nj+j));
            }/*end of for ii*/

            *(res+ i*Nj+j) = resp;
        }/*end of for i*/
    }/*end for j*/

    return 0;
}

