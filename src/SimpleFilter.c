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


/* Apply a small kernel: 3,5,7,9 point large on an image */
int SimpleFilter(double *img, int Ni, int Nj,\
        double *kernel, int Nii, int Njj,\
        double *res)
{
    int i,j,ii,jj;
    int ni, nj;
    int Nii2=Nii/2, Njj2=Njj/2;
    double resp;

   if (img == NULL || kernel == NULL || res == NULL)
   {
       printf("An image, kernel and return image required\n");
       return(1);
    }

    if(Ni < Nii || Nj < Njj)
    {
        printf("The kernel is larger than the image!\n");
            return(1);
    }

    /* for( i = Nii2; i< Ni-Nii2; i++) */
    for( i =0; i< Ni; i++)
    {
        /*for(j = Njj2; j<Nj -Njj2 ; j++) */
        for(j =0; j<Nj; j++)
        {
            resp = 0.0;
            for( ii = 0; ii < Nii; ii++)
            {
                for(jj = 0; jj<Njj; jj++)
                {
                    /* the addressing is linear, not array of arrays as usual
                     * then the index is: i*Nj+j, but now we have to transform
                     * i -> i+ii-Nii2
                     * j -> j+jj-Njj2 */
                    /* this way the edge takes the outermost pixels multiple times
                     * into account, and we lose some time. But the algorithm is more
                     * general than taking only odd size kernels */
                    ni = i + ii - Nii2;
                    nj = j + jj - Njj2;

                    if( ni > 0 && ni < Ni && nj > 0 && nj <Nj )
                        resp += *(kernel + ii*Njj+jj) * \
                            (*(img+ ni*Nj + nj));
                }
            }

            *(res+ i*Nj +j) = resp;
        }
    }
    return 0;
}

