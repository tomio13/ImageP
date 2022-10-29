#include<stdio.h>
#include<stdlib.h>
#include<limits.h>

/* C helper functions for image manipulation in python
 * Author:  Tamas Haraszti
 * Date 2019
 * License: LGPL 3 * compile:
 * gcc -shared -Wl,-soname,distance.so -fPIC -o distance.so distance-transform.c
 */

/* Hit_Miss:        A jit_miss filter for images
 *                  based on a 3x3 kernel, where the values are 0, 1 or undefined
 *                  If 1, the specific image point has to be 1
 *                  If 0, the specific image point has to be 0
 *                  If other, then it does not count
 *  Parameters:     img,        a numpy int array
 *                  kernel,     a numpy int array
 *                  Ni, Nj      the image dimensions
 *                  Nki, Nkj    the kernel dimensions
 *                  res         numpy int array for the result
 *  Return:             -1 on error, or 0 for success
 */
int Hit_Miss( int *img, int Ni, int Nj,
            int *kernel, int Nki, int Nkj,
            int *res)
{
    int i,j, ii, jj;
    int ni, nj, imgi, ki;
    int Nki2 = Nki/2, Nkj2 = Nkj/2;
    int r;

    if (img == NULL || kernel == NULL || res == NULL)
   {
       printf("An image, kernel and return image required\n");
       return(1);
    }

    if(Ni < Nki || Nj < Nkj)
    {
        printf("The kernel is larger than the image!\n");
            return(1);
    }

    /* we loop in the image and in the kernel
     * and we leave the edges to be defined by the caller who allocated res...
     */
    for( i= Nki2; i < Ni-Nki2; i++){
        for( j= Nkj2; j < Nj-Nkj2; j++){
            r = 1;
            for( ii=0; ii< Nki; ii++){
                for( jj=0; jj< Nkj; jj++){
                    ni = i + ii - Nki2;
                    nj = j + jj - Nkj2;
                    ki = *(kernel+ ii*Nkj + jj);
                    imgi = *(img+ ni*Nj + nj);

                    /* if kernel is set but the image is wrong,
                     * then result is 0, and break the loop
                     * else: go on
                     */
                    if( (ki == 1 && imgi !=1) ||
                    ( ki == 0 && imgi != 0) ){
                        r = 0;
                        ii = Nki;
                        break;
                    }
                    /* undefined kernel points and matching ones go on
                    */
                }
            }
            /* end kernel loop */
            *(res + i*Nj + j) = r;
        }
    }
    return(0);
}


/* Thinning         Thinnihg is removing the hit_miss filter for images
 *                  based on a 3x3 kernel, where the values are 0, 1 or undefined
 *                  If 1, the specific image point has to be 1
 *                  If 0, the specific image point has to be 0
 *                  If other, then it does not count
 *                  The resulted pixel is then subtracted from the image
 *  Parameters:     img,        a numpy int array
 *                  kernel,     a numpy int array
 *                  Ni, Nj      the image dimensions
 *                  Nki, Nkj    the kernel dimensions
 *                  res         numpy int array for the result
 *  Return:             -1 on error, or 0 for success
 */
int Thinning( int *img, int Ni, int Nj,
            int *kernel, int Nki, int Nkj,
            int *res)
{
    int i,j, ii, jj;
    int ni, nj, imgi, ki;
    int Nki2 = Nki/2, Nkj2 = Nkj/2;
    int r;

    if (img == NULL || kernel == NULL || res == NULL)
   {
       printf("An image, kernel and return image required\n");
       return(1);
    }

    if(Ni < Nki || Nj < Nkj)
    {
        printf("The kernel is larger than the image!\n");
            return(1);
    }

    /* we loop in the image and in the kernel
     * and we leave the edges to be defined by the caller who allocated res...
     */
    for( i= Nki2; i < Ni-Nki2; i++){
        for( j= Nkj2; j < Nj-Nkj2; j++){
            r = 1;
            for( ii=0; ii< Nki; ii++){
                for( jj=0; jj< Nkj; jj++){
                    ni = i + ii - Nki2;
                    nj = j + jj - Nkj2;
                    ki = *(kernel+ ii*Nkj + jj);
                    imgi = *(img+ ni*Nj + nj);

                    /* if kernel is set but the image is wrong,
                     * then result is 0, and break the loop
                     * else: go on
                     */
                    if( (ki == 1 && imgi !=1) ||
                    ( ki == 0 && imgi != 0) ){
                        r = 0;
                        ii = Nki;
                        break;
                    }
                    /* undefined kernel points and matching ones go on
                    */
                }
            }
            /* end kernel loop */
            *(res + i*Nj + j) = *(img + i*Nj +j) - r;
        }
    }
    return(0);
}
