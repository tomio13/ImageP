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

/*for sorting we need comparison */
/* it has to return <0 if a<b; =0 if a==b, >0 if a>b: */
__inline static void sorting( double *a, int N)
{
    int i, j;
    double val;
    int done;

    /*insertion sorting: see Wikipedia 8)*/
    for(i=1; i<N; i++)
    {
        val = *(a+i);
        j = i-1;
        done = 0;
        while(done == 0)
        {
            /* move jth element up first to
             * the place of val, then below,
             * as long as either this is the 
             * beginning or the next element is
             * smaller than val.
             * Then insert val to that point.*/
            if (*(a+j) > val)
            {
                *(a+j+1) = *(a+j);
                j --;
                
                if( j < 0)
                    done = 1;
            }
            else
                done = 1;
        }
        /* bubble over, finish "swapping":*/
        *(a+j+1) = val;

    }/*end for i*/

    /*A verbose output for testing:*/
/*    for(i=0; i<N; i++)
        printf("%.1f\t", *(a+i));
    printf("\n");
*/
}

/* A rank filter takes a window around a pixel (+/- n points), and
 * sorts them to order.
 * It subtitutes the pixel with min, max or the median falue of the region
 * Parameters:
 * img:     an image array
 * Ni, Nj   the size of the image
 * N        the size of the kernel around a pixel (+/-N) in X direction
 * M        the size of the kernel around a pixel (+/-N) in Y
 * Result   the result image, a double array with the same size as img
 * t        the type of filtering: min, max or median (i,x,m)
 *
 * returns 0 on success, -1 on error
 */

int RankFilter(double *img, int Ni, int Nj,\
            int N, int M, double* Result, char t )
{
    int N2, Nij;
    int i, j, ni;
    double *kernel;
    int Nk, ii, jj;

    if(img == NULL || Result == NULL)
    {
        printf("Invalid image arrays!\n");
        return -1;
    }
    

    if( N<1|| M<0 || Ni < N || Nj < M)
    {
        printf("Invalid dimensions N, Ni, Nj: %d, %d, %d\n", N, Ni, Nj);
        return -1;
    }
    
    if( t != 'i' && t!= 'x' && t !='m')
    {
        printf("Invalid filtering selected: %c\n", t);
        return -1;
    }

    /*the size of the kernel overall:*/
    N2 = (2*N+1)*(2*M+1);
    Nij = Ni*Nj; /* linear addressing */

    if( (kernel = (double*)malloc(N2*sizeof(double))) == NULL)
    {
        printf("Memory allocation error!\n");
        return -1;
    }

    /*Now do the job: go through the image, and group the pixels */
    /* There is no reason touching the edges... or we have to check
     * that the indices do not go out */
    for( ni=0; ni< Nij; ni++)
    {
        /*these are integer operations*/
        i =  ni / Nj;
        j = ni - i*Nj;

            /* fill up the kernel array:*/
            Nk = 0;
            for( ii=i-N; ii < i+N+1; ii++)
            {
                for(jj=j-M; jj<j+M+1; jj++)
                {
                    if( ii > 0 && ii < Ni && jj >0 && jj <Nj)
                    {
                        *(kernel+Nk) = *(img + ii*Nj + jj);
                        Nk ++; /* this should not exceed the limit */
                    }
                }
            }/* end of for ii */
            
            /*sort them into ascending order */
            sorting(kernel, Nk);
            if( t == 'i' )
                *(Result + ni) =  *(kernel);
            else if( t == 'x')
                *(Result + ni) = *( kernel + Nk-1);
            else
                *(Result + ni) = *(kernel + Nk/2);
    }/*end for ni */

    free(kernel);
    return 0;
}
