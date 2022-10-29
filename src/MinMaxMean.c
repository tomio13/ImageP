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

/* MinMaxMeanStd: calculate all these from an array
 * and return the result in one run.
 * Parameters:
 * img:     a numpy array
 * N:       size of the array 
 *          (dimensionality does not matter)
 * Min, Max, Mean, Std: pointers to variables to receive the result 
 *
 * Return value:    0 on success, -1 on error */
int MinMaxMeanVar(double *img, int N,\
                    double *Min,\
                    double *Max,\
                    double *Mean,\
                    double *Var)
{
    int i;
    double s, sum, sum2;
    double cv;

    if( img == NULL || N < 2 )
    {
        printf("Invalid array\n");
        return -1;
    }

    sum = 0.0;
    *Min = *img;
    *Max = *img;

    for( i=0; i<N; i++)
    {
        cv = *(img +i);

        if(cv > (*Max) )
            *Max = cv;
        else if(cv < (*Min))
            *Min = cv;

        sum += cv;
    } /* now we have some basic values */
    *Mean = sum / (double)N;

    sum = 0.0;
    sum2 = 0.0;
    for(i=0; i < N; i++)
    {
        /*the difference from mean: */
        s = (*(img+i) - (*Mean));
        sum += s ;
        sum2 += s*s;
    }
    *Var = (sum2 - sum*sum/(double)(N))/(double)(N-1);

    return 0;
}
