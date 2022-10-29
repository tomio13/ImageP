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


/* bwfloodlist:
 * Take an image, and a pixel in it
 * If this pixel is more than zero,
 * find all pixels around it that form a confluent area
 * Mark them into a response image
 * Do this recursively
 *
 *
 * Parameters: img: image, Ni, Nj the shape
 *          x,y the i,j indices of the selected pixel
 * result: an int array of Ni*Nj (similar to the image)
 * NewI: the new value to write into res (0 is allowed!)
 * eight: 0 or nonzero if eight pixel neighbours should be taken
 * Return: N - the number of pixels marked or -1 on error
 */
int bwfloodlist( int *img, int Ni, int Nj, \
                int x, int y, int gap,\
                int* Results, \
                int NewI, int eight)
{
    int N, Nindx, NNindx; /* hits, current lookup index, and maximum of the list */
    int i=0,j=0, Mini, Minj, Maxi, Maxj;
    int nx, ny, nxy; /*new x, new y, and the linear version of them nxy */
    int *indximg;

    if( x<0 || x >= Ni || y<0 || y >= Nj || gap<0 || img==NULL || Results==NULL)
    {
        printf("Invalid pixel indices or gap: %d, %d, %d", x,y,gap);
        return -1;
    }

    /* If this pixel is already empty, then return zero.
     * This should stop the recursion here */
    if( *(img+x*Nj+y) == 0)
    {
        return 0;
    }
    /*else: */
        /*O.k., our pixel is fine; now delete it, and mark in Results */
        /* Not deleting would make a circular reference below */
    NNindx = Ni*Nj;
    if( (indximg = (int*)malloc( NNindx*sizeof(int))) == NULL)
    {
        printf("memory error!\n");
        return -1;
    }

    /*first initialize to the actual point: */
    *indximg = x*Nj+y;
    N = 0;
    Nindx = 1;

    /*we process until we have something in here */
    while (Nindx > 0)
    {
        /*for simplicity we stored the index as a linear one */
        /* Now we need it back : */
        nxy = *(indximg);
        nx = nxy/ Nj;
        ny = nxy - nx *Nj;
        /* taken, then erase it */
        *indximg = 0;

        /*This is a hit, so deal with it. multiple hits are
         * cleaned up at the end */
        /* we can use the linear address at this point: */
            *(img + nxy ) = 0;
            *(Results + nxy ) = NewI;
            N ++;

            /* And check the neighbours: */

            /* define edges such, that the +/- 1 is checked by default: */
            Mini = (nx-1-gap) < 0 ? 0 : nx-1-gap;
            Minj = (ny-1-gap) < 0 ? 0 : ny-1-gap;
            Maxi = (nx+2+gap) > Ni ? Ni : nx+2+gap;
            Maxj = (ny+2+gap) > Nj ? Nj : ny+2+gap;

                /* Investigate the neighbours */
                /* gap allows a larger area to be called directly,
                 * even if there are empty points between */
                if(eight)
                {
                    for(i = Mini; i< Maxi; i++)
                    {
                        for(j = Minj; j< Maxj; j++)
                        {
                            if( *( img + i*Nj + j) != 0)
                            {
                                /*it is a + pixel, => add to the list */
                                *(indximg + Nindx) = i*Nj+j;
                                /*with the memory recollection below this can not
                                 * grow beyond the original image*/
                                Nindx ++;
                            }
                        }
                    }

                }
                else
                {
                    /*4 neighbours: up, down, left, right only. This runs through the already
                     * deletep pixel another twice */
                    for(j=ny, i=Mini; i< Maxi; i++)
                    {
                        if(*(img + i*Nj + j) != 0)
                        {
                            *(indximg + Nindx) = i*Nj+j;
                            Nindx ++;
                        }
                    }

                    for(i=nx, j=Minj; j< Maxj; j++)
                    {
                        if(*(img + i*Nj + j) != 0)
                        {
                            *(indximg + Nindx) = i*Nj+j;
                            Nindx ++;
                        }
                    }
                }/*end if(eight) */

            /*Clean up a bit. The list contains a zero at the beginning
             * and may contain multiple of nxy, which we just have
             * investigated and sorted out. Remove them:
             * copy all other valid values to the front of the list */
            for( i=0, j=0; i < Nindx; i++)
            {
                if( *(indximg + i) != 0 && *(indximg + i) != nxy )
                {
                    *(indximg + j) = *(indximg+i);
                    j ++;
                }
            }
            Nindx = j;
        }/* end of while*/

    free(indximg);

    return N;
}


/* bwlabel is using bwfloodlist to mark all confluent patches in an image
 * Parameters:
 * img :    an integer image (Numpy array, one dim, confluent)
 * Ni, Nj:  the size of the image, img is Ni*Nj large
 * res :    the result image, possibly an empti integer array
 *          (same size as img!)
 * MinSize: minimum this much of points have to be present in the patch
 * MaxSize: maximum this much of points have to be present in the patch
 * gap:     this size of gaps, missing points are tolerated
 * eight:   if 1 then all 8 direct neighbours are checked otherwise
 *          only 4 (up, down, left and right)
 * Return value:
 *  the number of marked patches (numbering starts with 1)    */
int bwlabel( int * img, int Ni, int Nj, int* res,\
            int MinSize, int MaxSize, int gap, int eight)
{
    int i, mark, Nij;
    int x,y, n, j, k;

    if( Ni < 2 || Nj<2 || gap<0 || img==NULL || res==NULL || MinSize<0 || MaxSize <= MinSize)
    {
        printf("Invalid image or parameters: Ni %d, Nj %d, gap %d, MinSize %d, MaxSize %d\n", Ni,Nj,gap, MinSize, MaxSize);
        return -1;
    }

    Nij = Ni*Nj;
    mark = 1;

    for(i=0; i<Nij; i++)
    {
        /* Call bwfloodlist if the points is nonzero */
        if( *(img + i) != 0 )
        {
            /* get the indices of the current position:*/
            x = i / Nj;
            y = i - Nj*x;

            n = bwfloodlist(img, Ni, Nj, x,y, gap, res, mark, eight);

            if( n > 0)
            {
                /*if not large enough or too large we have to remove the patch*/
                if( n < MinSize || n > MaxSize )
                {
                    /*it can not be before i */
                    /*and there is maximum n points */
                    for(j=i, k=0; j < Nij && k < n; j++)
                    {
                        if( *(res+j) == mark )
                        {
                            *(res+j) = 0;
                            k ++;
                        }
                    }/*end of cleaning*/
                }
                else
                {
                    /*it is a valid hit, change mark */
                    mark ++;
                }
            }/*End if hits were found */

        }
    }/*end main loop*/

    return (mark-1);
}
