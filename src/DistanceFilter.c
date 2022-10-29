#include<stdio.h>
#include<stdlib.h>
#include<limits.h>

/* C helper functions for image manipulation in python
 * Author:  Tamas Haraszti
 * Date 2018
 * License: LGPL 3 * compile:
 * gcc -shared -Wl,-soname,distance.so -fPIC -o distance.so distance-transform.c
 */

/* DistanceFilter1DL1:   calculate distance transform for a 1D stripe
 *                      set each pixels to the number of pixels to
 *                      the closest <= 0 pixel
 *                      It overwrites the input array
 *                      Only positive pixels are considered as set value
 *  Parameters:
 *  L:                  an array of pixels (int)
 *  N:                  length of array
 *  squared:            if > 0 set the square distance, else linear one
 *
 *  Return:             -1 on error, or 0 for success
 */
int DistanceFilter1DL1(int* L, int N, int squared)
{
    int i,j=0;
    int hit=0;

    if(L == NULL || N < 1){
        printf("Invalid input array!\n");
        return(-1);
    }
    if(N == 1){
        *L = 0;
        return(0);
    }

    /* We run up along the line, and from the first 1
     * we are setting an increasing integer value
     * if it is 1 again, we reset it again
     * The key point: j = 1, because the first nonzero pixel is
     * 1 pixel away of the neighbouring 0 one.
     */
    /* hit indicates if we have already encountered something.
     * if not, do not change the array.
     */
    hit = 0;
    j = 1;
    for(i= 0; i < N; i++){
        if(*(L+i) > 0){
            if (hit != 0){
                *(L+i) = j;
                j++;
            }
        } else{
            j=1;
            hit = 1;
        }
    }

    /* Now we have a set, now we go backwards, and
     * repeat, but this time taking the minimum between
     * the actual value and the potential new value
     */
    /* let us generate the right side distance too:
     */
    hit = 0;
    for(i = N-2; i >= 0; i--){
        /* if the higher neighbor + 1 is less, then take that
         * else leave the actual one */
        if (*(L+i) == 0) {
            hit = 1;
        }
        if (hit > 0) {
            *(L+i)= *(L+i) > (*(L+i+1)+1) ? *(L+i+1)+1 : *(L+i);
        }
    }
    if (squared > 0){
        for(i= 0; i < N; i++){
            if( *(L+i) < INT_MAX){
                *(L+i) = (*(L+i))*(*(L+i));
            }
        }
    }

    return(0);
}

/* DistanceFilter1D:        an Euclidean distance filter in 1D
 *                          It uses the DistanceFilter1DL1 above for the first run
 *
 *                          Calculate the lower parabolic envelop for L, then
 *                          a distance transform.
 *                          The transform is:
 *                          min((x-x')**2 + f(x')) for every x'
 *
 *                          Assume background is 0, valid values are >0
 *
 *  Parameters:
 *  L                       an array of 0 or >0 values; it would be overwritten
 *  D                       an empty array for the return values, same length as L
 *  N                       length of the array (int)
 *  Return value:           0 if o.k., -1 if failer
 */

int DistanceFilter1D( int* L, int* D, int N)
{
    int i, k=0, vk, s=0, ds=0;
    int *v = NULL, *z= NULL;
    int run= 1;

    if(N < 1 || L == NULL)
    {
        printf("Invalid array\n");
        return(-1);
    }

    if(N <2){
        *L = 0;
        return(0);
    }

    if((v = (int*)malloc((N+1)*sizeof(int))) == NULL){
        printf("Allocation of V got an error!\n");
        return(-1);
    }
    if((z = (int*)malloc((N+1)*sizeof(int))) == NULL){
        printf("Allocation of Z got an error\n");
        free(v);
        return(-1);
    }

    /* Initialize the values */
    for( i=0; i <N; i++){
        *(v+i) = 0;
        // *(z+i)=INT_MAX;
        *(z+i)=0;
    }

    k = 0;
    *z = INT_MIN;
    *(z+1) = INT_MAX;
    *v = 0;

    for (i=1; i < N; i++){
        /* INT_MAX elements make s infinite */
        if(*(L+i) == INT_MAX) {
            continue;
        }

        run = 1;
        // while(run == 1 && k > -1) {
        /* to agree with the original paper */
        while(run == 1) {
            vk = *(v+k);
            ds = 2*(i - vk);

            s = (int)((double)(i*i - vk*vk + *(L+i) - *(L+vk))/(double)ds);

            /* The original set is such that *z = - inf, *(z+1) is inf
             * thus, for k=0 this cannot fulfill,
             * for undefined z values it will
             * s cannot be 0, so where z is 0 it is infinite
             */
            if(s <= *(z+k) ){
                k --;
            }
            else{
                run = 0;
            }
       }
       /* the while loop is to end */
       k ++;
       *(v+k) = i;
       *(z+k) = s;
       *(z+k+1) = INT_MAX;
    }

    k = 0;
    for(i= 0; i < N; i++) {
        while(*(z+k+1) < i) {
            k++ ;
        }

        vk = *(v+k);
        *(D+i) = (i-vk)*(i-vk) + *(L+ vk);
    }

    free(v);
    free(z);
    return(0);
}
