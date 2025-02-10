#!/usr/bin/env python
""" This script was written to batch process images of a bead trapped
    by an optical tweezer. The result was crappy, but the script is a good
    example of the ImageP package.

    Use it with proper switches for setting paramters.


    Author: Tamas Haraszti
    Waranty: none
    last modified: 2010 July
"""

import numpy as nu
#from matplotlib import use
#use('GTK')
#for file only output (remote scripts etc.):
#use('cairo.png')
#use

from matplotlib import pyplot as pl
pl.ioff()

from ImageP import *


import glob
import os, sys

try:
    from cPickle import dump, load
except:
    from pickle import dump, load
#end try

from time import time, ctime

#######################################################################
# Argument parsing, and variable defaults:
from optparse import OptionParser
parser = OptionParser(version='0.1')
parser.usage = "Usage:\n %prog -d datadir -f *.imgextension -l objsize ... other options"

parser.add_option('-d', '--dir', dest='indir',\
            help='Path to the input / working directory',\
            default = './')

parser.add_option('-f', '--file', dest='fmask',\
            help='File mask. Defaults to *.tif', default='*.tif')

parser.add_option('-b', '--background', dest='bkg',\
            help='Background data file name (pickle object)')
parser.add_option('-M', '--meanbackground', dest='mbkg',\
            help='Take the mean off the image and kill negative pixels', action='store_true')

parser.add_option('-R', '--ROI', dest='ROI', nargs=4, type='int')

parser.add_option('-N', '--number', dest='N', type='int',\
        help='Maximum number of images to be processed')

parser.add_option('-l','--lobj', dest='lo', type='int', default=33,\
        help="Size of the particle in pixels; default: 33")

parser.add_option('-L','--lnoise', dest='ln', type='float', default=0.5,\
        help="Scale of the noise in pixels. Default: 0.5")

parser.add_option('-s','--stepsize', dest='step', type='float', default=0.0,\
        help="Step limit of tracking (in pixels). If <= 0.0 (default) then sets 2*l")

parser.add_option('-g', '--maxgap', dest='maxgap', type='int', default=0,\
        help="Maximal number of empty hits to be jumped over, default=any")

parser.add_option('-m', '--minlength', dest='minlength', type='int', default=2,\
        help='Minimum length of a trajectory')

parser.add_option('-t', '--threshold', dest='th', type='float', default=0.667,\
        help='Relative intensity threshold above which peaks are searched; default: 0.667')

parser.add_option('-e', '--edge', dest='Edge', action='store_true',\
         help="Keep features near the edge. A margin of diameter/2")

parser.add_option('-r','--resume', dest='res', action='store_true',\
        help="Resume previous tracking if res-end.dat exists")

parser.add_option('-p','--no-pickle', dest='nores', action='store_true',\
            help="disable temporary dumping during processing")

parser.add_option('-v', '--verbose', dest='verbose', action='store_true')
parser.add_option('-q', '--quiet', dest='verbose', action='store_false')

(op, args) = parser.parse_args()
########################################################################
# default values:
verbose = op.verbose if op.verbose else False
MaxGap = op.maxgap
MinLength = op.minlength

indir = op.indir
fmask = op.fmask

if not os.path.exists(indir):
    print("Path does not exist!")
    sys.exit(1)
#end if

if verbose:
    print("Data: %s" %(os.path.join(indir,fmask)))
#collect all image file names into a list:
lst = glob.glob(os.path.join(indir, fmask))
#Be sure, that they are in sequence:
lst.sort()
if len(lst) < 1:
    print("Files not found")
    sys.exit(1)
#end if
#define number of images:
N = len(lst)

if op.N:
    if N > op.N :
        lst = lst[:op.N]
        N = op.N
    #end if
    print("N: %d of images to be processed" %N)
#end if

mbkg = False
if op.mbkg :
    mbkg = True
#mean background

if op.bkg :
    try:
        fp = open(op.bkg, 'rb')
        bkg = load(fp)
        fp.close()
    except IOError:
        print("Background file does not exist")
        sys.exit(1)
else:
    bkg = None
#end if

if op.Edge:
    CutMargin = False
else:
    CutMargin = True
#end if

#Particle size:
diam = op.lo
#it is better to have particles with odd pixel diameter
if diam%2 == 0 :
    diam += 1
#end if

#now set the tracking step limit
stepsize = op.step if op.step > 0.0 else 2.0*diam

#parameters for the bandpass filter and PeakFind
#this is taken from the Matlab code of Rainer Kurre...
lobject = diam
lnoise = op.ln
th = op.th

ROI = op.ROI if op.ROI else []
RN = len(ROI) if op.ROI else 0

#generate information:
if verbose:
    pl.ion()
#end if

#before we really start, print a header, and put it to a report:
fp = open('report.rep','at')
txt = "New run at: %s\n" %(ctime(time()))
fp.write(txt)
txt = "Particle tracking with the following parameters:\n"
fp.write(txt);
print(txt)

if bkg != None:
    txt= "Background to substract: %s\n" %op.bkg
else:
    txt ="No background subtraction\n"
fp.write(txt)
print(txt)

if mbkg:
    fp.write('Mean background is subtracted from image, keeping higher intensities only\n')

#end if

if CutMargin:
    txt = "Edge of the image is ignored\n"
else:
    txt = "Edge of the image is kept\n"
fp.write(txt)
print(txt)

txt = "Number of images to process:\t%d\n" %N
fp.write(txt); print(txt)

txt= "Object size parameter (pixels):\t%.3f\n" %diam
print(txt)
fp.write(txt)
txt ="Noise parameter (pixels):\t%.3f\n" %lnoise
print(txt)
fp.write(txt)

txt= "Relative threshold:\t%.4f\n" %th
print(txt)
fp.write(txt)

if ROI != []:
    txt= "ROI defined:"
    for i in ROI:
        txt = "%s\t%d" %(txt,i)

    txt = "%s\n" %txt
else:
    txt= "ROI not defined\n"
#end if
fp.write(txt); print(txt)

txt= "Minimum trace size (points):\t%d\n" %MinLength
fp.write(txt); print(txt)

txt= "Maximum gap size (frames):\t%d\n" %MaxGap
fp.write(txt); print(txt)

txt= "Maximum step size (pixels):\t%d\n" %stepsize
fp.write(txt); print(txt)
if op.nores:
    txt = "Disable temporary storage during processing\n"
    fp.write(txt); print(txt)
    nores = True
else:
    txt = "Temporary dumps are generated during processing\n"
    fp.write(txt); print(txt)
    nores = False
#ebd if op.nores

if op.res:
    txt = "Resume previous tracking if possible\n"
    fp.write(txt); print(txt)

    try:
        rp = open('res-end.dat','rb')
        res = load(rp)
        rp.close()
        del(rp)
    except IOError:
        txt = "Unable to resume, res-end.dat read error\n"
        fp.write(txt); print(txt)
        res = []
        N0 = 0

    else:
        N0 = len(res)

        if N0 > N:
            txt = "Tracking of %d is already done. To repeat, please drop the -r switch" %N0
            fp.write(txt); print(txt)
            sys.exit(0)
        #end if N0
else:
    res=[]
    N0 =0

fp.write('End of info\n\n')
fp.close()
#header done

###############################################################################
# Processing starts here:

#we have to truncate the background:
#if invalid, we can crash the program here
if bkg != None and RN == 4:
    bkg = bkg[ROI[0]:ROI[1], ROI[2]:ROI[3]]
#end if

figure = pl.figure(num=1)
plt = figure.add_subplot(1,1,1)

tm = time()
#for verbosity we need a counter
#and another one for dumping current results
i10 = 1
isave = 0
Nsave = 1000

#Now, run through the stack:
for i in range(N0,N):
    name = lst[i]
    #Talk back a bit:
    if i10 == 10 or verbose:
        print("%d: %s is processed, %.1f seconds elapsed" %(i, name, time()-tm))
        i10 = 1
    else:
        i10 += 1
    #end if

    #Load the image:
    try:
        img = read_img(name)
    except:
        print("unable to read: %s" %name)
        continue
    if img is None or img.size == 0:
        print("Invalid image, skipping file: %s" %name)
        continue

    #Set the area of interest:
    if RN == 4:
        if verbose:
            print("Cutting image to: ", ROI)
        #end if
        #if ROI invalid, we have a problem...
        img = img[ROI[0]:ROI[1], ROI[2]:ROI[3]]
    #end if

    if bkg is not None :
        img = img - bkg
        img = img*(img > 0)

    elif mbkg:
        img = img - img.mean()
        img[ img < 0] = 0
    else:
        #take off the offset for the convolution filtering:
        img = img - img.min()

    #Filter the image:
    img2 = BPass(img, lobject, lnoise)
    # kill negative background (added at 2022-05-05)
    img2[img2 < 0] = 0

    #Feedback: plot the filtered image:
    #Comment this out for speed.
    if verbose :
        plt.cla()
        plt.imshow(img2, interpolation='nearest', origin='lower')
        pl.title(name)
        pl.draw()
    #end if

    #if there is nothing left, we have a problem
    #but let the other routines take care of that at the moment

    #Find the rough position in the filtered image
    #the second parameter is a relative treshold (0-1)
    #the third is the minimum radius: point +/i width is investigated
    #things smaller than this will be neglected!
    #We increase this with 1 pixel to make sure...
    pos = PeakFind(img2, th, int(diam/2), verbose=verbose)

    #Refine the position and extract some more information as well:
    #here the third parameter is the window size around the maximum
    #which should be taken into accound to calculate the center of mass
    #etc.
    #The key is: diam defines the window around the maximum spot, thus
    #it has a direct effect on how we get the result.
    #Do not change it here, let the user really select, because this
    #number is also acting as a window on which pixels are taken into account
    #Peak pos applies the mask around the center defined by PeakFind,
    #actually, for bright particles with dark hallow, it is better
    #to have some higher radius, so the PeakPos defined the background
    #bt the shadow part...

    newpos = PeakPos(img2, pos, size= diam, circular=True,\
                    CutEdge=CutMargin, verbose=False)

    #res.append(PeakPos(img2,pos,size=diam))
    res.append(newpos)

    if verbose:
        print("%d positions found" %(newpos['X'].size))
    #end if

    if not nores and isave == Nsave :
        #fp = open('res-end.dat','wb')
        try:
            fp = open('res-end.dat','wb')
            dump(res, fp, protocol=2)
            fp.close()
        except:
            print("Error creating pickle file!")
            nores = True

        print("Positions dumped to res-end.dat at %d" %i)

        isave = 1
    else:
        isave += 1
    #end if
#End for name

print("Finding particles finished in %.1f seconds" %(time()-tm))

#Save a copy of the result:
fn = 'res-end.dat'
print("Dumping position data (not tracked) to: %s" %fn)
try:
    fp = open('res-end.dat','wb')
    dump(res, fp, protocol=2)
    fp.close()
except:
    print("can not dump results data")
#end try

#res = nu.load('res-end.dat')

#create a track (destroys res).
#Parameters: res, step size between images, minimum length of a track,
#maximum number of images with missing hits (gap)
track = ParticleTrack(res,\
        maxstep=stepsize,\
        minlength=MinLength,
        maxgap = MaxGap,
        verbose=verbose)

if track == [] :
    print("No results\n");
    exit(0);
#end if empty track

fn = "track-data.dat"
print("Dumping track data to %s" %fn)

try:
    fp = open(fn,'wb')
    dump(track, fp, protocol= 2)
    fp.close()
except:
    print("can not dump binary track data")

print("Plot a summary")
#the background corrected image may be less explanatory:
if bkg is not None :
    img = read_img(name)
    if RN == 4:
        img = img[ROI[0]:ROI[1],ROI[2]:ROI[3]]
    #end if
#end if

plt.cla()
pl.gray()
plt.imshow(img, origin='lower', interpolation='nearest')
plt.axis('image')
plt.axis('off')
colors = ['red','green','blue','cyan','magenta','yellow','white']
NC = len(colors)

for i in range(len(track)):
    plt.plot(track[i]['X'],track[i]['Y'],'+', color=colors[i%NC])
    txt = '%d' %i
    plt.text(track[i]['X'][0]-2*diam,track[i]['Y'][0],txt,\
            color= colors[i%NC],\
            horizontalalignment='left')
#end for
pl.draw()
pl.savefig('SummaryPlot.png', dpi=150)
pl.draw()


#Some summaries: Plot the tracks
for i in range(len(track)):
    plt.cla()
    plt.plot(track[i]['X'],track[i]['Y'],'+')
    print("%d-th track plotted, %d points\n" %(i,len(track[i]['X'])))
    pl.title('Tracks')
    pl.draw()
    txt = "%05dth-track.png" %i
    pl.savefig(txt, dpi=150)
    pl.draw()
#end for

#make another figure, plot the mod 1 residuals
#this histogram should be flat

#Create the histogram:
if N > 50:
    fig2 = pl.figure(num=2)
    plt2 = fig2.add_subplot(1,1,1)

    for i in range(len(track)):
        plt2.cla()
        hi1 = hist(nu.asarray(track[i]['X'])%1, 15)
        hi2 = hist(nu.asarray(track[i]['Y'])%1, 15)

        plt2.plot(hi1['midpoints'],hi1['dist'])
        plt2.plot(hi2['midpoints'],hi2['dist'])

        #force the scale of Y to 0 ... Ymax:
        ax = plt2.axis()
        ax = (ax[0], ax[1], 0.0, ax[3])
        plt2.axis(ax)
        pl.draw()

        txt = "%03d-histogram.png" %i
        pl.savefig(txt,dpi=150)
        pl.draw()
#end if

#save tracks:
outname = "tracks-table.dat"
print("saving data to: %s" %outname)
fp = open('tracks-table.dat',"w")
fp.write('#N: %d\n' %N)
fp.write('#X -> J, Y -> I\n')
fp.write('#All positions are in pixels\n')
#fp.write('#index\tX(pixels)\tY(pixels)\tI0\tR2\timage(factor)\n')

#save everything!
#changed 2011.06
#one possibility
#keys = a[0].keys()
#sorting reverse makes: indx, Y, X, R2, Mark ...
#keys.sort(reverse=True)
#or define:
txtkeys = ["indx","X", "Y", "I0", "R2", "E", "Err", "Mark"]
keys= list(txtkeys)
txtkeys.append('bead')
#header = '\t'.join( map( repr, txtkeys))
header = '\t'.join(txtkeys)
fp.write(header); fp.write('\n')

N = len(track[0][keys[0]])

for i in range(len(track)):
    #pull up the actual data table:
    b = []
    for j in keys:
        b.append(track[i][j])
    #end for j
    #bead:
    b.append([i]*N)

    #now put them out:
    for j in zip(*b):
        txt = '\t'.join(map(repr, j))

        #txt = "%d\t%.4f\t%.4f\t%.4f\t%.4f\t%d\n" %(track[i]['indx'][j], \
         #       track[i]['X'][j], \
         #       track[i]['Y'][j], \
         #       track[i]['I0'][j],\
         #       track[i]['R2'][j], i)
        #we have the string, dump it:
        fp.write(txt); fp.write('\n')
    #end for j
#end for i
