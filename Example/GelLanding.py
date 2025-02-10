#!/usr/bin/env python
""" bright field image stacks from optical microscopy showing microgels landing on a
    glass surface
    Use the sharp points to count the landed objects. Try estimating the number of
    gels comparing areas.

    The program identifies non moving objects in focus and provide a count of them.
    Using the last image, and assuming that is the sharpest with most particles,
    it estimates a size for a single particle as the mean of the part of the distribution
    below its mean. Such thing may only work if there is a multimodal size distribution.
    Maybe some-time we can replace this part with something better.
    Then it divides every pixel area with the estimated single particle area,
    and provides the such modified rounded number of particles (the area division is
    rounded).

    The results are represented in a composite image, where red shows all objects,
    yellow those which did not move in comparison to the next image (in time).

    A table listing file names, an index along the time series, the estimated number
    of particles, the area used for normalization, and the corrected number of particles.
    Plotting the number or corrected number of particles vs. index gets a
    cumulative distribution of the process.

    Author:     Tomio
    License:    MIT
    date:       2024-02-05
    Warranty:   None
"""

from BatchAnalyzer import Report, ReadConf, SaveData, Plot
from glob import glob
from ImageP import *
from numpy import abs
import os
import sys

configfile= 'config.txt'
config= {'dir': './', 'outdir': 'dir',
         'fmask': '*.tif', 'dpi': 150, 'ext':'.png',
         'Wbg': 20, 'Ws': 1, 'threshold': -1,
         'invert': True,
         'MinSize': 100, 'MaxSize': 1000,
         "dilate": 1,
         }
# dir, outdir are the input and output folders
# Wbg is the width of the background Gaussian
# Ws is the sigman for the smoothing Gaussian
# threshold is a relative intensity threshold. if < 0 then Otsu's method
# is used.
# invert: use the negative of the images. Useful for bright-field images
# with dark objects in focus
#
# background: average the images and subtract this as background

# end of analysis

if __name__ == '__main__':
    args= sys.argv[1:]

    if len(args) > 0:
        if os.path.isfile(args[0]):
            configfile= args[0]
            print('received command line configuration file name:', configfile)
        else:
            print('invalid config file name')
            print('usage: program config.txt')
        # end if config
    # end if args
# end if main

config = ReadConf(configfile, config, simplify= True)
indir = os.path.abspath(config['dir'])
outdir = os.path.abspath(config['outdir']) \
                if config['outdir'] != 'dir' else indir

ext = config['ext']
dpi = int( config['dpi'])

if not os.path.isdir(outdir):
    # os.mkdir(outdir)
    os.makedirs(outdir)
#end create output dir?

lst = glob(os.path.join(indir, config['fmask']))
lst.sort()
N = len(lst)
conf_N = int(config['N']) if 'N' in config else -1

if conf_N > 0 and conf_N < N:
    N = conf_N
    lst = lst[:N]
# end truncating data

rep = Report(outdir, add_time= False, header='Cell landing analysis v 1.2')

rep.write('Dumping parameters set:', color='cyan')
for k,v in config.items():
    rep.write(k, v)
# end dumping parameters
rep.write('***********************************************')

rep.write('found', N,'images')
# Now, the actual work:
res =[]
if N == 0:
    rep.write('Images not found', color='red')
    rep.close()
    sys.exit(0)

# we need the threshold of that image for all of the others
# Then the threshold remains global for the whole stack!
# But then our index is messed up, so we count backwards
indx = N-1
first = True #first run
for fn in lst[::-1]:
    img = read_img(fn)

    ffn = os.path.splitext(fn)[0]
    fffn = os.path.split(ffn)[-1]

    if img is None:
        rep.write('Unable to read', fffn, color='red')
        continue

    rep.write('Read', fffn, color='cyan')
    res_row = {'file': fffn,
               'indx': indx}

    if 'invert' in config and config['invert'] == True:
        rep.write('inverting image')
        img = img.max() - img
    # end inversion

    # filters:

    if config['Wbg'] > 0:
        wbg = config['Wbg']
        ws = config['Ws']
        bimg = GaussDeblur(img,
                           int(3*wbg),
                           wbg,
                           int(3*max(ws, 1)),
                           ws,
                           True)
    elif config['Ws'] > 0:
        bimg = ConvFilter1D(img,
                            GaussKernel(int(3*max(config['Ws'],1)),
                                            config['Ws'])
                            )
    else:
        bimg = img
    # end filter

    ## now, we need some threshold
    if first== True:
        first= False
        th = config['threshold']
        if th > 0:
            th = th*bimg.max()
        else:
            th = graythresh(bimg)*bimg.max()

        rep.write('Global threshold:', th, color='blue')
        old_img = SimpleDilate(bimg > th, int(config['dilate']))

        continue
    # end defining threshold
    bin_img = bimg > th


    # trace those pixels which did not change:
    # bwimg = bwlabel(bin_img & old_img,
    #
    # use a different filter:
    # first identify spots on the image
    bwimg = bwlabel(bin_img,
                    MinSize= config['MinSize'],
                    MaxSize= config['MaxSize'])
    # remove those which moved
    # thus which have zero pixels in the old image
    for i in range(1, bwimg.max()+1):
        if (old_img[bwimg == i] == 0).any():
            bwimg[bwimg == i] = 0
    # end erasing existing patches
    # relabel the rest
    bwimg = bwlabel(bwimg > 0)

    out_path = os.path.join(outdir,
                            f'composite-image-{fffn}{ext}')
    # red: hits
    # green: filtered image
    # blue: found hits (size filtered)
    # display(composite(bin_bimg, bwimg>0, bimg),
    #
    # NEW
    # red: filtered results
    # green: size filtered hits
    display(composite(bin_img, bwimg>0),
            fname= out_path,
            dpi= dpi)

    N_row = bwimg.max()
    rep.write('Detected', N_row, 'objects', color='green')
    res_row['N']= N_row
    res_row['line_list'] = [(bwimg == i).sum() for i in range(1, N_row+1)]

    res.append(list(res_row.values()))
    indx -= 1
    old_img = SimpleDilate(bin_img, int(config['dilate']))
# end for

keys = list(res_row.keys())

# now, post process the rows
keys.append('area_1')
keys.append('corrected_N')

N = len(res)
first = True
line_indx = keys.index('line_list')
N_indx = keys.index('N')
unit_area = 0.0
# forwards on a backwards list
for i in range(N):
    # in this line of the results table,
    # take out all areas
    area_list = res[i].pop(line_indx)

    if len(area_list) < 1:
        res[i].append(0)
        res[i].append(0)
        continue

    # find a normalization factor, that is
    # a single particle size estimate:
    if first:
        first= False
        # try finding some sense in the data
        mean = sum(area_list)/float(len(area_list))
        # keep those which are small
        filtered_list = [j for j in area_list if j < mean]
        # take the average of those only
        one_size= sum(filtered_list)/float(len(filtered_list))
    # now, for every image:

    # record for every image the same size of a single particle
    res[i].append(one_size)

    corrected_N= [int(j/one_size + 0.5) for j in area_list]

    # correct for running backward
    out_path = os.path.join(outdir,
                           f'dot_areas-{N-i-1}.txt')

    SaveData(['dot_area', 'n_dots'],
             zip(*[area_list, corrected_N]),
             out_path,
             f'Dot area analysis for {lst[i]}',
             report= rep)

    res[i].append(sum(corrected_N))
# end looping on sizes

# line_list is used up, pop it from the keys
keys.pop(line_indx)

out_path = os.path.join(outdir,
                        'summary-table.txt')
SaveData(keys, res,
         out_path,
         'Results summary table of landing dots',
         report= rep)

rep.close()
# end processing images


