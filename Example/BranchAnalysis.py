#!/usr/bin/env python
""" Analyze linear structures in images to see how tubes / lines
    propagate and branch.
    Use a filtered skeleton to identify branching.

    Author:     Tamás Haraszti
    License:    Creative Commons 4 (BY)
    Date:       2024-01-05
    Warranty:   None

    This example is based on the ImageP and BatchAnalyzer packages
"""

from BatchAnalyzer import Report, ReadConf, SaveData
from glob import glob
from ImageP import *
from numpy import sqrt, array, zeros, quantile, any
import os
import sys

# default parameters
configfile= 'config.txt'
config = {'dir': './', 'outdir': 'Results', 'fmask':'*.tif', 'dpi': 150,
          'ext': '.png',
          'Ws': 1, 'Wb': 10, 'gamma': 1.5, 'blob_factor': 5, 'MinSize': 5,
          'threshold': -1.0, 'q': -1,
          'scaler': 1.0, 'scaler_file': ''
        }
# here MinSize is a minimum node length for the clean_skeleton


############ Functions ################
def clean_skeleton(skelimg, node_img, MinSize= 5):
    """ walk through the skeleton, and use points where multiple
        branches converge.
        Assume a skeleton shold not contain short side lines here
        and there, which are often a side effect of skeletoinzation.
        Try eliminating segments which are just sticking out or
        not connected to the rest at all.
        Thus, the skeleton is a complex mesh of connected lines.

        Parameters
        skelimig: a binary image containing a skeleton
        node_img: a binary image of the node points only
        MinSize: how long should be a side branch to count as real

        Return
        the cleaned skeleton image
    """

    nodeless = skelimg.copy()
    nodeless[node_img > 0] = 0
    display(composite(skelimg, nodeless))
    # we count all short pieces only, leave the long ones alone
    node_labels = bwlabel(nodeless, MaxSize = MinSize + 1)

    print('found', node_labels.max(), 'edges')

    for i in range(1, node_labels.max()+1):
        indx_i, indx_j = (node_labels == i).nonzero()

        # a 'real node' is one either standing free or ending
        # in branching points on both ends
        #
        # is this a real node or not?
        # identify the branching points at the ends
        low_end_i = max(indx_i[0]-1, 0)
        low_end_j = max(indx_j[0]-1,0)
        high_end_i = min(indx_i[0]+2, skelimg.shape[0])
        high_end_j = min(indx_j[0]+2, skelimg.shape[1])
        has_node_low = any(node_img[low_end_i:high_end_i, low_end_j:high_end_j] >0)
        # the other end
        low_end_i = max(indx_i[-1]-1, 0)
        low_end_j = max(indx_j[-1]-1,0)
        high_end_i = min(indx_i[-1]+2, skelimg.shape[0])
        high_end_j = min(indx_j[-1]+2, skelimg.shape[1])
        has_node_high = any(node_img[low_end_i:high_end_i, low_end_j:high_end_j] > 0)

        # if it is a segment or a free standing node, leave it
        #if (has_node_high and has_node_low) or \
        #        ((not has_node_high) and (not has_node_high)):
        # let us drop the free pieces shorter than maxsize:
        if (has_node_high and has_node_low):
            # print('passing node', i)
            continue
        # else:
        # one end is missing, erase it
        # print('cutting node:', i)
        # skelimg[SimpleDilate(node_labels == i)] = 0
        skelimg[indx_i, indx_j] = 0
        # end if to erase
    # end for nodes

    return(skelimg)
# end clean_skeleton


def remove_blobs(bimg, radius= 25, width= 5):
    """ do a blob removal based on the structure tensor method.
        Try avoiding the removal of complex structures with a blob like
        part within.

        return the clean binary image
    """
    # structure tensor:
    st = StructureTensor(bimg, radius= radius, width= width)
    bw_img = bwlabel(bimg)

    dmask = st['indx'] & (st['c'] < 0.5)
    masked_img = bimg.copy()

    for i in range(1, bw_img.max()):
        # ha a full structure is blob, then all
        # nonzero pixels cover a dmask pixel
        if all(dmask[bw_img == i] == 1):
            masked_img[bw_img == i] = 0

    return masked_img
# end remove_blobs


################### main program
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) >0:
        if os.path.isfile(args[0]):
            configfile= args[0]
            print('Received config file:', configfile)
    else:
        print('Please provide the name of the configuration file')
        sys.exit(0)
# end if
config = ReadConf(configfile, config, simplify= True)

indir = os.path.abspath(config['dir'])
outdir = indir if config['outdir'] == 'dir' else config['outdir']

if not os.path.isdir(outdir):
    os.mkdir(outdir)
# end if no outdir

rep = Report(outdir,
        'report.rep',
        header='Contour and node detection for fibrous scaffolds v. 0.5'
             )

rep.write('Configuration parameters', color='cyan')
for k,v in config.items():
    rep.write(k,':', v)
# end dumping configuration

ext= config['ext']
dpi = int(config['dpi'])

scaler = config['scaler']

scaler_dict= {}
if os.path.isfile(os.path.join(indir, config['scaler_file'])):
    with open(
            os.path.join(indir, config['scaler_file']),
            'rt',
            encoding='UTF-8') as fp:
        txtlines = fp.readlines()
        if txtlines and ',' in txtlines[0]:
            linelist = [i.split(',') for i in txtlines]
            scaler_dict = {i[0]:float(i[1]) for i in linelist if len(i) > 1}
            del(linelist, txtlines)
# end loading scaler_dict

# parameters for the Gaussian smoothing (background correction)
Ws = config['Ws']
Wb = config['Wb']


lst = glob(os.path.join(indir, config['fmask']))
if not lst:
    rep.write('No files found at:', indir, 'via', fmask, color='red')
    sys.exit(0)
# end if nothing found

lst.sort()

N = int(config['N']) if 'N' in config else len(lst)

if N < len(lst):
    lst = lst[:N]
# end trimming list

# collect a result set in a table
results = []
res_keys = []

for fn in lst:
    img = read_img(fn)
    fnn = os.path.split(fn)[-1]

    if img is None:
        rep.write('Unable to load image', fnn, color='red')
        continue
    # end if image not found
    rep.write('Loaded', fnn, color='green')
    resrow = {}
    resrow['filename']= fnn

    # set scaler:
    fnn_sc = os.path.splitext(fnn)[0]
    if scaler_dict != {} and fnn_sc in scaler_dict:
        scale = scaler_dict[fnn_sc]
        rep.write('Scaler is set to:', scale)
    else:
        scale = scaler
    # end adjusting scaler
    resrow['scaler'] = scale

    img2 = GaussDeblur(img, 3*Wb, Wb, min(3*Ws, 5), Ws, True)
    if config['gamma'] > 0 and config['gamma'] != 1.0:
        img2 = Compress(img2, config['gamma'])
    # end compressing the image

    fout = f'{fnn}-filtered-image{ext}'
    display(img2,
            fname= os.path.join(outdir, fout),
            dpi=dpi)

    # bimg = img2 > quantile(img2, 0.9)
    if 'q' in config and config['q'] > 0 and config['q'] < 1:
        th = quantile(img2, config['q'])

    elif config['threshold'] < 0:
        th = graythresh(img2)
        rep.write('Automatic threshold', th)
        th = th * img2.max()
    rep.write('Current threshold', th)

    bimg = img2 > th
    rep.write('Cleaning out blobs using the structure tensor')
    bimg_clean = remove_blobs(bimg,
                              radius= config['blob_factor']*min(5, 3*Ws),
                              width= config['blob_factor']*Ws)

    # dimg = DistanceFilter(bimg_clean)
    # skimg = FindLocalMaxima(bimg_clean)
    rep.write('skeletonization')
    skimg = Skel(bimg_clean)

    N, nodes = get_nodes(skimg)
    resrow['node points'] = N
    rep.write('Found', N,'nodes')

    sk_l = skimg.sum() * scale
    resrow['length'] = sk_l
    rep.write('Skeleton has a length of:', skimg.sum(), 'pixels', sk_l, 'microns')
    resrow['area'] = bimg_clean.sum()*scale*scale
    rep.write('Area of the binary structure:', resrow['area'], 'square microns')

    fout = f'{fnn}-skeleton-composite{ext}'
    display(composite(img2, skimg, nodes),
            fname= os.path.join(outdir, fout),
            dpi= dpi)
    rep.write('Saved image to:', fout)

    # try cleaning the image:
    rep.write('Correcting skeleton (cutting short non-connecting edges)')
    skimg2= clean_skeleton(skimg.copy(), nodes, MinSize= config['MinSize'])

    N, nodes2= get_nodes(skimg2)
    rep.write('New node number:', N,  color='cyan')
    sk_l2 = scale*skimg2.sum()
    resrow['corrected node number'] = N
    resrow['corrected length'] = sk_l2
    rep.write('Corrected skeleton length:', skimg2.sum(), 'pixels', sk_l2, 'microns')

    fout = f'{fnn}-corrected-skeleton-composite{ext}'
    display(composite(skimg, skimg2, nodes2),
            fname= os.path.join(outdir, fout),
            dpi= dpi)
    rep.write('Saved image to:', fout)

    if not res_keys:
        res_keys = list(resrow.keys())

    results.append(list(resrow.values()))
# end for images

# now, dump a summary table

if results and res_keys:
    SaveData(res_keys,
             results,
             filename= os.path.join(outdir, 'Summary-table.txt'),
             remark= 'Summary of node analysis for images',
             report= rep
             )
else:
    rep.write('No summary was generated', color='red')
rep.close()
# EOF
