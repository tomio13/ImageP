#!/usr/bin/env python
import setuptools # gets rid of the vcvarsall.bat issue
from setuptools import setup
from setuptools.dist import Distribution
from distutils.ccompiler import get_default_compiler
import os, sys

#We have some extension to be compiled into a dynamic lib. No real python C API
#is required, so we have to hack the setup a bit
#Let us go around: compile, link, then take the dynamic lib and put it
#to the install as a data file. Csource.py should pick it up.
PkgName="ImageP"
SourceList =['src/SimpleFilter', 'src/bwlabel','src/MinMaxMean',
             'src/RankFilter', 'src/SimpleFilter1D', 'src/hit_miss',
             'src/Perimeter',
             'src/PeakFind', 'src/SimpleErode', 'src/SimpleDilate',
             'src/DistanceFilter'
             ]

CSourceList = [f'{i}.c' for i in SourceList]
#end for

#some general declaration, variables:
ccname = get_default_compiler()
#where are the site packages:
#We need to tell where the package should go
# if nothing is set, python 3.10 creates ImageP...-data/data/
# not to land in there, try a relative path:
pathcore= '../../ImageP'

libname = 'Csources'
datafile = None

def build_src(libname, datafile, pathcore, ccname, CSourceList):

    if ccname == 'msvc':
        #experimenting showed this working on a win7 pc
        #from distutils.msvc9compiler import MSVCCompiler as CC
        from distutils.msvccompiler import MSVCCompiler as CC
        compile_flags = ["/Ox", "/MD"]
        objext = '.obj'
        #needs testing!
        libdirs = []
        libname = f'lib{libname}'
        deffile = f'/DEF:.\\src\\{libname}.def'
        link_args = ["/DLL","/nologo","/INCREMENTAL:NO", deffile]
        # link_postargs = [deffile]
        # pch.cpp contains the entry point for Win.
        CSourceList.insert(0, 'pch.cpp')
        outfile = f'{libname}.dll'
        datafile = [(pathcore,[outfile])]

    elif ccname == 'bcpp':
        print( "sorry, I do not know what to do with Borland C")
        CC = None

    elif ccname == 'emx':
        from distutils.emxccompiler import UnixCCompiler as CC
        print("sorry, I do not know what to do with EMX")
        CC = None

    elif ccname == 'mwerks':
        from distutils.mwerkscompiler import UnixCCompiler as CC
        print("sorry, I do not know what to do with mwerks Code warrior")
        CC = None
# pch.cpp contains the entry point for Win.

    else:
        #let us assume that cygwin and mingw needs the same options as gcc
        if ccname == 'cygwin' or ccname=='mingw32':
            from distutils.cygwinccompiler import UnixCCompiler as CC
            outfile = f'lib[libname].dll'
        else:
            #only 'unix' is left:
            from distutils.unixccompiler import UnixCCompiler as CC

        #compiler options:
        objext = '.o'
        compile_flags=['-Wall','-O3', '-fpic']
        libdirs = []

        #linking is:
        #gcc -shared -Wl,-soname,$OUTPUTFILE -o $OUTPUTFILE -l$LIB
        #On mac:
        #gcc -shared -dynamiclib MinMaxMean.c PeakFind.c Perimeter.c RankFilter.c SimpleDilate.c SimpleErode.c SimpleFilter.c SimpleFilter1D.c bwlabel.c -o libCsources.dylib

        #Is it a Mac?
        if "darwin" in sys.platform:
            outfile = f'lib{libname}.so'
            #outfile = "lib%s.dynlib" %libname
            print( "output: %s" %outfile)
            soname = "-dynamiclib"
            link_args=[soname]
        else:
            #Linux hopefully:
            outfile = f'lib{libname}.so'
            soname = f'-Wl,-soname,{outfile}'
            link_args=["-shared",soname]

        link_postargs=None
        #here we define where to copy and what:
        datafile = [(pathcore,[outfile])]

    #now try our best:
    try:

        print( "Calling the C compiler on the sources:")
        cc = CC()
        cc.compile(CSourceList, extra_postargs=compile_flags)
        print( "Done.")

        objlist = [f'{i}{objext}' for i in SourceList]

        print( "Calling the linker")

        #cc.link('',objlist, outfile, library_dirs=libdirs,\
        #I guess this way it should work:
        cc.link_shared_lib(objlist, libname, library_dirs=libdirs,\
                    extra_preargs=link_args, extra_postargs=link_postargs)

        print("building is done")

    except:
        print("Building the C extension failed.")
        print("Building without the C module:")
        datafile = None
    #End try

    return (libname, datafile)
#end build_src

#On windows one can still use a precompiled dll, if that works...
# dllfile ="src/Csources.dll"
# if os.path.isfile(dllfile):
#     print "dll found"

args = sys.argv
if len(args) > 1:
    if 'clean' in args:
        from glob import glob
        lst = list()
        lst.append(glob('libCsources*'))
        lst.append(glob('src/*.o'))
        lst.append(glob('src/*.obj'))
        lst.append(glob('ImageP/*.pyc'))

        print("cleaning up:")
        for l in lst:
            for i in l:
                print("delete:", i)
                os.remove(i)
    else:
        libname, datafile = build_src(libname, datafile, pathcore, ccname, CSourceList)

#end if args

print("dynamic lib: ",datafile)
#now back to the normal setup:

setup(name=PkgName,
        version='0.38',
        description= 'Package for particle tracking etc. in images',
            author= 'Tomio',
        author_email= 'haraszti@dwi.rwth-aachen.de',
        url='https://launchpad.net/imagep',
        ext_modules = None,
    #in this case we get all as a subpackage of ImageP:
    #the sources sit in the ImageP subfolder, C sources in src
        package_dir= {PkgName:'ImageP'},
        packages= [PkgName],
        # this parameter should allow that we have content other than python files
        include_package_data= True,
        data_files= datafile,
      # data_files= None,
        zip_safe= False
        )
