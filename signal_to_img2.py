import numpy as np
import matplotlib
matplotlib.use('agg',force=True)
from matplotlib import pyplot as plt
from ssqueezepy import ssq_cwt, issq_cwt, cwt
from ssqueezepy.visuals import imshow
from tqdm import tqdm
import os
import pandas as pd
import argparse
import pathlib
import numba
import warnings
warnings.filterwarnings("error")

@numba.njit(nogil=True)
def _any_nans(a):
    for x in a:
        if ( (np.isnan(x)) or (x>1000)): return True
    return False

@numba.jit
def any_nans(a):
    if not a.dtype.kind=='f': return False
    return _any_nans(a.flat)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--source',type=str,required=True)
parser.add_argument('--dest',type=str)
parser.add_argument('--csvpath',type=str,required=True)
parser.add_argument('--acc',type=int,default=0)

args = parser.parse_args()
kw = dict(wavelet=('morlet', {'mu': 4.5}), nv=16, scales='log',norm=0.1)
pkw = dict(abs=1)
dir=args.dest
df = pd.read_csv(args.csvpath, skipinitialspace=True)
total=df.shape[0]
for parameter,label in tqdm(zip(df['Run code'],df['Damaged region']),total=total):
    df2 = pd.read_csv(os.path.join(args.source,parameter)+'.csv')
    for key in df2.keys():
        if key == 'Time':
            continue
        try:
            item = df2[key].to_numpy()
            if any_nans(item):
                print(parameter)
                break
            Tx, *_ = ssq_cwt(item, **kw)
            _Tx = np.pad(Tx, [[4, 4]])  # improve display of top- & bottom-most freqs
            fullpath = os.path.join(args.dest,str(label),parameter)
            p = pathlib.Path(fullpath)
            p.mkdir(parents=True, exist_ok=True)
            imshow(_Tx, norm=(0, 4e-1), borders=False, ticks=False,show=False,save=os.path.join(fullpath,key)+'.png',**pkw)
            del item, Tx, _Tx, fullpath,p
        except:
            print(parameter)
            break
    #%%# Synchrosqueeze ##########################################################
#kw = dict(wavelet=('morlet', {'mu': 4.5}), nv=32, scales='log')
#Tx, *_ = ssq_cwt(x, t=ts, **kw)
#Wx, *_ = cwt(x, t=ts, **kw)
#pkw = dict(abs=1)
#_Tx = np.pad(Tx, [[4, 4]])  # improve display of top- & bottom-most freqs
#imshow(_Tx, norm=(0, 4e-1), borders=False, ticks=False,show=False,save="proviamo.png",**pkw)
