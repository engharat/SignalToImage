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
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from joblib import Parallel
from scipy.interpolate import UnivariateSpline

#2852
warnings.filterwarnings("error")
def enlarge(a,new_length):
    old_indices = np.arange(0,len(a))
    new_indices = np.linspace(0,len(a)-1,new_length)
    spl = UnivariateSpline(old_indices,a,k=3,s=0)
    return spl(new_indices)

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
parser.add_argument('--noise',type=float)
parser.add_argument('--txt',action="store_true")
parser.add_argument('--plot',action="store_true")
parser.add_argument('--enlarge',action="store_true")

args = parser.parse_args()
#kw = dict(wavelet=('morlet', {'mu': 4.5}), nv=32, scales='log')
kw = dict(wavelet=('morlet', {'mu': 45}), nv=32, scales='log')
pkw = dict(abs=1)
dir=args.dest
df = pd.read_csv(args.csvpath, skipinitialspace=True)
total=df.shape[0]

def main_func(parameter,label):
    df2 = pd.read_csv(os.path.join(args.source,parameter))
    for key in df2.keys():
        if key == 'Time':
            continue
        try:
            item = df2[key].to_numpy()
            if args.noise:
                noise = np.random.normal(0, item.std(),item.size) * args.noise
                item += noise
            if args.enlarge:
                item = enlarge(item, 2852)
            if any_nans(item):
                print(parameter)
                break
            fullpath = os.path.join(args.dest,str(label),parameter)
            p = pathlib.Path(fullpath)
            p.mkdir(parents=True, exist_ok=True)
            if args.txt:
                np.savetxt(os.path.join(fullpath,key)+'.txt',item)
#            imshow(_Tx, norm=(0, 4e-1), borders=False, ticks=False,show=False,save=os.path.join(fullpath,key)+'.png',**pkw)
                del item, fullpath,p
            elif args.plot:
                plt.axis('off')
#                plt.axis("tight") 
#                plt.axis("image")
                plt.plot(item)
                plt.savefig(os.path.join(fullpath,key)+'.png',bbox_inches='tight')
                plt.clf()
            else:
                Tx, *_ = ssq_cwt(item, **kw)
                _Tx = Tx #_Tx = np.pad(Tx, [[4, 4]])  # improve display of top- & bottom-most freqs
                imshow(_Tx, norm=(0, np.abs(_Tx).max()/2), borders=False, ticks=False,show=False,save=os.path.join(fullpath,key)+'.png',**pkw)
                del item, Tx, _Tx, fullpath,p
        except:
            print(parameter)
            break

##SCRIPT TO USE TQDM WITH PARALLEL

class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()





###MAIN FUNC CALLING
ProgressParallel(n_jobs=8,total=total)(delayed(main_func)(parameter, label) for parameter,label in zip(df['Run code'],df['Tag']))

