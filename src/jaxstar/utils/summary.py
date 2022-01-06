__all__ = ["summary_hdi", "summary_pct", "summary_stats", "summarize_results"]

#%%
import numpy as np
import pandas as pd
import os
from arviz import hdi

def summary_hdi(varr, prob=0.68, peak_prob=0.2):
    low, upp = hdi(varr, hdi_prob=prob)
    _low, _upp = hdi(varr, hdi_prob=peak_prob)
    val = 0.5 * (_low + _upp)
    return val, upp-val, val-low

def summary_pct(varr, pcts=[16, 50, 84]):
    low, val, upp = np.percentile(varr, pcts)
    return val, upp-val, val-low

def summary_stats(postdir, names, keys, stat='pct', **kwargs):
    if stat=='pct':
        summary = summary_pct
    else:
        summary = summary_hdi

    dout = pd.DataFrame({})
    for i,name in enumerate(names):
        filename = postdir + str(name) + '_samples.csv'
        if not os.path.exists(filename):
            print ("# output for %s does not exist."%name)
            continue
        dp = pd.read_csv(filename)
        _dic = {"name": name}
        for k in keys:
            _k = 'iso_' + k
            _dic[_k], _dic[_k+"_upp"], _dic[_k+"_low"] = summary(np.array(dp[k]), **kwargs)
        dout = dout.append([_dic])

    return dout.reset_index(drop=True)

def summarize_results(postdir, dinput, keys, obskeys, stat='pct', **kwargs):
    d = summary_stats(postdir, dinput.kepid, keys, stat=stat, **kwargs).rename({"name": "kepid"}, axis='columns')
    d = pd.merge(d, dinput[["kepid"]+obskeys+[_k+"_err" for _k in obskeys]+["binflag"]], on='kepid')
    for key in obskeys:
        d["d"+key] = d["iso_"+key] - d[key]
        d["dsigma"+key] = d["d"+key] / d[key+"_err"]
    d["dsigmaobs"] = np.sqrt(np.sum(np.array(d[["dsigma"+k for k in obskeys]])**2, axis=1))
    return d
