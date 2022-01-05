__all__ = ["summary_hdi", "summary_pct"]

#%%
import numpy as np
import pandas as pd
from arviz import hdi

def summary_hdi(varr, prob=0.68, peak_prob=0.2):
    low, upp = hdi(varr, hdi_prob=prob)
    _low, _upp = hdi(varr, hdi_prob=peak_prob)
    val = 0.5 * (_low + _upp)
    return val, upp-val, val-low

def summary_pct(varr, pcts=[16, 50, 84]):
    low, val, upp = np.percentile(varr, pcts)
    return val, upp-val, val-low

def summarize_results():
    return None
