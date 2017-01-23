#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd

t = pd.read_csv(sys.argv[1])
t = t[t.C == 0]
del t['test'], t['C']

t2 = pd.read_csv(sys.argv[2])
t2 = t2[t2.C == 0]
del t2['test'], t2['C']

j = t.merge(t2, on=("data", "n_features", "n"))

j['dev'] = j['dev_x'] - j['dev_y']
print j.groupby("data").aggregate(np.mean)

