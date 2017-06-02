# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt


xvar = np.arange(0, 5000, 1)
#yvar = math.sin(xvar) + math.sin(xvar/4)
#xvar = np.linspace(-np.pi, np.pi, 201)

periods = xvar.size

x = xvar#[:500]
y = np.sin(x/50.0) * 2000 + 2000 # + np.sin(x/4.0))
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('regular sinusoid')
plt.axis('tight')
plt.show()

index = pd.date_range('2016-1-1 12:00', periods=periods, freq='min')
out = pd.DataFrame(y, index=index, columns=['value'])
out.index.name = 'time'

out['flag'] = False
out.flag[4000] = True
out.flag[3000] = True
                
#out.to_csv("stm_sinewave.csv")

print "Done."

