#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 21:55:06 2017

@author: tomba
"""

import migraine_processing as mig
import pandas as pd
import numpy as np


df = mig.getCaseData(mig.CASE_NAMES[1])


best = 0

for i in range(1, df.value.size):
    res = df.value[i]-df.value[i-1]
    best = max(res, best)

print best
    