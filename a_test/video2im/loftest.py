# # -*- coding: utf8 -*-

from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

instances = [
 (0, 200),
 (0, 201),
 (0, 202),
 (0, 203),
 (0, 204),
 (0, 205),
 (0, 206),
 (0, 207),
 (0, 208),
 (0, 208),
 (0, 210),
 (0, 211),
 (0, 212),
 (0, 213),
 (0, 180),
 (0, 190),
 (0, 220),
 (0, 230),
 (0, 222),
 (0, 233),
 (0, 37),
 (0, 38),
 (0, 39),
 (0, 41),
 (0, 45),
 (0, 52),
 (0, 200),
 (0, 200),
 (0, 200),
 (0, 195)]

from LOF import outliers
lof = outliers(5, instances)

for outlier in lof:
    print (outlier["lof"],outlier["instance"])

from matplotlib import pyplot as p

x,y = zip(*instances)
p.scatter(x,y, 20, color="#0000FF")

for outlier in lof:
    value = outlier["lof"]
    instance = outlier["instance"]
    color = "#FF0000" if value > 1 else "#00FF00"
    p.scatter(instance[0], instance[1], color=color, s=(value-1)**2*10+20)

p.show()
plt.savefig('123.png')