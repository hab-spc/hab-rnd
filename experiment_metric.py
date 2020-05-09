import pandas as pd

from counts_analysis.c_utils import COUNTS_CSV
from validate_exp.stat_fns import *

x = '{} gtruth raw count'
y = '{} predicted raw count'

df = pd.read_csv(COUNTS_CSV['counts'])
cls = 'Lingulodinium Polyedra'
cls = 'Prorocentrum micans'
data = df.groupby('class').get_group(cls)
data = data[[x.format('pier'), y.format('pier'), x.format('lab'), y.format('lab')]]

camera = 'pier'
gtruth, pred = data[x.format(camera)], data[y.format(camera)]

print('Testing {}'.format(wape.__name__))
score = wape(gtruth, pred)
print(score)

print('Testing {}'.format(mase.__name__))
score = mase(gtruth, pred)
print(score)
