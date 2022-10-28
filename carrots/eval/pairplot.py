from loaders import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sns.set(style='ticks', color_codes=True)

print('Loading data')
data, _ = load_conditional_train()
carrots = pd.DataFrame(data, columns=['T8_Acc_X', 'T8_Acc_Y', 'T8_Acc_Z',
                                      'T8_AngVel_X', 'T8_AngVel_Y', 'T8_AngVel_Z',
                                      'RA_Acc_X', 'RA_Acc_Y', 'RA_Acc_Z',
                                      'RA_AngVel_X', 'RA_AngVel_Y', 'RA_AngVel_Z',
                                      'LA_Acc_X', 'LA_Acc_Y', 'LA_Acc_Z',
                                      'LA_AngVel_X', 'LA_AngVel_Y', 'LA_AngVel_Z',
                                      'RL_Acc_X', 'RL_Acc_Y', 'RL_Acc_Z',
                                      'RL_AngVel_X', 'RL_AngVel_Y', 'RL_AngVel_Z',
                                      'LL_Acc_X', 'LL_Acc_Y', 'LL_Acc_Z',
                                      'LL_AngVel_X', 'LL_AngVel_Y', 'LL_AngVel_Z'])
sample = carrots.sample(n=1600)

print('Generating Pairplot. This might take a few minutes')
graph = sns.pairplot(sample)

print('Saving figure')
plt.savefig('pairplot.png')

plt.show()