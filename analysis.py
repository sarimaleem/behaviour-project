import sys, os, glob
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(style='ticks', font='Arial', font_scale=1, rc={
    'axes.linewidth': 0.25, 
    'axes.labelsize': 7, 
    'axes.titlesize': 7, 
    'xtick.labelsize': 6, 
    'ytick.labelsize': 6, 
    'legend.fontsize': 6, 
    'xtick.major.width': 0.25, 
    'ytick.major.width': 0.25,
    'text.color': 'Black',
    'axes.labelcolor':'Black',
    'xtick.color':'Black',
    'ytick.color':'Black',} )
sns.plotting_context()

# directories:
#project_dir = '/Users/jwdegee/Box Sync/undergrad_instructions/'
#fig_dir = os.path.join(project_dir, 'figs')

# load:
df = pd.read_csv('data.csv')

# subselect rows:
df = df.loc[df['trial']<421,:].reset_index(drop=True)

# subselect columns:
columns = ['subj_idx', 'session', 'trial', 'block_type', 'stimulus', 'difficulty_b2', 'rt', 'correct', 'choice', 'hit', 'fa', 'miss', 'cr', 'cr2', 'miss2']
df = df.loc[:,columns]

# rename some columns:
df = df.rename(columns={'block_type': 'reward',
                        'difficulty_b2': 'difficulty',})

# make integers:
columns = ['session', 'trial', 'reward', 'stimulus', 'difficulty', 'correct', 'choice', 'hit', 'fa', 'miss', 'cr']
df.loc[:,columns] = df.loc[:,columns].astype(int)

# set RT to NaN on all but hit trials:
df.loc[df['hit']!=1, 'rt'] = np.NaN

# print some info on subject and sessions:
subjects = [subj for subj, d in df.groupby('subj_idx')]
nr_sessions = [len(np.unique(d['session'])) for subj, d in df.groupby('subj_idx')]
print('subjects: {}'.format(len(subjects)))
print('sessions: {}'.format(sum(nr_sessions)))
print('trials: {}'.format(df.shape[0]))

# check out dataframe:
print(df.head())
print(df.tail())

# the old way of computing stuff:
correct = []
for s in np.unique(df['subj_idx']):
    temp = df.loc[df['subj_idx']==s, 'correct']
    correct.append(temp.mean())

# the new and better way:
res = df.groupby(['subj_idx']).mean()['correct'].reset_index()

# this becomes especially helpful when we want to do this separately for reward block and difficulty:
res = df.groupby(['subj_idx', 'difficulty']).mean()['correct'].reset_index()

# let's plot the result:
fig = plt.figure()
ax = fig.add_subplot(111)
sns.pointplot(x='difficulty', y='correct', units='subj_idx', data=res, ax=ax)
#fig.savefig(os.path.join(fig_dir, 'performance.pdf'))

#FIXME plot the same for hit-rates, and compare to the overall fa-rate
# hit-rate: nr_hit / (nr_hit + nr_miss2)
# fa_rate: nr_fa / (nr_fa + nr_cr2)

#FIXME plot hit-rates both as a function of reward and difficulty





