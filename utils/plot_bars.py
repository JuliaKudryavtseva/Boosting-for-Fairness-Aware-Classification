import matplotlib.pyplot as plt
import numpy as np


def plot_bars(df, title=None, skip_std=None):

    df_mean = df.mean(skipna=True).reset_index().pivot('Classifier', 'Score').droplevel(0, axis=1)
    df_mean.columns = list(df_mean.columns)
    df_mean.index = list(df_mean.index)
    
    df_err = 2 * df.std(skipna=True).reset_index().pivot('Classifier', 'Score').droplevel(0, axis=1)
    df_err.columns = list(df_err.columns)
    df_err.index = list(df_err.index)
    
    bar_plot = df_mean.T.plot.bar(figsize=(10,8), rot=0, ecolor='black', ylim=[0,1], yticks=np.linspace(0,1,21), title=title)
    plt.grid()
    plt.show()
    
    return df_mean, df_err