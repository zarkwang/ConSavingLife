import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_paths(path_data,shock_time,income_levels,plot_name):

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    for j in range(len(path_data)): 
        subset = path_data[j]

        behavior = subset['original'][plot_name]
        life_length = len(behavior)
        ages = np.arange(life_length)
        ax[j].plot(ages,behavior,label='original')

        for t in shock_time:
            behavior = subset[f'shock_at_{t}'][plot_name]
            life_start = life_length - len(behavior)
            ages = np.arange(life_start,life_length)
            ax[j].plot(ages,behavior,label=f'get money t={t}')

        ax[j].set_title(f'income = {income_levels[j]}')

    ax[0].set_ylabel(plot_name)

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.show()


def instant_consumption(path_data,shock_time,income_levels,forward=True):

    df_instant_consume = pd.DataFrame({'shock_age':shock_time})

    for j in range(len(path_data)):

        instant_consumption = []

        for t in shock_time:
            original_consume = path_data[j]['original']['consumption'][t]
            if forward == True:
                new_consume = path_data[j][f'shock_at_{t}']['consumption'][0]
            else:
                 new_consume = path_data[j][f'shock_at_{t}']['consumption'][t]
            instant_consumption += [new_consume - original_consume]

        df_instant_consume[f'income={income_levels[j]}'] = instant_consumption


    print('Instant increase in consumption when income shock happens')
    print(df_instant_consume)
