import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
import tkinter as tk
from tkinter import filedialog

font = {'size': 40}

matplotlib.rc('font', **font)
pd.set_option('display.max_rows', 500, 'display.max_columns', 20, 'display.width', 1000)
plt.rcParams.update({"axes.grid": True})

########################################################
# evaluation paramaters

SAVE = 1
PLOT = 1
EVAL = 1
EXPORT = 0

curr = 0

t_pretrigger_controller = 0
n_cycles = 5

step_from = 0
step_to = 15
step_size = 1.5
step_travel_time = 0.1

spec = 0.1
spec_sync_slow = 0.1
spec_sync_fast = 0.1

velocity_trigger = 7

id = 'ET_'
weight = '45N'
cycles = '4.5 Mio.'
SETNR = 1
date = '07.02.2023'

author = 'LOJI'

if SETNR == 1:
    lbl1 = "0001"
    lbl2 = "0002"
    lbl3 = "0003"
    temperature = 'RT'
    setstr = 'SET-0001-0002-0003'
    title_accuracy = 'Position accuracy - PP A-Samples - Endtest / ' + temperature + ' / ' \
                     + weight + ' load' + ' / after ' + cycles + ' cycles'

    title_sync = 'Position synchronicity - PP A-Samples - Endtest -' + setstr + ' / ' + temperature + ' / ' \
                 + weight + ' load' + ' / after ' + cycles + ' cycles'
    condition = 'standard'
elif SETNR == 2:
    lbl1 = "ADK2-0002"
    lbl2 = "ADL2-0002"
    lbl3 = "ADM2-0002"
    temperature = '70°C'
    setstr = 'AD*2/000' + str(SETNR)
    title_accuracy = 'Position accuracy - TypeFHR - 1st LTT - SET#000' + str(SETNR) + ' / ' + temperature + ' / ' \
                     + weight + ' load' + ' / after ' + cycles + ' cycles'

    title_sync = 'Position synchronicity - TypeFHR - 1st LTT - SET#000' + str(SETNR) + ' / ' + temperature + ' / ' \
                 + weight + ' load' + ' / after ' + cycles + ' cycles'
    condition = 'standard'
elif SETNR == 3:
    lbl1 = "ADK3-0007"
    lbl2 = "ADL3-0007"
    lbl3 = "ADM3-0007"
    temperature = '70°C'
    setstr = 'AD*3/0007'
    title_accuracy = 'Position accuracy - TypeFHR - 1st LTT - SET#000' + str(SETNR) + ' / ' + temperature + ' / ' \
                     + weight + ' load' + ' / after ' + cycles + ' cycles'

    title_sync = 'Position synchronicity - TypeFHR - 1st LTT - SET#000' + str(SETNR) + ' / ' + temperature + ' / ' \
                 + weight + ' load' + ' / after ' + cycles + ' cycles'
    condition = 'SS guidance'


#########################################################
#Dataframe - accururacy/cycle

def main():
    root = tk.Tk()
    root.withdraw()
    files_path = filedialog.askopenfilenames(title='Select input files')
    #file_path = root.tk.splitlist(file_path)

    if SAVE == 1 or EXPORT == 1:
        root = tk.Tk()
        root.withdraw()
        FOLDEROUT = filedialog.askdirectory(title='Select output folder')
        if SAVE == 1:
            pdf = matplotlib.backends.backend_pdf.PdfPages(FOLDEROUT + '/' + id + 'PP A-Samples - Endtest -' + setstr + '_'
                                                           + cycles +'-cycles_'+temperature+'_'+weight+'.pdf')

    counter = 0

    for files_path in files_path:
        file_name = files_path.split('/')
        file = file_name[-1]
        print(file)

        csv_import = pd.read_csv(files_path, parse_dates=True, infer_datetime_format=True, skiprows=11, header=None, index_col=0,
                                 names=['position axis1 [mm]', 'position axis2 [mm]', 'position axis3 [mm]', 'time [s]'])

        df_import = pd.DataFrame(csv_import)

        for column in df_import:
            df_import[column] = pd.to_numeric(df_import[column], errors='coerce')

        if counter == 0:
            DF = df_import
        else:
            DF = DF.merge(df_import, how='outer')
        counter = counter + 1

    del df_import
    del csv_import

    DF["time [s]"] = pd.to_datetime(DF.index)

    DF["time [s]"] = (DF["time [s]"] - DF["time [s]"][0]).dt.total_seconds()
    ZD = DF["time [s]"].diff()
    COUNT = 0
    for i in range(len(ZD)):
        if ZD[i] > 0.001:
            COUNT += 1
    print(COUNT)

    #select only valid measurements
    frequency = round(1/float(DF["time [s]"][1] - DF["time [s]"][0]))

    # smooth Position signal from Laser-sensors

    DF.reset_index(drop=True, inplace=True)

    #differentiate position signal --> velocity
    DF['velocity axis1 [mm/s]'] = DF['position axis1 [mm]'].diff()/DF['time [s]'].diff()
    DF['velocity axis2 [mm/s]'] = DF['position axis2 [mm]'].diff()/DF['time [s]'].diff()
    DF['velocity axis3 [mm/s]'] = DF['position axis3 [mm]'].diff()/DF['time [s]'].diff()

    # velocity filtering to detect triggers better
    smooth_pts = round(frequency / 10)
    DF['velocity axis1 [mm/s]'] = DF['velocity axis1 [mm/s]'].rolling(window=smooth_pts).mean()
    DF['velocity axis2 [mm/s]'] = DF['velocity axis2 [mm/s]'].rolling(window=smooth_pts).mean()
    DF['velocity axis3 [mm/s]'] = DF['velocity axis3 [mm/s]'].rolling(window=smooth_pts).mean()
    DF = DF.iloc[smooth_pts - 1:, :]

    #find velocity >= 5 mm/s or >=-5 mm/s
    DF['velocity_trigger'] = False
    DF.loc[DF['velocity axis1 [mm/s]'] > velocity_trigger, 'velocity_trigger'] = True
    DF.loc[DF['velocity axis1 [mm/s]'] < -velocity_trigger, 'velocity_trigger'] = True

    DF['step_trigger'] = False
    DF.loc[DF['velocity_trigger'].diff() == 1, 'step_trigger'] = True

    noise_triggers = DF.loc[DF['step_trigger'] == True,'time [s]'].diff()
    noise_triggers_idx = noise_triggers.loc[noise_triggers < 15/frequency].index

    DF.loc[noise_triggers_idx,'step_trigger'] = False

    ranges_idx = list(DF.loc[DF['step_trigger'] == True].index)

    DF['step_ranges'] = False

    step_trigger_idx = list(DF.loc[DF['step_trigger'] == True].index)

    for i in range(len(DF.loc[DF['step_trigger'] == True])):
        if i % 2 == 0:
            step_trigger_idx[i] = step_trigger_idx[i] - round(step_travel_time*frequency)
        else:
            step_trigger_idx[i] = step_trigger_idx[i] + round(step_travel_time*frequency)

    DF.loc[step_trigger_idx, 'step_ranges'] = True

    DF['sync_slow'] = False
    DF['sync_fast'] = False
    DF['accuracy'] = False

    step_trigger_cnt = int(((step_to - step_from) / step_size * 4) + 4)

    if len(DF.loc[DF['step_ranges'] == True]) % n_cycles == 0:
        print(str(len(DF.loc[DF['step_ranges'] == True])) + ' / ' + str(step_trigger_cnt*n_cycles) + ' trigger detected')
    else:
        print('Invalid number of triggers detected! --> ' + str(len(DF.loc[DF['step_ranges'] == True])))


    # offset correction 1/Frequency before 1st trigger
    offset_idx = DF.loc[DF['step_ranges'] == True].index[0]
    offset_axis1 = DF.loc[offset_idx - frequency:offset_idx, 'position axis1 [mm]'].mean(axis=0)
    DF['position axis1 [mm]'] = -(DF['position axis1 [mm]'] - offset_axis1)
    offset_axis2 = DF.loc[offset_idx - frequency:offset_idx, 'position axis2 [mm]'].mean(axis=0)
    DF['position axis2 [mm]'] = -(DF['position axis2 [mm]'] - offset_axis2)
    offset_axis3 = DF.loc[offset_idx - frequency:offset_idx, 'position axis3 [mm]'].mean(axis=0)
    DF['position axis3 [mm]'] = -(DF['position axis3 [mm]'] - offset_axis3)

    DF['sync_diff'] = DF[['position axis1 [mm]', 'position axis2 [mm]', 'position axis3 [mm]']].max(axis=1) - DF[
        ['position axis1 [mm]', 'position axis2 [mm]', 'position axis3 [mm]']].min(axis=1)

    smooth_pts = round(frequency / 50)
    DF['sync_diff'] = DF['sync_diff'].rolling(window=smooth_pts).mean()
    for cycle_id in range(n_cycles, 0, -1):
        idx = DF.loc[DF['step_ranges'] == True][(cycle_id-1) * (step_trigger_cnt):
                                                (cycle_id) * (step_trigger_cnt)].index
        idx_start = idx[0]
        idx_end = idx[-1]
        DF.loc[idx_start:idx_end, 'cycle no.'] = int(cycle_id)
        DF.loc[idx_start:idx_end, 't_cycle [s]'] = DF.loc[idx_start:idx_end, 'time [s]'] - DF.loc[idx_start, 'time [s]']

        idx = DF.loc[DF['step_ranges'] == True][(cycle_id - 1) * (step_trigger_cnt):
                                                (cycle_id) * (step_trigger_cnt) - 4].index
        idx_start = idx[0]
        idx_end = idx[-1]
        DF.loc[idx_start:idx_end, 'accuracy'] = True

        for step_id in range(int(len(idx)), -2, -2):
            if step_id == int(len(idx)):
                DF.loc[idx_end:idx_end + frequency, 'stepsAxis1'] = DF.loc[idx_end:idx_end + frequency,
                                                                   'position axis1 [mm]'].mean(axis=0)
                DF.loc[idx_end:idx_end + frequency, 'stepsAxis2'] = DF.loc[idx_end:idx_end + frequency,
                                                                   'position axis2 [mm]'].mean(axis=0)
                DF.loc[idx_end:idx_end + frequency, 'stepsAxis3'] = DF.loc[idx_end:idx_end + frequency,
                                                                   'position axis3 [mm]'].mean(axis=0)
                DF.loc[idx_end:idx_end + frequency, 'target position [mm]'] = 0     #(step_to - step_from) / step_size
                DF.loc[idx_end:idx_end + frequency, 'movement'] = 'down'
            elif step_id == 0:
                DF.loc[idx_start - frequency:idx_start, 'stepsAxis1'] = DF.loc[idx_start - frequency:idx_start,
                                                                        'position axis1 [mm]'].mean(axis=0)
                DF.loc[idx_start - frequency:idx_start, 'stepsAxis2'] = DF.loc[idx_start - frequency:idx_start,
                                                                        'position axis2 [mm]'].mean(axis=0)
                DF.loc[idx_start - frequency:idx_start, 'stepsAxis3'] = DF.loc[idx_start - frequency:idx_start,
                                                                        'position axis3 [mm]'].mean(axis=0)
                DF.loc[idx_start - frequency:idx_start, 'target position [mm]'] = 0
                DF.loc[idx_start - frequency:idx_start, 'movement'] = 'up'
            else:
               DF.loc[idx[step_id-1]:idx[step_id], 'stepsAxis1'] = DF.loc[idx[step_id-1]:idx[step_id], 'position axis1 [mm]'].mean(axis=0)
               DF.loc[idx[step_id-1]:idx[step_id], 'stepsAxis2'] = DF.loc[idx[step_id-1]:idx[step_id], 'position axis2 [mm]'].mean(axis=0)
               DF.loc[idx[step_id-1]:idx[step_id], 'stepsAxis3'] = DF.loc[idx[step_id-1]:idx[step_id], 'position axis3 [mm]'].mean(axis=0)
               DF.loc[idx[step_id-1]:idx[step_id], 'target position [mm]'] = step_id * step_size / 2

               if step_id > (step_to - step_from) / step_size * 2:
                  DF.loc[idx[step_id-1]:idx[step_id], 'movement'] = 'down'
               else:
                  DF.loc[idx[step_id-1]:idx[step_id], 'movement'] = 'up'

        DF['movement'] = DF['movement'].astype('category')

        DF['target position [mm]'] = -abs(DF['target position [mm]'] - step_to) + step_to

        idx = DF.loc[DF['step_ranges'] == True][((cycle_id) * (step_trigger_cnt)-5):
                                                ((cycle_id) * (step_trigger_cnt))-3].index
        idx_start = idx[0]
        idx_end = idx[-1]
        DF.loc[idx_start:idx_end, 'sync_slow'] = True

        idx = DF.loc[DF['step_ranges'] == True][((cycle_id) * (step_trigger_cnt)-4):
                                                (cycle_id) * (step_trigger_cnt)].index
        idx_start = idx[0]
        idx_end = idx[-1]
        DF.loc[idx_start:idx_end, 'sync_fast'] = True

        DF['deviation axis1 [mm]'] = DF['stepsAxis1'] - DF['target position [mm]']
        DF['deviation axis2 [mm]'] = DF['stepsAxis2'] - DF['target position [mm]']
        DF['deviation axis3 [mm]'] = DF['stepsAxis3'] - DF['target position [mm]']

########################################
    # PLOTS
    sns.set(rc={"figure.figsize": (20, 9), "lines.markersize": 5})
    sns.set_context("talk")

    minor_ticks = np.arange(step_from, step_to+1, 1)

    if EVAL:
        fig, axes = plt.subplots(1, 1)

        sns.lineplot(data=DF, x='time [s]', y='position axis1 [mm]', color='lightblue')
        sns.lineplot(data=DF, x='time [s]', y='stepsAxis1', color='blue')

        sns.lineplot(data=DF, x='time [s]', y='position axis2 [mm]', color='bisque')
        sns.lineplot(data=DF, x='time [s]', y='stepsAxis2', color='orange')

        sns.lineplot(data=DF, x='time [s]', y='position axis3 [mm]', color='lightgreen')
        sns.lineplot(data=DF, x='time [s]', y='stepsAxis3', color='green')
        sns.lineplot(data=DF, x='time [s]', y='target position [mm]', color='red')

        fig, ax = plt.subplots(2, 1, sharex=True)
        sns.lineplot(data=DF, x='time [s]', y='position axis1 [mm]', color='lightblue', ax=ax[0])
        sns.lineplot(data=DF, x='time [s]', y='stepsAxis1', color='royalblue', ax=ax[0])
        sns.lineplot(data=DF, x='time [s]', y='velocity axis1 [mm/s]', color='royalblue', ax=ax[1])
        sns.lineplot(data=DF, x='time [s]', y='step_ranges', color='red', ax=ax[1])


    if PLOT:

# lineplot - motion profile - accuracy
        hue = None
        style = None

        fig, ax = plt.subplots(1, 1)
        fig.subplots_adjust(right=0.8)
        ax.set_ylabel('position [mm]')
        fig.suptitle('Position accuracy measurement -  motion profile - PP A-Samples - Endtest -' + setstr)
        sns.lineplot(data=DF.loc[(DF['accuracy'] == True) & (DF['cycle no.'] == 1)], x='time [s]',
                     y='position axis1 [mm]', hue=hue, style=style, label=lbl1, palette='Blues')
        sns.lineplot(data=DF.loc[(DF['accuracy'] == True) & (DF['cycle no.'] == 1)], x='time [s]',
                     y='position axis2 [mm]', hue=hue, style=style, label=lbl2, palette='Oranges')
        sns.lineplot(data=DF.loc[(DF['accuracy'] == True) & (DF['cycle no.'] == 1)], x='time [s]',
                     y='position axis3 [mm]', hue=hue, style=style, label=lbl3, palette='Greens')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.9))
        im = plt.imread(get_sample_data('C:/Repos/env/pictures/layout.png'))
        newax = fig.add_axes([0.68, 0.1, 0.25, 0.25], anchor='NE', zorder=-1)
        newax.imshow(im)
        newax.axis('off')
        m_info = '\n'.join((r'Measurement-Info: ',
                                     r'',
                                     r'serial no.: %s' % (setstr, ),
                                     r'author: %s' % (author, ),
                                     r'condition: %s' % (condition, ),
                                     r'temperature: %s' % (temperature, ),
                                     r'weight: %s' % (weight, ),
                                     r'cycle cnt.: %s' % (cycles, ),
                                     r'date: %s' % (date, )))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        ax.text(1.01, 0.72, m_info, transform=ax.transAxes, fontsize=18,
                verticalalignment='top', bbox=props)

# lineplot - multiple axis up/down together - average + standard deviation
        legend = 'brief'
        ci = 'sd'

        fig, ax = plt.subplots(1, 1, sharex=True)
        #fig.subplots_adjust(top=0.9, right=0.9)
        fig.suptitle(title_accuracy)
        if weight != "25N":
            ax.axhline(spec, ls='--', color='red')
            ax.axhline(-spec, ls='--', color='red')
        ax.set_ylabel('deviation [mm]')
        ax.set_ylim(-2 * spec, 2 * spec)
        ax.set_xticks(minor_ticks)
        sns.lineplot(data=DF, x='target position [mm]', y='deviation axis1 [mm]', ci=ci, legend=legend,
                     palette='Blues', label=lbl1)
        sns.lineplot(data=DF, x='target position [mm]', y='deviation axis2 [mm]', ci=ci, legend=legend,
                     palette='Oranges', label=lbl2)
        sns.lineplot(data=DF, x='target position [mm]', y='deviation axis3 [mm]', ci=ci, legend=legend,
                     palette='Greens', label=lbl3)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# lineplot - multiple axis up/down separated - average + standard deviation
        style = 'movement'
        ci = 'sd'
        legend = 'brief'

        fig, ax = plt.subplots(1, 1, sharex=True)
        #fig.subplots_adjust(top=0.9, right=0.9)
        fig.suptitle(title_accuracy)
        if weight != "25N":
            ax.axhline(spec, ls='--', color='red')
            ax.axhline(-spec, ls='--', color='red')
        ax.set_ylabel('deviation [mm]')
        ax.set_ylim(-2 * spec, 2 * spec)
        ax.set_xticks(minor_ticks)
        sns.lineplot(data=DF, x='target position [mm]', y='deviation axis1 [mm]', ci=ci, legend=legend, style=style,
                     palette='Blues', label=lbl1)
        sns.lineplot(data=DF, x='target position [mm]', y='deviation axis2 [mm]', ci=ci, legend=legend, style=style,
                     palette='Oranges', label=lbl2)
        sns.lineplot(data=DF, x='target position [mm]', y='deviation axis3 [mm]', ci=ci, legend=legend, style=style,
                     palette='Greens', label=lbl3)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# lineplot - multiple axis plots for each cycle
        hue = 'cycle no.'
        style = 'movement'
        legend = 'full'

        fig, ax = plt.subplots(1, 1)
        #fig.subplots_adjust(top=0.9, right=0.9)
        fig.suptitle(title_accuracy)
        if weight != "25N":
            ax.axhline(spec, ls='--', color='red')
            ax.axhline(-spec, ls='--', color='red')
        ax.set_ylim(-2 * spec, 2 * spec)
        ax.set_ylabel('deviation [mm]')
        ax.set_xticks(minor_ticks)
        sns.lineplot(data=DF, x='target position [mm]', y='deviation axis1 [mm]', hue=hue,
                     style=style, legend=legend, palette='Blues')
        sns.lineplot(data=DF, x='target position [mm]', y='deviation axis2 [mm]', hue=hue,
                     style=style, legend=legend, palette='Oranges')
        sns.lineplot(data=DF, x='target position [mm]', y='deviation axis3 [mm]', hue=hue,
                     style=style, legend=legend, palette='Greens')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# barplot - subplots for single axis up/down together + standard deviation
        hue = None
        ci = 'sd'

        fig, ax = plt.subplots(3, 1, sharex=True)
        #fig.subplots_adjust(top=0.9, right=0.9)

        if weight != "25N":
            ax[0].axhline(spec, ls='--', color='red')
            ax[0].axhline(-spec, ls='--', color='red')

            ax[1].axhline(spec, ls='--', color='red')
            ax[1].axhline(-spec, ls='--', color='red')

            ax[2].axhline(spec, ls='--', color='red')
            ax[2].axhline(-spec, ls='--', color='red')

        ax[0].set_xlabel(None)
        ax[1].set_xlabel(None)

        fig.suptitle(title_accuracy)
        bp = sns.barplot(data=DF, x='target position [mm]', y='deviation axis1 [mm]', hue=hue, ci=ci,
                         palette='Blues', label=lbl1, ax=ax[0])
        bp.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=hue)
        bp = sns.barplot(data=DF, x='target position [mm]', y='deviation axis2 [mm]', hue=hue, ci=ci,
                         palette='Oranges', label=lbl2, ax=ax[1])
        bp.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=hue)
        bp = sns.barplot(data=DF, x='target position [mm]', y='deviation axis3 [mm]', hue=hue, ci=ci,
                         palette='Greens', label=lbl3, ax=ax[2])
        bp.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=hue)

        plt.setp(ax[0].get_xticklabels()[1::2], visible=False)
        plt.setp(ax[1].get_xticklabels()[1::2], visible=False)
        plt.setp(ax[2].get_xticklabels()[1::2], visible=False)

        ax[0].set_ylim(-spec * 1.5, spec * 1.5)
        ax[1].set_ylim(-spec * 1.5, spec * 1.5)
        ax[2].set_ylim(-spec * 1.5, spec * 1.5)

        ax[0].grid(which='major', linestyle='-')
        ax[1].grid(which='major', linestyle='-')
        ax[2].grid(which='major', linestyle='-')

# barplot - single axis up/down separated - average only
        hue = 'movement'
        ci = None

        fig, ax = plt.subplots(3, 1, sharex=True)
        #fig.subplots_adjust(top=0.9, right=0.9)
        if weight != "25N":
            ax[0].axhline(spec, ls='--', color='red')
            ax[0].axhline(-spec, ls='--', color='red')

            ax[1].axhline(spec, ls='--', color='red')
            ax[1].axhline(-spec, ls='--', color='red')

            ax[2].axhline(spec, ls='--', color='red')
            ax[2].axhline(-spec, ls='--', color='red')

        ax[0].set_xlabel(None)
        ax[1].set_xlabel(None)

        fig.suptitle(title_accuracy)
        bp = sns.barplot(x='target position [mm]', y='deviation axis1 [mm]', data=DF, hue=hue, ci=ci, palette='Blues',
                         ax=ax[0])
        bp.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=hue)
        bp = sns.barplot(x='target position [mm]', y='deviation axis2 [mm]', data=DF, hue=hue, ci=ci, palette='Oranges',
                         ax=ax[1])
        bp.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=hue)
        bp = sns.barplot(x='target position [mm]', y='deviation axis3 [mm]', data=DF, hue=hue, ci=ci, palette='Greens',
                         ax=ax[2])

        bp.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=hue)

        plt.setp(ax[0].get_xticklabels()[1::2], visible=False)
        plt.setp(ax[1].get_xticklabels()[1::2], visible=False)
        plt.setp(ax[2].get_xticklabels()[1::2], visible=False)

        ax[0].set_ylim(-spec * 1.5, spec * 1.5)
        ax[1].set_ylim(-spec * 1.5, spec * 1.5)
        ax[2].set_ylim(-spec * 1.5, spec * 1.5)

        ax[0].grid(which='major', linestyle='-')
        ax[1].grid(which='major', linestyle='-')
        ax[2].grid(which='major', linestyle='-')

# barplot - single axis for each cycle
        hue = 'cycle no.'
        ci = None

        fig, ax = plt.subplots(3, 1, sharex=True)
        #fig.subplots_adjust(top=0.9, right=0.9)
        fig.suptitle(title_accuracy)
        if weight != "25N":
            ax[0].axhline(spec, ls='--', color='red')
            ax[0].axhline(-spec, ls='--', color='red')

            ax[1].axhline(spec, ls='--', color='red')
            ax[1].axhline(-spec, ls='--', color='red')

            ax[2].axhline(spec, ls='--', color='red')
            ax[2].axhline(-spec, ls='--', color='red')

        ax[0].set_xlabel(None)
        ax[1].set_xlabel(None)

        bp = sns.barplot(x='target position [mm]', y='deviation axis1 [mm]', data=DF, hue=hue, ci=ci, palette='Blues',
                         ax=ax[0])
        bp.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=hue)
        bp = sns.barplot(x='target position [mm]', y='deviation axis2 [mm]', data=DF, hue=hue, ci=ci, palette='Oranges',
                         ax=ax[1])
        bp.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=hue)
        bp = sns.barplot(x='target position [mm]', y='deviation axis3 [mm]', data=DF, hue=hue, ci=ci, palette='Greens',
                         ax=ax[2])
        bp.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=hue)

        plt.setp(ax[0].get_xticklabels()[1::2], visible=False)
        plt.setp(ax[1].get_xticklabels()[1::2], visible=False)
        plt.setp(ax[2].get_xticklabels()[1::2], visible=False)

        ax[0].set_ylim(-spec * 1.5, spec * 1.5)
        ax[1].set_ylim(-spec * 1.5, spec * 1.5)
        ax[2].set_ylim(-spec * 1.5, spec * 1.5)

        ax[0].grid(which='major', linestyle='-')
        ax[1].grid(which='major', linestyle='-')
        ax[2].grid(which='major', linestyle='-')

################################### synchronicity

# lineplot - motion profile - synchronicity
        hue = None
        style = None

        fig, ax = plt.subplots(1, 1)
        fig.subplots_adjust(right=0.8)
        ax.set_ylabel('position [mm]')
        ax.set_title('slow movement (1 mm/s)   /   fast movement (30 mm/s)')
        fig.suptitle('Position synchronicity measurement -  motion profile - PP A-Samples - Endtest -' + setstr)
        sns.lineplot(data=DF.loc[(DF['accuracy'] == False) & (DF['cycle no.'] == 1)], x='time [s]',
                     y='position axis1 [mm]', hue=hue, style=style, label=lbl1, palette='Blues')
        sns.lineplot(data=DF.loc[(DF['accuracy'] == False) & (DF['cycle no.'] == 1)], x='time [s]',
                     y='position axis1 [mm]', hue=hue, style=style, label=lbl2, palette='Oranges')
        sns.lineplot(data=DF.loc[(DF['accuracy'] == False) & (DF['cycle no.'] == 1)], x='time [s]',
                     y='position axis2 [mm]', hue=hue, style=style, label=lbl3, palette='Greens')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.9))
        im = plt.imread(get_sample_data('C:/Repos/env/pictures/layout.png'))
        newax = fig.add_axes([0.68, 0.1, 0.25, 0.25], anchor='NE', zorder=-1)
        newax.imshow(im)
        newax.axis('off')
        m_info = '\n'.join((r'Measurement-Info: ',
                                     r'',
                                     r'serial no.: %s' % (setstr,),
                                     r'author: %s' % (author,),
                                     r'condition: %s' % (condition,),
                                     r'temperature: %s' % (temperature,),
                                     r'weight: %s' % (weight,),
                                     r'cycle cnt.: %s' % (cycles,),
                                     r'date: %s' % (date,)))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        ax.text(1.01, 0.72, m_info, transform=ax.transAxes, fontsize=18,
                verticalalignment='top', bbox=props)


        # lineplot - slow movement - synchronicity
        hue = None
        style = None

        for i in range(0, n_cycles, 1):
            fig, ax = plt.subplots(2, 2)
            fig.suptitle(title_sync)
            ax[0, 0].set_title('slow movement (1 mm/s) / cycle no. ' + str(i+1))
            ax[0, 1].set_title('fast movement (30 mm/s) / cycle no. ' + str(i+1))
            sync_slow_max = round(DF.loc[(DF['sync_slow'] == True) & (DF['cycle no.'] == i + 1), 'sync_diff'].max(), 3)
            sync_fast_max = round(DF.loc[(DF['sync_fast'] == True) & (DF['cycle no.'] == i + 1), 'sync_diff'].max(), 3)
            sns.lineplot(data=DF.loc[(DF['sync_slow'] == True) & (DF['cycle no.'] == i + 1)], x='time [s]',
                         y='position axis1 [mm]', hue=hue, style=style, palette='Blues', ax=ax[0, 0])
            sns.lineplot(data=DF.loc[(DF['sync_slow'] == True) & (DF['cycle no.'] == i + 1)], x='time [s]',
                         y='position axis2 [mm]', hue=hue, style=style, palette='Oranges', ax=ax[0, 0])
            sns.lineplot(data=DF.loc[(DF['sync_slow'] == True) & (DF['cycle no.'] == i + 1)], x='time [s]',
                         y='position axis3 [mm]', hue=hue, style=style, palette='Greens', ax=ax[0, 0])
            sns.lineplot(data=DF.loc[(DF['sync_fast'] == True) & (DF['cycle no.'] == i + 1)], x='time [s]',
                         y='position axis1 [mm]', hue=hue, style=style, palette='Blues', ax=ax[0, 1])
            sns.lineplot(data=DF.loc[(DF['sync_fast'] == True) & (DF['cycle no.'] == i + 1)], x='time [s]',
                         y='position axis2 [mm]', hue=hue, style=style, palette='Oranges', ax=ax[0, 1])
            sns.lineplot(data=DF.loc[(DF['sync_fast'] == True) & (DF['cycle no.'] == i + 1)], x='time [s]',
                         y='position axis3 [mm]', hue=hue, style=style, palette='Greens', ax=ax[0, 1])
            sns.lineplot(data=DF.loc[(DF['sync_slow'] == True) & (DF['cycle no.'] == i + 1)], x='time [s]',
                         y='sync_diff', hue=hue, style=style, label='max. synchronicity diff.: ' + str(sync_slow_max),
                         color='purple', ax=ax[1, 0])
            sns.lineplot(data=DF.loc[(DF['sync_fast'] == True) & (DF['cycle no.'] == i + 1)], x='time [s]',
                         y='sync_diff', hue=hue, style=style, label='max. synchronicity diff.: ' + str(sync_fast_max),
                         color='purple', ax=ax[1, 1])
            ax[1, 0].set_ylim(-spec/10, 2 * spec)
            ax[1, 1].set_ylim(-spec/10, 2 * spec)
            if weight != "25N":
                ax[1, 0].axhline(spec_sync_slow, ls='--', color='red')
                ax[1, 1].axhline(spec_sync_fast, ls='--', color='red')
            ax[0, 0].set_ylabel('position [mm]')
            ax[0, 0].set_xlabel(None)
            ax[0, 1].set_ylabel(None)
            ax[0, 1].set_xlabel(None)
            ax[1, 0].set_ylabel('sync. diff. [mm]')
            ax[1, 1].set_ylabel(None)

################################### CPA-Log

    if curr == 1:
        root2 = tk.Tk()
        root2.withdraw()
        files_path2 = filedialog.askopenfilenames(title='Select input files')

        counter = 0

        for files_path2 in files_path2:
            file_name2 = files_path2.split('/')
            file2 = file_name2[-1]
            print(file2)

            df_controller = open(files_path2, 'r').read().split('\n')

            df_controller = df_controller[27:]
            for i in range(0, len(df_controller)):
                df_controller[i] = df_controller[i].split('\t')

            df_controller = pd.DataFrame(df_controller)
            df_controller.columns = ['time [s]', 'pos_encoder1', 'pos_encoder2', 'pos_encoder3', 'pos_target',
                                     'current_lifter1', 'current_lifter2', 'current_lifter3']

            df_controller['n_cycle'] = counter + 1  # I'd put counter alone, easier to filter later
            df_controller = df_controller[:-1]
            if counter == 0:
                DF_controller = df_controller

            else:
                DF_controller = DF_controller.merge(df_controller, how='outer')

            counter = counter + 1

        for column in DF_controller:
            DF_controller[column] = pd.to_numeric(DF_controller[column], errors='coerce')

        DF_controller['time [s]'] = DF_controller['time [s]'] - DF_controller['time [s]'][0]
        DF_controller = DF_controller.loc[DF_controller['time [s]'] > t_pretrigger_controller]
        DF_controller = DF_controller.reset_index(drop=True)
        DF_controller['time [s]'] = DF_controller['time [s]'] - DF_controller['time [s]'][0]

        fig, axes = plt.subplots(1, 1, sharey='row')
        axes.plot(DF_controller['time [s]'], DF_controller['current_lifter1'], linewidth=2,
                  markersize=1, label=lbl1 + '')
        axes.plot(DF_controller['time [s]'], DF_controller['current_lifter2'], linewidth=2,
                  markersize=1, label=lbl2 + '')
        axes.plot(DF_controller['time [s]'], DF_controller['current_lifter3'], linewidth=2,
                  markersize=1, label=lbl3 + '')
        axes.set_xlabel('time [s]')
        axes.set_ylabel('current [A]')

        fig.suptitle(file)
        plt.legend()

    if SAVE:
        for plots in plt.get_fignums():
            pdf.savefig(plt.figure(plots), bbox_inches='tight')
        pdf.close()

    if PLOT:
        plt.show()

    if EXPORT == 1:
        df_export = DF[['target position [mm]', 'cycle no.', 'movement']]
        df_export = df_export.drop_duplicates()
        df_export_idx = df_export.index
        df_export = DF.iloc[df_export_idx]
        df_export.to_csv(FOLDEROUT + '/' + id + 'PP A-Samples - Endtest -' + setstr +'_' + condition + '_' + cycles + '-cycles_'
                  + temperature + '_' + weight + '.csv', index=False)
        print(FOLDEROUT + '/' + id + 'PP A-Samples - Endtest -' + setstr + '_' + condition + '_' + cycles + '-cycles_'
                  + temperature + '_' + weight + '.csv export successful!')

if __name__ == "__main__":
    main()