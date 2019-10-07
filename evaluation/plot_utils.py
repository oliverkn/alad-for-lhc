import matplotlib.pyplot as plt
import numpy as np


def plot_hist(hist_data_list, labels, settings, output_file=None):
    f, ax_arr = plt.subplots(23 // 3 + 1, 3, figsize=(18, 40))

    for i, name in enumerate(hist_data_list[0].keys()):
        ax = ax_arr[int(i / 3), i % 3]
        fsettings = settings[name]

        for hist_data, label in zip(hist_data_list, labels):
            if name not in hist_data:
                continue

            x = hist_data[name]['bin_edges']
            y = hist_data[name]['pdf']
            y = np.append(y, y[-1])
            ax.step(x, y, label=label, where='post')

        ax.set_yscale(fsettings['yscale'])
        ax.set_title(name)
        ax.legend()

    if output_file is not None:
        print('saving fig to ' + output_file)
        plt.savefig(output_file)

    plt.show()

