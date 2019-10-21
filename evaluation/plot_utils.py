import matplotlib.pyplot as plt
import numpy as np

def plot_stack_hist2(hist_data_list, labels, weights, settings, output_file=None, all_lin=False):
    f, ax_arr = plt.subplots(23 // 3 + 1, 3, figsize=(18, 40))

    for i, name in enumerate(hist_data_list[0].keys()):
        ax = ax_arr[int(i / 3), i % 3]
        fsettings = settings[name]

        bins_array = []
        y_array = []
        pdf_array = []

        for hist_data, label in zip(hist_data_list, labels):
            if name not in hist_data:
                continue

            bins = hist_data[name]['bin_edges']
            pdf = hist_data[name]['pdf']

            bins_array.append(bins)
            y_array.append(bins[:-1])
            pdf_array.append(pdf)

        ax.hist(y_array, bins_array, weights=pdf)



        if all_lin is False:
            ax.set_yscale(fsettings['yscale'])
        ax.set_title(name)
        ax.legend()

    if output_file is not None:
        print('saving fig to ' + output_file)
        plt.savefig(output_file)

    plt.show()

def plot_hist(hist_data_list, labels, settings, output_file=None, all_lin=False):
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

            norm = np.sum(y)

            ax.step(x, y, label='%s (a=%3f)' % (label, norm), where='post')

        if all_lin is False:
            ax.set_yscale(fsettings['yscale'])
        ax.set_title(name)
        ax.legend()

    if output_file is not None:
        print('saving fig to ' + output_file)
        plt.savefig(output_file)

    plt.show()


def plot_stacked_hist(hist_data_list, labels, weights, settings, output_file=None, all_lin=False):
    f, ax_arr = plt.subplots(23 // 3 + 1, 3, figsize=(18, 40))

    for i, name in enumerate(hist_data_list[0].keys()):
        ax = ax_arr[int(i / 3), i % 3]
        fsettings = settings[name]

        # for hist_data, label in zip(hist_data_list, labels):
        for i in range(len(hist_data_list)):
            if name not in hist_data_list[0]:
                continue

            label = labels[i]
            x = hist_data_list[i][name]['bin_edges']
            y = hist_data_list[0][name]['pdf'] * weights[0]

            for j in range(1, i + 1):
                y = y + hist_data_list[j][name]['pdf'] * weights[j]

            # y = y / np.sum(y)
            norm = np.sum(y)
            y = np.append(y, y[-1])

            ax.step(x, y, label='%s (a=%3f)' % (label, norm), where='post')

        if all_lin is False:
            ax.set_yscale(fsettings['yscale'])
        ax.set_title(name)
        ax.legend()

    if output_file is not None:
        print('saving fig to ' + output_file)
        plt.savefig(output_file)

    plt.show()
