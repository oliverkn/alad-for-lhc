import os
import numpy as np
import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm


def do_roc(scores, true_labels, file_name='', directory='', plot=True):
    """ Does the ROC curve

    Args:
            scores (list): list of scores from the decision function
            true_labels (list): list of labels associated to the scores
            file_name (str): name of the ROC curve
            directory (str): directory to save the jpg file
            plot (bool): plots the ROC curve or not
    Returns:
            roc_auc (float): area under the under the ROC curve
            thresholds (list): list of thresholds for the ROC
    """
    fpr, tpr, _ = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)  # compute area under the curve
    if plot:
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % (roc_auc))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")

        plt.savefig(directory + file_name + 'roc.png')
        plt.close()

    return roc_auc


def do_cumdist(scores, file_name='', directory='', plot=True):
    N = len(scores)
    X2 = np.sort(scores)
    F2 = np.array(range(N)) / float(N)
    if plot:
        plt.figure()
        plt.xlabel("Anomaly score")
        plt.ylabel("Percentage")
        plt.title("Cumulative distribution function of the anomaly score")
        plt.plot(X2, F2)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + file_name + 'cum_dist.png')


def get_percentile(scores, dataset):
    if dataset == 'kdd':
        # Highest 20% are anomalous
        per = np.percentile(scores, 80)
    elif dataset == "arrhythmia":
        # Highest 15% are anomalous
        per = np.percentile(scores, 85)
    else:
        c = 90
        per = np.percentile(scores, 100 - c)
    return per


def do_hist(scores, true_labels, directory, dataset, random_seed, display=False):
    plt.figure()
    idx_inliers = (true_labels == 0)
    idx_outliers = (true_labels == 1)
    hrange = (min(scores), max(scores))
    plt.hist(scores[idx_inliers], 50, facecolor=(0, 1, 0, 0.5),
             label="Normal samples", density=True, range=hrange)
    plt.hist(scores[idx_outliers], 50, facecolor=(1, 0, 0, 0.5),
             label="Anomalous samples", density=True, range=hrange)
    plt.title("Distribution of the anomaly score")
    plt.legend()
    if display:
        plt.show()
    else:
        plt.savefig(directory + 'histogram_{}_{}.png'.format(random_seed, dataset),
                    transparent=True, bbox_inches='tight')
        plt.close()


def predict(scores, threshold):
    return scores >= threshold


def make_meshgrid(x_min, x_max, y_min, y_max, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def save_grid_plot(samples, samples_rec, name_model, dataset, nb_images=50,
                   grid_width=10):
    args = name_model.split('/')[:-1]
    directory = os.path.join(*args)
    if not os.path.exists(directory):
        os.makedirs(directory)
    samples = (samples + 1) / 2
    samples_rec = (samples_rec + 1) / 2

    figsize = (32, 32)
    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(grid_width, grid_width)
    gs.update(wspace=0.05, hspace=0.05)
    list_samples = []
    for x, x_rec in zip(np.split(samples, nb_images // grid_width),
                        np.split(samples_rec, nb_images // grid_width)):
        list_samples += np.split(x, grid_width) + np.split(x_rec, grid_width)
    list_samples = [np.squeeze(sample) for sample in list_samples]
    for i, sample in enumerate(list_samples):
        if i >= nb_images * 2:
            break
        ax = plt.subplot(gs[i])
        if dataset == 'mnist':
            plt.imshow(sample, cmap=cm.gray)
        else:
            plt.imshow(sample)
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
    plt.savefig('{}.png'.format(name_model))
