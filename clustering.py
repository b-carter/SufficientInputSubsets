from matplotlib import pyplot as plt
from matplotlib import gridspec
import seaborn as sns
sns.set(style='whitegrid')
import numpy as np
import pandas as pd
import os
import random
from collections import Counter, defaultdict
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

from rationale_objects import Image, BeerReview, Rationale, SIS_RATIONALE_KEY


def mnist_load_images_from_dir(dirpath):
    images = []
    for filename in os.listdir(dirpath):
        if not filename.endswith('.json'):
            continue
        with open(os.path.join(dirpath, filename), 'r') as f:
            img = Image.from_json(f)
            images.append(img)
    return images


# R1 and R2 are list of 2-dim tuples (x, y) where (x, y) is the coordinate in the rationale
def energy_dist(R1, R2, i=None, j=None, memo=None):
    R1 = np.asarray(R1)
    R2 = np.asarray(R2)
    if i is not None and memo is not None and i in memo:
        R1_term = memo[i]
    else:
        R1_term = np.mean(cdist(R1, R1, metric='euclidean'))
        if i is not None and memo is not None:
            memo[i] = R1_term
    if j is not None and memo is not None and j in memo:
        R2_term = memo[j]
    else:
        R2_term = np.mean(cdist(R2, R2, metric='euclidean'))
        if j is not None and memo is not None:
            memo[j] = R2_term
    R1_R2_term = np.mean(cdist(R1, R2, metric='euclidean'))
    dist_sq = 2*R1_R2_term - R1_term - R2_term
    if np.isclose(0, dist_sq):
        dist_sq = 0
    dist = np.sqrt(dist_sq)
    return dist


def compute_energy_dists(all_rationales):
    energy_dist_matrix = np.zeros((len(all_rationales), len(all_rationales)))
    memo = {}
    for i in range(len(all_rationales)):
        r1 = all_rationales[i]
        for j in range(i+1):
            r2 = all_rationales[j]
            dist = energy_dist(r1, r2, i=i, j=j, memo=memo)
            energy_dist_matrix[i, j] = dist
            energy_dist_matrix[j, i] = dist
    return energy_dist_matrix


def save_matrix(matrix, fpath):
    # np.savetxt(fpath, matrix)
    np.savez_compressed(fpath, matrix=matrix)


def load_matrix(fpath):
    # return np.loadtxt(fpath)
    return np.load(fpath)['matrix']


# Returns list of pixel coordinates (x, y) in each SIS (used in energy distance clustering)
def mnist_get_all_sis(images, rationale_key=SIS_RATIONALE_KEY):
    all_sis = [i.get_rationales(rationale_key) for i in images]
    all_sis = [i.get_elms() for sublist in all_sis for i in sublist]
    all_sis = [[np.unravel_index(i, (28, 28)) for i in r] for r in all_sis]
    return all_sis


# Returns list of (28 x 28) arrays, each of which represents SIS image (masking non-SIS elements)
def mnist_get_all_sis_images(images, rationale_key=SIS_RATIONALE_KEY):
    all_sis_images = []
    for image in images:
        for rationale in image.get_rationales(rationale_key):
            x_rationale = image.get_x_rationale_only([rationale])
            all_sis_images.append(x_rationale.reshape(28, 28))
    return np.array(all_sis_images)


def mnist_get_all_sis_and_images(images, rationale_key=SIS_RATIONALE_KEY):
    return (
        mnist_get_all_sis(images, rationale_key=rationale_key),
        mnist_get_all_sis_images(images, rationale_key=rationale_key),
    )


# Compute DBSCAN
def cluster(dist_matrix, eps=0.5, min_samples=15, verbose=True):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(dist_matrix)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    if verbose:
        print('Estimated number of clusters: %d' % n_clusters_)
        print('# core samples: ', sum((1 for i in core_samples_mask if i)))

    cluster_to_core_idxs = defaultdict(list)
    cluster_to_noncore_idxs = defaultdict(list)
    for i in range(dist_matrix.shape[0]):
        cluster_label = labels[i]
        if core_samples_mask[i]:  # core sample
            cluster_to_core_idxs[cluster_label].append(i)
        else:
            cluster_to_noncore_idxs[cluster_label].append(i)

    return cluster_to_core_idxs, cluster_to_noncore_idxs, labels


# Visualize all core samples per cluster
def visualize_mnist_clustering(cluster_to_core_idxs, cluster_to_noncore_idxs, all_sis_images,
                               num_examples=15, title=None, savepath=None,
                               cluster_to_name_str=None):
    all_clusters = set(list(cluster_to_core_idxs.keys()) + list(cluster_to_noncore_idxs.keys()))
    all_clusters = sorted(list(all_clusters))
    if all_clusters[0] == -1:  # move misc. cluster to the end
        tmp = all_clusters.pop(0)
        all_clusters.append(tmp)

    nrow = len(all_clusters)
    ncol = num_examples

    fig = plt.figure(figsize=((ncol+3)/2.0, (nrow+2)/2.0))
    gs = gridspec.GridSpec(nrow, ncol,
             wspace=0.05, hspace=0.05,
             top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1),
             left=0.5/(ncol+1), right=1-0.5/(ncol+1))

    np.random.seed(1234)  # for reproducibility
    for r, cluster in enumerate(all_clusters):
        if cluster_to_name_str is not None and cluster in cluster_to_name_str:
            cluster_name = cluster_to_name_str[cluster]
        elif cluster == -1:
            cluster_name = 'Misc.'
        else:
            cluster_name = '$C_{%d}$' % (cluster + 1)
        if cluster in cluster_to_core_idxs:
            core_idxs = cluster_to_core_idxs[cluster]
        else:
            core_idxs = []
        if cluster in cluster_to_noncore_idxs:
            noncore_idxs = cluster_to_noncore_idxs[cluster]
        else:
            noncore_idxs = []
        core_idxs = np.asarray(core_idxs)
        noncore_idxs = np.asarray(noncore_idxs)
        if len(core_idxs) >= num_examples:  # randomly sample all examples form the core_idxs
            idxs_to_plot = core_idxs[np.random.choice(np.arange(core_idxs.shape[0]), size=num_examples, replace=False)]
        elif len(core_idxs) > 0:  # use all core idxs
            idxs_to_plot = core_idxs
        elif cluster == -1:
            idxs_to_plot = []
        else:
            raise ValueError('Should always have at least one core sample.')
        # Sample any remaining images from noncore_idxs
        if len(idxs_to_plot) < num_examples:
            num_needed = num_examples - len(idxs_to_plot)
            additional_idxs = noncore_idxs[np.random.choice(
                                                      np.arange(len(noncore_idxs)),
                                                      size=num_needed,
                                                      replace=False)]
            idxs_to_plot = list(idxs_to_plot) + list(additional_idxs)
        images_to_plot = [all_sis_images[i] for i in idxs_to_plot]
        for c, img in enumerate(images_to_plot):
            ax = plt.subplot(gs[r,c])  #axes[r, c]
            ax.imshow(img, cmap='gray')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            ax.grid(False)
            if c == 0:  # set row label
                if cluster_to_name_str is not None:
                    size = 16
                    labelpad = 125
                    horizontalalignment = 'left'
                else:
                    size = 20
                    labelpad = None
                    horizontalalignment = 'center'
                ax.set_ylabel('%s   ' % cluster_name, rotation=0,
                        verticalalignment='center', size=size,
                        horizontalalignment=horizontalalignment,
                        labelpad=labelpad)

    if title is not None:
        plt.suptitle(title, y=1.025, size=22)

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.show()


def get_label_to_cluster(labels):
    cluster_label_to_cluster = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_label_to_cluster[label].append(i)
    return cluster_label_to_cluster


def text_print_clustering(cluster_label_to_cluster,
                          cluster_label_to_core_sample_idxs,
                          all_suffic_rationales_tokenized,
                          index_to_token,
                          num_random_examples=10,
                          num_core_examples=None,
                          random_seed=1234):
    np.random.seed(random_seed)
    for label, idxs in sorted(cluster_label_to_cluster.items()):
        print('-- Cluster %d --' % label)

        # Print core samples for cluster
        if label in cluster_label_to_core_sample_idxs:
            core_sample_idxs = cluster_label_to_core_sample_idxs[label]
            core_sample_idxs_to_print = np.random.choice(np.asarray(core_sample_idxs), size=len(core_sample_idxs), replace=False)
            if num_core_examples is not None:
                core_sample_idxs_to_print = core_sample_idxs_to_print[:num_core_examples]
            for i in core_sample_idxs_to_print:
                rationale_tokens = all_suffic_rationales_tokenized[i]
                rationale_words = [index_to_token[t] for t in rationale_tokens]
                rationale_str = ' '.join(rationale_words)
                rationale_str = '*** %s ***' % rationale_str  # highlight core samples
                print(rationale_str)
            core_sample_idxs_set = set(list(core_sample_idxs))
            non_core_idxs = [i for i in idxs if i not in core_sample_idxs_set]
        else:
            non_core_idxs = idxs

        # Print random examples in cluster
        if len(non_core_idxs) < num_random_examples:
            idxs_to_print = non_core_idxs
        else:
            idxs_to_print = np.random.choice(np.asarray(non_core_idxs), size=num_random_examples, replace=False)
        for i in idxs_to_print:
            rationale_tokens = all_suffic_rationales_tokenized[i]
            rationale_words = [index_to_token[t] for t in rationale_tokens]
            rationale_str = ' '.join(rationale_words)
            print(rationale_str)

        print('')


def get_all_rationale_strs_by_cluster(cluster_label_to_cluster,
                                      cluster_label_to_core_sample_idxs,
                                      all_suffic_rationales_tokenized,
                                      index_to_token,
                                      shuffle_lists=True,
                                      random_seed=1234):
    random.seed(random_seed)
    cluster_to_all_rationale_strs = {}
    for label, idxs in sorted(cluster_label_to_cluster.items()):
        all_rationale_strs = []  # list of final rationale strings, e.g. 'w1 w2'

        # Core samples
        if label in cluster_label_to_core_sample_idxs:
            core_sample_idxs = cluster_label_to_core_sample_idxs[label]
            for i in core_sample_idxs:
                rationale_tokens = all_suffic_rationales_tokenized[i]
                rationale_words = [index_to_token[t] for t in rationale_tokens]
                rationale_str = ' '.join(rationale_words)
                all_rationale_strs.append(rationale_str)
            core_sample_idxs_set = set(list(core_sample_idxs))
            non_core_idxs = [i for i in idxs if i not in core_sample_idxs_set]
        else:
            non_core_idxs = idxs

        # Non-core samples
        for i in non_core_idxs:
            rationale_tokens = all_suffic_rationales_tokenized[i]
            rationale_words = [index_to_token[t] for t in rationale_tokens]
            rationale_str = ' '.join(rationale_words)
            all_rationale_strs.append(rationale_str)

        # randomize all_rationale_strs so ordering is arbitrary before
        #  they are displayed downstream, for when there are ties
        if shuffle_lists:
            random.shuffle(all_rationale_strs)
        cluster_to_all_rationale_strs[label] = all_rationale_strs

    return cluster_to_all_rationale_strs


def text_print_clustering_by_freq(cluster_label_to_cluster,
                                  cluster_label_to_core_sample_idxs,
                                  all_suffic_rationales_tokenized,
                                  index_to_token,
                                  num_top=None):
    cluster_to_all_rationale_strs = get_all_rationale_strs_by_cluster(
        cluster_label_to_cluster,
        cluster_label_to_core_sample_idxs,
        all_suffic_rationales_tokenized,
        index_to_token,
        shuffle_lists=True,
    )
    for label, rationale_strs in sorted(cluster_to_all_rationale_strs.items()):
        counter = Counter(rationale_strs)

        print('-- Cluster %d --' % label)
        for r, freq in counter.most_common(num_top):
            print('%s\t%d' % (r, freq))

        print('')


def _latex_table_text_cluster_data(rationale_strs, num_sis=4,
                                   include_freq=False):
    counter = Counter(rationale_strs)
    data = {}
    for i, (rationale_str, freq) in enumerate(counter.most_common(num_sis)):
        key = 'SIS %d' % (i+1)
        val = rationale_str
        data[key] = val
        if include_freq:
            data['%s Freq' % key] = freq
    return data


def latex_table_text_clusters(cluster_label_to_cluster,
                              cluster_label_to_core_sample_idxs,
                              all_suffic_rationales_tokenized,
                              index_to_token,
                              num_sis=4,
                              show_noise_cluster=True,
                              include_freq=False,
                              increase_cluster_nums=True,
                              composition_data=None,
                              composition_title='Composition'):
    cluster_to_all_rationale_strs = get_all_rationale_strs_by_cluster(
        cluster_label_to_cluster,
        cluster_label_to_core_sample_idxs,
        all_suffic_rationales_tokenized,
        index_to_token,
        shuffle_lists=True,
    )
    df_data = {}
    for label, rationale_strs in sorted(cluster_to_all_rationale_strs.items()):
        if label == -1 and not show_noise_cluster:
            continue
        cluster_data = _latex_table_text_cluster_data(
            rationale_strs,
            num_sis=num_sis,
            include_freq=include_freq,
        )
        # if composition_data is not None:
        #     cluster_data['__%s' % composition_title] = composition_data[label]
        if increase_cluster_nums and label != -1:
            out_label = label + 1
        else:
            out_label = label
        cluster_key = '$C_{%d}$' % out_label
        if composition_data is not None:
            cluster_key += ' (%s %s)' % \
                    (composition_title, composition_data[label])
        df_data[cluster_key] = cluster_data

    # Create DataFrame
    df = pd.DataFrame(df_data).transpose()
    # Replace NaNs with empty string in df
    df = df.replace(np.nan, '-', regex=True)

    return df


# Helper for cleaning output of latex table
def clean_latex_table_output(latex_table, clean_freq_cols=False,
                             make_figure=True, fix_escape=True):
    latex_table_clean = latex_table.replace('\_\_', '')
    latex_table_clean = latex_table_clean.replace(
            '\\begin{tabular}', '\\begin{tabularx}{\\textwidth}')
    latex_table_clean = latex_table_clean.replace(
            '{llrlllllll}', '{XXXXXXXXXX}')
    latex_table_clean = latex_table_clean.replace(
            '{llrllllll}', '{XXXXXXXXX}')
    latex_table_clean = latex_table_clean.replace(
            '{llrlrlrlr}', '{XXXXXXXXX}')
    latex_table_clean = latex_table_clean.replace(
            '\\end{tabular}', '\\end{tabularx}')
    latex_table_clean = latex_table_clean.replace('Cluster ', '')
    latex_table_clean = latex_table_clean.replace(r'{}', 'Cluster')
    if clean_freq_cols:
        freq_idx_to_replace = 1
        while True:
            freq_col = 'SIS %d Freq' % freq_idx_to_replace
            if freq_col in latex_table_clean:
                latex_table_clean = latex_table_clean.replace(freq_col, 'Freq.')
                freq_idx_to_replace += 1
            else:
                break
    latex_table_clean = latex_table_clean.replace('\\toprule\n', '')
    latex_table_clean = latex_table_clean.replace('\\bottomrule\n', '')
    if fix_escape:
        latex_table_clean = latex_table_clean.replace('%', '\\%')
    if make_figure:
        latex_table_clean = '''\\begin{{figure}}
    \\begingroup  % for font size
    \\footnotesize
    {}
    \\endgroup
    \\caption[]{{}}
  \label{{}}
\\end{{figure}}
'''.format(latex_table_clean)
    return latex_table_clean


## Helpers for text clustering

def text_get_rationale_tokens(review, rationale, sort=True):
    elms = rationale.get_elms()
    if sort:
        elms = sorted(elms)
    rationale_tokens = np.asarray(review.x[review.num_pad:])[elms]
    return rationale_tokens


def text_get_all_sis_tokenized(reviews, rationale_key=SIS_RATIONALE_KEY):
    all_sis_tokenized = []
    for review in reviews:
        for rationale in review.get_rationales(rationale_key):
            all_sis_tokenized.append(
                text_get_rationale_tokens(review, rationale)
            )
    return all_sis_tokenized


def jaccard_dist(s1, s2):
    intersection_size = len(s1.intersection(s2))
    union_size = len(s1.union(s2))
    return 1.0 - float(intersection_size) / union_size


def compute_jaccard_dist_matrix(all_sis_tokenized):
    rationale_dists = np.zeros((len(all_sis_tokenized), len(all_sis_tokenized)))
    for i in range(len(all_sis_tokenized)):
        r1 = set(all_sis_tokenized[i])
        for j in range(0, i+1):
            r2 = set(all_sis_tokenized[j])
            dist = jaccard_dist(r1, r2)
            rationale_dists[i, j] = dist
            rationale_dists[j, i] = dist
    return rationale_dists
