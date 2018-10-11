import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import colorsys

from sis import coeff_determination_metric, find_sub_list, retokenize_annotation


ASPECT_TO_COLOR = { 0: 'red',
                    1: 'blue',
                    2: 'green' }


def rgb_to_hsl(r, g, b):
    r /= 255.0
    g /= 255.0
    b /= 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    h *= 360.0
    s *= 100.0
    l *= 100.0
    return (h, s, l)


def highlight_annot(review, rationale, index_to_token, aspect, hsl=True,
                     underline_annots=True):
    review_len = review.get_num_tokens()
    words = review.to_text(index_to_token, str_joiner=None)
    aspects = [None for i in range(review_len)]
    if hsl:
        hsl_values = [None for i in range(review_len)]
        # linearly-spaced interval, rounded
        hsl_interval = np.rint(np.linspace(60, 95, num=len(rationale),
                                           endpoint=True))
        # if coloring text, should use [15, 85] range for interval
    if underline_annots:
        is_annot = [False for i in range(review_len)]
        annot_idxs = []
        for start, end in review.get_annotation_idxs():
            for i in range(start, end + 1):
                is_annot[i] = True

    for rank, i in enumerate(rationale):
        aspects[i] = aspect
        if hsl:
            # want 0 --> first element of rationale.elms (most important word)
            # rank = K - 1 - rank
            hsl_values[i] = hsl_interval[rank]

    formatted_words = []
    for i, w in enumerate(words):
        style = ''
        if aspects[i] is not None:
            asp = aspects[i]
            if hsl:
                color = ASPECT_TO_COLOR[asp]
                if color == 'red':
                    hsl_maker = lambda x: 'hsl(0, 100%%, %d%%)' % int(x)
                    background_color = 'hsl(0, 100%, 96%)';
                elif color == 'green':
                    hsl_maker = lambda x: 'hsl(120, 100%%, %d%%)' % int(x)
                    background_color = 'hsl(120, 100%, 96%)';
                else:  # color == 'blue'
                    hsl_maker = lambda x: 'hsl(240, 100%%, %d%%)' % int(x)
                    background_color = 'hsl(240, 100%, 96%)';
                style += 'background-color:%s;' % (hsl_maker(hsl_values[i]))
            else:
                style += 'color:%s;' % (ASPECT_TO_COLOR[asp])
        if underline_annots and is_annot[i]:
            style += 'text-decoration:underline; text-decoration-color:%s;' % \
                        (ASPECT_TO_COLOR[aspect])
        formatted_w = '<span style="%s">%s</span>' % (style, w)
        formatted_words.append(formatted_w)

    html_out = ' '.join(formatted_words)
    return html_out

def highlight_multi_rationale(review, rationales, index_to_token,
                                color_palette=sns.color_palette('dark'),
                                hsl=True, underline_annots=True,
                                underline_color='black'):
    review_len = review.get_num_tokens()
    words = review.to_text(index_to_token, str_joiner=None)
    word_to_rationale = [None for i in range(review_len)]
    if hsl:
        hsl_values = [None for i in range(review_len)]
    if underline_annots:
        is_annot = [False for i in range(review_len)]
        annot_idxs = []
        for start, end in review.get_annotation_idxs():
            for i in range(start, end + 1):
                is_annot[i] = True

    for i, rationale in enumerate(rationales):
        if hsl:
            # linearly-spaced interval, rounded
            hsl_interval = np.rint(np.linspace(60, 90, num=len(rationale),
                                               endpoint=True))
        for rank, c in enumerate(rationale):
            assert(word_to_rationale[c] is None)  # rationales must be disjoint
            word_to_rationale[c] = i
            if hsl:
                hsl_values[c] = hsl_interval[rank]

    formatted_words = []
    for i, w in enumerate(words):
        style = ''
        if word_to_rationale[i] is not None:
            rationale_idx = word_to_rationale[i]
            try:
                color = color_palette[rationale_idx]
            except:  # ran out of colors in palette, default to black/gray
                color = sns.color_palette('gray')[1]
            if hsl:
                h, s, l = rgb_to_hsl(*color)
                hsl_maker = lambda x: 'hsl(%.4f, %.4f%%, %d%%)' % (h, s, int(x))
                style += 'background-color:%s;' % (hsl_maker(hsl_values[i]))
            else:
                style += 'color:rgb(%.4f, %.4f, %.4f);' % \
                    (color[0]*255.0, color[1]*255.0, color[2]*255.0)
        if underline_annots and is_annot[i]:
            style += 'text-decoration:underline; text-decoration-color:%s;' % \
                        (underline_color)
        formatted_w = '<span style="%s">%s</span>' % (style, w)
        formatted_words.append(formatted_w)

    html_out = ' '.join(formatted_words)
    return html_out

def highlight_annot_tf(decoded_seq, rationale, color='red', shading=True,
                        joiner='', hide_elms=[]):
    in_rationale = [False for i in range(len(decoded_seq))]
    if shading:
        hsl_values = [None for i in range(len(decoded_seq))]
        # linearly-spaced interval, rounded
        hsl_interval = np.rint(np.linspace(60, 95, num=len(rationale),
                                           endpoint=True))
        # if coloring text, should use [15, 85] range for interval

    for rank, i in enumerate(rationale):
        in_rationale[i] = True
        if shading:
            # want 0 --> first element of rationale.elms (most important word)
            # rank = K - 1 - rank
            hsl_values[i] = hsl_interval[rank]

    formatted_seq = []
    for i, elem in enumerate(decoded_seq):
        if i in hide_elms:
            continue
        style = ''
        if in_rationale[i]:
            if shading:
                if color == 'red':
                    hsl_maker = lambda x: 'hsl(0, 100%%, %d%%)' % int(x)
                    background_color = 'hsl(0, 100%, 96%)'
                elif color == 'green':
                    hsl_maker = lambda x: 'hsl(120, 100%%, %d%%)' % int(x)
                    background_color = 'hsl(120, 100%, 96%)'
                else:  # color == 'blue'
                    hsl_maker = lambda x: 'hsl(240, 100%%, %d%%)' % int(x)
                    background_color = 'hsl(240, 100%, 96%)'
                style += 'background-color:%s;' % (hsl_maker(hsl_values[i]))
            else:
                style += 'color:%s;' % (color)
        formatted_elem = '<span style="%s">%s</span>' % (style, elem)
        formatted_seq.append(formatted_elem)

    html_out = joiner.join(formatted_seq)
    return html_out

def make_legend(num_sis, color_palette=sns.color_palette(), labels=None):
    html = ''
    def make_legend_li(rgb, text):
        html = ''
        html += '''<li style="float:left; margin-right:10px;">
        <span style="border:none;float:left;width:25px;height:16px;margin:3px;
        background-color:rgb(%.4f, %.4f, %.4f)"></span> %s</li>
        ''' % (rgb[0], rgb[1], rgb[2], text)
        return html
    html += '<p><ul style="list-style:none;">'
    for i, color in enumerate(color_palette[:num_sis]):
        rgb = np.array(list(color)) * 255.0
        if labels is not None:
            text = labels[i]
        else:
            text = 'SIS %d' % (i+1)
        html += make_legend_li(rgb, text)
    html += '</ul></p>'
    return html

def save_html(html, filepath, header=None):
    with open(filepath, 'w') as outfile:
        outfile.write('<html>\n<body>\n')
        if header:
            outfile.write(header + '<hr>\n')
        outfile.write(html)
        outfile.write('\n</body>\n</html>')

def plot_predictive_dist(dist, bins=25, vertlines=[], title='', savepath=None):
    plt.hist(dist, bins=bins)
    for x in vertlines:
        plt.axvline(x=x, c='black')
    if title != '':
        plt.title(title)
    plt.xlabel('Predicted Score')
    plt.ylabel('Frequency')
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
        plt.clf()
        plt.cla()
        plt.close()
    else:
        plt.show()

# `data` is list of (vals, bins, label) tuples
def plot_hist(data, title='', xlabel='', ylabel='', normed=True,
               savepath=None, legend_loc='upper right'):
    for vals, bins, label in data:
        plt.hist(vals, bins=bins, normed=normed, alpha=0.5, label=label)
    if xlabel != '':
        plt.xlabel(xlabel)
    if ylabel != '':
        plt.ylabel(ylabel)
    if title != '':
        plt.title(title)
    plt.legend(loc=legend_loc)
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
        plt.clf()
        plt.cla()
        plt.close()
    else:
        plt.show()

def plot_scatter(x, y, xlabel='', ylabel='', title='', upper_lim=None,
                  savepath=None):
    assert(len(x) == len(y))
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=50, alpha=0.2)
    if xlabel != '':
        plt.xlabel(xlabel)
    if ylabel != '':
        plt.ylabel(ylabel)
    if title != '':
        plt.title(title)
    if upper_lim is None:
        upper_lim = max(np.max(x), np.max(y))
    plt.xlim(0, upper_lim)
    plt.ylim(0, upper_lim)
    plt.plot([0, upper_lim], [0, upper_lim], '--', c='black')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
        plt.clf()
        plt.cla()
        plt.close()
    else:
        plt.show()

# For visualizing weights per feature (e.g. from integrated gradients)
def plot_bar_weights(weights, xs=None, title='', xlabel='', ylabel='',
                      savepath=None):
    if xs is None:
        xs = list(range(len(weights)))
    plt.bar(xs, weights)
    if xlabel != '':
        plt.xlabel(xlabel)
    if ylabel != '':
        plt.ylabel(ylabel)
    if title != '':
        plt.title(title)
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    else:
        plt.show()


# Visualize all SISes for some examples
# `rows` is a list of image grids
def visualize_mnist_sis_collection(rows, title=None, savepath=None):
    nrow = len(rows)
    ncol = max((len(r) for r in rows))+1

    fig = plt.figure(figsize=((ncol+3)/2.0, (nrow+2)/2.0))
    gs = gridspec.GridSpec(nrow, ncol,
             wspace=0.05, hspace=0.05,
             top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1),
             left=0.5/(ncol+1), right=1-0.5/(ncol+1))

    for r, images in enumerate(rows):
        for c, img in enumerate(images):
            if c != 0:
                c += 1
            ax = plt.subplot(gs[r,c])
            ax.imshow(img, cmap='gray')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            ax.grid(False)

    if title is not None:
        plt.suptitle(title, y=1.025, size=22)

    if savepath is not None:
        plt.savefig(savepath, dpi=600, bbox_inches='tight')
    plt.show()
