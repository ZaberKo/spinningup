import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path


def smooth(y, radius, mode='two_sided', valid_only=False):
    '''Smooth signal y, where radius is determines the size of the window.

    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]
    valid_only: put nan in entries where the full-sized window is not available
    '''
    assert mode in ('two_sided', 'causal')
    if len(y) < 2 * radius + 1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius + 1)
        out = np.convolve(y, convkernel, mode='same') / \
            np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel, mode='full') / \
            np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius + 1]
        if valid_only:
            out[:radius] = np.nan
    return out


def plot(path, r=0):
    df = pd.read_csv(os.path.join(path, "progress.txt"), delimiter="\t")
    plt.figure(dpi=300, figsize=(9, 3))
    smoothed_return = smooth(df["AverageTestEpRet"], r)
    smoothed_std = smooth(df["StdTestEpRet"], r)
    plt.plot(df["Epoch"], smoothed_return,
             label="return", lw=0.4)
    plt.fill_between(df["Epoch"], smoothed_return-smoothed_std,
                     smoothed_return+smoothed_std, alpha=0.2)
    plt.legend()
    plt.show()


def plot_all(path, r=0, plot_avg=True):

    plt.figure(dpi=300, figsize=(9, 3))
    eval_return_arr = []
    for exp_path in Path(path).expanduser().iterdir():
        print(exp_path)
        df = pd.read_csv(os.path.join(
            exp_path, "progress.txt"), delimiter="\t")

        eval_return_arr.append(df["AverageTestEpRet"])
        smoothed_return = smooth(df["AverageTestEpRet"], r)
        smoothed_std = smooth(df["StdTestEpRet"], r)
        plt.plot(df["Epoch"], smoothed_return, lw=0.4)
        # plt.fill_between(df["Epoch"],smooth(df["MinTestEpRet"],r),smooth(df["MaxTestEpRet"],r),alpha=0.2)
        plt.fill_between(df["Epoch"], smoothed_return-smoothed_std,
                         smoothed_return+smoothed_std, alpha=0.2)
    _min_len = min([len(d) for d in eval_return_arr])
    avg_return = np.array([d[:_min_len] for d in eval_return_arr]).mean(axis=0)
    smoothed_avg_return = smooth(avg_return, r)
    # plt.plot(np.arange(1, len(smoothed_avg_return)+1), smoothed_avg_return)
    # plt.legend()
    plt.show()
    
    if plot_avg:
        plt.figure(dpi=300, figsize=(9, 3))
        plt.plot(np.arange(1, len(smoothed_avg_return)+1), smoothed_avg_return)
        plt.show()

def plot_all_alpha(path, r=0):

    plt.figure(dpi=300, figsize=(9, 3))

    for exp_path in Path(path).expanduser().iterdir():
        print(exp_path)
        df = pd.read_csv(os.path.join(
            exp_path, "progress.txt"), delimiter="\t")

        plt.plot(df["Epoch"], smooth(df["Alpha"],r), lw=0.4)

    plt.show()


def plotly_plot_all(path, r=0, plot_avg=True):
    eval_return_arr = []
    fig=go.Figure()
    colors=px.colors.qualitative.Plotly
    i =0
    for exp_path in Path(path).expanduser().iterdir():
        print(exp_path)
        df = pd.read_csv(os.path.join(
            exp_path, "progress.txt"), delimiter="\t")

        eval_return_arr.append(df["AverageTestEpRet"])
        smoothed_return = smooth(df["AverageTestEpRet"], r)
        
        smoothed_std = smooth(df["StdTestEpRet"], r)

        color=colors[i]
        
        fig.add_scatter(
            x=df["Epoch"], y=smoothed_return, line_color=color,
            name=f"exp{i}"
        )

        fig.add_scatter(
            x=pd.concat([df["Epoch"], df["Epoch"][::-1]]),
            y=np.concatenate([smoothed_return+smoothed_std, (smoothed_return-smoothed_std)[::-1]]),
            fill="toself",
            fillcolor=color,
            opacity=0.2,
            line_color='rgba(255,255,255,0)',
            showlegend=False,
        )
        i+=1
    fig.update_layout(xaxis=dict(title="epoch", dtick=50),yaxis=dict(title="epoch"), showlegend=True)
    fig.show()
    
    if plot_avg:
        _min_len = min([len(d) for d in eval_return_arr])
        avg_return = np.array([d[:_min_len] for d in eval_return_arr]).mean(axis=0)
        smoothed_avg_return = smooth(avg_return, r)
        fig=px.line(x=np.arange(1, len(smoothed_avg_return)+1), y=smoothed_avg_return)
        fig.update_layout(xaxis=dict(title="epoch", dtick=50),yaxis=dict(title="epoch"), showlegend=True)
        fig.show()