#!/usr/bin/env python3

import math 
import pandas as pd 
import seaborn as sns
import numpy as np  
# import matplotlib
import matplotlib.pyplot as plt


def plot_scatter(csv_path="/work/karame.mp/src/benchmark/build_new/result.csv"):
    data = pd.read_csv(csv_path)

    print(list(data["n"]))

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(x="n", y="speed_up", data=data,
                        # hue="alg_name",
                        # hue="chunks",
                        # legend="full",
                        # palette=palette,
                            alpha=0.8,
                            # s=4,
                            ax=ax)

    sns.lineplot(x="n", y="speed_up", data=data,
                    # hue="alg_name",
                    # hue="chunks",ax
                    legend=None,
                    #  palette=palette,
                    ax=ax)

   # Plot a line at y=1 
    sns.lineplot(x=[data["n"].min(), data["n"].max()],
                 y=[1, 1], ax=ax, color="black", linestyle="dashed")


    ax.set_xscale("log", base=2)
    ax.set_xticks((list(data["n"])))
    # plt.xscale("log", base=2)

    # print(ax.set_xscale("log", base=2))

    ax.set_title("Par Speed_up with adaptive_chunk!!")

    plt.savefig("plot3.png", dpi=140)

plot_scatter("/work/karame.mp/src/benchmark/build_new/result.csv")

