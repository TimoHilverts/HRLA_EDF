import matplotlib.pyplot as plt
import time
import os

from .PostProcessor import PostProcessor
from .plot_empirical_probabilities import plot_empirical_probabilities

class Comparator:
    def __init__(self, data_filenames):
        self.post_processors = [PostProcessor(data_filename) for data_filename in data_filenames]

    def plot_empirical_probabilities_per_a(self, dpi, tol, running=False, layout="12"):
        # Ensure layout is valid
        if layout not in ["12", "13", "23", "32", "22"]:
            raise ValueError("layout must be one of '12', '13', '23', '32' or '22'")

        # Create subplots
        tiles = (int(layout[0]), int(layout[1]))
        figsize = {"12": (5, 2.8), "13": (9, 2.5), "23": (9, 4.5), "32": (5, 6), "22": (5, 4)}
        fig, axs = plt.subplots(*tiles, figsize=figsize[layout], sharex=True, sharey=True)

        # Plot a curve per post_processor and per value of a
        for post_processor in self.post_processors:
            for i, a in enumerate(post_processor.As):
                ax = axs[i // 2][i % 2] if layout[0] != "1" else axs[i]
                plot_empirical_probabilities(post_processor, i, ax, i, layout, dpi, tol, running, "a")

        # Create figure legend
        ax = axs[0][0] if layout[0] != "1" else axs[0]
        handles, labels = ax.get_legend_handles_labels()
        plt.tight_layout()
        fig.legend(handles, labels, loc='lower center', ncol=len(self.post_processors), bbox_to_anchor=(0.5, 0))
        bottom = {"12": 0.35, "13": 0.35, "23": 0.2, "32": 0.15, "22": 0.2}
        plt.subplots_adjust(bottom=bottom[layout])

        # Save and show plot
        if not os.path.exists("output"):
            os.makedirs("output")
        if not os.path.exists("output/plots"):
            os.makedirs("output/plots")
        plotname = f"output/plots/Comparator_{time.time()}.png"
        plt.savefig(plotname, dpi=300)
        print(f"[SUCCESS] Saving plot to {plotname}")

    def plot_empirical_probabilities_per_a_e(self, dpi, tols, As, running=False):
        # Ensure layout is valid
        layout = "12"
        if layout not in ["12", "13", "23", "32", "22"]:
            raise ValueError("layout must be one of '12', '13', '23', '32' or '22'")

        # Create subplots
        tiles = (int(layout[0]), int(layout[1]))
        figsize = {"12": (5, 2.2), "13": (9, 2.5), "23": (9, 4.5), "32": (5, 6), "22": (5, 4)}
        fig, axs = plt.subplots(*tiles, figsize=figsize[layout], sharex=True, sharey=True)

        # Plot a curve per post_processor and per value of a
        for i, (tol, a) in enumerate(zip(tols, As)):
            for post_processor in self.post_processors:
                for j, a_temp in enumerate(post_processor.As):
                    if a_temp != a:
                        continue
                    ax = axs[i // 2][i % 2] if layout[0] != "1" else axs[i]
                    plot_empirical_probabilities(post_processor, j, ax, i, layout, dpi, tol, running, "a_tol")

        # Create figure legend
        ax = axs[0][0] if layout[0] != "1" else axs[0]
        handles, labels = ax.get_legend_handles_labels()
        plt.tight_layout()
        fig.legend(handles, labels, loc='lower center', ncol=len(self.post_processors), bbox_to_anchor=(0.5, 0))
        bottom = {"12": 0.35, "13": 0.35, "23": 0.2, "32": 0.15, "22": 0.2}
        plt.subplots_adjust(bottom=bottom[layout])

        # Save and show plot
        if not os.path.exists("output"):
            os.makedirs("output")
        if not os.path.exists("output/plots"):
            os.makedirs("output/plots")
        plotname = f"output/plots/Comparator_{time.time()}.png"
        plt.savefig(plotname, dpi=300)
        print(f"[SUCCESS] Saving plot to {plotname}")

    def plot_empirical_probabilities_per_d_e(self, dpi, tols, ds, running=False):
        # Ensure layout is valid
        layout = "12"
        if layout not in ["12", "13", "23", "32", "22"]:
            raise ValueError("layout must be one of '12', '13', '23', '32' or '22'")

        # Create subplots
        tiles = (int(layout[0]), int(layout[1]))
        figsize = {"12": (5, 2.2), "13": (9, 2.5), "23": (9, 4.5), "32": (5, 6), "22": (5, 4)}
        fig, axs = plt.subplots(*tiles, figsize=figsize[layout], sharex=True, sharey=True)

        # Plot a curve per post_processor and per value of a
        for i, (tol, d) in enumerate(zip(tols, ds)):
            for post_processor in self.post_processors:
                if d != post_processor.d:
                    continue
                ax = axs[i // 2][i % 2] if layout[0] != "1" else axs[i]
                for j in range(len(post_processor.As)):
                    plot_empirical_probabilities(post_processor, j, ax, i, layout, dpi, tol, running, "d_tol")

        # Create figure legend
        ax = axs[0][0] if layout[0] != "1" else axs[0]
        handles, labels = ax.get_legend_handles_labels()
        plt.tight_layout()
        fig.legend(handles, labels, loc='lower center', ncol=len(self.post_processors[0].As), bbox_to_anchor=(0.5, 0))
        bottom = {"12": 0.35, "13": 0.35, "23": 0.2, "32": 0.15, "22": 0.2}
        plt.subplots_adjust(bottom=bottom[layout])

        # Save and show plot
        if not os.path.exists("output"):
            os.makedirs("output")
        if not os.path.exists("output/plots"):
            os.makedirs("output/plots")
        plotname = f"output/plots/Comparator_{time.time()}.png"
        plt.savefig(plotname, dpi=300)
        print(f"[SUCCESS] Saving plot to {plotname}")

    def plot_empirical_probabilities_per_d(self, dpi, tols, running=False):
        fig, axs = plt.subplots(1, 2, figsize=(5, 2.2), sharex=True, sharey=True)

        # Plot a curve per post_processor and per value of a
        for pi, post_processor in enumerate(self.post_processors):
            for i, a in enumerate(post_processor.As):
                plot_empirical_probabilities(post_processor, i, axs[pi], pi, "12", dpi, tols[pi], running, "d")

        # Create figure legend
        ax = axs[0]
        handles, labels = ax.get_legend_handles_labels()
        plt.tight_layout()
        fig.legend(handles, labels, loc='lower center', ncol=len(self.post_processors[0].As), bbox_to_anchor=(0.5, 0))
        plt.subplots_adjust(bottom=0.35)

        # Save and show plot
        if not os.path.exists("output"):
            os.makedirs("output")
        if not os.path.exists("output/plots"):
            os.makedirs("output/plots")
        plotname = f"output/plots/Comparator_d_{time.time()}.png"
        plt.savefig(plotname, dpi=300)
        print(f"[SUCCESS] Saving plot to {plotname}")
