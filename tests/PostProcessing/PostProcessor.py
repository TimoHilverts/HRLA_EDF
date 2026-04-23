import time
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tabulate import tabulate
import multiprocess as mp
import dill

from .plot_empirical_probabilities import plot_empirical_probabilities

class PostProcessor:
    def __init__(self, data_filename, U=None, dU=None):
        print("Started Loading Data for PostProcessor", end="... ", flush=True)
        start = time.time()
        with open(data_filename, 'rb') as handle:
            data = dill.load(handle)
        end = time.time()
        print(f"Finished in {end - start} seconds")

        # Process loaded data
        self.samples = data.get("samples")
        self.title = data.get("title")
        self.U = U if U is not None else data.get("U")
        self.dU = dU if dU is not None else data.get("dU")
        self.d = data.get("d")
        self.M = data.get("M")
        self.N = data.get("N")
        self.K = data.get("K")
        self.h = data.get("h")
        self.As = data.get("As")
        self.sim_annealing = data.get("sim_annealing")


    def plot_empirical_probabilities(self, dpi, layout="23", tols=[1,2,3,4,5,6], running=False):
        # Ensure layout is valid
        if layout not in ["13", "23", "32", "22"]:
            raise ValueError("layout must be one of '13', '23', '32' or '22'")

        # Create subplots
        tiles = (int(layout[0]), int(layout[1]))
        figsize = {"13": (9, 2.5), "23": (9, 4.5), "32": (5, 6), "22": (5, 4)}
        fig, axs = plt.subplots(*tiles, figsize=figsize[layout], sharex=True, sharey=True)

        def plot_tol_curve(p_idx):
            # Retrieve tolerance and axis
            tol = tols[p_idx]
            ax = axs[p_idx // int(layout[1])][p_idx % int(layout[1])] if layout[0] != "1" else axs[p_idx]
            # Plot a curve per value of a
            for i, a in enumerate(self.As):
                plot_empirical_probabilities(self, i, ax, p_idx, layout, dpi, tol, running)

        # Plot each tolerance plot
        for i in tqdm(range(len(tols)), desc="Plotting Empirical Probabilities"):
            plot_tol_curve(i)

        # Create figure legend
        ax = axs[0][0] if layout[0] in ["2", "3"] else axs[0]
        handles, labels = ax.get_legend_handles_labels()
        plt.tight_layout()
        fig.legend(handles, labels, loc='lower center', ncol=len(self.As), bbox_to_anchor=(0.5, 0))
        bottom = {"13": 0.35, "23": 0.2, "32": 0.15, "22": 0.2}
        plt.subplots_adjust(bottom=bottom[layout])

        # Save and show plot
        if not os.path.exists("output"):
            os.makedirs("output")
        if not os.path.exists("output/plots"):
            os.makedirs("output/plots")
        plotname = f"output/plots/{self.title}_{time.time()}.png"
        plt.savefig(plotname, dpi=300)
        print(f"[SUCCESS] Saving plot to {plotname}")

    def compute_tables(self, measured, dpi, mode="mean", running=True):
        # Check validity of mode
        if mode not in ["mean", "std", "best"]:
            raise ValueError("mode must be one of 'mean', 'std' or 'best'")

        # Function to compute the wanted quantity
        def compute_task(task_nb):
            a = self.As[task_nb]
            bests_sub = ["" for i in range(len(measured) + 1)]
            bests_sub[0] = f"a={a}"
            for i, k in enumerate(measured):
                if running:
                    bsts = [min([min([self.U(x[kk-1]) for kk in range(dpi, k + dpi, dpi)]) for x in samples[task_nb]]) for samples in self.samples]
                else:
                    bsts = [min([U(x[k-1]) for x in samples[task_nb]]) for samples in self.samples]
                if mode == "mean":
                    bests_sub[i+1] = f"& {np.mean(bsts):.8f}"
                elif mode == "std":
                    bests_sub[i+1] = f"& {np.std(bsts):.8f}"
                elif mode == "median":
                    bests_sub[i+1] = f"& {np.median(bsts):.8f}"
                elif mode == "best":
                    bests_sub[i+1] = f"& {-np.min(bsts):.8f}"

            return bests_sub

        # Compute for each value of a in parallel
        with mp.Pool() as pool:
            bests = pool.map(compute_task, range(len(self.As)))
        
        # Add header row with number of iterations
        bests = [[""] + [f"K={K}" for K in measured], *bests]

        # Function to get transpose of matrix
        transpose = lambda X: [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]

        # Print the results
        bests[0][0] = mode
        print()
        print(tabulate(bests, headers="firstrow"))
        bests[0][0] = f"{mode}.T"
        print()
        print(tabulate(transpose(bests), headers="firstrow"))

    def get_best(self, measured, dpi):
        # Function to compute the wanted quantity
        def compute_task(task_nb):
            a = self.As[task_nb]
            best = None
            best_val = np.inf
            for i, k in enumerate(measured):
                for samples in self.samples:
                    for kk in range(dpi, k + dpi, dpi):
                        for x in samples[task_nb]:
                            val = self.U(x[kk-1])
                            if val < best_val:
                                best_val = val
                                best = x[kk-1]
            return best

        # Compute for each value of a in parallel
        with mp.Pool() as pool:
            bests = pool.map(compute_task, range(len(self.As)))
        
        for i, a in enumerate(self.As):
            print(f"Best value for a={a}: {bests[i]}")

    def plot_curves(self, dpi):
        # Create subplots — now 1x3 to include std
        fig, axs = plt.subplots(1, 3, figsize=(13, 4.5))

        methods = ["mean", "std", "best"]

        def plot_method_curve(p_idx):
            method = methods[p_idx]
            ax = axs[p_idx]

            for i, a in enumerate(self.As):
                # Collect per-iteration revenue values across all runs
                # Shape: (n_runs, K//dpi)
                all_run_revenues = []

                for samples in self.samples:
                    # Running best revenue for this run
                    run_best_U = np.inf
                    run_curve = np.zeros(self.K // dpi)

                    for k in range(0, self.K, dpi):
                        bests_at_k = [-self.U(x[k]) for x in samples[i]]
                        best_at_k = max(bests_at_k)

                        if method == "best":
                            # Running best: carry forward if new value is worse
                            if k > 0:
                                run_curve[k // dpi] = max(best_at_k, run_curve[k // dpi - 1])
                            else:
                                run_curve[k // dpi] = best_at_k
                        else:
                            # For mean and std: just record the best revenue at this iteration
                            run_curve[k // dpi] = best_at_k

                    all_run_revenues.append(run_curve)

                all_run_revenues = np.array(all_run_revenues)  # shape: (n_runs, K//dpi)

                if method == "mean":
                    to_plot = np.mean(all_run_revenues, axis=0)
                elif method == "std":
                    to_plot = np.std(all_run_revenues, axis=0)
                elif method == "best":
                    to_plot = np.max(all_run_revenues, axis=0)

                ax.plot(
                    range(0, self.K, dpi),
                    to_plot,
                    label=rf"$a={a}$" if not self.sim_annealing else r"$\overline{a}=$" + rf"${a}$"
                )

            ax.set_xlim(0, self.K - 1)
            ax.text(
                0.95, 0.93, method,
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
            ax.set_xlabel(r"Iteration count ($K$)")

            if method == "std":
                ax.set_ylabel("Std. dev. of best revenue (€)")
            else:
                ax.set_ylabel("Best revenue found (€)")

        for i in tqdm(range(len(methods)), desc="Plotting Curves"):
            plot_method_curve(i)

        # Shared legend at the bottom
        ax = axs[0]
        handles, labels = ax.get_legend_handles_labels()
        plt.tight_layout()
        fig.legend(
            handles, labels,
            loc='lower center',
            ncol=len(self.As),
            bbox_to_anchor=(0.5, 0)
        )
        plt.subplots_adjust(bottom=0.3)

        # Save
        if not os.path.exists("output"):
            os.makedirs("output")
        if not os.path.exists("output/plots"):
            os.makedirs("output/plots")
        plotname = f"output/plots/{self.title}_{time.time()}.png"
        plt.savefig(plotname, dpi=300)
        print(f"[SUCCESS] Saving plot to {plotname}")
        

    def plot_best_revenue_vs_K(self, Ks):
        """
        Plot best revenue found up to specified iteration counts.
        """

        plt.figure(figsize=(8,5))

        for i, a in enumerate(self.As):

            best_revenue_curve = []
            running_best_U = np.inf

            for k in Ks:

                bests_U = []

                for samples in self.samples:
                    for x in samples[i]:
                        val = self.U(x[k-1])
                        bests_U.append(val)

                current_best_U = min(bests_U)

                running_best_U = min(running_best_U, current_best_U)

                best_revenue_curve.append(-running_best_U)

            plt.plot(Ks, best_revenue_curve, marker="o", label=rf"$a={a}$")

        plt.xlabel("Iterations $K$")
        plt.ylabel("Best revenue found (€)")
        plt.title("HRLA convergence: Best revenue vs iteration count")
        plt.grid(True)
        plt.legend()

        if not os.path.exists("output"):
            os.makedirs("output")
        if not os.path.exists("output/plots"):
            os.makedirs("output/plots")

        plotname = f"output/plots/{self.title}_best_revenue_vs_K_{time.time()}.png"
        plt.savefig(plotname, dpi=300)

        print(f"[SUCCESS] Saving plot to {plotname}")
        plt.show()

        

    def get_running_best_curve(self, a_idx=0, dpi=1):
        """
        Single-pass computation of the running best revenue curve for one value of a.

        Parameters
        ----------
        a_idx : int
            Index of the chosen a-value in self.As.
        dpi : int
            Keep one point every dpi iterations.

        Returns
        -------
        Ks : list[int]
            Iteration counts.
        best_revenues : np.ndarray
            Best revenue found up to each stored K.
        best_x_final : np.ndarray
            Best contract vector found up to the final iteration.
        best_revenue_final : float
            Corresponding best revenue.
        """
        if a_idx < 0 or a_idx >= len(self.As):
            raise ValueError(f"a_idx must be between 0 and {len(self.As)-1}")

        if dpi <= 0:
            raise ValueError("dpi must be positive")

        Ks = list(range(dpi, self.K + 1, dpi))
        n_points = len(Ks)

        # For each independent run, store its running-best curve
        per_run_curves = []
        best_U_global = np.inf
        best_x_global = None

        for samples in self.samples:
            # samples[a_idx] = trajectories / particles / chains for this a
            trajectories = samples[a_idx]

            # running best within this run
            run_best_U = np.inf
            run_curve = np.empty(n_points, dtype=float)

            for j, k in enumerate(Ks):
                # only inspect the newly reached iteration k
                current_iter_best_U = np.inf
                current_iter_best_x = None

                for x in trajectories:
                    val = self.U(x[k - 1])
                    if val < current_iter_best_U:
                        current_iter_best_U = val
                        current_iter_best_x = x[k - 1]

                if current_iter_best_U < run_best_U:
                    run_best_U = current_iter_best_U

                run_curve[j] = -run_best_U  # convert objective to revenue

                if current_iter_best_U < best_U_global:
                    best_U_global = current_iter_best_U
                    best_x_global = current_iter_best_x.copy()

            per_run_curves.append(run_curve)

        per_run_curves = np.array(per_run_curves)  # shape: (n_runs, n_points)
        best_revenues = np.max(per_run_curves, axis=0)

        return Ks, best_revenues, best_x_global, -best_U_global

    def plot_running_best_revenue(self, a_idx=0, dpi=1):
        """
        Plot the running best revenue curve for one chosen value of a.
        """
        Ks, best_revenues, best_x_final, best_revenue_final = self.get_running_best_curve(
            a_idx=a_idx, dpi=dpi
        )

        plt.figure(figsize=(8, 5))
        plt.plot(Ks, best_revenues, marker="o", markersize=3, linewidth=1.8)
        plt.xlabel(r"Iteration count $K$")
        plt.ylabel("Best revenue found (€)")
        plt.title(rf"Running best revenue for $a={self.As[a_idx]}$")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if not os.path.exists("output"):
            os.makedirs("output")
        if not os.path.exists("output/plots"):
            os.makedirs("output/plots")

        plotname = f"output/plots/{self.title}_running_best_aidx{a_idx}_{time.time()}.png"
        plt.savefig(plotname, dpi=300)
        print(f"[SUCCESS] Saving plot to {plotname}")
        plt.show()

        print(f"Best revenue up to K={self.K}: {best_revenue_final:.4f}")
        print(f"Best contract vector: {best_x_final}")

    def get_best_overall(self, a_idx=0, dpi=1, to_x=None):
        """
        Return the single best point found over all runs for one chosen a-value.

        Parameters
        ----------
        a_idx : int
            Index of the chosen a-value in self.As.
        dpi : int
            Check one point every dpi iterations.
        to_x : callable or None
            Optional mapping from z-space to x-space.

        Returns
        -------
        best_z : np.ndarray
            Best vector found in z-space.
        best_x : np.ndarray or None
            Corresponding vector in x-space if to_x is provided.
        best_revenue : float
            Corresponding best revenue.
        best_k : int
            Iteration at which the best point was found.
        """
        if a_idx < 0 or a_idx >= len(self.As):
            raise ValueError(f"a_idx must be between 0 and {len(self.As)-1}")
        if dpi <= 0:
            raise ValueError("dpi must be positive")

        best_U = np.inf
        best_z = None
        best_k = None

        for samples in self.samples:
            trajectories = samples[a_idx]

            for z_traj in trajectories:
                for k in range(dpi, self.K + 1, dpi):
                    z = z_traj[k - 1]
                    val = self.U(z)

                    if val < best_U:
                        best_U = val
                        best_z = z.copy()
                        best_k = k

        best_x = to_x(best_z) if to_x is not None else None

        return best_z, best_x, -best_U, best_k

    def plot_running_best_gap_vs_K(self, R_star, dpi=1):
        """
        Plot the running-best optimality gap R_star - R(K) for each value of a.

        Parameters
        ----------
        R_star : float
            Best known revenue over all experiments.
        dpi : int
            Keep one point every dpi iterations.
        """
        if dpi <= 0:
            raise ValueError("dpi must be positive")

        Ks = list(range(dpi, self.K + 1, dpi))
        n_points = len(Ks)

        plt.figure(figsize=(8, 5))

        for a_idx, a in enumerate(self.As):
            # For each independent run, compute its running-best revenue curve
            per_run_curves = []

            for samples in self.samples:
                trajectories = samples[a_idx]

                run_best_U = np.inf
                run_curve = np.empty(n_points, dtype=float)

                for j, k in enumerate(Ks):
                    current_iter_best_U = np.inf

                    for z in trajectories:
                        val = self.U(z[k - 1])
                        if val < current_iter_best_U:
                            current_iter_best_U = val

                    if current_iter_best_U < run_best_U:
                        run_best_U = current_iter_best_U

                    run_curve[j] = -run_best_U   # convert objective to revenue

                per_run_curves.append(run_curve)

            per_run_curves = np.array(per_run_curves)   # shape: (n_runs, n_points)

            # Best revenue across runs at each K
            best_revenues = np.max(per_run_curves, axis=0)

            # Gap to the best known revenue
            gaps = R_star - best_revenues

            plt.plot(Ks, gaps, linewidth=2, label=rf"$\bar{{a}}={a}$")

        plt.xlabel(r"Iteration count ($K$)")
        plt.ylabel(r"Gap to best known revenue $R^* - R(K)$ (€)")
        plt.title(rf"Running-best gap to $R^*={R_star:.4f}$")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        if not os.path.exists("output"):
            os.makedirs("output")
        if not os.path.exists("output/plots"):
            os.makedirs("output/plots")

        plotname = f"output/plots/{self.title}_running_best_gap_{time.time()}.png"
        plt.savefig(plotname, dpi=300)
        print(f"[SUCCESS] Saving plot to {plotname}")
        plt.show()

    def plot_running_best_revenue_zoom(self, dpi=1, zoom_K=5000):
        """
        Plot running best revenue vs iteration K for each value of a,
        together with a zoomed-in panel for early iterations.

        Parameters
        ----------
        dpi : int
            Keep one point every dpi iterations.
        zoom_K : int
            Upper limit of the zoomed region.
        """

        if dpi <= 0:
            raise ValueError("dpi must be positive")

        Ks = list(range(dpi, self.K + 1, dpi))
        n_points = len(Ks)

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        for a_idx, a in enumerate(self.As):

            per_run_curves = []

            for samples in self.samples:

                trajectories = samples[a_idx]

                run_best_U = np.inf
                run_curve = np.empty(n_points)

                for j, k in enumerate(Ks):

                    current_iter_best_U = np.inf

                    for z in trajectories:
                        val = self.U(z[k - 1])
                        if val < current_iter_best_U:
                            current_iter_best_U = val

                    if current_iter_best_U < run_best_U:
                        run_best_U = current_iter_best_U

                    run_curve[j] = -run_best_U   # convert objective to revenue

                per_run_curves.append(run_curve)

            per_run_curves = np.array(per_run_curves)

            best_revenues = np.max(per_run_curves, axis=0)

            # Full convergence
            axs[0].plot(Ks, best_revenues, linewidth=2, label=rf"$a={a}$")

            # Zoomed convergence
            axs[1].plot(Ks, best_revenues, linewidth=2, label=rf"$a={a}$")

        # Full plot
        axs[0].set_xlabel("Iterations $K$")
        axs[0].set_ylabel("Best revenue found (€)")
        axs[0].set_title("Convergence behaviour")
        axs[0].grid(True, alpha=0.3)

        # Zoom plot
        axs[1].set_xlim(0, zoom_K)
        axs[1].set_xlabel("Iterations $K$")
        axs[1].set_title(f"Zoom: first {zoom_K} iterations")
        axs[1].grid(True, alpha=0.3)

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(self.As))

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)

        if not os.path.exists("output"):
            os.makedirs("output")
        if not os.path.exists("output/plots"):
            os.makedirs("output/plots")

        plotname = f"output/plots/{self.title}_running_best_zoom_{time.time()}.png"
        plt.savefig(plotname, dpi=300)

        print(f"[SUCCESS] Saving plot to {plotname}")
        plt.show()

    def get_best_across_all_as(self, dpi=1, to_x=None):
        """
        Return the single best point found over all runs and all values of a.

        Parameters
        ----------
        dpi : int
            Check one point every dpi iterations.
        to_x : callable or None
            Optional mapping from z-space to x-space.

        Returns
        -------
        best_a : float
            Value of a for which the best point was found.
        best_z : np.ndarray
            Best vector found in z-space.
        best_x : np.ndarray or None
            Corresponding vector in x-space if to_x is provided.
        best_revenue : float
            Corresponding best revenue.
        best_k : int
            Iteration at which the best point was found.
        """
        if dpi <= 0:
            raise ValueError("dpi must be positive")

        best_U = np.inf
        best_a = None
        best_z = None
        best_k = None

        for a_idx, a in enumerate(self.As):
            for samples in self.samples:
                trajectories = samples[a_idx]

                for z_traj in trajectories:
                    for k in range(dpi, self.K + 1, dpi):
                        z = z_traj[k - 1]
                        val = self.U(z)

                        if val < best_U:
                            best_U = val
                            best_a = a
                            best_z = z.copy()
                            best_k = k

        best_x = to_x(best_z) if to_x is not None else None

        return best_a, best_z, best_x, -best_U, best_k

    def plot_running_best_revenue_all(self, dpi=1):
        """
        Plot running best revenue vs iteration K for each value of a
        in one full convergence figure.
        """
        if dpi <= 0:
            raise ValueError("dpi must be positive")

        Ks = list(range(dpi, self.K + 1, dpi))
        n_points = len(Ks)

        plt.figure(figsize=(8, 5))

        for a_idx, a in enumerate(self.As):
            per_run_curves = []

            for samples in self.samples:
                trajectories = samples[a_idx]

                run_best_U = np.inf
                run_curve = np.empty(n_points)

                for j, k in enumerate(Ks):
                    current_iter_best_U = np.inf

                    for z in trajectories:
                        val = self.U(z[k - 1])
                        if val < current_iter_best_U:
                            current_iter_best_U = val

                    if current_iter_best_U < run_best_U:
                        run_best_U = current_iter_best_U

                    run_curve[j] = -run_best_U  # revenue

                per_run_curves.append(run_curve)

            per_run_curves = np.array(per_run_curves)
            best_revenues = np.max(per_run_curves, axis=0)

            plt.plot(Ks, best_revenues, linewidth=2, label=rf"$a={a}$")

        plt.xlabel(r"Iteration count ($K$)")
        plt.ylabel("Best revenue found (€)")
        plt.title("Convergence behaviour")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        if not os.path.exists("output"):
            os.makedirs("output")
        if not os.path.exists("output/plots"):
            os.makedirs("output/plots")

        plotname = f"output/plots/{self.title}_running_best_full_{time.time()}.png"
        plt.savefig(plotname, dpi=300)
        print(f"[SUCCESS] Saving plot to {plotname}")
        plt.show()


    def plot_running_best_revenue_zoom_only(self, dpi=1, zoom_K=5000):
        """
        Plot running best revenue vs iteration K for each value of a,
        restricted to the first zoom_K iterations.
        """
        if dpi <= 0:
            raise ValueError("dpi must be positive")

        Ks = list(range(dpi, self.K + 1, dpi))
        n_points = len(Ks)

        plt.figure(figsize=(8, 5))

        for a_idx, a in enumerate(self.As):
            per_run_curves = []

            for samples in self.samples:
                trajectories = samples[a_idx]

                run_best_U = np.inf
                run_curve = np.empty(n_points)

                for j, k in enumerate(Ks):
                    current_iter_best_U = np.inf

                    for z in trajectories:
                        val = self.U(z[k - 1])
                        if val < current_iter_best_U:
                            current_iter_best_U = val

                    if current_iter_best_U < run_best_U:
                        run_best_U = current_iter_best_U

                    run_curve[j] = -run_best_U  # revenue

                per_run_curves.append(run_curve)

            per_run_curves = np.array(per_run_curves)
            best_revenues = np.max(per_run_curves, axis=0)

            plt.plot(Ks, best_revenues, linewidth=2, label=rf"$a={a}$")

        plt.xlim(0, zoom_K)
        plt.xlabel(r"Iteration count ($K$)")
        plt.ylabel("Best revenue found (€)")
        plt.title(f"Convergence behaviour: first {zoom_K} iterations")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        if not os.path.exists("output"):
            os.makedirs("output")
        if not os.path.exists("output/plots"):
            os.makedirs("output/plots")

        plotname = f"output/plots/{self.title}_running_best_zoomonly_{time.time()}.png"
        plt.savefig(plotname, dpi=300)
        print(f"[SUCCESS] Saving plot to {plotname}")
        plt.show()

    def summarize_best_of_M(self, a_idx=0, dpi=1):
        """
        Summarize the best-of-M performance for one chosen a-value.

        Interpretation:
        - self.samples contains repeated outer experiments
        - within each outer experiment, samples[a_idx] contains the M runs
        - for each outer experiment, we take the best revenue among those M runs
        - then we summarize these best-of-M revenues across outer experiments

        Parameters
        ----------
        a_idx : int
            Index of the chosen a-value in self.As.
        dpi : int
            Check one point every dpi iterations.

        Returns
        -------
        summary : dict
            Dictionary containing:
            - M
            - a
            - mean_best_revenue
            - std_best_revenue
            - best_revenue
            - best_k
        """
        if a_idx < 0 or a_idx >= len(self.As):
            raise ValueError(f"a_idx must be between 0 and {len(self.As)-1}")
        if dpi <= 0:
            raise ValueError("dpi must be positive")

        best_of_M_per_rep = []
        best_k_global = None
        best_U_global = np.inf

        for rep_samples in self.samples:
            trajectories = rep_samples[a_idx]

            rep_best_U = np.inf
            rep_best_k = None

            for z_traj in trajectories:
                for k in range(dpi, self.K + 1, dpi):
                    z = z_traj[k - 1]
                    val = self.U(z)

                    if val < rep_best_U:
                        rep_best_U = val
                        rep_best_k = k

            best_of_M_per_rep.append(-rep_best_U)

            if rep_best_U < best_U_global:
                best_U_global = rep_best_U
                best_k_global = rep_best_k

        best_of_M_per_rep = np.array(best_of_M_per_rep, dtype=float)

        summary = {
            "M": self.M,
            "a": self.As[a_idx],
            "mean_best_revenue": float(np.mean(best_of_M_per_rep)),
            "std_best_revenue": float(np.std(best_of_M_per_rep)),
            "best_revenue": float(np.max(best_of_M_per_rep)),
            "best_k": int(best_k_global) if best_k_global is not None else None,
        }

        return summary

    @staticmethod
    def compare_M_table(data_filenames, a_idx=0, dpi=1, U=None, dU=None):
        """
        Compare multiple experiment files with different values of M
        for one fixed a-value.

        Parameters
        ----------
        data_filenames : list[str]
            List of dill result files, each generated with a different M.
        a_idx : int
            Index of the chosen a-value in self.As.
        dpi : int
            Check one point every dpi iterations.

        Returns
        -------
        rows : list[list]
            Table rows for further use if needed.
        """
        rows = []

        for filename in data_filenames:
            pp = PostProcessor(filename, U=U, dU=dU)
            summary = pp.summarize_best_of_M(a_idx=a_idx, dpi=dpi)

            rows.append([
                summary["M"],
                f'{summary["mean_best_revenue"]:.4f}',
                f'{summary["std_best_revenue"]:.4f}',
                f'{summary["best_revenue"]:.4f}',
                summary["best_k"],
            ])

        rows.sort(key=lambda x: x[0])

        headers = [
            "M",
            "Mean best revenue (€)",
            "Std. dev. (€)",
            "Best revenue (€)",
            "Best iteration K",
        ]

        print()
        print(tabulate(rows, headers=headers, tablefmt="github"))

        return rows

    @staticmethod
    def plot_running_best_revenue_for_h(data_filenames, hs=None, dpi=1, zoom_K=None, a_idx=0, U=None, dU=None):
        """
        Compare convergence curves (Mean, Std, Best) for different step sizes h 
        using separate experiment files.
        """
        if dpi <= 0:
            raise ValueError("dpi must be positive")

        # 1. Setup the 1x3 subplot figure
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        methods = ["mean", "std", "best"]
        
        # We will collect handles for the legend from the first plot
        legend_handles = []

        # 2. Iterate through each file (each representing a different h)
        for file_idx, filename in enumerate(data_filenames):
            pp = PostProcessor(filename, U=U, dU=dU)

            if a_idx < 0 or a_idx >= len(pp.As):
                raise ValueError(f"a_idx must be between 0 and {len(pp.As)-1} for file {filename}")

            # Define iteration points
            Ks = list(range(dpi, pp.K + 1, dpi))
            n_points = len(Ks)
            
            # Prepare to store the running best curve for every single replication in this file
            all_run_curves = [] # Shape will be (n_replications, n_points)

            for rep_samples in pp.samples:
                trajectories = rep_samples[a_idx]
                run_best_U = np.inf
                run_curve = np.empty(n_points)

                for j, k in enumerate(Ks):
                    # Find best in current iteration across all chains/particles
                    current_iter_vals = [pp.U(z[k - 1]) for z in trajectories]
                    current_iter_best_U = min(current_iter_vals)

                    # Update running best (optimization is minimizing U)
                    if current_iter_best_U < run_best_U:
                        run_best_U = current_iter_best_U

                    run_curve[j] = -run_best_U  # Convert to revenue

                all_run_curves.append(run_curve)

            all_run_curves = np.array(all_run_curves)
            
            # Labeling logic
            label = rf"$h={hs[file_idx]}$" if hs is not None else rf"$h={pp.h}$"

            # 3. Plot this h-value's data across all 3 subplots
            for m_idx, method in enumerate(methods):
                ax = axs[m_idx]
                
                if method == "mean":
                    to_plot = np.mean(all_run_curves, axis=0)
                elif method == "std":
                    to_plot = np.std(all_run_curves, axis=0)
                else: # "best"
                    to_plot = np.max(all_run_curves, axis=0)

                line, = ax.plot(Ks, to_plot, linewidth=2, label=label)
                
                # Capture handles for legend only once
                if m_idx == 0:
                    legend_handles.append(line)

        # 4. Formatting and aesthetics
        for m_idx, method in enumerate(methods):
            ax = axs[m_idx]
            ax.set_title(method.capitalize())
            ax.set_xlabel(r"Iteration count ($K$)")
            ax.grid(True, alpha=0.3)
            
            if method == "std":
                ax.set_ylabel("Std. dev. of revenue (€)")
            else:
                ax.set_ylabel("Revenue (€)")
            
            if zoom_K is not None:
                ax.set_xlim(0, zoom_K)

            # Add the "wheat" box label like in your plot_curves def
            ax.text(
                0.95, 0.5, method,
                transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

        # Shared legend at the bottom
        plt.tight_layout()
        fig.legend(
            handles=legend_handles, 
            labels=[h.get_label() for h in legend_handles],
            loc='lower center',
            ncol=len(data_filenames),
            bbox_to_anchor=(0.5, -0.05)
        )
        plt.subplots_adjust(bottom=0.2)

        # 5. Save the output
        os.makedirs("output/plots", exist_ok=True)
        plotname = f"output/plots/h_comparison_triple_{int(time.time())}.png"
        plt.savefig(plotname, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] Saving multi-panel plot to {plotname}")
        plt.show()

    @staticmethod
    def compare_h_table(data_filenames, a_idx=0, dpi=1, U=None, dU=None):
        """
        Compare multiple experiment files with different values of h
        for one fixed a-value.

        Parameters
        ----------
        data_filenames : list[str]
            List of dill result files, each generated with a different h.
        a_idx : int
            Index of the chosen a-value in self.As.
        dpi : int
            Check one point every dpi iterations.

        Returns
        -------
        rows : list[list]
            Table rows for further use if needed.
        """
        rows = []

        for filename in data_filenames:
            pp = PostProcessor(filename, U=U, dU=dU)
            summary = pp.summarize_best_of_M(a_idx=a_idx, dpi=dpi)

            rows.append([
                pp.h,
                f'{summary["mean_best_revenue"]:.4f}',
                f'{summary["std_best_revenue"]:.4f}',
                f'{summary["best_revenue"]:.4f}',
                summary["best_k"],
            ])

        rows.sort(key=lambda x: x[0])

        headers = [
            "h",
            "Mean best revenue (€)",
            "Std. dev. (€)",
            "Best revenue (€)",
            "Best iteration K",
        ]

        print()
        print(tabulate(rows, headers=headers, tablefmt="github"))

        return rows