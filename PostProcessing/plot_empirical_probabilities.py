import numpy as np

def plot_empirical_probabilities(post_processor, i, ax, p_idx, layout, dpi, tol, running=False, comp="eps"):
    # Compute each empirical probability
    probs = np.zeros(post_processor.K // dpi)
    for k in range(0, post_processor.K, dpi):
        bests_U = [min([post_processor.U(x[k]) for x in samples[i]]) for samples in post_processor.samples]
        probs[k // dpi] = len(list(filter(lambda u: u <= tol, bests_U))) / post_processor.M

        # Take the running best if running is specified
        if running and k // dpi > 0:
            probs[k // dpi] = max(probs[k // dpi], probs[k // dpi - 1])

    # Plot the computed curve
    if comp == "eps" or comp == "d" or comp == "d_tol":
        label = rf"$a={post_processor.As[i]}$" if not post_processor.sim_annealing else r"$\overline{a}=$" + rf"${post_processor.As[i]}$"
    elif comp == "a" or comp == "a_tol":
        label = rf"{post_processor.title}"
    else:
        raise NotImplementedError(f"Unknown comparison type {comp}")
    ax.plot(range(0, post_processor.K, dpi), probs, label=label)

    # Define axis limits
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, post_processor.K-1)

    # Add a badge 
    if comp == "eps":
        badge = rf"$\varepsilon={tol}$"
    elif comp == "a":
        badge = rf"$a={post_processor.As[i]}$"
    elif comp == "a_tol":
        badge = rf"$a={post_processor.As[i]}, \varepsilon={tol}$"
    elif comp == "d_tol":
        badge = rf"$d={post_processor.d}, \varepsilon={tol}$"
    elif comp == "d":
        badge = rf"$\varepsilon={tol}, d={post_processor.d}$"
    else:
        raise NotImplementedError(f"Unknown comparison type {comp}")
    ax.text(0.95, 0.93, badge, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Set axis labels depending on positive
    if p_idx // int(layout[1]) == int(layout[0]) - 1 :
        ax.set_xlabel(r"Iteration count ($k$)")
    if p_idx % int(layout[1]) == 0:
        ax.set_ylabel(rf"$P(U(X_k)-U^*\leq \varepsilon)$")