from src.utils.mini_tools import load_data, save_data
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

PATH = "c:\\Users\\u0166838\\OneDrive - KU Leuven\\Documents\\Doc\\Code\\SCOPE"
# PATH = "/lustre1/scratch/372/vsc37290/SCOPE"
save_folder = "figures"

color_legend = {
    "separate": "#1F77B4",   # Muted blue
    "kmeans_q": "#2CA02C",     # Fresh green
    "dtr": "#D62728",        # Soft red/pink

    "optimal": "#9D9D9D",      # Neutral grey
    "random": "#BAB0AC",       # Light grey
}


linestyle_legend = {
    "separate": "dashdot",       # fine-grained separation
    "dtr": "solid",             # strong, primary line
    "kmeans_q": "dotted",      # unique pattern, between dotted & dashed
}

marker_legend = {
    "dtr": "s",
    "separate": "^",
    "kmeans_q": "o",
    "optimal": "None",
    "random": "None"
}

label_legend = {
    "dtr-S": r"SCOPE-S ($\mathit{ours}$)",
    "dtr-T": r"SCOPE-T ($\mathit{ours}$)",
    "dtr-RAS": r"SCOPE-RA ($\mathit{ours}$)",
    "separate-S": r"SEP-S ($\mathit{de\ Leoni\ et\ al.}$)",
    "separate-T": "SEP-T",
    "separate-RAS": "SEP-RA",
    "kmeans_q": r"KMeans-Q ($\mathit{Branchi\ et\ al.}$)",
    "optimal": "Upper Bound",
    "random": "Random"
}

label_legend_ablation = {
    "dtr": r"SCOPE ($\mathit{ours}$)",
    "separate": "SEP",
    "kmeans_q": r"KMeans-Q",
    "optimal": "Upper Bound",
    "random": "Random"
}

font_axes = 20
font_title = 20
font_sub_title = 20
font_ticks = 14
font_legend=13

ax_x_title = r"$\delta$"
ax_y_title = "Gain over bank policy (%)"

def load_results(training_sizes, delta_levels, methods, model_category, cross_fit, num_iterations=5, model_specific="xgb", n_stages=2, dataset="SimBank", confounding_type="point"):
    folder_to_add = ""
    if dataset == "bpic17":
        conf_suffix = "_case" if confounding_type == "case" else ""
        folder_to_add = os.path.join("bpic17" + conf_suffix, str(n_stages))
    
    results_dict = {}
    avg_uplift_dict = {}
    se_uplift_dict = {}

    for training_size in training_sizes:
        print("\n" + "="*60)
        print(f"Training size: {training_size}")
        print("="*60)
        
        results_dict[training_size] = {}
        avg_uplift_dict[training_size] = {}
        se_uplift_dict[training_size] = {}

        for method in methods:
            category = model_category if method != "kmeans_q" else "rl"
            print('\nMethod:', method)
            
            to_add_cross_fit = "cross_fitted_" if "dtr-AIPWE" in method and cross_fit else ""
            to_add_model_specific = model_specific + "_" if (method != "kmeans_q" and model_specific != "xgb" and model_specific != "lstm" and model_specific != "vanilla_nn") else ""
            print(to_add_model_specific)
            results_dict[training_size][method] = {}
            avg_uplift_dict[training_size][method] = {}
            se_uplift_dict[training_size][method] = {}

            target = "effect" if any(x in method for x in ["DR", "AIPWE", "RA"]) else "outcome"

            for delta in delta_levels:
                print("Delta:", delta)
                results_dict[training_size][method][delta] = []
                uplift_eval_list = []

                for i in range(num_iterations):
                    if method in ["random", "optimal"]:
                        if method == "optimal":
                            file_path = os.path.join(
                                PATH, "res", folder_to_add, str(training_size), str(int(100 * delta)), method,
                                f"{training_size}_{int(100 * delta)}_{method}"
                            )
                        else:
                            file_path = os.path.join(
                                PATH, "res", folder_to_add, str(training_size), str(int(100 * delta)), method,
                                f"{training_size}_{int(100 * delta)}_{i}_{method}"
                            )
                        profit_eval = load_data(file_path + "_profit.pkl")
                        uplift_eval = load_data(file_path + "_uplift.pkl")
                    else:
                        file_path = os.path.join(
                            PATH, "res", folder_to_add, str(training_size), str(int(100 * delta)), method, "eval",
                            f"{training_size}_{int(100 * delta)}_{method}_{target}_{category}"
                        )
                        profit_eval = load_data(file_path + f"_{to_add_model_specific}{i}_{to_add_cross_fit}profit_eval.pkl")
                        uplift_eval = load_data(file_path + f"_{to_add_model_specific}{i}_{to_add_cross_fit}uplift_eval.pkl")

                    results_dict[training_size][method][delta].append({
                        "profit_eval": profit_eval,
                        "uplift_eval": uplift_eval
                    })
                    uplift_eval_list.append(uplift_eval)

                # compute mean & SE
                avg_uplift_eval = sum(uplift_eval_list) / len(uplift_eval_list)
                n = len(uplift_eval_list)
                s = (sum((x - avg_uplift_eval) ** 2 for x in uplift_eval_list) / (n - 1)) ** 0.5
                se_uplift_eval = s / (n ** 0.5)

                avg_uplift_dict[training_size][method][delta] = avg_uplift_eval
                se_uplift_dict[training_size][method][delta] = se_uplift_eval

                print(f"  Avg uplift: {avg_uplift_eval:.4f}, SE: {se_uplift_eval:.4f}")

    return results_dict, avg_uplift_dict, se_uplift_dict

def plot_results_delta_levels_by_training_size_grid(avg_uplift_dict, se_uplift_dict, methods, delta_levels, training_sizes, model_specific="xgb"):
    """
    Plot average uplift by delta levels and methods for multiple training sizes in a grid.

    Parameters
    ----------
    avg_uplift_dict : dict
        Nested dictionary: avg_uplift_dict[training_size][method][delta].
    se_uplift_dict : dict
        Nested dictionary: se_uplift_dict[training_size][method][delta].
    methods : list[str]
        List of methods to plot.
    delta_levels : list[float]
        Delta levels to plot.
    training_sizes : list[int]
        List of training sizes to display in grid.
    """
    n_sizes = len(training_sizes)
    # Special case: if exactly 3 training sizes, force 1 row and 3 columns
    if n_sizes == 3:
        n_rows = 1
        n_cols = 3
    else:
        n_cols = min(2, n_sizes)
        n_rows = (n_sizes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), sharey=True)

    # Ensure axes is always a 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    for idx, training_size in enumerate(training_sizes):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        training_title = f"{training_size // 1000}K" if training_size >= 1000 else str(training_size)
        ax.set_title(training_title, fontsize=font_sub_title)

        # ✅ Only set y-label for plots on the leftmost column
        if col == 0:
            ax.set_ylabel(ax_y_title, fontsize=font_axes)

        ax.grid(True, color='gray', linestyle='-', linewidth=0.3, alpha=0.15)

        for method in methods:
            if method in ["random", "optimal"]:
                value = avg_uplift_dict[training_size][method]
                if isinstance(value, dict):
                    value = list(value.values())[0]
                elif isinstance(value, (list, np.ndarray)):
                    value = value[0]

                if method == "optimal":
                    label = f"Upper Bound: {value:.3f}"
                else:
                    label = f"Random: {value:.3f} ± {se_uplift_dict[training_size][method][next(iter(se_uplift_dict[training_size][method]))]:.3f}"
                color = color_legend.get(method, 'black')
                marker = marker_legend.get(method, 'o')
                ax.plot([], [], marker=marker, color=color, linestyle="None", label=label)
                continue

            avg_uplifts = [avg_uplift_dict[training_size][method][delta] for delta in delta_levels]
            se_uplifts = [se_uplift_dict[training_size][method][delta] for delta in delta_levels]

            avg_uplifts = np.array(avg_uplifts)
            se_uplifts = np.array(se_uplifts)

            label = label_legend.get(method.split('-')[0] + '-' + method.split('-')[1], method) if method != "kmeans_q" else label_legend.get(method, method)
            # color = color_legend.get(method.split('-')[0] + '-' + method.split('-')[1], 'black') if method != "kmeans_q" else color_legend.get(method, 'black')
            color = color_legend.get(method.split('-')[0], 'solid') if method != "kmeans_q" else color_legend.get(method, method)
            linestyle = linestyle_legend.get(method.split('-')[0], 'solid') if method != "kmeans_q" else linestyle_legend.get(method, method)
            # linestyle = linestyle_legend.get(method.split('-')[1], 'solid') if method != "kmeans_q" else linestyle_legend.get(method, method)
            marker = marker_legend.get(method.split('-')[0], 'o') if method != "kmeans_q" else marker_legend.get(method, method)
            # marker = marker_legend.get(model_specific, 'o') if method != "kmeans_q" else marker_legend.get(method, method)

            ax.plot(delta_levels, avg_uplifts, marker=marker, label=label,
                    color=color, linestyle=linestyle, linewidth=1, markersize=6)
            ax.fill_between(delta_levels,
                            avg_uplifts - se_uplifts,
                            avg_uplifts + se_uplifts,
                            alpha=0.2,
                            color=color)

        ax.set_xticks(delta_levels)

        ax.tick_params(axis='x', labelsize=font_ticks)  # set font size for x-axis ticks
        ax.tick_params(axis='y', labelsize=font_ticks)  # set font size for y-axis ticks

        # ✅ Only show x-axis label and ticks on the bottom row
        if row == n_rows - 1:
            # ax.set_xlabel(r"Confounding level ($\delta$)", fontsize=font_axes)
            ax.set_xlabel(ax_x_title, fontsize=font_axes)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])

        # ✅ Only show legend in the last subplot
        if idx == len(training_sizes) - 1:
            ax.legend(fontsize=font_legend)

    # Remove unused subplots
    for idx in range(len(training_sizes), n_rows * n_cols):
        fig.delaxes(axes.flatten()[idx])

    # fig.suptitle(r"Performance across confounding levels for different $\mathit{training\ sizes}$", fontsize=font_title)

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder,"training_sizes.pdf"), bbox_inches="tight")
    plt.show()

def plot_results_delta_levels_by_learner_grid(avg_uplift_dict, se_uplift_dict, methods, delta_levels, training_size, model_specific="xgb"):
    """
    Plot average uplift by delta levels and methods, arranged by causal learner type (S-, T-, RA-learner).

    Parameters
    ----------
    avg_uplift_dict : dict
        Nested dictionary: avg_uplift_dict[training_size][method][delta].
    se_uplift_dict : dict
        Nested dictionary: se_uplift_dict[training_size][method][delta].
    methods : list[str]
        List of methods to plot. Must include learner info in name, e.g., 'xgb-S', 'rf-T', etc.
    delta_levels : list[float]
        Delta levels to plot.
    training_size : int
        Fixed training size to display.
    """
    # --- Group methods by causal learner type (S, T, RA, etc.)
    learner_groups = {}
    for method in methods:
        if "-" in method:
            learner_type = method.split("-")[1]
            learner_groups.setdefault(learner_type, []).append(method)

    learner_types = list(learner_groups.keys())
    n_learners = len(learner_types)

    # ✅ Updated: use up to 3 columns when possible
    n_cols = min(3, n_learners)
    n_rows = (n_learners + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), sharey=True)

    # Ensure axes is always 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    for idx, learner_type in enumerate(learner_types):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        learner_title = "RA" if "RA" in learner_type else learner_type
        ax.set_title(f"{learner_title}-Learner", fontsize=font_sub_title)
        # ✅ Only set y-label for plots on the leftmost column
        if col == 0:
            ax.set_ylabel(ax_y_title, fontsize=font_axes)
        ax.grid(True, color='gray', linestyle='-', linewidth=0.3, alpha=0.15)

        for method in learner_groups[learner_type]:
            if method in ["random", "optimal"]:
                value = avg_uplift_dict[training_size][method]
                if isinstance(value, dict):
                    value = list(value.values())[0]
                elif isinstance(value, (list, np.ndarray)):
                    value = value[0]

                if method == "optimal":
                    label = f"Upper Bound: {value:.3f}"
                else:
                    label = f"Random: {value:.3f} ± {se_uplift_dict[training_size][method][next(iter(se_uplift_dict[training_size][method]))]:.3f}"
                color = color_legend.get(method, 'black')
                marker = marker_legend.get(method, 'o')
                ax.plot([], [], marker=marker, color=color, linestyle="None", label=label)
                continue

            avg_uplifts = [avg_uplift_dict[training_size][method][delta] for delta in delta_levels]
            se_uplifts = [se_uplift_dict[training_size][method][delta] for delta in delta_levels]

            avg_uplifts = np.array(avg_uplifts)
            se_uplifts = np.array(se_uplifts)

            # Extract visual styles from legends
            base_name = "-".join(method.split("-")[:2])
            # label = label_legend.get(base_name, method)
            label = label_legend_ablation.get(method.split('-')[0], method)
            # color = color_legend.get(base_name, 'black')
            color = color_legend.get(method.split('-')[0], 'solid') if method != "kmeans_q" else color_legend.get(method, method)
            linestyle = linestyle_legend.get(method.split('-')[0], 'solid')
            # linestyle = linestyle_legend.get(method.split('-')[1], 'solid') if method != "kmeans_q" else linestyle_legend.get(method, method)
            # marker = marker_legend.get(method.split('-')[1], 'o')
            marker = marker_legend.get(method.split('-')[0], 'o') if method != "kmeans_q" else marker_legend.get(method, method)
            # marker = marker_legend.get(model_specific, 'o') if method != "kmeans_q" else marker_legend.get(method, method)

            ax.plot(delta_levels, avg_uplifts, marker=marker, label=label,
                    color=color, linestyle=linestyle, linewidth=1, markersize=6)
            ax.fill_between(delta_levels,
                            avg_uplifts - se_uplifts,
                            avg_uplifts + se_uplifts,
                            alpha=0.2,
                            color=color)

        ax.set_xticks(delta_levels)

        ax.tick_params(axis='x', labelsize=font_ticks)  # set font size for x-axis ticks
        ax.tick_params(axis='y', labelsize=font_ticks)  # set font size for y-axis ticks

        # ✅ Only show x-axis label and ticks on bottom row
        if row == n_rows - 1:
            ax.set_xlabel(ax_x_title, fontsize=font_axes)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])

        # ✅ Show legend in every subplot
        # ax.legend(fontsize=9)
        # ✅ Only show legend in the last subplot
        if idx == n_learners - 1:
            ax.legend(fontsize=font_legend)

    # Remove any unused subplots
    for idx in range(len(learner_types), n_rows * n_cols):
        fig.delaxes(axes.flatten()[idx])

    # for ax_row in axes:
    #     for ax in ax_row:
    #         ax.tick_params(labelleft=True)


    # fig.suptitle(r"Performance across confounding levels for different $\mathit{learners}$", fontsize=font_title)
    # plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder,"learners.pdf"), bbox_inches="tight")
    plt.show()

def plot_results_delta_levels_by_model_grid(model_spec_dict_avg, model_spec_dict_se, delta_levels, training_size):
    """
    Plot average uplift by delta levels and methods, arranged by model type (xgb, rf, lstm, etc.).

    Parameters
    ----------
    model_spec_dict_avg : dict
        Nested dictionary: model_spec_dict_avg[model][training_size][method][delta].
    model_spec_dict_se : dict
        Nested dictionary: model_spec_dict_se[model][training_size][method][delta].
    delta_levels : list[float]
        Delta levels to plot.
    training_size : int
        Fixed training size to display.
    """

    # --- Group models
    models = list(model_spec_dict_avg.keys())
    n_models = len(models)

    # ✅ Keep same grid logic (max 3 columns)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), sharey=True)

    # Ensure axes is always a 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    
    # ax_x_title = "Confounding Level (Δ)"
    # ax_y_title = "Gain over Bank Policy"

    # --- Iterate over models
    for idx, model in enumerate(models):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        if model == "xgb":
            title = "XGBoost"
        elif model == "rf":
            title = "Random Forest"
        elif model == "lstm":
            title = "MLP/LSTM"
        else:
            title = f"{model.upper()} Model"

        ax.set_title(title, fontsize=font_sub_title)
        if col == 0:
            ax.set_ylabel(ax_y_title, fontsize=font_axes)
        ax.grid(True, color='gray', linestyle='-', linewidth=0.3, alpha=0.15)

        # --- Get model data
        if training_size not in model_spec_dict_avg[model]:
            print(f"⚠️ Skipping {model}: no data for training size {training_size}")
            continue

        avg_dict = model_spec_dict_avg[model][training_size]
        se_dict = model_spec_dict_se[model][training_size]

        # --- Plot each method
        for method in avg_dict.keys():
            if method in ["random", "optimal"]:
                value = avg_dict[method]
                if isinstance(value, dict):
                    value = list(value.values())[0]
                elif isinstance(value, (list, np.ndarray)):
                    value = value[0]

                if method == "optimal":
                    label = f"Upper Bound: {value:.3f}"
                else:
                    label = f"Random: {value:.3f} ± {se_dict[method][next(iter(se_dict[method]))]:.3f}"
                color = color_legend.get(method, 'black')
                marker = marker_legend.get(method, 'o')
                ax.plot([], [], marker=marker, color=color, linestyle="None", label=label)
                continue

            avg_uplifts = np.array([avg_dict[method][delta] for delta in delta_levels])
            se_uplifts = np.array([se_dict[method][delta] for delta in delta_levels])

            # Extract visual styles from legends
            base_name = "-".join(method.split("-")[:2])
            # label = label_legend.get(base_name, method)
            label = label_legend_ablation.get(method.split('-')[0], method)
            # color = color_legend.get(base_name, 'black')
            color = color_legend.get(method.split('-')[0], 'solid') if method != "kmeans_q" else color_legend.get(method, method)
            linestyle = linestyle_legend.get(method.split('-')[0], 'solid')
            # linestyle = linestyle_legend.get(method.split('-')[1], 'solid') if method != "kmeans_q" else linestyle_legend.get(method, method)
            # marker = marker_legend.get(method.split('-')[1], 'o')
            marker = marker_legend.get(method.split('-')[0], 'o') if method != "kmeans_q" else marker_legend.get(method, method)
            # marker = marker_legend.get(model, 'o') if method != "kmeans_q" else marker_legend.get(method, method)

            # --- Plot curve
            ax.plot(delta_levels, avg_uplifts, marker=marker, label=label,
                    color=color, linestyle=linestyle, linewidth=1.5, markersize=6)

            # --- Add SE shading
            ax.fill_between(delta_levels,
                            avg_uplifts - se_uplifts,
                            avg_uplifts + se_uplifts,
                            alpha=0.2,
                            color=color)

        # --- Format ticks and labels
        ax.set_xticks(delta_levels)
        ax.tick_params(axis='x', labelsize=font_ticks)
        ax.tick_params(axis='y', labelsize=font_ticks)

        if row == n_rows - 1:
            ax.set_xlabel(ax_x_title, fontsize=font_axes)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])

        if idx == n_models - 1:
            ax.legend(fontsize=font_legend)

    # --- Remove unused axes
    for idx in range(len(models), n_rows * n_cols):
        fig.delaxes(axes.flatten()[idx])

    # --- Title and layout
    # fig.suptitle(r"Performance across confounding levels for different $\mathit{base\ models}$", fontsize=font_title)
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder,"base_models.pdf"), bbox_inches="tight")
    plt.show()

def plot_results_delta_levels_by_stage_grid(
    avg_uplift_dict,
    se_uplift_dict,
    methods,
    n_stages_list,
    delta_levels,
    train_size=10000,
    model_specific="xgb"
):
    """
    Plot average uplift by delta levels and methods for multiple training sizes in a subplot grid.

    Parameters
    ----------
    avg_uplift_dict : dict
        Nested dictionary: avg_uplift_dict[stage][train_size][method][delta].
    se_uplift_dict : dict
        Nested dictionary: se_uplift_dict[stage][train_size][method][delta].
    methods : list[str]
        List of methods to plot.
    n_stages_list : list[int]
        List of stages for x-axis.
    delta_levels : list[float]
        List of delta values to create subplots.
    """

    num_deltas = len(delta_levels)

    # Determine subplot grid size
    if num_deltas == 3:
        rows, cols = 1, 3
    else:
        cols = min(3, num_deltas)
        rows = int(np.ceil(num_deltas / cols))

    # fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), sharey=True, squeeze=False)
    # fig, axes = plt.subplots(rows, cols,
    #                      figsize=(7.2 * cols, 5.5 * rows),
    #                      dpi=130,
    #                      sharey=True)

    for idx, delta in enumerate(delta_levels):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        for method in methods:
            avg_uplifts, se_uplifts = [], []

            for stage in n_stages_list:
                avg_val = avg_uplift_dict[stage][train_size][method][delta]
                se_val = se_uplift_dict[stage][train_size][method][delta]

                # ensure scalar
                if isinstance(avg_val, (list, np.ndarray)):
                    avg_val = np.mean(avg_val)
                if isinstance(se_val, (list, np.ndarray)):
                    se_val = np.mean(se_val)

                avg_uplifts.append(avg_val)
                se_uplifts.append(se_val)

            avg_uplifts = np.array(avg_uplifts)
            se_uplifts = np.array(se_uplifts)

            # Style selection (same logic as before)
            if method == 'random':
                label, color, linestyle, marker = "Random", 'lightgrey', 'dashed', 'd'
            elif method == 'optimal':
                label, color, linestyle, marker = "Upper Bound", 'grey', 'dashed', 'd'
            else:
                label = label_legend.get(method.split('-')[0] + '-' + method.split('-')[1], method) if method != "kmeans_q" else label_legend.get(method, method)
                color = color_legend.get(method.split('-')[0], 'black')
                linestyle = linestyle_legend.get(method.split('-')[0], 'solid')
                marker = marker_legend.get(method.split('-')[0], 'o')

            ax.plot(
                n_stages_list,
                avg_uplifts,
                marker=marker,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=1.5,
                markersize=6
            )

            ax.fill_between(
                n_stages_list,
                avg_uplifts - se_uplifts,
                avg_uplifts + se_uplifts,
                color=color,
                alpha=0.2
            )

        # Axis formatting
        if c == 0:
            ax.set_ylabel(ax_y_title, fontsize=font_axes)
        ax.set_xlabel("Number of decision points", fontsize=font_axes)
        ax.set_title(f"δ = {delta}", fontsize=font_title)
        ax.grid(True, color='gray', linestyle='-', linewidth=0.3, alpha=0.15)
        ax.tick_params(axis='x', labelsize=font_ticks)
        ax.tick_params(axis='y', labelsize=font_ticks)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Only add legend to top right subplot
        if idx == len(delta_levels) - 1:
            ax.legend(fontsize=font_legend, loc='lower right')

    # Hide any unused subplots if delta_levels doesn't fill the grid
    for j in range(num_deltas, rows * cols):
        fig.delaxes(axes[j // cols][j % cols])

    # fig.suptitle(r"Performance across confounding levels for different $\mathit{numbers\ of\ stages}$", fontsize=font_title)
    plt.tight_layout()
    if "S" in methods[0] and model_specific == "xgb":
        plt.savefig(os.path.join(save_folder,"numbers_of_decision_points.pdf"), bbox_inches="tight")
    else:
        learner = methods[0].split('-')[1]
        plt.savefig(os.path.join(save_folder,("numbers_of_decision_points" + "_" + learner + "_" + model_specific + ".pdf")), bbox_inches="tight")
    plt.show()