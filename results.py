# %%
# import pandas as pd
# import wandb

# api = wandb.Api()
# entity, project = "frozenwolf", "CL-ASR"
# runs = api.runs(entity + "/" + project)

# summary_list, config_list, name_list = [], [], []
# for run in runs:
#     # .summary contains output keys/values for
#     # metrics such as accuracy.
#     #  We call ._json_dict to omit large files
#     summary_list.append(run.summary._json_dict)

#     # .config contains the hyperparameters.
#     #  We remove special values that start with _.
#     config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

#     # .name is the human-readable name of the run.
#     name_list.append(run.name)
#     print(run.name)

# runs_df = pd.DataFrame(
#     {"summary": summary_list, "config": config_list, "name": name_list}
# )

# runs_df


# %%
import pandas as pd
import matplotlib.pyplot as plt

n_langs = 9

wandb_desc = pd.read_csv("~/Downloads/wandb_export_2025-05-18T22_08_33.973+05_30.csv")
wandb_desc = wandb_desc[wandb_desc['lang']>=n_langs]
wandb_desc = wandb_desc[wandb_desc['lang']<20]
wandb_desc = wandb_desc[wandb_desc['Tags'] != '1 gpu']

wandb_desc = wandb_desc[~wandb_desc['Name'].isin(['icy-tree-152', 'balmy-wave-159', 'different-sun-162', 'winter-totem-185', 'expert-moon-167', 'copper-pine-202', 'carbonite-admiral-188', 'volcanic-gorge-198'])]

# %%
def automated_notes(x):
    mode = str(x['epochs']) + " "
    if "naive" in x['Notes']:
        mode += 'naive'
    elif "lwf" in x['Notes']:
        mode += 'lwf'
        mode += f" kd: {x['cl_config.knowledge_distillation']}"
    elif "ewc" in x['Notes']:
        mode += 'ewc'
        mode += f" lambda: {x['cl_config.e_lambda']}"
    elif "mas" in x['Notes']:
        mode += 'mas'
        mode += f" mas_ctx: {x['cl_config.mas_ctx']}"
        
    return mode

def automated_epoch(x):
    ep = int(x['epochs'])
    return ep


def automated_cl_type(x):
    if "naive" in x['Notes']:
        mode = 'naive'
    elif "lwf" in x['Notes']:
        mode = 'lwf'
    elif "ewc" in x['Notes']:
        mode = 'ewc'
    elif "mas" in x['Notes']:
        mode = 'mas'
        
    return mode

wandb_desc['Notes'] = wandb_desc.apply(automated_notes, axis=1)

wandb_desc['CL_mode'] = wandb_desc.apply(automated_cl_type, axis=1)

wandb_desc['epochs'] = wandb_desc.apply(automated_epoch, axis=1)


# %%
wandb_desc.sort_values(['Notes'])
# wandb_desc[['Name', 'ID', 'Notes']]
wandb_desc['Notes'].value_counts()

# %%
##ignore run names = icy-tree-152, balmy-wave-159, different-sun-162, winter-totem-185, expert-moon-167, copper-pine-202

wandb_desc[['CL_mode', 'Notes', 'epochs', 'Tags', 'lang', 'Name']].sort_values(['CL_mode', 'epochs'])

# %%
wandb_notes = {}
wb_notes_list = []
wandb_notes_info = {}
### Name, Notes column
for i in range(len(wandb_desc)):
    name = wandb_desc.iloc[i]["Name"]
    notes = wandb_desc.iloc[i]["Notes"]
    tags = wandb_desc.iloc[i]["Tags"]
    if wandb_desc.iloc[i]["lang"] < n_langs:
        continue
    if tags == "2 gpu":
        continue
    if notes != "nan":
        wandb_notes[name] = notes.replace('"','')
        wb_notes_list.append(notes.replace('"',''))
        wandb_notes_info[name] = wandb_desc.iloc[i][["CL_mode", "epochs", "lang"]].to_dict()
        
        

wandb_notes, wandb_notes_info

# %%
import pandas as pd
import matplotlib.pyplot as plt

wandb_csv = pd.read_csv("~/Downloads/wandb_export_2025-05-18T22_09_44.261+05_30.csv")

# %%
wandb_csv

# %%
graphs = {}

for col in wandb_csv.columns:
    if col == "lang" or "__MIN" in col or "__MAX" in col:
        continue
    
    run_name = col.split(" - ")[0].rstrip().lstrip()
    
    if run_name not in wandb_notes:
        continue
    graph_name = col.split(" - ")[1].rstrip().lstrip()
    if run_name not in graphs:
        graphs[run_name] = {}
    
    graphs[run_name][graph_name] = wandb_csv[col]

# %%
run_name, graphs.keys()

# %%
for name in graphs:
    print(name, len(graphs[name]), wandb_notes[name], wandb_notes_info[name])


# %%
wandb_notes_info

cl_selected_runs = {"ewc": {}, "mas": {}, "lwf": {}, "naive": {}}
epoch_selected_runs = {1:{}, 2:{}, 5:{}, 10:{}}

for run_name in wandb_notes_info:
    cl_mode = wandb_notes_info[run_name]['CL_mode']
    cl_selected_runs[cl_mode][run_name] = wandb_notes[run_name]
    epoch = wandb_notes_info[run_name]['epochs']
    epoch_selected_runs[epoch][run_name] = wandb_notes[run_name]
        
cl_selected_runs

# default:
default = ['mas mas_ctx: 1.0', 
          'ewc lambda: 5.0', 
          'lwf kd: 0.1', 
          'naive']

def return_keys_in_list(wandb_notes_selected, keys):
    selected = {}
    for run_name in wandb_notes_selected:
        for k in keys:
            if k in wandb_notes_selected[run_name]:
                selected[run_name] = wandb_notes_selected[run_name]
                break
    return selected

# mas mas_ctx: 0.3
# lwf kd: 0.1
# ewc lambda: 5.0


return_keys_in_list(epoch_selected_runs[10], default)

# %%
epoch_selected_runs

# %%
temp = ['lwf kd: 0.1', 'naive']
return_keys_in_list(epoch_selected_runs[5], temp), list(return_keys_in_list(epoch_selected_runs[10], temp).keys())

# %%
from pprint import pprint

perf_metrics = [
"test/perf_<lang>_<mode>_avg_wer",
"test/perf_<lang>_<mode>_noisy_wer",
"test/perf_<lang>_<mode>_wer",
]

# modes = ["ctc", "rnnt"]


modes = [ "ctc"]


LANGUAGES = ['hindi','bengali','marathi','telugu','tamil','urdu','gujarati','kannada','odia','malayalam','punjabi','sanskrit'][:n_langs]

# ### subplot for each lang 1 column with two line plots (ctc, rnnt), 3 columns (avg, noisy, wer)
# plt.figure(figsize=(20, 60))

# run_name1 = 'misty-pond-198'
# run_name2 = 'tough-shape-176'

# for lang_idx, lang in enumerate(LANGUAGES):
#     for pidx, perf_metric in enumerate(perf_metrics):
#         plt.subplot(len(LANGUAGES), 3, lang_idx*3 + 1 + pidx)
        
#         #### spacing between rows
#         plt.subplots_adjust(hspace=0.5)
#         for mode_idx, mode in enumerate(modes):
#             perf_metric_ = perf_metric.replace("<lang>", lang).replace("<mode>", mode)
#             pf = "normal"
#             if "noisy" in perf_metric_:
#                 pf = "noisy"
#             elif "avg" in perf_metric_:
#                 pf = "average"
                
#             plt.title(f"{lang}-{pf}")
#             plt.ylabel("WER")
#             plt.xticks(range(len(LANGUAGES)), LANGUAGES, rotation=45)
#             plt.plot(range(len(LANGUAGES)), graphs[run_name1][perf_metric_][:n_langs], label=f"{wandb_notes[run_name1].split(' ')[1]}")
#             plt.scatter(range(len(LANGUAGES)), graphs[run_name1][perf_metric_][:n_langs])
#             plt.plot(range(len(LANGUAGES)), graphs[run_name2][perf_metric_][:n_langs], label=f"{wandb_notes[run_name2].split(' ')[1]}")
#             plt.scatter(range(len(LANGUAGES)), graphs[run_name2][perf_metric_][:n_langs])
#             plt.legend(loc='lower right')

# plt.show()

# %%
def plot_graph(wandb_notes_selected, mode, hide_runname=False, save=None):
    
    plt.figure(figsize=(20, 60))

    plt.title("WER vs Language\n\n")
    for lang_idx, lang in enumerate(LANGUAGES):
        for pidx, perf_metric in enumerate(perf_metrics):
            plt.subplot(len(LANGUAGES), 3, lang_idx*3 + 1 + pidx)
            #### spacing between rows
            plt.subplots_adjust(hspace=0.5)
            for run_idx, run_name in enumerate(list(wandb_notes_selected.keys())):
                perf_metric_ = perf_metric.replace("<lang>", lang).replace("<mode>", mode)
                if hide_runname:
                    plt.title(f"{lang} - {perf_metric_} - {wandb_notes_selected[run_name]}")
                else:
                    pm = "normal"
                    if "noisy" in perf_metric_:
                        pm = "noisy"
                    elif "avg" in perf_metric_:
                        pm = "average"
                    plt.title(f"{lang} - {pm}")
                plt.ylabel("WER")
                plt.xticks(range(len(LANGUAGES)), LANGUAGES, rotation=45)
                if perf_metric_ not in graphs[run_name]:
                    print(f"Skipping {run_name} - {perf_metric_}")
                    continue
                plt.plot(range(len(LANGUAGES)), graphs[run_name][perf_metric_][:n_langs], label=wandb_notes_selected[run_name])
                plt.scatter(range(len(LANGUAGES)), graphs[run_name][perf_metric_][:n_langs])
                plt.legend(loc='lower right')
                plt.grid(True)

    
    if save is not None:
        plt.tight_layout()
        plt.savefig(
    save,
    format="pdf",
    bbox_inches="tight",
    dpi=1200,  # max resolution (even though PDF is vector)
    transparent=True  # removes background for Overleaf consistency
)
    else:
        plt.tight_layout()
        plt.show()
        
    plt.close()

    
    
import matplotlib.pyplot as plt

def plot_graph_avg_nly(wandb_notes_selected, mode, save=None):
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle("Average WER per Language", fontsize=16)
    plt.subplots_adjust(hspace=0.4)

    for lang_idx, lang in enumerate(LANGUAGES):
        row, col = divmod(lang_idx, 3)
        ax = axes[row][col]

        for run_idx, run_name in enumerate(wandb_notes_selected):
            perf_metric = f"test/perf_{lang}_{mode}_avg_wer"
            if perf_metric not in graphs[run_name]:
                print(f"Skipping {run_name} - {perf_metric}")
                continue

            wer_vals = graphs[run_name][perf_metric][:n_langs]
            ax.plot(range(len(wer_vals)), wer_vals, label=wandb_notes_selected[run_name])
            ax.scatter(range(len(wer_vals)), wer_vals)

        ax.set_title(f"{lang}")
        ax.set_ylabel("WER")
        ax.set_xticks(range(len(LANGUAGES)))
        ax.set_xticklabels(LANGUAGES, rotation=45)
        ax.grid(True)
        ax.legend(loc='upper left', fontsize=8)

    if save is not None:
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(
            save,
            format="pdf",
            bbox_inches="tight",
            dpi=1200,
            transparent=True
        )
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    plt.close()




# %%
def calc_scores(wandb_notes_selected, mode, metric="avg"): # metric="avg", "", "noisy"
    bwt_scores = {}
    avg_scores = {}
    min_scores = {}
    max_scores = {}


    for run_name in wandb_notes_selected.keys():
        all_perf = []
        bwt_scores[run_name] = {}
        avg_scores[run_name] = {}
        min_scores[run_name] = {}
        max_scores[run_name] = {}
        for lang in LANGUAGES:
            perf_metric_ = f"test/perf_{lang}_{mode}_{metric}_wer".replace("__","_")  # adjust based on actual format
            if perf_metric_ not in graphs[run_name]:
                continue
            
            perf = graphs[run_name][perf_metric_][:n_langs]  # Length should be equal to len(LANGUAGES)
            all_perf.append(perf.tolist())

            if len(perf) != len(LANGUAGES):
                print(f"Skipping BWT for {run_name} - {perf_metric_}, length mismatch")
                continue
            
            bwt = 0
            avg = 0
            min_ = 1000
            max_ = 0
    
            c = 0
            for i in range(LANGUAGES.index(lang), len(LANGUAGES)):
                avg += perf[i]
                c+=1
                if perf[i] < min_:
                    min_ = perf[i]     
                if perf[i] > max_:
                    max_ = perf[i]



            avg_scores[run_name][lang] = avg/c
            min_scores[run_name][lang] = min_
            max_scores[run_name][lang] = max_

        
        for t, (lang, perf) in enumerate(zip(LANGUAGES, all_perf)):
            # print(t, lang, perf)
            bwt_scores[run_name][lang] = 0
            # print("LANG", lang, perf)
            for prev_lang in range(t):
                bwt_scores[run_name][lang] += all_perf[prev_lang][prev_lang] - all_perf[prev_lang][t]
                # print(all_perf[prev_lang][prev_lang], all_perf[prev_lang][t])
            bwt_scores[run_name][lang] /= max(t, 1)
            
            # print(lang, bwt_scores[run_name][lang])
            # print("---------")
            
    return bwt_scores, avg_scores, min_scores, max_scores



# %%

def rename_wandb_notes(wandb_notes_selected):
    for k,v in list(wandb_notes_selected.items()):
        del wandb_notes_selected[k]
        v = v.split(" ")[1]
        wandb_notes_selected[k] = v
        
    return wandb_notes_selected

def remove_epoch(wandb_notes_selected):
    for k,v in list(wandb_notes_selected.items()):
        del wandb_notes_selected[k]
        v = ' '.join(v.split(" ")[1:])
        wandb_notes_selected[k] = v
        
    return wandb_notes_selected

def only_epoch_cl(wandb_notes_selected):
    for k,v in list(wandb_notes_selected.items()):
        del wandb_notes_selected[k]
        v = ' '.join(v.split(" ")[:2])
        wandb_notes_selected[k] = v
        
    return wandb_notes_selected




import numpy as np
import matplotlib.pyplot as plt

def updated_plot_stats(wandb_notes_selected, mode, metrics = ["avg", "", "noisy"], save=None):
    import os
    if save is not None:
        if not os.path.exists(save):
            os.mkdir(save)
        
    metric_titles = {"avg": "Avg", "": "Normal", "noisy": "Noisy"}

    langs = list(next(iter(calc_scores(wandb_notes_selected, mode, "avg")[1].values())).keys())
    x = np.arange(len(langs))
    print(metrics)
    # --- 1. WER Line Plot ---
    fig, axs = plt.subplots(1, len(metrics), figsize=(10, 8//len(metrics)), sharey=True)
    if len(metrics) == 1:
        axs = [axs]
    for col_idx, metric in enumerate(metrics):
        _, avg_scores, _, _ = calc_scores(wandb_notes_selected, mode, metric)
        ax = axs[col_idx]
        for run_name, desc in wandb_notes_selected.items():
            scores = [avg_scores[run_name][lang] for lang in langs]
            ax.plot(langs, scores, marker='o', label=desc)
        ax.set_title(f"{metric_titles[metric]} WER")
        ax.set_xlabel("Language")
        ax.set_ylabel("WER (Lower = Better)")
        ax.tick_params(labelleft=True)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)
        if col_idx == 0:
            ax.legend()
    plt.suptitle("WER", fontsize=14)
    plt.tight_layout()
    
    if save is not None:
        plt.savefig(
            os.path.join(save, "wer_line_plot.pdf"),
            format="pdf",
            bbox_inches="tight",
            dpi=1200,  # max resolution (even though PDF is vector)
            transparent=True  # removes background for Overleaf consistency
        )
    else:
        plt.show()
        
    plt.close()

    # --- 2. Shaded Min/Max Area Plot ---
    if len(metrics) == 1:
        axs = [axs]
    fig, axs = plt.subplots(1, len(metrics), figsize=(10, 8//len(metrics)), sharey=True)
    if len(metrics) == 1:
        axs = [axs]
    for col_idx, metric in enumerate(metrics):
        _, avg_scores, min_scores, max_scores = calc_scores(wandb_notes_selected, mode, metric)
        ax = axs[col_idx]
        for run_name, desc in wandb_notes_selected.items():
            avg_vals = [avg_scores[run_name][lang] for lang in langs]
            min_vals = [min_scores[run_name][lang] for lang in langs]
            max_vals = [max_scores[run_name][lang] for lang in langs]
            ax.plot(x, avg_vals, marker='o', label=desc)
            ax.fill_between(x, min_vals, max_vals, alpha=0.2)
        ax.set_title(f"{metric_titles[metric]} WER with Shading")
        ax.set_xlabel("Language")
        ax.set_xticks(x)
        ax.tick_params(labelleft=True)
        ax.set_xticklabels(langs, rotation=45)
        ax.grid(True)
        if col_idx == 0:
            ax.set_ylabel("WER")
            ax.legend()
    plt.suptitle("WER Min/Max", fontsize=14)
    plt.tight_layout()
    if save is not None:
        plt.savefig(
            os.path.join(save, "wer_shaded_plot.pdf"),
            format="pdf",
            bbox_inches="tight",
            dpi=1200,  # max resolution (even though PDF is vector)
            transparent=True  # removes background for Overleaf consistency
        )
    else:
        plt.show()

    plt.close()

    # --- 3. Error Bar Plot ---
    if len(metrics) == 1:
        axs = [axs]
    fig, axs = plt.subplots(1, len(metrics), figsize=(10, 8//len(metrics)), sharey=True)
    if len(metrics) == 1:
        axs = [axs]
    for col_idx, metric in enumerate(metrics):
        _, avg_scores, min_scores, max_scores = calc_scores(wandb_notes_selected, mode, metric)
        ax = axs[col_idx]
        for i, (run_name, desc) in enumerate(wandb_notes_selected.items()):
            avg_vals = np.array([avg_scores[run_name][lang] for lang in langs])
            min_vals = np.array([min_scores[run_name][lang] for lang in langs])
            max_vals = np.array([max_scores[run_name][lang] for lang in langs])
            lower = avg_vals - min_vals
            upper = max_vals - avg_vals
            ax.errorbar(x + i * 0.1, avg_vals, yerr=[lower, upper], fmt='o-', capsize=5, label=desc)
        ax.set_title(f"{metric_titles[metric]} WER Error Bars")
        ax.set_xlabel("Language")
        ax.set_xticks(x)
        ax.tick_params(labelleft=True)
        ax.set_xticklabels(langs, rotation=45)
        ax.grid(True)
        if col_idx == 0:
            ax.set_ylabel("WER")
            ax.legend()
    plt.suptitle("WER Min–Avg–Max", fontsize=14)
    plt.tight_layout()
    if save is not None:
        plt.savefig(
            os.path.join(save, "wer_error_bars_plot.pdf"),
            format="pdf",
            bbox_inches="tight",
            dpi=1200,  # max resolution (even though PDF is vector)
            transparent=True  # removes background for Overleaf consistency
        )
    else:
        plt.show()

    plt.close()

    # --- 4. BWT Plot ---
    if len(metrics) == 1:
        axs = [axs]
    fig, axs = plt.subplots(1, len(metrics), figsize=(10, 8//len(metrics)), sharey=True)
    if len(metrics) == 1:
        axs = [axs]
    for col_idx, metric in enumerate(metrics):
        bwt_scores, _, _, _ = calc_scores(wandb_notes_selected, mode, metric)
        ax = axs[col_idx]
        for run_name, desc in wandb_notes_selected.items():
            scores = [bwt_scores[run_name][lang] for lang in langs]
            ax.plot(x, scores, marker='o', label=desc)
        ax.set_title(f"{metric_titles[metric]} BWT")
        ax.set_xlabel("Language")
        ax.set_xticks(x)
        ax.tick_params(labelleft=True)
        ax.set_xticklabels(langs, rotation=45)
        ax.grid(True)
        if col_idx == 0:
            ax.set_ylabel("BWT")
            ax.legend()
    plt.suptitle("Backward Transfer (BWT)", fontsize=14)
    plt.tight_layout()
    if save is not None:
        plt.savefig(
            os.path.join(save, "bwt_plot.pdf"),
            format="pdf",
            bbox_inches="tight",
            dpi=1200,  # max resolution (even though PDF is vector)
            transparent=True  # removes background for Overleaf consistency
        )
    else:
        plt.show()

    plt.close()

    # --- 5. Box Plots: WER per Segment, Grouped by Run, with Different Colors ---
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    fig, axs = plt.subplots(1, len(metrics), figsize=(10, 8//len(metrics)))
    if len(metrics) == 1:
        axs = [axs]
    for col_idx, metric in enumerate(metrics):
        _, avg_scores, _, _ = calc_scores(wandb_notes_selected, mode, metric)
        ax = axs[col_idx]

        n_langs = len(langs)
        segments = [n_langs // 3, 2 * n_langs // 3, n_langs]
        segment_labels = [f"{seg}" for seg in segments]
        run_names = list(wandb_notes_selected.keys())
        num_runs = len(run_names)

        # Generate unique colors for each run
        color_map = cm.get_cmap('tab10', num_runs)
        run_colors = [mcolors.to_hex(color_map(i)) for i in range(num_runs)]

        width = 0.2
        positions = []
        all_data = []
        box_colors = []

        for seg_idx, seg in enumerate(segments):
            for run_idx, run_name in enumerate(run_names):
                wer_values = [avg_scores[run_name][lang] for lang in langs[:seg]]
                all_data.append(wer_values)
                positions.append(seg_idx * (num_runs + 1) + run_idx)
                box_colors.append(run_colors[run_idx])

        # Create boxplot
        box = ax.boxplot(
            all_data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showmeans=True,
            boxprops=dict(color='black'),
            meanprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black'),
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            flierprops=dict(marker='x', color='gray', alpha=0.5),
        )

        # Apply colors per box
        for patch, color in zip(box['boxes'], box_colors):
            patch.set_facecolor(color)

        # X-axis labels at group centers
        group_centers = [(i * (num_runs + 1) + (num_runs - 1) / 2) for i in range(len(segments))]
        ax.set_xticks(group_centers)
        ax.set_xticklabels(segment_labels)
        ax.set_title(f"{metric_titles[metric]} WER Box Plot")
        ax.set_xlabel("Languages")
        ax.set_ylabel("WER")
        ax.grid(True)

        # Legend for runs
        legend_handles = [
            plt.Line2D([], [], color=run_colors[i], marker='s', linestyle='None', label=wandb_notes_selected[run])
            for i, run in enumerate(run_names)
        ]
        ax.legend(legend_handles, [wandb_notes_selected[run] for run in run_names], loc='lower left')

    plt.suptitle("WER Box Plot", fontsize=14)
    plt.tight_layout()
    if save is not None:
        plt.savefig(
            os.path.join(save, "wer_box_plot.pdf"),
            format="pdf",
            bbox_inches="tight",
            dpi=1200,  # max resolution (even though PDF is vector)
            transparent=True  # removes background for Overleaf consistency
        )
    else:
        plt.show()
        
    plt.close()




    
 

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def enforce_shared_ylim(axs):
    axs_iter = axs if isinstance(axs, (list, np.ndarray)) else [axs]
    ymins, ymaxs = [], []
    for ax in axs_iter:
        ymin, ymax = ax.get_ylim()
        ymins.append(ymin)
        ymaxs.append(ymax)
    common_ymin = min(ymins)
    common_ymax = max(ymaxs)
    for ax in axs_iter:
        ax.set_ylim(common_ymin, common_ymax)

def updated_plot_stats_multi(wandb_notes_selected, mode, save=None):
    import os
    if save is not None:
        if not os.path.exists(save):
            os.mkdir(save)
    metrics = ["", "noisy"]
    metric_titles = {"": "Normal", "noisy": "Noisy"}
    run_names = list(wandb_notes_selected.keys())
    num_runs = len(run_names)
    langs = list(calc_scores(wandb_notes_selected, mode, "avg")[1][run_names[0]].keys())
    x = np.arange(len(langs))

    # --- 1. WER Line Plot ---
    if num_runs  == 4:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
        axs = axs.flatten()
    else:
        fig, axs = plt.subplots(1, num_runs, figsize=(6 * num_runs, 5), sharey=True)
    for idx, run_name in enumerate(run_names):
        ax = axs[idx] if num_runs > 1 else axs
        desc = wandb_notes_selected[run_name]
        for metric in metrics:
            _, avg_scores, _, _ = calc_scores({run_name: desc}, mode, metric)
            scores = [avg_scores[run_name][lang] for lang in langs]
            ax.plot(langs, scores, marker='o', label=metric_titles[metric])
        ax.set_title(desc)
        ax.set_xlabel("Language")
        ax.tick_params(labelleft=True)
        ax.set_ylabel("WER" if idx == 0 else "")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)
        ax.legend()
    enforce_shared_ylim(axs)
    fig.suptitle("WER (Normal vs Noisy)")
    plt.tight_layout()
    if save is not None:
        plt.savefig(
            os.path.join(save, "wer_line_plot.pdf"),
            format="pdf",
            bbox_inches="tight",
            dpi=1200,  # max resolution (even though PDF is vector)
            transparent=True  # removes background for Overleaf consistency
        )
    else:
        plt.show()
        
    plt.close()

    # --- 2. Shaded Min/Max Plot ---
    if num_runs  == 4:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
        axs = axs.flatten()
    else:
        fig, axs = plt.subplots(1, num_runs, figsize=(6 * num_runs, 5), sharey=True)
    for idx, run_name in enumerate(run_names):
        ax = axs[idx] if num_runs > 1 else axs
        desc = wandb_notes_selected[run_name]
        for metric in metrics:
            _, avg_scores, min_scores, max_scores = calc_scores({run_name: desc}, mode, metric)
            avg_vals = [avg_scores[run_name][lang] for lang in langs]
            min_vals = [min_scores[run_name][lang] for lang in langs]
            max_vals = [max_scores[run_name][lang] for lang in langs]
            ax.plot(x, avg_vals, marker='o', label=metric_titles[metric])
            ax.fill_between(x, min_vals, max_vals, alpha=0.2)
        ax.set_title(desc)
        ax.set_xlabel("Language")
        ax.set_ylabel("WER" if idx == 0 else "")
        ax.tick_params(labelleft=True)
        ax.set_xticks(x)
        ax.set_xticklabels(langs, rotation=45)
        ax.grid(True)
        ax.legend()
    enforce_shared_ylim(axs)
    fig.suptitle("WER Min/Max (Normal vs Noisy)")
    plt.tight_layout()
    if save is not None:
        plt.savefig(
            os.path.join(save, "wer_shaded_plot.pdf"),
            format="pdf",
            bbox_inches="tight",
            dpi=1200,  # max resolution (even though PDF is vector)
            transparent=True  # removes background for Overleaf consistency
        )
    else:
        plt.show()
        
    plt.close()

    # --- 3. Error Bar Plot ---
    if num_runs  == 4:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
        axs = axs.flatten()
    else:
        fig, axs = plt.subplots(1, num_runs, figsize=(6 * num_runs, 5), sharey=True)
    for idx, run_name in enumerate(run_names):
        ax = axs[idx] if num_runs > 1 else axs
        desc = wandb_notes_selected[run_name]
        for i, metric in enumerate(metrics):
            _, avg_scores, min_scores, max_scores = calc_scores({run_name: desc}, mode, metric)
            avg_vals = np.array([avg_scores[run_name][lang] for lang in langs])
            min_vals = np.array([min_scores[run_name][lang] for lang in langs])
            max_vals = np.array([max_scores[run_name][lang] for lang in langs])
            lower = avg_vals - min_vals
            upper = max_vals - avg_vals
            ax.errorbar(x + i * 0.1, avg_vals, yerr=[lower, upper], fmt='o-', capsize=5, label=metric_titles[metric])
        ax.set_title(desc)
        ax.set_xlabel("Language")
        ax.set_ylabel("WER" if idx == 0 else "")
        ax.tick_params(labelleft=True)
        ax.set_xticks(x)
        ax.set_xticklabels(langs, rotation=45)
        ax.grid(True)
        ax.legend()
    enforce_shared_ylim(axs)
    fig.suptitle("WER (Normal vs Noisy)")
    plt.tight_layout()
    if save is not None:
        plt.savefig(
            os.path.join(save, "wer_error_bars_plot.pdf"),
            format="pdf",
            bbox_inches="tight",
            dpi=1200,  # max resolution (even though PDF is vector)
            transparent=True  # removes background for Overleaf consistency
        )
    else:
        plt.show()
        
    plt.close()

    # --- 4. BWT Plot ---
    if num_runs  == 4:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
        axs = axs.flatten()
    else:
        fig, axs = plt.subplots(1, num_runs, figsize=(6 * num_runs, 5), sharey=True)
    for idx, run_name in enumerate(run_names):
        ax = axs[idx] if num_runs > 1 else axs
        desc = wandb_notes_selected[run_name]
        for metric in metrics:
            bwt_scores, _, _, _ = calc_scores({run_name: desc}, mode, metric)
            scores = [bwt_scores[run_name][lang] for lang in langs]
            ax.plot(x, scores, marker='o', label=metric_titles[metric])
        ax.set_title(desc)
        ax.set_xlabel("Language")
        ax.set_ylabel("BWT" if idx == 0 else "")
        ax.set_xticks(x)
        ax.tick_params(labelleft=True)
        ax.set_xticklabels(langs, rotation=45)
        ax.grid(True)
        ax.legend()
    enforce_shared_ylim(axs)
    fig.suptitle("BWT (Normal vs Noisy)")
    plt.tight_layout()
    
    if save is not None:
        plt.savefig(
            os.path.join(save, "bwt_plot.pdf"),
            format="pdf",
            bbox_inches="tight",
            dpi=1200,  # max resolution (even though PDF is vector)
            transparent=True  # removes background for Overleaf consistency
        )
    else:
        plt.show()
        
    plt.close()

    # --- 5. Box Plot per Segment ---
    if num_runs  == 4:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
        axs = axs.flatten()
    else:
        fig, axs = plt.subplots(1, num_runs, figsize=(6 * num_runs, 5), sharey=True)
    color_map = cm.get_cmap('tab10', len(metrics))
    box_colors = [mcolors.to_hex(color_map(i)) for i in range(len(metrics))]
    segments = [len(langs) // 3, 2 * len(langs) // 3, len(langs)]
    segment_labels = [f"{seg}" for seg in segments]

    for idx, run_name in enumerate(run_names):
        ax = axs[idx] if num_runs > 1 else axs
        desc = wandb_notes_selected[run_name]
        all_data = []
        positions = []
        color_list = []
        for seg_idx, seg in enumerate(segments):
            for metric_idx, metric in enumerate(metrics):
                _, avg_scores, _, _ = calc_scores({run_name: desc}, mode, metric)
                wer_values = [avg_scores[run_name][lang] for lang in langs[:seg]]
                all_data.append(wer_values)
                positions.append(seg_idx * (len(metrics) + 1) + metric_idx)
                color_list.append(box_colors[metric_idx])
        box = ax.boxplot(
            all_data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showmeans=True,
            boxprops=dict(color='black'),
            meanprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black'),
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            flierprops=dict(marker='x', color='gray', alpha=0.5),
        )
        for patch, color in zip(box['boxes'], color_list):
            patch.set_facecolor(color)

        centers = [(i * (len(metrics) + 1) + (len(metrics) - 1) / 2) for i in range(len(segments))]
        ax.set_xticks(centers)
        ax.set_xticklabels(segment_labels)
        ax.set_title(desc)
        ax.tick_params(labelleft=True)
        ax.set_xlabel("Languages")
        ax.set_ylabel("WER" if idx == 0 else "")
        ax.grid(True)
        handles = [
            plt.Line2D([], [], color=box_colors[i], marker='s', linestyle='None', label=metric_titles[metrics[i]])
            for i in range(len(metrics))
        ]
        ax.legend(handles=handles)
    enforce_shared_ylim(axs)
    fig.suptitle("WER Box Plot (Normal vs Noisy)")
    plt.tight_layout()
    if save is not None:
        plt.savefig(
            os.path.join(save, "wer_box_plot.pdf"),
            format="pdf",
            bbox_inches="tight",
            dpi=1200,  # max resolution (even though PDF is vector)
            transparent=True  # removes background for Overleaf consistency
        )
    else:
        plt.show()

    plt.close()


import matplotlib.pyplot as plt

# Assume LANGUAGES and n_langs are defined
LANGUAGES = ['hindi','bengali','marathi','telugu','tamil','urdu','gujarati','kannada','odia','malayalam','punjabi','sanskrit'][:n_langs]
modes = ["ctc", "rnnt"]

# Run names and their labels
run_name1 = 'misty-pond-198'
run_name2 = 'tough-shape-176'

fig, axes = plt.subplots(3, 3, figsize=(18, 12))
fig.suptitle("Average WER per Language (CTC vs RNN-T)", fontsize=16)
plt.subplots_adjust(hspace=0.4)

for lang_idx, lang in enumerate(LANGUAGES):
    row, col = divmod(lang_idx, 3)
    ax = axes[row][col]
    
    for mode in modes:
        perf_metric = f"test/perf_{lang}_{mode}_avg_wer"

        if perf_metric not in graphs[run_name1] or perf_metric not in graphs[run_name2]:
            print(f"Skipping {lang} - {mode}")
            continue

        # Plot for run 1
        vals1 = graphs[run_name1][perf_metric][:n_langs]
        ax.plot(range(len(vals1)), vals1, label=f"{wandb_notes[run_name1].split(' ')[1]} - {mode}")
        ax.scatter(range(len(vals1)), vals1)

        # Plot for run 2
        vals2 = graphs[run_name2][perf_metric][:n_langs]
        ax.plot(range(len(vals2)), vals2, label=f"{wandb_notes[run_name2].split(' ')[1]} - {mode}")
        ax.scatter(range(len(vals2)), vals2)

    ax.set_title(f"{lang}")
    ax.set_ylabel("WER")
    ax.set_xticks(range(len(LANGUAGES)))
    ax.set_xticklabels(LANGUAGES, rotation=45)
    ax.grid(True)
    ax.legend(loc='upper left', fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(
        "rnnt_ctc_lwf_naive.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=1200,  # max resolution (even though PDF is vector)
        transparent=True  # removes background for Overleaf consistency
    )
plt.close()



# wandb_notes_selected = cl_selected_runs["ewc"]
# Median (black line)

# Mean (red circle)

# Whiskers extending to min/max within 1.5 IQR

# Outliers, if any, marked as gray



wandb_notes_selected = return_keys_in_list(epoch_selected_runs[5], default)
mode = "ctc"
wandb_notes_selected = rename_wandb_notes(wandb_notes_selected)
### WER vs Language
plot_graph_avg_nly(wandb_notes_selected, mode, save="avg_nly_wer_vs_lang.pdf")


wandb_notes_selected = return_keys_in_list(epoch_selected_runs[5], default)
mode = "ctc"
wandb_notes_selected = rename_wandb_notes(wandb_notes_selected)
### WER vs Language
plot_graph(wandb_notes_selected, mode, save="wer_vs_lang.pdf")

updated_plot_stats(wandb_notes_selected, "rnnt", metrics = ["avg"], save="rnnt_benchmark")
updated_plot_stats(wandb_notes_selected, "ctc", metrics = ["avg"], save="ctc_benchmark")



temp = ["lwf kd: 0.1",  "naive"]
run_name = return_keys_in_list(epoch_selected_runs[5], temp)
run_name = rename_wandb_notes(run_name)
#### Normal vs Noisy graphs for lwf and naive
updated_plot_stats_multi(run_name, mode, save="lwf_naive_normal_noisy")

temp = ["ewc", "mas", "lwf kd: 0.1",  "naive"]
run_name = return_keys_in_list(epoch_selected_runs[5], default)
run_name = rename_wandb_notes(run_name)
#### Normal vs Noisy graphs for ewc, mas, lwf and naive
updated_plot_stats_multi(run_name, mode, save="all_comparison_noisy")


temp = ["mas",  "naive"]
wandb_notes_selected = return_keys_in_list(epoch_selected_runs[5], temp)
wandb_notes_selected = remove_epoch(wandb_notes_selected)
#### MAS ablation for different config
updated_plot_stats(wandb_notes_selected, mode, metrics = ["avg",], save="mas_ablation")

temp = ["lwf",  "naive"]
wandb_notes_selected = return_keys_in_list(epoch_selected_runs[5], temp)
wandb_notes_selected = remove_epoch(wandb_notes_selected)
#### lwf ablation for different config
updated_plot_stats(wandb_notes_selected, mode, metrics = ["avg",], save="lwf_ablation")

temp = ["ewc",  "naive"]
wandb_notes_selected = return_keys_in_list(epoch_selected_runs[5], temp)
wandb_notes_selected = remove_epoch(wandb_notes_selected)
#### ewc ablation for different config
updated_plot_stats(wandb_notes_selected, mode, metrics = ["avg",], save="ewc_ablation")


# %%
wandb_notes_selected = return_keys_in_list(epoch_selected_runs[5], default)
wandb_notes_selected = rename_wandb_notes(wandb_notes_selected)
#### lwf vs ewc vs mas vs naive
updated_plot_stats(wandb_notes_selected, mode, metrics = ["avg"], save="lwf_vs_ewc_vs_mas_vs_naive")


temp = ['lwf kd: 0.1']
wandb_notes_selected = return_keys_in_list(wandb_notes, temp)
wandb_notes_selected = only_epoch_cl(wandb_notes_selected)
#### LwF epoch vs WER
updated_plot_stats(wandb_notes_selected, mode, metrics = ["avg",], save="lwf_epoch_vs_wer")


temp = ['ewc lambda: 5.0']
wandb_notes_selected = return_keys_in_list(wandb_notes, temp)
wandb_notes_selected = only_epoch_cl(wandb_notes_selected)
#### EWC epoch vs WER
updated_plot_stats(wandb_notes_selected, mode, metrics = ["avg",], save="ewc_epoch_vs_wer")


temp = ['mas mas_ctx: 0.3']
wandb_notes_selected = return_keys_in_list(wandb_notes, temp)
wandb_notes_selected = only_epoch_cl(wandb_notes_selected)
#### MAS epoch vs WER
updated_plot_stats(wandb_notes_selected, mode, metrics = ["avg",], save="mas_epoch_vs_wer")


temp = ['naive']
wandb_notes_selected = return_keys_in_list(wandb_notes, temp)
wandb_notes_selected = only_epoch_cl(wandb_notes_selected)
#### naive epoch vs WER
updated_plot_stats(wandb_notes_selected, mode, metrics = ["avg",], save="naive_epoch_vs_wer")



