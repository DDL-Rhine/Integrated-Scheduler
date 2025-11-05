# python exp/main/plot_workload.py
import json
import matplotlib.pyplot
import pandas
import seaborn
import matplotlib.pyplot as plt 
import numpy as np 

def init_plot(): 
    # seaborn.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['legend.labelspacing'] = 0.4
    # matplotlib.rcParams['legend.columnspacing'] = -5
    fig, ax = matplotlib.pyplot.subplots()
    ax.grid(linestyle='-', linewidth=1, alpha=0.5)
    fig.set_size_inches(w=6, h=4)
    return fig, ax 

fontsize = 24
linewidth=6 
markersize=24

total_policy_list =  [ "optimus", "optimus-FreezeOut", "simple_pollux", "simple_pollux-FreezeOut", "pollux", "icefrog", "simple_icefrog", 'simple_icefrog-batch-fixed', 'lucid', 'lucid-FreezeOut']
# total_policy_list = ["optimus-FreezeOut",  "simple_icefrog-batch-fixed"]
def transform_policy_order(policy): 
    return 'policy' + str(total_policy_list.index(policy))

# policy_list = [ "optimus-FreezeOut", "simple_pollux", "simple_icefrog"]
# policy_list = [ "optimus", "simple_pollux", "simple_icefrog"]
# policy_list = [ "optimus", "optimus-FreezeOut", "simple_icefrog-batch-fixed", "simple_icefrog"]
# policy_list = ["lucid-FreezeOut",  "optimus-FreezeOut", "simple_pollux-FreezeOut", "simple_icefrog-batch-fixed"]
policy_list = ["lucid",  "optimus-FreezeOut", "simple_pollux-FreezeOut", "simple_icefrog-batch-fixed"]
legend_policy_list = [transform_policy_order(policy) for policy in policy_list]
 
# policy_list = ["simple_icefrog", "simple_pollux", "pollux"]
if False: # 'Freeze' in policy_list[0]: 
    policy_labels=["Optimus+", "Pollux+", "IceFrog"]
    policy_labels=["Optimus+", "Pollux+", "IceFrog", "IceFrog+"]
else: 
    policy_labels=["Lucid", "Optimus", "Pollux", "IceFrog"]

# policy_list = ["optimus", "tiresias"]
# jobload_list = [0.5, 1.0, 1.5, 2.0]
# jobload_list = [1.0, 2.0, 4.0, 6.0]
# jobload_list = [0.5, 1.0, 2.0, 4.0] # , 4.0, 6.0]
jobload_list = [1.0, 2.0, 3.0, 4.0]
# y_list = [1, 2, 3, 4]
# y_list = [0.5, 1, 2]
y_list = [0, 2, 4, 6, 8]

root = '0_results/main/'
records = []
for policy in policy_list:
    for jobload in jobload_list:
        with open("{}/workload-{}/{}/summary.json".format(root, jobload, policy)) as f:
            summary = json.load(f)
        for workload, jcts in summary["jcts"].items():
            path = f"./workloads-1.0/{workload}.csv" if jobload == 1.0 else f"./workloads-{jobload}/{workload}.csv"
            df = pandas.read_csv(path)
            # print(path)
            # print(policy, jobload)
            # for row in df.itertuples(): 
            #     print(row.name)
            #     print(row.time, jcts[row.name])
            jct = sum(jcts.values()) / len(jcts)
            print('length ', len(jcts), len(df), policy, workload)
            # print()
            # makespan = max(row.time + jcts[row.name] for row in df.itertuples())
            
            # if jobload == 1.0 and 'lucid' in policy: 
            #     jct = jct / 3
            #     print(policy, jct)
                
            records.append({
                "workload": workload,
                "policy": transform_policy_order(policy),
                "jobload": jobload_list.index(jobload) + 1,
                "jct": jct,
                # "makespan": max(row.time + jcts[row.name] for row in df.itertuples()),
            })
            # print(path, sum(jcts.values()) / len(jcts),  max(row.time + jcts[row.name] for row in df.itertuples()))
# import pdb; pdb.set_trace() 
df = pandas.DataFrame.from_records(records)

df = df.groupby(["policy", "jobload", "workload"]).mean().reset_index()
# import pdb; pdb.set_trace() 
# for workload in ['0.5', '1.0', '1.5', '2.0']: 
# for workload in [str(load) for load in jobload_list]: 
for workload in [1, 2, 3, 4]:
    workload_df = df[df.jobload == workload]
    lucid_policy = legend_policy_list[0]
    optimus_policy = legend_policy_list[1]
    pollux_policy = legend_policy_list[2]
    icefrog_policy = legend_policy_list[3]
    
    optimus_jct = workload_df[workload_df.policy == optimus_policy].jct.mean()
    icefrog_jct = workload_df[workload_df.policy == icefrog_policy].jct.mean()
    pollux_jct = workload_df[workload_df.policy == pollux_policy].jct.mean()
    lucid_jct = workload_df[workload_df.policy == lucid_policy].jct.mean()
    print('workload {}, optimus accelerate {}'.format(workload,  optimus_jct / icefrog_jct))
    print('workload {}, pollux accelerate {}'.format(workload, pollux_jct / icefrog_jct))
    print('workload {}, lucid accelerate {}'.format(workload, lucid_jct / icefrog_jct))
    # print('workload {}, jct accelerate {}'.format(workload, icefrog_jct))
    
# import pdb; pdb.set_trace() 
    


base_hour=3600
df.jct /= base_hour
# df.makespan /= base_hour


fig, ax1 = init_plot() 

if False: 
    jct_info = dict() 
    for jobload in df.jobload.values: 
        if jobload not in jct_info: 
            jct_info[jobload] = dict() 
        for policy in df.policy.values:
            if policy not in jct_info[jobload]: 
                jct_info[jobload][policy] = 0
    for jobload, jct, policy in zip(df.jobload, df.jct, df.policy): 
        jct_info[jobload][policy] += jct


    for jobload in np.unique(df.jobload.values): 
        base = jct_info[jobload]['icefrog']
        # print('compared to optimus {}'.format(jct_info[jobload]['optimus'] / base))
        # print('compared to pollux {}'.format(jct_info[jobload]['pollux'] / base))
        print('compared to optimus {}'.format(1 - base / jct_info[jobload]['optimus']))
        print('compared to pollux {}'.format(1 - base / jct_info[jobload]['pollux']))
    exit(0)
custom_colors = ["tab:red", "tab:blue", "tab:orange", "tab:green"]
# seaborn.lineplot(x=df.jobload, y=df.jct, linewidth=linewidth, hue=df.policy, hue_order=legend_policy_list, ci=95, ax=ax1, legend=False)
seaborn.lineplot(x=df.jobload, y=df.jct, linewidth=linewidth, hue=df.policy, palette=custom_colors, hue_order=legend_policy_list, ci=95, ax=ax1, legend=False)
ax1.set_ylim(0, 10)
markersize = markersize/4*3
# import pdb; pdb.set_trace()
# ax1.add_legend(legend_order=legend_policy_list)
ax1.lines[1].set_marker("o")
ax1.lines[1].set_markersize(markersize)
ax1.lines[1].set_linestyle("-")


ax1.lines[2].set_marker("v")
ax1.lines[2].set_markersize(markersize)
ax1.lines[2].set_linestyle("--")

ax1.lines[3].set_marker("X")
ax1.lines[3].set_markersize(markersize)
ax1.lines[3].set_linestyle(":")

ax1.lines[0].set_marker("s")
ax1.lines[0].set_markersize(markersize)
ax1.lines[0].set_linestyle("-.")


# ax1.set_xticks([0.5, 1.0, 1.5, 2.0])
ax1.set_xticks([1, 2, 3, 4])
ax1.set_xticklabels(["0.5x", "1.0x", "1.5x", "2.0x"], fontsize=fontsize)
# ax1.set_xticklabels([f"{load}x" for load in jobload_list], fontsize=fontsize)
# ax1.set_xticks([0.5])
# ax1.set_xticklabels(["0.5x"])

# y_list = [0, 1, 2]
# y_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# ax1.set_yscale('log', base=1)
y_list = [0, 1, 2, 3, 4, 5]
# import pdb; pdb.set_trace() 
ax1.set_yticks(y_list)
ax1.set_yticklabels([str(y) for y in y_list], fontsize=fontsize)

ax1.set_ylim(0, max(y_list))
# import pdb; pdb.set_trace() 

# ax1.set_xlabel("Relative Job Load", fontsize=fontsize)
ax1.set_xlabel("", fontsize=0)
ax1.set_ylabel("Avg JCT (hours)", fontsize=fontsize)


# fig.legend(handles=ax1.lines, labels=policy_labels,
#            fontsize=9, loc=5, bbox_to_anchor=(0.9, 0.6),  ncol=1)
# fig.legend(handles=ax1.lines, labels=policy_labels,
#            fontsize=9, loc=5, ncol=1)

if False: # 'Freeze' in policy_list[0] or 'Dynamic' in policy_list[0]: 
    fig.legend(handles=ax1.lines, labels=policy_labels,
            fontsize=fontsize, loc=5, bbox_to_anchor=(0.65, 0.7),  ncol=1) #, fancybox=True, shadow=False)
else: 
    fig.legend(handles=ax1.lines, labels=policy_labels,
            handlelength=0.5, columnspacing=1.2,
            fontsize=fontsize, loc=5, bbox_to_anchor=(0.88, 0.75),  ncol=2) #, fancybox=True, shadow=False)


import os
if not os.path.exists("1_images/workload-density/"): 
    os.makedirs("1_images/workload-density/")
import sys
if len(sys.argv) < 2 or sys.argv[1] == 'False': 
    pdf = False 
else: 
    pdf = True 
    

add_freeze = 'Freeze' in policy_list[0]
print(pdf)
# save_path = '1_images/workload-density'
save_path = '1_images/impact-of-sched/'
if pdf: 
    fig.savefig("{}/simulator-jobload{}.pdf".format(save_path, "" if add_freeze else "freeze"), dpi=300, bbox_inches='tight')
    print("{}/simulator-jobload{}.pdf".format(save_path, "" if add_freeze else "freeze"))
else: 
    fig.savefig("{}/simulator-jobload{}.png".format(save_path, "" if add_freeze else "-nofreeze"), dpi=300, bbox_inches='tight')
    print("{}/simulator-jobload{}.png".format(save_path, "" if add_freeze else "-nofreeze"))
# print("1_images/workload-density/simulator-jobload.png")
# matplotlib.pyplot.show()
