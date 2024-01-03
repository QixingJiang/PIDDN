import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import json
# Function to clean column names
def clean_column_names(df):
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace('\s+', '_', regex=True)


def load_json_logs(json_logs):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics, e.g. memory, bbox_mAP
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            for line in log_file:
                log = json.loads(line.strip())
                # skip lines without `epoch` field
                if 'epoch' not in log:
                    continue
                epoch = log.pop('epoch')
                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)
                for k, v in log.items():
                    log_dict[epoch][k].append(v)
    return log_dicts


siamese_json = "/docker_host2/mulframe_pcb/new_anno_workdir/siamese_faster_rcnn_swin-t/20230722_040624.log.json",


dual_json = "/docker_host2/mulframe_pcb/new_anno_workdir/dual_faster_rcnn/20230722_040922.log.json",

metric = 'loss'
start_epoch = 5
epochs = 300
eval_interval = 1


siamese_dicts = load_json_logs(siamese_json)[0]
dual_dicts = load_json_logs(dual_json)[0]


epochs = list(siamese_dicts.keys())
# xs = np.arange(
#     int(start_epoch),
#     max(epochs) + 1, int(eval_interval))
xs = np.arange(
    1,
    max(epochs) + 1, 1)
ys = []
zs = []
for epoch in epochs:
    ys += [siamese_dicts[epoch][metric][0]]
    zs += [dual_dicts[epoch][metric][0]]
ax = plt.gca()
custom_ticks = np.linspace(0, 300, 6, dtype=int)
ax.set_xticks(custom_ticks)


plt.xlabel('epoch')
plt.ylabel("Train loss")
plt.plot(xs, ys, label="Siamese backbone")
plt.plot(xs, zs, label="Dual backbone")
# 显示图例
plt.legend()
plt.title("Train Loss curves on PairPCB train")
plt.savefig("test_curve.png")



# pdb.set_trace()
# # Plot mAP@0.5 curves
# plt.figure()
# #lable属性为曲线名称，自己可以定义
# plt.plot(siamese_json['mAP'], label="Siamese backbone")
# plt.plot(dual_results['mAP'], label="Dual backbone")
# plt.xlabel("Epoch")
# plt.ylabel("mAP@0.5")
# plt.legend()
# plt.title("mAP@0.5 Comparison")
# plt.savefig("test_curve.png")


