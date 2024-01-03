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


factor1_logs = "piddn_workdir/faster_swin_affine_ciou2_recti/20231127_080624.log.json",
factor2_logs = "piddn_workdir/faster_swin_affine_ciou3/20231127_081259.log.json",
factor3_logs = "piddn_workdir/faster_swin_affine_ciou4/20231128_025007.log.json",
factor4_logs = "piddn_workdir/faster_swin_affine_ciou5/20231128_025101.log.json",
factor5_logs = "piddn_workdir/faster_swin_affine_ciou6/20231128_030011.log.json",

metric = 'mAP'
start_epoch = 5
epochs = 300
eval_interval = 5


dict1 = load_json_logs(factor1_logs)[0]
dict2 = load_json_logs(factor2_logs)[0]
dict3 = load_json_logs(factor3_logs)[0]
dict4 = load_json_logs(factor4_logs)[0]
dict5 = load_json_logs(factor5_logs)[0]

epochs = list(dict1.keys())
xs = np.arange(
    int(start_epoch),
    max(epochs) + 1, int(eval_interval))
ys1 = []
ys2 = []
ys3 = []
ys4 = []
ys5 = []
for epoch in xs:
    ys1 += [dict1[epoch][metric][0]]
    ys2 += [dict2[epoch][metric][0]]
    ys3 += [dict3[epoch][metric][0]]
    ys4 += [dict4[epoch][metric][0]]
    ys5 += [dict5[epoch][metric][0]]
ax = plt.gca()
custom_ticks = np.linspace(0, 300, 6, dtype=int)
ax.set_xticks(custom_ticks)
# ax.set_yticks([0.1, 0.2, 0.3, 0.4])

plt.xlabel('epoch')
plt.ylabel("mAP@0.5")
plt.plot(xs, ys1, label="factor1")
plt.plot(xs, ys2, label="factor2")
plt.plot(xs, ys3, label="factor3")
plt.plot(xs, ys4, label="factor4")
plt.plot(xs, ys5, label="factor5")
# 显示图例
plt.legend()
# plt.title("Train Loss curves on PairPCB train")
plt.savefig("my_curve.png")



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


