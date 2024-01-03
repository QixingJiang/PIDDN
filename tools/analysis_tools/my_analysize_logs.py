import json
import matplotlib.pyplot as plt
import argparse
'''
  --mode: val/train
  --select: mAP/loss/something else...
  --json_paths: the json file (log file) path
  --out_dir:the out saved picture path
  --epoch_num: the total epoch numbers 
  --pic_name: the out saved picture name 

example: 
 python tools/analysis_tools/my_analysize_logs.py --mode val --select mAP --json_paths test_pcb/20221011_111126.log.json --line_names mAP --out_dir ./ --epoch_num 120 --pic_name mAP 
'''
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='val')
parser.add_argument("--select", type=str, default='bbox_mAP_50')
parser.add_argument("--json_paths", type=str, nargs='+')
parser.add_argument("--line_names", type=str, nargs='+')
parser.add_argument("--out_dir", type=str, default='./')
parser.add_argument("--epoch_num", type=int, default=20)
parser.add_argument("--pic_name", type=str, default="result")
args = parser.parse_args()

select = args.select
pic_name = args.pic_name
mode = args.mode  # Ñ¡ÔñlogÎÄ¼þÖÐµÄÄ£Ê½
json_paths = args.json_paths
line_names = args.line_names
out_dir = args.out_dir
epoch_num = args.epoch_num

plt.figure(figsize=(12, 8), dpi=300)
for i, json_path in enumerate(json_paths):
    epoch_now = 0
    x = []  # ´æ·Åepoch
    y = []  # ´æ·ÅÖ¸±ê
    y_min = 1000000  # ´æ·ÅÖ¸±ê×î´óÖµ   ap²»»á³¬¹ý1  »æÖÆloss¿É×ÔÓÉ¸ü¸Ä
    y_max = -1  # ´æ·ÅÖ¸±ê×îÐ¡Öµ   ap²»»áÐ¡ÓÚ-1  »æÖÆloss¿É×ÔÓÉ¸ü¸Ä
    x_min = 0  # ³öÏÖ×îÐ¡ÖµµÄepoch
    x_max = 0  # ³öÏÖ×î´óÖµµÄepoch
    isFirst = True
    with open(json_path, 'r') as f:
        for jsonstr in f.readlines():
            if epoch_now == epoch_num:
                break
            if isFirst:  # mmdetectionÉú³ÉµÄlog  jsonÎÄ¼þµÚÒ»ÐÐÊÇÅäÖÃÐÅÏ¢  Ìø¹ý
                isFirst = False
                continue
            row_data = json.loads(jsonstr)
            if row_data['mode'] == mode:  # Ñ¡Ôñtrain»òÕßvalÄ£Ê½ÖÐµÄÖ¸±êÊý¾Ý
                epoch_now = epoch_now + 1
                item_select = float(row_data[select])
                x_select = int(row_data['epoch'])
                x.append(x_select)
                y.append(item_select)
                if item_select >= y_max:  # Ñ¡Ôñ×î´óÖµ  ÎªÊ²Ã´²»ÓÃnumpy.argminÄØ£¿  ÒòÎªepoch¿ÉÄÜ²»´Ó1¿ªÊ¼  xminºÍymin¿ÉÄÜÆ¥Åä´íÎó  ±È½ÏÂé·³
                    y_max = item_select
                    x_max = x_select
                if item_select <= y_min:  # Ñ¡Ôñ×î´óÖµ
                    y_min = item_select
                    x_min = x_select

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.plot(x, y, label=line_names[i])
    plt.plot(x_min, y_min, 'g-p', x_max, y_max, 'r-p')
    show_min = '[' + str(x_min) + ' , ' + str(y_min) + ']'
    show_max = '[' + str(x_max) + ' , ' + str(y_max) + ']'
    plt.annotate(show_min, xy=(x_min, y_min), xytext=(x_min, y_min))
    plt.annotate(show_max, xy=(x_max, y_max), xytext=(x_max, y_max))

plt.xlabel('epoch')
plt.legend()
plt.ylabel(select)

# plt.ylim(0.8, 1.0)  # ÉèÖÃyÖá×ø±ê·¶Î§


plt.savefig(args.out_dir + '/' + pic_name + '.jpg', dpi=300)