config=/opt/data/private/jqx/mulframe_pcb/configs/htc/swin_t_htc_single_pcb.py
out_dir='/opt/data/private/jqx/mulframe_pcb/anchor_result/'




python optimize_anchors.py $config --input-shape=[256, 256] --algorithm k-means\
--output-dir=$out_dir
