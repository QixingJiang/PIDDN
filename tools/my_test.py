import os
import pdb
txt_dir = '/docker_host2/mulframe_pcb/test_seq.txt'
count = 0
all_img = []
with open(txt_dir, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if "image_path" in line:
            image = line.split('image_path: ')[-1].strip('\n')
            all_img.append(image)
with open('all_test_img.txt', 'w') as w:
    for i in all_img:
        w.write(i+'\n')
            