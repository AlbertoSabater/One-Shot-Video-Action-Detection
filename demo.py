#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 18:11:02 2020

@author: asabater
"""

import prediction_utils
import os
import pickle
import os
import numpy as np
from prediction_utils import get_video_distances, calculate_distances
from eval_utils import get_current_timeline, render_video
from PIL import Image, ImageFont
import matplotlib.pyplot as plt




# Comment to use GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''




store_timelines = True
plot_timelines = True
store_video = False



path_model = './trained_networks/0701_0156_model_7/'
model, model_params = prediction_utils.load_model(path_model) 
model.set_encoder_return_sequences(True)
model_params['skip_frames'] = []
model_params['max_seq_len'] = 0   






raw_data_path = './data_samples/'

# df -> All data from spreadsheet
actions_data = pickle.load(open(os.path.join(raw_data_path, 'actions_data_v2.pckl'), 'rb'))
# video_preds -> { video_name: (tempos, preds) } -> All video predictions
video_skels = pickle.load(open(os.path.join(raw_data_path, 'video_skels_v2.pckl'), 'rb'))





pred_files = actions_data.preds_file.drop_duplicates().tolist()
pf = np.random.choice(pred_files)
# pf = '153607'


FRAMES_BEFOR_ANCHOR = 32


dist_params = { 'anchor_strategy': 'pos_-1_-2_-3', 'last_anchor': False, 'dist_to_anchor_func': 'mean'}
metric_thr = { 'cos': {'med':0.5, 'good':0.0, 'excel':0.0}, 'js': {'med':0.52, 'good':0.0, 'excel':0.0} }



data_anchor, data_target, anchors_info, targets_info = get_video_distances(pf, 
                                   actions_data, video_skels, model, model_params, 
                                   batch = None, 
                                   in_memory_callback=False, cache={})

# Crop the beginning of the video
init_frame = anchors_info.init_frame.min() - FRAMES_BEFOR_ANCHOR
data_anchor = data_anchor[data_anchor.num_frame >= init_frame]
data_target = data_target[data_target.num_frame >= init_frame]
data_target = data_target.assign(tl_frame=list(range(len(data_target))))


data_target = calculate_distances(data_anchor, data_target, anchors_info, 
                                          metric_thr)


timelines = []
if store_timelines or plot_timelines:
    for metric, thr in metric_thr.items():
        if store_timelines or plot_timelines:
            timeline = get_current_timeline(None, data_target, metric, 
                                        label_left=metric, label_right='', 
                                        width=100, plot=False)
            timelines.append(timeline)
    
    timelines_filename = pf + '_timeline.png'
    timelines_img = np.vstack(timelines)
    if store_timelines:
        timelines_img = Image.fromarray(timelines_img)
        timelines_img.save(timelines_filename)     
    if plot_timelines:
        plt.figure(dpi=150)
        plt.imshow(timelines_img)
        plt.axis('off')
        plt.show()



if store_video:
    max_width, max_height = 1280, 720
    output_video_filename = pf + '_vid.mp4'
    font = ImageFont.truetype("NotoSerif-Regular.ttf", int(max_height*0.033))
    render_video(data_anchor, data_target, anchors_info,
                 output_video_filename, metric_thr, max_width, max_height, 
                 font, output_fps=12)   


