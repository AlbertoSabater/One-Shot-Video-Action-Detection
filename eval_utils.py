#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 18:47:45 2020

@author: asabater
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import datetime



VLINE_W = 1.5
VLINE_REF = 2
REF_FRAMES_FREQ = 25
ANCHOR_COLOR = '#6600cc'
TARGET_COLOR = '#0066cc'
DETS_COLORS = {'med': '#F77800', 'good': '#ffcc66', 'excel': '#00cc66'}



def get_current_timeline(curr_frame, data_target, metric, label_left=None, label_right=None, 
                         width=100, plot=False):

    fig = plt.figure(figsize=(6,0.55), dpi=150)
    plt.axis('equal')
    plt.axis('off')

    total_frames = data_target.tl_frame.max()
    total_frames -= 1
    if curr_frame is None: curr_frame = total_frames
    curr_frame = curr_frame * width / total_frames
    
    # Total frames
    plt.hlines(y=0, xmin=0, xmax=width, color='grey', linestyle='-', linewidth=1,zorder=10)
    plt.vlines(0, -VLINE_REF, VLINE_REF, colors='k', linestyle='-', linewidth=1,zorder=10)
    plt.vlines(width, -VLINE_REF, VLINE_REF, colors='grey', linestyle='-', linewidth=1,zorder=10)
    # Current frames
    plt.hlines(y=0, xmin=0, xmax=curr_frame, color='k', linestyle='-', linewidth=1,zorder=10)
    plt.vlines(curr_frame, -VLINE_REF, VLINE_REF, colors='k', linestyle='-', linewidth=1,zorder=10)
    # Reference frames
    for i in range(REF_FRAMES_FREQ,total_frames,REF_FRAMES_FREQ):
        plt.vlines(i*width/total_frames, -VLINE_REF, VLINE_REF, colors='grey', 
                    linestyle=(0, (1, 1)), linewidth=1.3,zorder=0, alpha=0.6)
    
    
    # Anchors
    anchors = [ (g.tl_frame.min(),g.tl_frame.max()) for i,g in data_target.groupby('anchor_id') ]
    anchors = [ (init * width / total_frames, end * width / total_frames) for init,end in anchors ]
    for init, end in anchors:
        if init > curr_frame: continue
        end = min(end, curr_frame)
        
        plt.vlines(init, -VLINE_W, VLINE_W, colors=ANCHOR_COLOR, linestyle='-', alpha=0.6)
        plt.vlines(end, -VLINE_W, VLINE_W, colors=ANCHOR_COLOR, linestyle='-', alpha=0.6)
        plt.hlines(y=0, xmin=init, xmax=end, color=ANCHOR_COLOR, linestyle='-', linewidth=2, alpha=0.6,zorder=15)
    
    # Detections
    detections = data_target[data_target[metric+'_det']]
    if len(detections) == 0: detections = []
    else:
        detections = list(zip(detections.tl_frame.tolist(), detections[metric+'_det_level']))
                  # detections[[metric+'_med', metric+'_good', metric+'_excel']].apply(lambda x: np.array(['med', 'good', 'excel'])[x.values][0], axis=1).tolist()))
    # detections = [ (row.num_frame, color) for _,row in data_target[data_target[metric+'_det'].iterrows() ]
    detections = [ (det_frame * width / total_frames, color) for det_frame, color in detections ]
    for det_farame, color in detections:
        if det_farame > curr_frame: continue
        # plt.vlines(det, -VLINE_W, VLINE_W, colors=DETS_COLOR, linestyle='-', alpha=0.6)
        plt.vlines(det_farame, -VLINE_W, VLINE_W, colors=DETS_COLORS[color], linestyle='-', alpha=0.6)
    
    # # Targets
    targets = [ (g.tl_frame.min(),g.tl_frame.max()) for i,g in data_target.groupby('target_id') ]
    targets = [ (init * width / total_frames, end * width / total_frames) for init,end in targets ]
    for init, end in targets:
        if init > curr_frame: continue
        end = min(end, curr_frame)
        plt.vlines(init, -VLINE_W, VLINE_W, colors=TARGET_COLOR, linestyle='-', alpha=0.6)
        plt.vlines(end, -VLINE_W, VLINE_W, colors=TARGET_COLOR, linestyle='-', alpha=0.6)
        plt.hlines(y=0, xmin=init, xmax=end, color=TARGET_COLOR, linestyle='-', linewidth=2, alpha=0.6,zorder=15)
    
    # if label_left is not None:
    #     plt.text(0,2.5, label_left)    
    # if label_right is not None:
    #     plt.text(100,3, label_right, ha='right', fontsize=7)
    
    plt.text(0, -4.5, data_target.num_frame.min().astype(int), fontsize=7)
    plt.text(100,-4.5, data_target.num_frame.max().astype(int), fontsize=7, ha='right')
    
    canvas = FigureCanvas(fig)
    canvas.draw()       # draw the canvas, cache the renderer
    w, h = canvas.get_width_height()
    frame = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, 3)
    
    if not plot:
        plt.close(fig)
        
    return frame



from skvideo.io import FFmpegWriter
from tqdm import tqdm


def print_3d_skeleton(ax, skel, color=None):
    ax.scatter(skel[:,0], skel[:,1], skel[:,2], alpha=0.3, s=10)        # , c=color
    
    connecting_joint = [1, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11]
    for i in range(25):
        p1, p2 = i, connecting_joint[i]
        ax.plot((skel[p1][0],skel[p2][0]), 
                (skel[p1][1],skel[p2][1]), 
                (skel[p1][2],skel[p2][2]), c=color) 
        
def draw_skel_bbox(skel, ax, c, label):
    xmin, xmax, zmin, zmax = [skel[:,0].min(), skel[:,0].max(), skel[:,2].min(), skel[:,2].max()]
    y_mean = skel[:,1].mean()
    ax.plot([xmin, xmax], [y_mean,y_mean], [zmin, zmin], c=c)
    ax.plot([xmin, xmax], [y_mean,y_mean], [zmax, zmax], c=c)
    ax.plot([xmin, xmin], [y_mean,y_mean], [zmin, zmax], c=c)
    ax.plot([xmax, xmax], [y_mean,y_mean], [zmin, zmax], c=c)      
    ax.text(xmin+0.01, y_mean, zmin-0.08, label)

       
def render_video(data_anchor, data_target, anchors_info, 
                 output_video_filename, metric_thr, 
                 max_width, max_height, font, output_fps=12):
    
    
    data_target['anchor_skel_raw'] = [ min(data_anchor.iterrows(), key=lambda r: abs(r[1].num_frame - row.num_frame))[1].skels_raw for _,row in data_target.iterrows() ]
    
    
    total_coords = np.array(data_target.anchor_skel_raw.tolist() + data_target.skels_raw.tolist())
    X,Y,Z = total_coords[:,:,0], total_coords[:,:,2], total_coords[:,:,1]
    # max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    max_p, min_p = 99, 1
    max_range = np.array([np.percentile(X, max_p)-np.percentile(X, min_p), 
                          np.percentile(Y, max_p)-np.percentile(Y, min_p), 
                          np.percentile(Z, max_p)-np.percentile(Z, min_p)]).max() / 2.0
    # print([np.percentile(X, max_p), np.percentile(X, min_p), 
    #         np.percentile(Y, max_p), np.percentile(Y, min_p), 
    #         np.percentile(Z, max_p), np.percentile(Z, min_p)])
    # mid_x, mid_y, mid_z = (X.max()+X.min()) * 0.5, (Y.max()+Y.min()) * 0.5, (Z.max()+Z.min()) * 0.5
    mid_x, mid_y, mid_z = (np.percentile(X, max_p)+np.percentile(X, min_p)) * 0.5, \
                            (np.percentile(Y, max_p)+np.percentile(Y, min_p)) * 0.5, \
                            (np.percentile(Z, max_p)+np.percentile(Z, min_p)) * 0.5
    x_lim = (mid_x - max_range, mid_x + max_range)
    y_lim = (mid_y - max_range, mid_y + max_range)
    z_lim = (mid_z - max_range, mid_z + max_range)
    
    x_floor, y_floor = np.arange(*x_lim, 0.05), np.arange(*y_lim, 0.05)
    x_floor, y_floor = np.meshgrid(x_floor,y_floor)  
    z_floor = np.full(x_floor.shape, np.percentile(Z, 5))
    del total_coords; del X; del Y; del Z
    
    

    sess = anchors_info.iloc[0]['patient'] + '/' + anchors_info.iloc[0]['session'] + ' | ' + anchors_info.iloc[0]['preds_filename']
    max_tl_frame, max_num_frame = data_target.tl_frame.max(), int(data_target.num_frame.max())
    num_ex, label_ex = anchors_info.iloc[0].ex_num, anchors_info.iloc[0].action

    
    writer = FFmpegWriter(output_video_filename,
 								   inputdict={'-r': str(output_fps)},
 								   outputdict={'-r': str(output_fps)})   
    
    # for _,row in tqdm(data_target[60:90].iterrows(), total=len(data_target)):
    for _,row in tqdm(data_target.iterrows(), total=len(data_target)):

        fig = plt.Figure(figsize=(18,9), dpi=150)
        ax = fig.add_subplot(1,1,1, projection='3d')        
        ax.view_init(10, -60)
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_zlim(*z_lim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z') 
        
        
        # Plot skeleton and floor
        print_3d_skeleton(ax, row.anchor_skel_raw[:, [0,2,1]], color=ANCHOR_COLOR)
        print_3d_skeleton(ax, row.skels_raw[:, [0,2,1]], color=TARGET_COLOR)
        ax.plot_wireframe(x_floor, y_floor, z_floor, alpha=0.2, rcount=80)

        
        # Print gt bounding boxes
        if row.is_anchor:
            draw_skel_bbox(row.anchor_skel_raw[:, [0,2,1]], ax, c=ANCHOR_COLOR, label='Recording Anchor')
        if row.is_target:
            draw_skel_bbox(row.skels_raw[:, [0,2,1]], ax, c=TARGET_COLOR, label='Recording Anchor')

        
        # Print metric distances
        
        xmin, xmax, zmin, zmax = [row.skels_raw[:,0].min(), row.skels_raw[:,0].max(), row.skels_raw[:,1].min(), row.skels_raw[:,1].max()]
        y_mean = row.skels_raw[:,2].mean()
        for i,(metric,thr) in enumerate(metric_thr.items()):
            if not row[metric+'_det']: 
                c = 'k'; weight = None
            else: 
                c = DETS_COLORS[row[metric+'_det_level']]; weight = 'bold'

            ax.text(xmin+0.01, y_mean, zmin-((i+2)*0.08), 
                    ' - {}: {:.2f}'.format(metric, row[metric+'_dist']), 
                    weight=weight, color=c)        
        
        

        # Get plot as Image
        canvas = FigureCanvas(fig)
        canvas.draw()       # draw the canvas, cache the renderer
        w, h = canvas.get_width_height()
        frame = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, 3)
        frame = frame[:, ~np.all(np.all(frame==255, axis=0), axis=1), :]
        frame = frame[~np.all(np.all(frame==255, axis=1), axis=1), :, :]
        
        h_orig, w_orig = frame.shape[:2]
        w_rel, h_rel = w_orig / max_width, h_orig / max_height
            
        frame = Image.fromarray(frame)
        if w_rel < h_rel: new_size = (int(w_orig/h_rel), int(h_orig/h_rel))
        else: new_size = (int(w_orig/w_rel), int(h_orig/w_rel))        
        frame = frame.resize(new_size, Image.ANTIALIAS)
        frame_new = Image.new("RGB", (max_width, max_height), (255, 255, 255))
        frame_new.paste(frame, (max_width-new_size[0], max_height-new_size[1]))
        frame = frame_new; del frame_new
        
        
        timelines = []
        for metric in metric_thr.keys():
            timeline = get_current_timeline(row.tl_frame, data_target, metric, 
                                        label_left=metric, 
                                        label_right=None, 
                                        width=100, plot=False)
            timelines.append(timeline)        
        timelines = np.concatenate(timelines, axis=0)
        timelines = timelines[:, ~np.all(np.all(timelines==255, axis=0), axis=1), :]
        # timelines = timelines[~np.all(np.all(timelines==255, axis=1), axis=1), :, :]
        timelines = Image.fromarray(timelines)        
        tl_width = int(max_width*0.5)
        tl_height = int(timelines.size[1] * tl_width / timelines.size[0])
        timelines = timelines.resize((tl_width, tl_height), Image.ANTIALIAS)
        frame.paste(timelines, (max_width-tl_width-18, 0))        
        

        draw = ImageDraw.Draw(frame)
        # Session
        draw.text((20,0), sess, fill=(0,0,0), font=font)
        # num_frame
        # draw.text((20,25), "#{}".format(num_frame+min_frame), fill=(0, 0, 0), font=font)
        draw.text((20,30), "{:<3} / {} || {:<3} / {}".format(row.tl_frame, max_tl_frame, int(row.num_frame), max_num_frame), fill=(0, 0, 0), font=font)
        # timestamp
        draw.text((20,60), str(datetime.datetime.fromtimestamp(row.ts)), fill=(0, 0, 0), font=font)
        # Exercise
        draw.text((20,90), "#{} | {}".format(num_ex, label_ex), fill=(0, 0, 0), font=font)
 
            
        frame = np.array(frame)
        writer.writeFrame(frame)  
        # break
    
    writer.close()


