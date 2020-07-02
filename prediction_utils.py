#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 18:11:25 2020

@author: asabater
"""

import json
import os
import pickle
from data_generator import load_scaler, get_pose_data_v2, average_wrong_frame_skels
import pandas as pd
import numpy as np




def load_model(path_model, return_sequences=True):
    model_params = json.load(open(path_model + '/model_params.json'))
    # model_params['triplet'] = True
    
    # model = tf.keras.models.load_model(path_model + 'model', compile=False)
    
    weights = sorted([ w for w in os.listdir(path_model + 'weights') if 'index' in w ])
    losses = [ float(w.split('-')[2][8:15]) for w in weights ]
    weights = weights[losses.index(min(losses))][:-6]
    print(weights)
    
    if model_params.get('use_gru',False) == True and 'decoder_v2' not in path_model:
        model_params['use_gru'] = False
    
    if 'ae_tcn' in model_params['model_name']:
        from models.autoencoder_tcn import AutoEncoderTCN
        model = AutoEncoderTCN(prediction_mode=return_sequences, **model_params)
    else:
        raise ValueError('model_name not handled:', model_params['model_name'])
    
    model.load_weights(path_model + 'weights/' + weights).expect_partial()
    
    model.build((None, None, model_params['num_feats']))
    
    
    scale_data = model_params['scale_data']
    if scale_data: 
        print(' * Loading data scaler')
        model_params['scaler'] = pickle.load(open(path_model + 'scaler.pckl', 'rb'))
    else: model_params['scaler'] = None
    
    
    for data_key in ['use_jcd_features', 'use_speeds', 'use_coords_raw', 
                     'use_coords', 'use_jcd_diff', 'use_bone_angles',
                     'tcn_batch_norm']:
        if data_key not in model_params: model_params[data_key] = False
    
    return model, model_params



def get_video_distances(pf, actions_data, video_skels, model, model_params, batch=None, in_memory_callback=False, cache={}):

    anchors = actions_data[(actions_data.preds_file == pf) & (actions_data.is_therapist == 'y')]
    anchors_preds_filename = anchors.preds_filename.drop_duplicates().tolist()[0]
    
    if not in_memory_callback or not anchors_preds_filename in cache:
        anchors_tempos, anchors_num_frames, anchors_skels_raw = video_skels[anchors_preds_filename]
        anchors_skels_raw = average_wrong_frame_skels(anchors_skels_raw)
        anchors_skels = get_pose_data_v2(anchors_skels_raw, validation=True, **model_params)
        # anchors_skels = np.expand_dims(anchors_skels, axis=0)    
        # anchor_preds = np.array(model.get_embedding(anchors_skels))[0]
        anchors_y_true = [ any((anchors.init_frame < num_frame) & (anchors.end_frame > num_frame)) \
                          for num_frame in anchors_num_frames ]
            
            
        # anchors_ids = [ any((anchors.init_frame < num_frame) & (anchors.end_frame > num_frame)) \
                          # for num_frame in anchors_num_frames ]
    
        data_anchor = pd.DataFrame({'ts': anchors_tempos, 'num_frame': anchors_num_frames, 
                                    'skels_raw': list(anchors_skels_raw),
                                    'skels_feats': list(anchors_skels),
                                    # 'emb': list(anchor_preds),
                                    'emb': None,
                                    'y_true': anchors_y_true,
                                    'action_id': None})
        for _,row in anchors.iterrows(): data_anchor.loc[(data_anchor.num_frame > row.init_frame) & (data_anchor.num_frame < row.end_frame), 'action_id'] = row.comments
        if in_memory_callback: cache[anchors_preds_filename] = data_anchor
    else:
        data_anchor = cache[anchors_preds_filename]
    

    targets = actions_data[(actions_data.preds_file == pf) & (actions_data.is_therapist == 'n')]
    targets_preds_filename = anchors_preds_filename.replace('keypoints', 'keypointschild')
    if not targets_preds_filename in cache:
        targets_tempos, targets_num_frames, targets_skels_raw = video_skels[targets_preds_filename]
        targets_skels_raw = average_wrong_frame_skels(targets_skels_raw)
        targets_skels = get_pose_data_v2(targets_skels_raw, validation=True, **model_params)
        # targets_skels = np.expand_dims(targets_skels, axis=0)    
        # targets_preds = np.array(model.get_embedding(targets_skels))[0]
        targets_is_target = [ any((targets.init_frame < num_frame) & (targets.end_frame > num_frame)) \
                          for num_frame in targets_num_frames ]
        targets_is_anchor = [ any((anchors.init_frame < num_frame) & (anchors.end_frame > num_frame)) \
                          for num_frame in targets_num_frames ]
            
    
        data_target = pd.DataFrame({'ts': targets_tempos, 'num_frame': targets_num_frames, 
                                    'skels_raw': list(targets_skels_raw),
                                    'skels_feats': list(targets_skels),
                                    # 'emb': list(targets_preds),
                                    'emb': None,
                                    'tl_frame': None,
                                    'is_anchor': targets_is_anchor,
                                    'is_target': targets_is_target,
                                    'action_id': None,
                                    # 'tl_frame': list(range(len(targets_preds))),
                                    'target_id': None, 'anchor_id': None
                                    })
        
        # Add targets ids
        for _,row in targets.iterrows(): data_target.loc[(data_target.num_frame > row.init_frame) & (data_target.num_frame < row.end_frame), 'target_id'] = row.comments
        for _,row in anchors.iterrows(): data_target.loc[(data_target.num_frame > row.init_frame) & (data_target.num_frame < row.end_frame), 'anchor_id'] = row.comments
        cache[targets_preds_filename] = data_target
    else:
        data_target = cache[targets_preds_filename]



    # anchor_preds = np.array(model.get_embedding(anchors_skels))[0]
    # targets_preds = np.array(model.get_embedding(targets_skels))[0]
    
    data_anchor.loc[:,'emb'] = pd.Series(list(np.array(model.get_embedding(np.expand_dims(np.array(data_anchor.skels_feats.tolist()), axis=0), batch=batch))[0]))
    data_target.loc[:,'emb'] = pd.Series(list(np.array(model.get_embedding(np.expand_dims(np.array(data_target.skels_feats.tolist()), axis=0), batch=batch))[0]))

    return data_anchor, data_target, anchors, targets



def get_anchor_embs_by_strategy(data_anchor, anchors_info, anchor_strategy):
    if anchor_strategy == 'last': # Last anchor frame
        anchor_embs = { row.end_frame:[data_anchor[data_anchor.action_id == row.comments].emb.iloc[-1]] for _,row in anchors_info.iterrows() }
    elif anchor_strategy.startswith('perc'):    # Pick embeddings by position percentage
        percs = list(map(float, anchor_strategy[5:].split('_')))
        anchor_embs = {}
        for _,row in anchors_info.iterrows():
            frame_embs = data_anchor[data_anchor.action_id == row.comments].emb
            anchor_embs[row.end_frame] = [ frame_embs.iloc[int((len(frame_embs)-1)*perc)] for perc in percs ]
    elif anchor_strategy.startswith('pos'):    # Pick embeddings by position percentage
        positions = list(map(int, anchor_strategy[4:].split('_')))
        anchor_embs = {}
        for _,row in anchors_info.iterrows():
            frame_embs = data_anchor[data_anchor.action_id == row.comments].emb
            anchor_embs[row.end_frame] = [ frame_embs.iloc[pos] for pos in positions ]
    else: raise ValueError('anchor_strategy "{}" not implemented'.format(anchor_strategy))
    return anchor_embs


from scipy.spatial import distance
def get_distance(dist_func, emb1, emb2, verbose=False):
    if dist_func in ['euc', 'euclidean']:
        dist = distance.euclidean(emb1, emb2)
    elif dist_func in ['cos', 'cosine']:
        dist = distance.cosine(emb1, emb2)
    elif dist_func in ['js', 'jensenshannon']:
        dist = distance.jensenshannon(emb1, emb2)
    else:
        raise ValueError('Distance "{}" not handled'.format(dist_func))
    if verbose: print(dist_func, dist)
    return dist



def calculate_distances(data_anchor, data_target, anchors_info, metric_thr, 
                        anchor_strategy='last',        # how to pick the anchors
                        dist_to_anchor_func='mean',    # Mean or median od the distances to anchor
                        last_anchor=True,            # Use last anchor or all of them
                        top=False,                      # Use only best distances [perc0.2]
                        ):      
    
    anchor_embs = get_anchor_embs_by_strategy(data_anchor, anchors_info, anchor_strategy)
        
        
    min_frame = min(anchor_embs.keys())
    detections = { metric:[] for metric,_ in metric_thr.items() }
    for _,row in data_target.iterrows():
        
        
        anchors_to_compare = { anchor_frame:embs for anchor_frame,embs in anchor_embs.items() if anchor_frame < row.num_frame }
        if len(anchors_to_compare) == 0: # Skip comparison if there are no anchors
            for metric,_ in metric_thr.items(): detections[metric].append(None)
            continue
        
        if last_anchor:     # Use only the last anchor
            anchors_to_compare = dict([max(anchor_embs.items(), key=lambda x: x[0])])
        
        # Flatten the anchor embeddings
        anchors_to_compare = sum(anchors_to_compare.values(), [])
        
        for metric, thrs in metric_thr.items():
            metric_dists = sorted([ get_distance(metric, atc, row.emb) for atc in anchors_to_compare ])
            
            if top != False:
                if top.startswith('perc'):
                    # num_goods = max(1, int(len(metric_dists)*float(top[4:])))
                    top_perc = float(top[4:])
                    num_goods = int(len(metric_dists)*top_perc)
                    if num_goods == 0: num_goods = 1 if top_perc > 0 else -1
                    metric_dists = metric_dists[:num_goods] if num_goods>0 else metric_dists[num_goods:]
                    
                else: raise ValueError('top distance param not handled:', top)
                
            if dist_to_anchor_func == 'mean':
                dist = np.mean(metric_dists)
            elif dist_to_anchor_func == 'median':
                dist = np.median(metric_dists)
            elif dist_to_anchor_func == 'min':
                dist = np.min(metric_dists)
            detections[metric].append(dist)
                
        

    for metric,thr in metric_thr.items(): 
        data_target.loc[:,metric+'_dist'] = detections[metric]
        data_target.loc[:,metric+'_det'] = data_target[metric+'_dist']<=thr['med']
        
        data_target.loc[:,metric+'_med'] = data_target[metric+'_dist'].apply(lambda x: x<=thr['med'] and x>thr['good'])
        data_target.loc[:,metric+'_good'] = data_target[metric+'_dist'].apply(lambda x: x<=thr['good'] and x>thr['excel'])
        data_target.loc[:,metric+'_excel'] = data_target[metric+'_dist'].apply(lambda x: x<=thr['excel'])

        data_target.loc[:,metric+'_det_level'] = None
        if len(data_target.loc[data_target[metric+'_det'],metric+'_det_level']) > 0:
            data_target.loc[data_target[metric+'_det'],metric+'_det_level'] =  data_target[data_target[metric+'_det']][[metric+'_med', metric+'_good', metric+'_excel']].apply(lambda x: np.array(['med', 'good', 'excel'])[x.values][0], axis=1).tolist()

    return data_target



