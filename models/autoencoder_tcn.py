#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 10:21:43 2020

@author: asabater
"""

from tcn import TCN, tcn_full_summary
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, RepeatVector, TimeDistributed, Lambda, Masking # BatchNormalization
from tensorflow.keras.layers import Input
import tensorflow as tf
import tensorflow.keras.backend as K
import ast
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from models.autoencoder_lstm import DecoderLSTM

import numpy as np

class EncoderTCN(Model):
    def __init__(self, num_feats, nb_filters, kernel_size, nb_stacks, use_skip_connections, 
                 lstm_dropout, padding, dilations,
                 masking=False, 
                 prediction_mode=False,
                 tcn_batch_norm = False,
                 **kwargs
                 ):
        super(EncoderTCN, self).__init__()
        # self.encoder_layers = [Input((None, num_feats,))]
        self.encoder_layers = []
        
        # Add masking layer
        if masking: 
            print('MASKING')
            self.encoder_layers.append(Masking())
        
        num_tcn = len(dilations)
        for i in range(num_tcn-1):
            l = TCN(
                    nb_filters = nb_filters, 
                    kernel_size = kernel_size,
                    nb_stacks = nb_stacks,
                    use_skip_connections = use_skip_connections,
                    padding = padding,
                    dilations = dilations[i],
                    dropout_rate = lstm_dropout,
                    return_sequences=True,
                    use_batch_norm = tcn_batch_norm
                    )
            self.encoder_layers.append(l)        
            print('TCN', i, dilations[i], l.receptive_field)
        
        l = TCN(
                nb_filters = nb_filters, 
                kernel_size = kernel_size,
                nb_stacks = nb_stacks,
                use_skip_connections = use_skip_connections,
                padding = padding,
                dilations = dilations[-1],
                dropout_rate = lstm_dropout,
                return_sequences=prediction_mode,
                use_batch_norm = tcn_batch_norm
                )
        self.encoder_layers.append(l)
        print('TCN', -1, dilations[-1], l.receptive_field)
        
        for l in self.encoder_layers: print(l)
                
        self.encoder = Sequential(self.encoder_layers)

    def call(self, x):
        encoder = self.encoder(x)
        return encoder
    def get_embedding(self, x):
        emb = self.encoder(x)
        return emb


class DecoderTCN(Model):
    def __init__(self, num_feats, nb_filters, kernel_size, nb_stacks, use_skip_connections, 
                 lstm_dropout, padding, dilations,
                 tcn_batch_norm = False,
                 **kwargs):
        super(DecoderTCN, self).__init__()
        self.rep = Lambda(self.repeat_vector)
        self.decoder_layers = []
        # num_tcn = len(dilations)
        for d in dilations[::-1]:
            d = d[::-1]
            # d = [1]*len(d)
            l = TCN(
                    nb_filters = nb_filters, 
                    kernel_size = kernel_size,
                    nb_stacks = nb_stacks,
                    use_skip_connections = use_skip_connections,
                    padding = padding,
                    dilations = d,
                    dropout_rate = lstm_dropout,
                    return_sequences=True,
                    use_batch_norm = tcn_batch_norm
                    )
            self.decoder_layers.append(l)
            print('TCN decoder', d, l.receptive_field)
            # self.decoder_layers.append(BatchNormalization())
            
        self.decoder_layers.append(TimeDistributed(Dense(num_feats, name='out_encoder')))
        self.decoder = Sequential(self.decoder_layers)
    def repeat_vector(self, args):
       layer_to_repeat = args[0]
       reference_layer = args[1]
       return RepeatVector(K.shape(reference_layer)[1])(layer_to_repeat)    
    def call(self, args):
        decoder = self.rep(args)
        decoder = self.decoder(decoder)
        return decoder




# conv_params -> nb_filters, kernel_size, nb_stacks, use_skip_connections
class AutoEncoderTCN(Model):
    def __init__(self, num_feats, conv_params, lstm_dropout,
                     masking, 
                     decoder, triplet, classification, clf_neurons=None, num_classes=None,
                     prediction_mode=False,
                     lstm_decoder = False, num_layers=None, num_neurons=None,
                     tcn_batch_norm = False,
                     use_gru = False,
                     **kwargs
                 ):
        super(AutoEncoderTCN, self).__init__()
        
        if len(conv_params) == 4:
            nb_filters, kernel_size, nb_stacks, use_skip_connections = conv_params
            padding = 'causal'
            dilations = [1, 2, 4, 8, 16, 32]
        elif len(conv_params) == 5:
            nb_filters, kernel_size, nb_stacks, use_skip_connections, padding = conv_params
            dilations = [1, 2, 4, 8, 16, 32]
        elif len(conv_params) == 6:
            nb_filters, kernel_size, nb_stacks, use_skip_connections, padding, dilations = conv_params
            if type(dilations) == int: 
                dilations = [dilations]
            if type(dilations) == str: dilations = ast.literal_eval(dilations)
            else:
                dilations = [ [ i for i in [1, 2, 4, 8, 16, 32] if i<= d ] for d in dilations ]

            print('dilations', dilations)
        else:
            raise ValueError('conv_params length not recognized', len(conv_params))
        
        self.encoder_net = EncoderTCN(
                                   num_feats = num_feats, 
                                   nb_filters=nb_filters, 
                                   kernel_size=kernel_size,
                                   nb_stacks=nb_stacks,
                                   use_skip_connections = use_skip_connections,
                                   padding = padding,
                                   dilations = dilations,
                                   lstm_dropout=lstm_dropout,
                                   masking=masking,
                                   prediction_mode = prediction_mode,
                                   tcn_batch_norm = tcn_batch_norm)

        if decoder:
            if not lstm_decoder:
                self.decoder_net = DecoderTCN(
                                       num_feats = num_feats, 
                                       nb_filters=nb_filters, 
                                       kernel_size=kernel_size,
                                       nb_stacks=nb_stacks,
                                       use_skip_connections = use_skip_connections,
                                       padding = padding,
                                       dilations = dilations,
                                       lstm_dropout=lstm_dropout,
                                       tcn_batch_norm = tcn_batch_norm)
            else:
                print(num_layers, num_neurons, num_feats, use_gru)
                self.decoder_net = DecoderLSTM(
                                    num_layers = num_layers,
                                    num_neurons = num_neurons, 
                                    lstm_recurrent_dropout = 0.0,
                                    lstm_dropout = lstm_dropout,
                                    num_feats = num_feats,
                                    use_gru = use_gru,
                                )
        self.decoder = decoder

        # if triplet: 
        # self.norm = Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='out_triplet')
        self.norm = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))
        self.triplet = triplet

        if clf_neurons != 0: self.clf_dense = Dense(clf_neurons, activation='relu')
        self.clf_neurons = clf_neurons

        if classification: self.clf_out = Dense(num_classes, activation='softmax', name='out_clf')         
        self.classification = classification
        
        # self.unify = Lambda(lambda x: x/tf.math.reduce_sum(x, axis=-1, keepdims=True))

    def call(self, x):
        encoder_raw = self.encoder_net(x)
        
        if self.clf_neurons != 0: encoder = self.clf_dense(encoder_raw)
        else: encoder = encoder_raw

        out = []    # return emb, clf, decoder
        if self.triplet:
            emb = self.norm(encoder)
            out.append(emb)
        if self.classification:
            clf = self.clf_out(encoder)    
            out.append(clf)
        if self.decoder:
            decoder = self.decoder_net([encoder_raw, x])
            out.append(decoder)

        return out
            
    def get_embedding(self, x, batch=None, unify=False):
        
        if batch is None or batch<=0:
            emb = self.encoder_net(x)
            if self.clf_neurons != 0: emb = self.clf_dense(emb)
            # if unify: emb = self.unify(emb)
            # else: emb = self.norm(emb)
            # emb = self.unify(emb)
            emb = self.norm(emb)
            return emb
        
        elif batch > 0:
            embs_data = []
            for num_sample in range(x.shape[0]):
                for num_frame in range(x.shape[1]):
                    start, end = max(0,num_frame-batch+1), num_frame+1
                    embs_data.append(pad_sequences(x[num_sample:num_sample+1, start:end], maxlen=batch, dtype='float32', padding='pre'))
            preds_batch = [ self.get_embedding(np.concatenate(embs_data[ind:ind+512]), batch=None) for ind in range(0, len(embs_data), 512) ]
            preds_batch = np.concatenate(preds_batch)
            # preds_batch = np.array(self.get_embedding(np.concatenate(embs_data), batch=None))
            preds_batch = preds_batch.reshape((x.shape[0], x.shape[1], preds_batch.shape[1]))
            return preds_batch
    
            # return np.column_stack(embs)
        else: raise ValueError('Incorrect prediction batch')
    
    def set_encoder_return_sequences(self, return_sequences):
        l = [ l for l in self.encoder_net.layers if type(l) == TCN ][-1]
        l.return_sequences = return_sequences


