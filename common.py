#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

class Config:
    '''where to write all the logging information during training(includes saved models)'''
    dataset = '/mnt/data/pictures/*.png'

    raw_ylt = '/home/jackey/ASD/'

    np_X = raw_ylt + 'Local_Data/Xtr.npy'
    np_y = raw_ylt + 'Local_Data/labeltr.npy'

    np_X_te = raw_ylt + 'Local_Data/Xte.npy'
    np_y_te = raw_ylt + 'Local_Data/labelte.npy'
    

    minibatch_size = 40
    nr_channel = 3
    # image_shape = ( 585, 415 )
    image_shape = ( 415, 585 )
    nr_class = 2
    nr_epoch = 160
    weight_decay = 2*1e-5

    borders = [30, 100, 120]
    lr_group = [1e-4, 1e-5, 5e-6, 1e-6]
    if len(borders) != len(lr_group) - 1:
        raise Exception("Incorrect pair: lr_group and borders")

    show_interval = 3
    snapshot_interval = 1
    test_interval = 1

    CUDA_VISIB_DEVI = '2, 3'
    
    
    
    
    
    
    overall_stat = 'overall_stat.txt'
    FP_file = '_FP_100.txt'
    AR_file = '_AR_100.txt'
    term_file = '_term.txt'
    
    test_file = 'test_feature.txt'
    new_file = 'new_feature.txt'
    trial_file = 'trial_feature.txt'

    conf_val = 0.99
    subtrack_down = 170
    supp_ratio = 0.95

    '''where to write model snapshots to'''
    # log_model_dir = os.path.join(log_dir, 'models')

    # exp_name = os.path.basename(log_dir)
    @property
    def input_shape(self):
        return (self.minibatch_size, self.nr_channel) + self.image_shape



config = Config()

def Check(x):
    if not os.path.exists(x):
        os.mkdir(x)




