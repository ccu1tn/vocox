"""
To evaluate the performance of the model
@author: Van-Tam
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
import numpy as np

import scipy.io as sio
from model_GNGenetic import sem_enc_model, chan_enc_model, Chan_Model, chan_dec_model, sem_dec_model

from pesq import pesq, pesq_batch
import random

sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()
num_cpus = os.cpu_count()
print("Number of CPU cores is", num_cpus)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True) # assign GPU memory dynamically

###############    define global parameters    ###############
def parse_args():
    parser = argparse.ArgumentParser(description="semantic communication systems for speech transmission")
    
    # parameter of frame
    parser.add_argument("--sr", type=int, default=8000, help="sample rate for wav file")
    parser.add_argument("--num_frame", type=int, default=128, help="number of frames in each batch")
    parser.add_argument("--frame_size", type=float, default=0.016, help="time duration of each frame")
    parser.add_argument("--stride_size", type=float, default=0.016, help="time duration of frame stride")
    
    # parameter of semantic coding and channel coding
    parser.add_argument("--sem_enc_outdims", type=list, default=[32, 128, 128, 128, 128, 128, 128],
                        help="output dimension of SE-ResNet in semantic encoder.")
    parser.add_argument("--chan_enc_filters", type=list, default=[128],
                        help="filters of CNN in channel encoder.")
    parser.add_argument("--chan_dec_filters", type=list, default=[128],
                        help="filters of CNN in channel decoder.")
    parser.add_argument("--sem_dec_outdims", type=list, default=[128, 128, 128, 128, 128, 128, 32],
                        help="output dimension of SE-ResNet in semantic decoder.")
    
    # path of tfrecords files
    parser.add_argument("--trainset_tfrecords_path", type=str, default="./output_tfrecords_1000audio/trainset.tfrecords",
                        help="tfrecords path of trainset.")
    parser.add_argument("--validset_tfrecords_path", type=str, default="./output_tfrecords_1000audio/validset.tfrecords",
                        help="tfrecords path of validset.")
    parser.add_argument("--saved", type=str, default="./saved_model_GNGenetic/600_epochs",
                        help="path of the saved models aftering training.")
    
    # parameter of wireless channel
    parser.add_argument("--snr_train_dB", type=int, default=10, help="snr in dB for training.") #bandau 10 chua test 8db
    
    # epoch and learning rate
    parser.add_argument("--num_epochs", type=int, default=600, help="training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate.")
    
    args = parser.parse_args()
    
    return args

args = parse_args()
print("Called with args:", args)

frame_length = int(args.sr*args.frame_size)
stride_length = int(args.sr*args.stride_size)
saved_model_dir = args.saved
rate = args.sr


print("**********frame_length: {}       stride_length: {}**********\n".format(frame_length,stride_length))

if __name__ == "__main__":
    
    @tf.function
    def map_function(example):

        feature_map = {"wav_raw": tf.io.FixedLenFeature([], tf.string)}
        parsed_example = tf.io.parse_single_example(example, features=feature_map)
        
        wav_slice = tf.io.decode_raw(parsed_example["wav_raw"], out_type=tf.int16)
        wav_slice = tf.cast(wav_slice, tf.float32) / 2**15

        return wav_slice
        
    @tf.function
    def test_step(_input, std):

            
        std = tf.cast(std, dtype=tf.float32)
        
        _output, batch_mean, batch_var = sem_enc(_input)
        _output = chan_enc(_output)
        _output = chan_layer(_output, std)
        _output = chan_dec(_output)
        _output = sem_dec([_output, batch_mean, batch_var])


        return _output
    pop_size=100
    elite_size=50
    noise_params=100
    set_session(sess)
    score = []
    for db in range(0,20):
        
        snr = pow(10, ( db / 10))
        std = np.sqrt(1 / (2*snr))
        std = tf.cast(std, dtype=tf.float32)

        ###############    Load system model    ###############
        
        # load semantic encoder
        sem_enc_h5 = saved_model_dir + "/sem_enc.h5"
        sem_enc = tf.keras.models.load_model(sem_enc_h5,compile=False)
                
        # load channel encoder
        chan_enc_h5 = saved_model_dir + "/chan_enc.h5"
        chan_enc = tf.keras.models.load_model(chan_enc_h5,compile=False )
                
        # load channel_decoder
        chan_dec_h5 = saved_model_dir + "/chan_dec.h5"
        chan_dec = tf.keras.models.load_model(chan_dec_h5,compile=False)
                
        # load semantic_decoder
        sem_dec_h5 = saved_model_dir + "/sem_dec.h5"
        sem_dec = tf.keras.models.load_model(sem_dec_h5,compile=False)


        # define channel model
        #chan_layer = Chan_Model(name="Channel_Model")
        chan_layer = Chan_Model("Chan_Model", pop_size, elite_size, noise_params)
        trainset = tf.data.TFRecordDataset(args.trainset_tfrecords_path)
        
        trainset = trainset.map(map_func=map_function, num_parallel_calls=num_cpus) # num_parallel_calls should be number of cpu cores
        
        #trainset = trainset.shuffle(buffer_size=args.batch_size*657, reshuffle_each_iteration=True)
            
        trainset = trainset.batch(batch_size=args.batch_size)
        
        trainset = trainset.prefetch(buffer_size=args.batch_size)
        
        #a = random.uniform(-0.1, 0.1)
        s = 0
        i = 0
        for step, _input in enumerate(trainset):
            i += 1
        
            _output, batch_mean, batch_var = sem_enc(_input)
            _output = chan_enc(_output)
            _output = chan_layer(_output, std)
            _output = chan_dec(_output)
            _output = sem_dec([_output, batch_mean, batch_var])
            _input2 = _input.numpy()
            _test2 = _output.numpy()

            for i in range(_input2.shape[0]):
                s1 = pesq(rate,_input2[i,:],_test2[i,:],'nb')
                s += s1 / _input2.shape[0]
                #break
                
                #print('*************PESQ Score: {}*************\n'.format(score))
        print('s',s)
        print('i',i)
        s = s/i
        print('*************PESQ Score: {}*************\n'.format(s))
        score.append(s)

    score = np.array(score)
    np.save('./results_PESQ/test_results_600_local_epochs.npy',score)



