#!/usr/bin/env python3

# Description: like imprint_weights, but using a pre-inferred output in .txt

import numpy as np
import os
import argparse
import collections

parser = argparse.ArgumentParser(description='Retrain the fully connected layer of a model using the HEF file and new samples')
parser.add_argument('--hef', help="HEF file path")
parser.add_argument('--npz_input', help="Path to the NPY to GET OLD weights&bias of the fully connected layer, that takes the output of the HEF as input")
parser.add_argument('--npz_output', help="Path to the NPY to SAVE NEW weights&bias of the fully connected layer, that takes the output of the HEF as input")
parser.add_argument('--input-images', help="Path to directory structure of samples of new classes to train the model on. The directory structure should be as follows: <input-images>/<#NUM#.class-name>/<image-name>.jpg where #NUM# is an integer (0, 1, ...) enumerating the class")  
args = parser.parse_args()

# TODO: run the bash file that applies the pipeline

if (not args.hef or not args.input_images):
    raise ValueError('You must define hef path and input images path in the command line. Run with -h for additional info')

images_path = args.input_images
new_weights_d = collections.OrderedDict()    
feature_instance_mtxs_by_class = {}
for class_dir in os.listdir(images_path):
    fname = os.path.join(images_path, class_dir, 'output.txt')
    if not os.path.exists(fname):
        print(f'No output.txt in the folder {class_dir}')
        continue
    class_num = int(class_dir.split('.')[0])     
    try:
        output_lines = open(fname).readlines()
        features = np.array([[int(x) for x in line.split(' ')[:-1]] for line in output_lines])
    except:
        raise ValueError('output file format not valid - expecting lines (one per instance) '+
                         'of space-delimited decimal values of the FE-inferred features vector')
    print(f'Using {features.shape[0]} instances for class {class_num}')    
    feature_vectors_centroid = np.mean(features, axis=0, keepdims=True)
    new_weights_d[class_num] = feature_vectors_centroid
    feature_instance_mtxs_by_class[class_num] = features
    
if new_weights_d=={}:
    print("No results to work with!")
else:        
    # TODO also auto-update the labels file!
    
    #new_weights = np.concatenate((new_weights_d[class_num] for class_num in class_nums), axis=0)
    new_weights = np.concatenate(list(new_weights_d.values()), axis=0)
    # Normalize!
    new_weights /= np.linalg.norm(new_weights, axis=1, keepdims=True) 
    
    weights = dict(np.load(args.npz_input))
    weights['fc.bias'] = np.concatenate((weights['fc.bias'], np.zeros(new_weights.shape[0])), axis=0)  
    weights['fc.weight']  = np.concatenate((weights['fc.weight'], new_weights), axis=0)
    np.savez(args.npz_output, **weights)  
    
    print('--> Imprint weight was successful!\n')
    for cls_num, features in feature_instance_mtxs_by_class.items():
        # print(f'testing with the class {cls_num} train instances.. infer')
        logits = features @ weights['fc.weight'].T
        winners = np.argmax(logits, axis=1)
        print(f'Testing cls_num={cls_num} train instances with the new weights. The top-1 classes are:')
        print(winners)
        print(np.sum(winners==1000), np.sum(winners==1001))
