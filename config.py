# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 22:48:41 2022

@author: Ali
"""

import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Interpretable CNN')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#choose wheter to train a CF model for a given base model or train a base model from scratch
#parser.add_argument('--create_counterfactual_combined' ,default = True,type=str2bool)## create CF model for a pretrained base model or train a new base model
parser.add_argument('--filter_visualization' ,default = True,type=str2bool) # find top k highest and lowest activation magnitudes for the target filter and the corresponding images

parser.add_argument('--user_evaluation' ,default = False,type=str2bool) # save images

# CF model args
parser.add_argument('--train_counterfactual_net' ,default = True, type=str2bool)## 
parser.add_argument('--train_all_classes' ,default = True, type=str2bool)## 
parser.add_argument('--dropout' ,default = False, type=str2bool)## dont use... not good results

parser.add_argument('--train_singular_counterfactual_net' ,default = False, type=str2bool)## 
parser.add_argument('--choose_subclass' ,default = False, type=str2bool)## choose subclass for training on

parser.add_argument('--counterfactual_PP' ,default = False, type=str2bool)## whether to generate filters for PP  or PN case 
parser.add_argument('--resume_counterfactual_net' ,default = False, type=str2bool)## False = train CF model from scratch; True = resume training CF model
parser.add_argument('--resume_from_epoch' ,default = 30, type=np.int32)## False = train CF model from scratch; True = resume training CF model
parser.add_argument('--test_counterfactual_net' ,default = False, type=str2bool)## 
parser.add_argument('--load_counterfactual_net',default = True, type=str2bool)
parser.add_argument('--resume', default =True, type=str2bool) # load saved weights for base model
parser.add_argument('--alter_class', default = 10, type = np.int32) # alter class #misclassified classes 9-170
parser.add_argument('--analysis_class', default = 6, type = np.int32) # class for which images are loaded and analyzed
parser.add_argument('--find_global_filters', default = False, type=str2bool) # perform statistical analysis to find the activation magnitude of all filters for the alter class and train images of alter class
#parser.add_argument('--alter_class_2', default = 0, type = np.int32) # alter class for 2nd example, 9, 170, 25, 125, 108
parser.add_argument('--cfe_epochs', default = 10, type = np.int32 ) #100 for mnist, 200 for CUB
parser.add_argument('--l1_weight', default = 2, type = np.float32) # 2 default
parser.add_argument('--save_logFile', default = True, type=str2bool) #

#parser.add_argument('--pretrained', default = False) # load self-pretrained model for cifar dataset... i.e. load base model already trained on cifar-10

# common args
parser.add_argument('--augmentation' ,default = True, type=str2bool)## 

#base model parameters
parser.add_argument('--dataset',default = 'fmnist')#NIST, BraTS,mnist, cifar10, CUB200, #cxr1000, #catsvsdogs, #VOC2010, #fmnist
parser.add_argument('--save_directory',default = './trained_weights/')
parser.add_argument('--train_using_builtin_fit_method',default = True)#for training base model easily
parser.add_argument('--train',default = False)
parser.add_argument('--fine_tune',default = False) # fine tune all weights after transfer learning step (CUB dataset)
parser.add_argument('--test', default = True)
parser.add_argument('--model',default = 'customCNN/')#customCNN, VGG16, resnet50,efficientnet, inceptionv3
parser.add_argument('--imagenet_weights',default = False) #use imageNet pretrained weights (True for CUB dataset)

KAGGLE = False

if KAGGLE: 
    args = parser.parse_known_args()[0]
    args.save_directory = "/kaggle/working/trained_weights/" 
    kaggle_load_dir = "/kaggle/input/train-cfe-model-kaggle/trained_weights/"
else: 
    args = parser.parse_args()

if (args.train_counterfactual_net and args.train_all_classes):
    dropout = ""
    if args.dropout:
        dropout = "dropout"
    weights_path = args.save_directory+args.model+args.dataset+'/all_clases/'+dropout+'/epochs_'+str(args.cfe_epochs)
    if KAGGLE:
        resume_path = kaggle_load_dir+args.model+args.dataset+'/all_clases/'+dropout+'/epochs_'+str(args.resume_from_epoch)
    else:
        resume_path = args.save_directory+args.model+args.dataset+'/all_clases/'+dropout+'/epochs_'+str(args.resume_from_epoch)
    pretrained_weights_path = args.save_directory+args.model+args.dataset+'/standard'
else:
    weights_path = args.save_directory+args.model+args.dataset+'/standard'
    pretrained_weights_path = weights_path
    resume_path = args.save_directory+args.model+args.dataset+'/standard/epochs_'+str(args.resume_from_epoch)
