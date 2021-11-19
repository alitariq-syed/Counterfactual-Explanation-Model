# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 12:23:57 2020

@author: Ali
"""
from tensorflow.python.platform import build_info as tf_build_info
print(tf_build_info.cuda_version_number)
# 9.0 in v1.10.0
print(tf_build_info.cudnn_version_number)