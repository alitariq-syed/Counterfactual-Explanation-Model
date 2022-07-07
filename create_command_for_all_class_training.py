# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 03:13:28 2022

@author: Ali
"""

import numpy as np

cmd=[]
for i in range(200):
    cmd.append("python train_CFE_model.py --alter_class "+str(i))
    
np.savetxt("commands_list.txt",cmd,fmt="%s")