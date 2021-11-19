# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 22:50:05 2020

@author: Ali
"""
#models10: implement idea 2: fine tune from forward only with fixed classes
#%%
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten,Softmax,GlobalAveragePooling2D
import numpy as np
import matplotlib.pyplot as plt


#%% 
#TODO 0. (fixed) bug in nan not being replaced by zero: tf.clip_by_value(activation_sums[f,:]/class_sums, 0, tf.reduce_max(activation_sums))
#TODO 1. (fixed) Debug target map creation
#TODO 2. (keep on adding) debug mean filters at end of epoch... what happen when new epoch starts.... do class sums keep on adding or start from 0
#TODO 3. (done) figure out way to save and restore model paramters such as filter_means/classes 
#TODO 4. Dont directly modify feature maps... instead use masked featuremaps to reduce loss only
#TODO 5. (done) Dont modify filter sums/classes when predicting only
#TODO 6. (partially done) positive template for empty input fmap should be zeros instead of activation at corner
class MySubClassModel(Model):

  def __init__(self,base_model,args, num_classes=10):
    super(MySubClassModel, self).__init__(name='my_model')
    self.num_classes = num_classes
    self.args = args
    #it should be a pre-trained model according to paper
    self.base_model = base_model# assume: without top layers (no classification layers)
    self.k = self.base_model.get_output_shape_at(0)[3] #number of output filters
    base_model.summary()
    if args.interpretable:
        output_shape = self.base_model.get_output_shape_at(0)
        self.n = output_shape[1] #target conv layer output size
        #self.k = output_shape[3] 
        
        self.filter_means = None
        self.templates = None
        
        self.n_square = np.power(self.n,2)
        self.tao = (0.5)/self.n_square #tao is positive constant #controls max value of fmap
        self.beta = 2 #beta is postive constant # controls mask spread
        
        #create templates corresponding to each feature map (i.e output shape, (none, 11,11,32) --> means 32 templates of size 11x11)
        self.t_p = tf.cast(self.create_templates(),tf.float32) #positive templates for each filter; shape: (kxnxnx(nxn))
        # single negative template: it should be appended with the positive templates, to be used during backpropagation if Image does not belong to target class
        self.t_n = tf.cast(tf.convert_to_tensor(np.ones((self.n,self.n))*-self.tao),tf.float32)
    
    
        """for loss computation"""
        self.alpha = self.n_square/(1+self.n_square)
        
        #prior prob of positive template:
        self.p_tp = self.alpha#self.alpha/self.n_square
        #prior prob of negative template:
        self.p_tn = 1-self.alpha
        
        #TODO: initialize with some value or load saved/predetermined categories depending on selected method
        if (args.resume and not args.fixed_classes):
            path = args.save_path
            activation_sums_1 = np.load(file = path+'/activations_sum1.npy')
            activation_sums_2 = np.load(file = path+'/activations_sum2.npy')
            class_sums_1 = np.load(file = path+'/class_sum1.npy')
            class_sums_2 = np.load(file = path+'/class_sum2.npy')


            self.activation_sums_1 = tf.Variable(initial_value=activation_sums_1,dtype=tf.dtypes.float32,trainable=False) # mean activation per class for each filter... then choose argmax as filter category
            self.activation_sums_2 = tf.Variable(initial_value=activation_sums_2,dtype=tf.dtypes.float32,trainable=False)
            self.class_sums_1 = tf.Variable(initial_value=class_sums_1,dtype=tf.dtypes.float32,trainable=False)
            self.class_sums_2 = tf.Variable(initial_value=class_sums_2,dtype=tf.dtypes.float32,trainable=False)
        elif not args.fixed_classes:
            self.activation_sums_1 = tf.Variable(initial_value=tf.zeros((self.k,num_classes),dtype=tf.float32),dtype=tf.dtypes.float32,trainable=False) # mean activation per class for each filter... then choose argmax as filter category
            self.activation_sums_2 = tf.Variable(initial_value=tf.zeros((self.k,num_classes),dtype=tf.float32),dtype=tf.dtypes.float32,trainable=False)
            self.class_sums_1 = tf.Variable(initial_value=tf.zeros((num_classes),dtype=tf.float32),dtype=tf.dtypes.float32,trainable=False)
            self.class_sums_2 = tf.Variable(initial_value=tf.zeros((num_classes),dtype=tf.float32),dtype=tf.dtypes.float32,trainable=False)
            
        # initialize class map and templates
        if args.fixed_classes:
            self.class_templates = self.create_class_templates()

    # Define your layers here.
        if args.fixed_classes:
            self.mask1 = computeMaskedOutput_fixed_class(self.t_p,self.class_templates,self.args.fixed_classes_reduce_loss)
            self.mask2 = computeMaskedOutput_fixed_class(self.t_p,self.class_templates,self.args.fixed_classes_reduce_loss)
        else:
            self.mask1 = computeMaskedOutput(self.t_p)
            self.mask2 = computeMaskedOutput(self.t_p)
    if args.dataset == 'cxr1000': #for VGG model
        self.conv1 = Conv2D(self.k, (3, 3), activation='relu',padding='same')
        self.maxpool = MaxPool2D()
        self.flatten = GlobalAveragePooling2D()
        self.dense_2 = Dense(num_classes, activation='sigmoid')
    else:
        self.conv1 = Conv2D(self.k, (3, 3), activation='relu',padding='same')
        #self.pool1 = MaxPool2D(pool_size=(2, 2))
        self.flatten = Flatten()
        #self.dense_1 = Dense(32, activation='relu')
        self.maxpool = MaxPool2D()
        self.dense_2 = Dense(num_classes, activation='softmax')

  def create_class_templates(self):
        k=self.k
        n_classes = self.num_classes
        n = self.n
        
        filters_per_class = tf.math.round(k/n_classes)        
        class_map = np.zeros((k),dtype=np.uint8)
        count=0
        for i in range(n_classes):
            for j in range(int(filters_per_class)):
                class_map[count]= i
                count+=1
                if count == k:
                    break
        
        class_templates = np.stack([np.zeros((n,n),dtype=np.float32) for i in range(k)])
        class_templates = np.stack([class_templates for i in range(n_classes)])
        
        class_templates = np.transpose(class_templates,[0,2,3,1])
        #class_templates dimensions = class,filters,fmapx,fmapy
        
        # create class_templates for each class
        for i in range(k):
            class_templates[class_map[i],:,:,i] = np.ones((n,n))
        
        class_templates = tf.convert_to_tensor(class_templates)
        return class_templates

  def create_templates(self):
    # t_p = np.zeros((32,self.k,self.n,self.n,self.n,self.n))
    # for l in range(32):
    #     for k in range(self.k):
    #         for i in range(self.n):
    #             for j in range(self.n):
    #                 t_p[l,k,i,j,:,:] = self.create_template_single(mu=np.array([i,j]))
    #     #plt.imshow(t_p[0,0,0,:,:],cmap='gray')
    # t_p_tensor = tf.convert_to_tensor(t_p)
    # t_p_tensor = tf.transpose(t_p_tensor,[0,2,3,1,4,5])
      
    #append negative template to t_p and use it for empty fmaps,,, check if it affects other code
    t_p = np.zeros((self.n+1,self.n+1,self.n,self.n))
    for i in range(self.n):
        for j in range(self.n):
            t_p[i,j,:,:] = self.create_template_single(mu=np.array([i,j]))
    t_p[i+1,j+1] = tf.zeros((self.n,self.n))
        #plt.imshow(t_p[0,0,0,:,:],cmap='gray')
    #t_p_tensor = tf.stack([t_p for i in range(self.k)])
    #t_p_tensor = tf.stack([t_p_tensor for i in range(32)])

    #t_p_tensor = tf.convert_to_tensor(t_p)
    #t_p_tensor = tf.transpose(t_p_tensor,[0,2,3,1,4,5])
    
    return t_p#,t_n
  
  def create_template_single(self,mu):
    t_p = np.zeros((self.n,self.n))
    for i in range(self.n):
        for j in range(self.n):
            t_p[i,j] = self.tao*max(1-self.beta*(np.linalg.norm(np.array([i, j]) - mu, ord=1)/self.n),-1)    
    return t_p

  def compute_loss(self, x_masked, X_relu, positive_templates, activation_sums, class_sums, gt, training):
      if gt is not None:
          x = x_masked
          
          gt_argmax = tf.argmax(gt,1,output_type=tf.dtypes.int32)
          #class_sums[gt_argmax[img]] +=1
          #classes,_ = np.histogram(gt_argmax)
          classes = tf.math.bincount(
            gt_argmax, weights=None, minlength=self.num_classes, maxlength=None, dtype=tf.dtypes.float32,
            name=None
            )          
          #class_sums += classes
          if training:
              class_sums.assign(class_sums+classes)

          #find the mean activations corresponding to classes for each filter
          #filter_means = np.zeros(activation_sums.shape)
          if self.filter_means is None:
              self.filter_means = tf.Variable(initial_value=tf.zeros(activation_sums.shape,dtype=tf.float32),dtype=tf.dtypes.float32,trainable=False)
          else:
              self.filter_means.assign(tf.zeros(activation_sums.shape,dtype=tf.float32))
              #tf.variable is requied becase we cannot assign new values to a tf.tensor
          
          #activation_sums = activation_sums.numpy()
          for f in range(x.shape[3]): #number of filters
              filter_1 = x[:,:,:,f]
              
              #filter_1_sums=activation_sums[f,:].numpy()
        
              # for img in range(x.shape[0]):
              #     ind = gt_argmax[img]
              #     filter_1_means[ind]+= tf.reduce_mean(filter_1[img])
              # filter_1_means = tf.convert_to_tensor(filter_1_means)
              if training:
                  for img in range(x.shape[0]):
                      # activation_sums[f,gt_argmax[img]] += tf.reduce_mean(filter_1[img])
                      activation_sums[f,gt_argmax[img]].assign(activation_sums[f,gt_argmax[img]]+tf.reduce_mean(filter_1[img]))
                      #class_sums[gt_argmax[img]] +=1 
                      
                #TODO:      #can be sped up:
          ######mean_fmap_activations = GlobalAveragePooling2D()(fmaps)
          ######for img in range(fmaps.shape[0]):#number of images/batch size
          ######    class_activation_sums[:,gt_argmax[img]] = class_activation_sums[:,gt_argmax[img]]+mean_fmap_activations[img,:]

              #filter_means[f,:].assign(activation_sums[f,:]/class_sums)
             # self.filter_means[f,:].assign(tf.math.divide_no_nan(activation_sums[f,:],class_sums))
                  

              #filter_means[f,:][tf.math.is_nan(filter_means[f,:])].assign(0)
              #filter_class = np.argmax(filter_means[f,:])
              
          ## filter categories assigned, now compute loss image/template pair in the batch
          self.filter_means.assign(tf.math.divide_no_nan(activation_sums,class_sums))
          filter_class = tf.argmax(self.filter_means,1,output_type=tf.dtypes.int32)
          # pos_filter_indices = tf.where(filter_class==gt_argmax.numpy())
          # neg_filters_indices = tf.where(filter_class!=gt_argmax.numpy())
          
          #implement the -MI loss 
          #Loss_f = −MI(X;T) = − Sigma_T(p(T)) Sigma_x(p(x|T)) log(p(x|T)/ p(x))

          #p(T_positive) = alpha/n2
          #p(T_negative) = 1- alpha
                    
          #The fitness between a feature map x and a template T is measured as the conditional ikelihood p(x|T):  
          #p(x|T) = 1/Z_T exp [tr(x · T)]
              #Z_T = Sigma_x exp(tr(x · T))
              #tr(x · T) = Sigma_ij (x_ij * t_ij)
          
          #p(x) = Sigma_T (p(T)p(x|T))
          
          # n_p = tf.math.count_nonzero(pos_filters)#number of positive filter/batch_images
          # n_n = x.shape[3] - n_p
          # p_T = tf.reduce_mean(n_p*self.p_tp + n_n*self.p_tn)
          #TODO: investigate and fix values of p_tp and p_tn
          
          ######### paper method attempt
          #combine positive and negative templates
          # trace_xT=0
          # for f in range(x.shape[3]): #number of filters
          #     templates=[]
          #     filter_1 = x[:,:,:,f]
          #     f_class = filter_class[f]
          #     for img in range(x.shape[0]):
          #         #for each filter, check if the batch_image and class are same, then stack positive template, else stack negative template
          #         if f_class == gt_argmax[img]:
          #             templates.append(x[img,:,:,f])#masked and then relu
          #         else:
          #             templates.append(tf.keras.layers.ReLU()(X_relu[img,:,:,f] * self.t_n)) #x-->relu and then masked... investigate which is proper way
          #     #reshape templates
          #     #templates = tf.transpose(templates,[0,2,3,1])
          #     trace_xT += tf.reduce_mean(templates)
          # Z_T = tf.math.exp(trace_xT)    
          
          #px_T
          #########
          
          #########own idea attempt
          pos_templates = tf.keras.layers.ReLU()(positive_templates)
          
          #create batch of target templates
          #target_templates=[]
          #target_templates=tf.zeros_like(positive_templates)
          if self.templates is None:
                  self.templates = tf.Variable(initial_value=tf.zeros_like(positive_templates),dtype=tf.dtypes.float32,trainable=False)
          else: 
                  self.templates[0:filter_1.shape[0]].assign(tf.zeros_like(positive_templates))
          for f in range(x.shape[3]): #number of filters
     
              filter_1 = x[:,:,:,f]
              f_class = filter_class[f]
              
              pos_indices = tf.where(f_class==gt_argmax)
              neg_indices = tf.where(f_class!=gt_argmax)

              # templates=tf.zeros_like(filter_1)

              
#               ind = tf.gather_nd(
#     pos_templates[:,:,:,f], tf.where(f_class==gt_argmax),  batch_dims=0, name=None
# )
              #for i in range(pos_indices.shape[0]):
              for i in range(len(pos_indices)):
                  self.templates[pos_indices[i][0],:,:,f].assign(pos_templates[pos_indices[i][0],:,:,f])
                  #templates[i[0],:,:].assign(tf.gather_nd(pos_templates[:,:,:,f], tf.where(f_class==gt_argmax),  batch_dims=0, name=None)[count])
              for i in range(len(neg_indices)):
                  self.templates[neg_indices[i][0],:,:,f].assign(tf.keras.layers.ReLU()(self.t_n))
              # for img in range(x.shape[0]):
              #     #for each filter, check if the batch_image and class are same, then stack positive template, else stack negative template
              #     if f_class == gt_argmax[img]:
              #         templates.append(pos_templates[img,:,:,f])#masked and then relu
              #         # p_T.append(self.p_tp)
              #     else:
              #         templates.append(tf.keras.layers.ReLU()(self.t_n)) #x-->relu and then masked... investigate which is proper way
              #         # p_T.append(self.p_tn)
                  
              #     #result= tf.cond(f_class == gt_argmax[img], lambda: templates_return_pos(pos_templates[img,:,:,f]), lambda: templates_return_neg(self.t_n))
              #     #templates.append(result)

              # #templates
              #target_templates.append(self.templates)#not working as intended
          #target_templates = tf.transpose(tf.convert_to_tensor(target_templates),[0,2,3,1])    
          #check =  tf.convert_to_tensor(target_templates)  

          #loss = tf.reduce_mean(tf.keras.losses.mae(target_templates,x))
         # loss = 1000*np.mean(abs(target_templates-x))
          #loss not getting reduced this way
          loss=self.templates[0:filter_1.shape[0]]
          # #%% plot all batch filters/ maps
          # for j in range(1):
          #       fig, axs = plt.subplots(8,4, figsize=(15, 15))#, facecolor='w', edgecolor='k')
          #       #fig.subplots_adjust(hspace = .5, wspace=.001)
          #       axs = axs.ravel()
          #       for i in range(32):
                
          #           axs[i].imshow(x_masked[j,:,:,i],cmap='gray')#contourf(np.random.rand(10,10),5,cmap=plt.cm.Oranges)
          #           axs[i].axis('off')
                    
          #       plt.show()
          # for j in range(1):
          #       fig, axs = plt.subplots(8,4, figsize=(15, 15))#, facecolor='w', edgecolor='k')
          #       #fig.subplots_adjust(hspace = .5, wspace=.001)
          #       axs = axs.ravel()
          #       for i in range(32):
                
          #           axs[i].imshow(loss[j,:,:,i],cmap='gray')#contourf(np.random.rand(10,10),5,cmap=plt.cm.Oranges)
          #           axs[i].axis('off')
                    
          #       plt.show()

          return loss, activation_sums, class_sums

      return 1
      
  def compute_loss_fixed_class(self, x_masked, X_relu, positive_templates,  gt, training):
      if gt is not None:
          x = x_masked
          
          gt_argmax = tf.argmax(gt,1,output_type=tf.dtypes.int32)
  
          batch = gt_argmax
          batch_templates=[]
          # for i in range(batch.shape[0]):
          #     batch_templates.append(self.class_templates[batch[i]])
          # batch_templates = tf.convert_to_tensor(batch_templates)
          
          batch_templates = tf.stack([self.class_templates[batch[i]] for i in range(batch.shape[0])])
          
          #testing:
          # empty_index = tf.where(tf.reduce_max(tf.reduce_max(x,axis=1),axis=1)==0.0)
          # #positive_templates[empty_index] = tf.zeros_like()

          pos_templates = tf.keras.layers.ReLU()(positive_templates)
          
          # if self.templates is None:
          #         self.templates = tf.Variable(initial_value=pos_templates,dtype=tf.dtypes.float32,trainable=False)
          # else: 
          #         self.templates[0:x.shape[0]].assign(pos_templates)
          # #handle positive templates for empty feature maps.... instead of default template at corner, it should be empty
          # for i in range(len(empty_index)):
          #   self.templates[empty_index[i][0],:,:,empty_index[i][1]].assign(tf.zeros((12,12)))
          
          # target_templates = self.templates*batch_templates
          target_templates = pos_templates*batch_templates
          loss=target_templates[0:x.shape[0]]

          return loss

      return 1
      


  def call(self, inputs, gt=None,training=False,filters_disabled = None):
    # Define your forward pass here,
    # using layers you previously defined (in `__init__`).
    loss1=0
    loss2=0
    if self.args.interpretable:
        if (self.args.filter_modified_directly and not self.args.fixed_classes_reduce_loss):
            raw_map = self.base_model(inputs)
            x1, X1_relu, positive_templates1 = self.mask1(raw_map)
            x3 = self.conv1(x1)
            x2, X2_relu, positive_templates2 = self.mask2(x3)
            x = self.maxpool(x2)
            x = self.flatten(x)
            #x = self.dense_1(x)
            x = self.dense_2(x)
        elif (self.args.filter_modified_directly and self.args.fixed_classes_reduce_loss):
            raw_map = self.base_model(inputs)
            x1, X1_relu, positive_templates1, loss1 = self.mask1(raw_map,gt,training)
            x3 = self.conv1(x1)
            x2, X2_relu, positive_templates2, loss2 = self.mask2(x3,gt,training,filters_disabled)
            x = self.maxpool(x2)
            x = self.flatten(x)
            #x = self.dense_1(x)
            x = self.dense_2(x)
        else:
            raw_map = self.base_model(inputs)
            x1, X1_relu, positive_templates1 = self.mask1(raw_map)
            x3 = self.conv1(raw_map)
            x2, X2_relu, positive_templates2 = self.mask2(x3)
            x = self.maxpool(x3)
            x = self.flatten(x)
            #x = self.dense_1(x)
            x = self.dense_2(x)
            
            x1=raw_map    
            x2=x3
        
        if gt is not None:
            if (self.args.loss_compute and not self.args.fixed_classes_reduce_loss):
                loss1, activation_sums_1, class_sums_1 = self.compute_loss(x1,X1_relu, positive_templates1,self.activation_sums_1,self.class_sums_1,gt,training)
                loss2, activation_sums_2, class_sums_2 = self.compute_loss(x2,X2_relu, positive_templates2, self.activation_sums_2,self.class_sums_2,gt,training)
                
                self.activation_sums_1 = activation_sums_1
                self.activation_sums_2 = activation_sums_2
                self.class_sums_1 = class_sums_1
                self.class_sums_2 = class_sums_2
            #elif (self.args.loss_compute and self.args.fixed_classes_reduce_loss):
            #    loss1 = self.compute_loss_fixed_class(x1,X1_relu, positive_templates1,gt,training)
            #    loss2 = self.compute_loss_fixed_class(x2,X2_relu, positive_templates2,gt,training)
                
            
        
    else:
        if self.args.full_standard:
            x1 = self.base_model(inputs)
            x = self.maxpool(x1)
            x = self.GlobalAveragePooling2D(x) #flatten(x)
            x = self.dense_2(x)
            
            positive_templates1=None
            x2 = None
            raw_map = x1
        else:
            x1 = self.base_model(inputs)
            raw_map = x1
            positive_templates1=None
            x2 = self.conv1(x1)
            if (not training and filters_disabled is not None):
                if not tf.is_tensor(filters_disabled):
                    disabled = x2.numpy()
                    disabled[:,:,:,filters_disabled]=tf.zeros_like((x2[:,:,:,filters_disabled]))
                    x2 = tf.convert_to_tensor(disabled)
            x = self.maxpool(x1)
            x = self.flatten(x)
            x = self.dense_2(x)
        
    return x, x1, x2, loss1,loss2, raw_map, positive_templates1

#%% method 4: custom layer
class computeMaskedOutput(tf.keras.layers.Layer):
  def __init__(self,t_p):
    super(computeMaskedOutput, self).__init__()
    self.t_p = t_p

  # def build(self, input_shape):
  #   self.kernel = self.add_weight("kernel",
  #                                 shape=[int(input_shape[-1]),
  #                                        self.num_outputs])

  def call(self, input):
    x = input
    out = tf.nn.max_pool_with_argmax(x, ksize=(x.shape[1],x.shape[2]), strides=(x.shape[1],x.shape[2]), 
                          output_dtype=tf.dtypes.int64, include_batch_in_index=True,padding='SAME')
    # out = tf.nn.max_pool_with_argmax(x, ksize=(x.shape[1],x.shape[2]), strides=(x.shape[1],x.shape[2]), 
    #                        output_dtype=tf.dtypes.int64, include_batch_in_index=False,padding='SAME')
    indices = tf.unravel_index(tf.reshape(out[1],[-1]),x.shape)
    #indices returning indices of shape (4,1024) --> 4=(batch,x,y,filters); 1024 = 32*32 (batch*filters)
    
    #The indices in argmax are flattened, so that a maximum value at position [b, y, x, c] becomes 
    #flattened index: (y * width + x) * channels + c
    # 800 = (0,2,5,0) = (2*32+5)*32+0
    
    #selected_templates = self.t_p[indices]
    #selected_templates=tf.stack([self.t_p[i] for i in indices])
    
    #faster
    templates=[]
    templates = tf.stack([self.t_p[indices[1,i],indices[2,i],:,:] for i in range(x.shape[0]*x.shape[3])])
    templates = tf.stack([templates[i*x.shape[3]:i*x.shape[3]+x.shape[3],:,:] for i in range(x.shape[0])])
    templates = tf.transpose(templates,[0,2,3,1])
 
    # templates = []#slower
    # for i in range(x.shape[0]):
    #     for j in range(x.shape[3]):
    #         st = indices[1:3,(i*x.shape[3] + j)]
    #         template = self.t_p[j,st[0],st[1],:,:]
    #         if tf.reduce_max(x[i,:,:,j]) == 0.0:#use blank template for empty fmaps
    #             template = tf.zeros_like(template)
    #         templates.append(template)
    # templates = tf.convert_to_tensor(templates)
    # templates = tf.stack([templates[i*x.shape[3]:i*x.shape[3]+x.shape[3],:,:] for i in range(x.shape[0])])
    # #templates = tf.stack([templates[i*x.shape[3]:i*x.shape[3]+x.shape[0],:,:] for i in range(x.shape[3])])
    # templates = tf.transpose(templates,[0,2,3,1])
    


    masked = x*templates
    masked = tf.keras.layers.ReLU()(masked)
    #convolution????
    
    #X_relu = tf.keras.layers.ReLU()(x) #original x after relu (for loss computation)
    # x is already relud from previous layer
    
    return masked, x, templates
#%% fixed class idea
class computeMaskedOutput_fixed_class(tf.keras.layers.Layer):
  def __init__(self,t_p,class_templates,reduce_loss):
    super(computeMaskedOutput_fixed_class, self).__init__()
    self.t_p = t_p
    self.class_templates=class_templates
    self.reduce_loss=reduce_loss
    self.indices_zero = None

  # def build(self, input_shape):
  #   self.kernel = self.add_weight("kernel",
  #                                 shape=[int(input_shape[-1]),
  #                                        self.num_outputs])

  def call(self, input, gt=None,training=False, filters_disabled=None):
    x = input
    out = tf.nn.max_pool_with_argmax(x, ksize=(x.shape[1],x.shape[2]), strides=(x.shape[1],x.shape[2]), 
                          output_dtype=tf.dtypes.int64, include_batch_in_index=True,padding='SAME')
    # out = tf.nn.max_pool_with_argmax(x, ksize=(x.shape[1],x.shape[2]), strides=(x.shape[1],x.shape[2]), 
    #                        output_dtype=tf.dtypes.int64, include_batch_in_index=False,padding='SAME')
    indices = tf.unravel_index(tf.reshape(out[1],[-1]),x.shape)
    #indices returning indices of shape (4,1024) --> 4=(batch,x,y,filters); 1024 = 32*32 (batch*filters)
    
    #The indices in argmax are flattened, so that a maximum value at position [b, y, x, c] becomes 
    #flattened index: (y * width + x) * channels + c
    # 800 = (0,2,5,0) = (2*32+5)*32+0

    #idea for efficiency
    ## create blank template as part of t_p... use this to stack for empty index
    #testing:
    empty_index = tf.where(tf.reduce_max(tf.reduce_max(x,axis=1),axis=1)==0.0)
    if self.indices_zero is None:
        self.indices_zero = tf.Variable(initial_value=indices,dtype=indices.dtype,trainable=False)
    else: 
        self.indices_zero[:,0:indices.shape[1]].assign(indices)
        
    for i in range(len(empty_index)):
    #for i in range(empty_index.shape[0]):
       self.indices_zero[1:3,empty_index[i][0]*x.shape[3]+empty_index[i][1]].assign((self.t_p.shape[0]-1,self.t_p.shape[0]-1))
    
    #faster, but fault templates when fmap is empty
    templates=[]
    templates = tf.stack([self.t_p[self.indices_zero[1,i],self.indices_zero[2,i],:,:] for i in range(x.shape[0]*x.shape[3])])
    templates = tf.stack([templates[i*x.shape[3]:i*x.shape[3]+x.shape[3],:,:] for i in range(x.shape[0])])
    templates = tf.transpose(templates,[0,2,3,1])
 

    #handle positive templates for empty feature maps.... instead of default template at corner, it should be empty
    #for i in range(len(empty_index)):
    #  templates[empty_index[i][0],:,:,empty_index[i][1]].assign(tf.zeros((12,12)))

    
    # templates = []#slower
    # for i in range(x.shape[0]):
    #     for j in range(x.shape[3]):
    #         #st = indices[1:3,(i*x.shape[3] + j)]
    #         template = self.t_p[indices[1,(i*x.shape[3] + j)],indices[2,(i*x.shape[3] + j)],:,:]
    #         if tf.reduce_max(x[i,:,:,j]) == 0.0:#use blank template for empty fmaps
    #             template = tf.zeros_like(template)
    #         templates.append(template)
    # templates = tf.convert_to_tensor(templates)
    # templates = tf.stack([templates[i*x.shape[3]:i*x.shape[3]+x.shape[3],:,:] for i in range(x.shape[0])])
    # templates = tf.transpose(templates,[0,2,3,1])

    
    masked = x*templates
    masked = tf.keras.layers.ReLU()(masked)

    #now multiply with respective fixed class filters
    #assign class templates for each image in current batch
    #input--> class templates, batch classes
    
    #this ensure that this operation is applied at training time only when gt is available
    batch_templates=[]
    target=None

    if (training and gt is not None):#fix gt check... do not use during testing
        #gt_argmax
        batch = tf.argmax(gt,1,output_type=tf.dtypes.int32)
        
        batch_templates = tf.stack([self.class_templates[batch[i]] for i in range(batch.shape[0])])
        if self.reduce_loss:
            target = templates*batch_templates
        else:
            masked = masked*batch_templates
    if (not training and filters_disabled is not None):
        # disabled = masked.numpy()
        # disabled[:,:,:,filters_disabled[0][0]:filters_disabled[-1][0]+1]=tf.zeros_like((masked[:,:,:,filters_disabled[0][0]:filters_disabled[-1][0]+1]))
        # masked = tf.convert_to_tensor(disabled)
        if tf.is_tensor(filters_disabled):
            disabled = masked.numpy()
            disabled[:,:,:,filters_disabled[0][0]:filters_disabled[-1][0]+1]=tf.zeros_like((masked[:,:,:,filters_disabled[0][0]:filters_disabled[-1][0]+1]))
            masked = tf.convert_to_tensor(disabled)
        else:
            disabled = masked.numpy()
            disabled[:,:,:,filters_disabled]=tf.zeros_like((masked[:,:,:,filters_disabled]))
            masked = tf.convert_to_tensor(disabled)
        # modified=np.zeros_like(masked)
        # modified[:,:,:,filters_disabled[0][0]:filters_disabled[-1][0]+1]=masked[:,:,:,filters_disabled[0][0]:filters_disabled[-1][0]+1]
        # masked = tf.convert_to_tensor(modified)
    
    return masked, x, templates, target
#%%
    
if __name__ == '__main__':
    model = MySubClassModel()