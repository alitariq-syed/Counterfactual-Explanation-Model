# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 22:50:05 2020

@author: Ali
"""

#%%
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten,Softmax
import numpy as np
import matplotlib.pyplot as plt


#%%
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
        
        self.n_square = np.power(self.n,2)
        self.tao = (0.5)/self.n_square #tao is positive constant
        self.beta = 4 #beta is postive constant
        
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
        self.activation_sums_1 = np.zeros((self.k,num_classes)) # mean activation per class for each filter... then choose argmax as filter category
        self.activation_sums_2 = np.zeros((self.k,num_classes))
        self.class_sums_1 = np.zeros((10))
        self.class_sums_2 = np.zeros((10))

        ### taking it outside the model as labels are not available....


    # Define your layers here.
        self.mask1 = computeMaskedOutput(self.t_p)
        self.mask2 = computeMaskedOutput(self.t_p)

    self.conv1 = Conv2D(self.k, (3, 3), activation='relu',padding='same')

    #self.pool1 = MaxPool2D(pool_size=(2, 2))
    self.flatten = Flatten()
    #self.dense_1 = Dense(32, activation='relu')
    self.maxpool = MaxPool2D()
    self.dense_2 = Dense(num_classes, activation='softmax')


  def create_templates(self):
    t_p = np.zeros((self.k,self.n,self.n,self.n,self.n))
    for k in range(self.k):
        for i in range(self.n):
            for j in range(self.n):
                t_p[k,i,j,:,:] = self.create_template_single(mu=np.array([i,j]))
    #plt.imshow(t_p[0,0,0,:,:],cmap='gray')
    t_p_tensor = tf.convert_to_tensor(t_p)

    #t_n = np.ones((self.n,self.n))*-self.tao
    
    return t_p_tensor#,t_n
  
  def create_template_single(self,mu):
    t_p = np.zeros((self.n,self.n))
    for i in range(self.n):
        for j in range(self.n):
            t_p[i,j] = self.tao*max(1-self.beta*(np.linalg.norm(np.array([i, j]) - mu, ord=1)/self.n),-1)    
    return t_p
      
  def compute_loss(self, x_masked, X_relu, positive_templates, activation_sums, class_sums, gt):
      if gt is not None:
          x = x_masked
          
          gt_argmax = tf.argmax(gt,1)
          #class_sums[gt_argmax[img]] +=1
          classes,_ = np.histogram(gt_argmax)
          class_sums += classes
          #find the mean activations corresponding to classes for each filter
          filter_means = np.zeros(activation_sums.shape)
          #activation_sums = activation_sums.numpy()
          for f in range(x.shape[3]): #number of filters
              filter_1 = x[:,:,:,f]
              
              #filter_1_sums=activation_sums[f,:].numpy()
        
              # for img in range(x.shape[0]):
              #     ind = gt_argmax[img]
              #     filter_1_means[ind]+= tf.reduce_mean(filter_1[img])
              # filter_1_means = tf.convert_to_tensor(filter_1_means)
              
              for img in range(x.shape[0]):
                  activation_sums[f,gt_argmax[img]] += tf.reduce_mean(filter_1[img])
                  #class_sums[gt_argmax[img]] +=1 
              filter_means[f,:] = activation_sums[f,:]/class_sums
              filter_means[f,:][np.isnan(filter_means[f,:])] = 0
              #filter_class = np.argmax(filter_means[f,:])
              
          ## filter categories assigned, now compute loss image/template pair in the batch
          filter_class = np.argmax(filter_means,1)
          pos_filter_indices = tf.where(filter_class==gt_argmax.numpy())
          neg_filters_indices = tf.where(filter_class!=gt_argmax.numpy())
          
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
          trace_xT=0
          trace_xT_2=0
          trace_xT_3=0
          
          loss = 0
          loss2 = 0
          loss3 = 0
          
          logZ_pos = tf.reduce_mean(tf.math.exp(tf.reduce_mean(tf.reduce_mean(x,axis=1,keepdims=True),axis=2,keepdims=True)),axis=0,keepdims=True)
          logZ_pos = tf.math.log(logZ_pos)
          #logZ_pos = tf.reshape(logZ_pos, x.shape[3])
          
          for f in range(x.shape[3]): #number of filters
              templates=[]
              p_T=[]
              
              templates_2=[]
              p_T_2=[]
              
              templates_3=[]
              p_T_3=[]

              filter_1 = x[:,:,:,f]
              f_class = filter_class[f]
              for img in range(x.shape[0]):
                  #for each filter, check if the batch_image and class are same, then stack positive template, else stack negative template
                  if f_class == gt_argmax[img]:
                      templates.append(x[img,:,:,f])#masked and then relu
                      # p_T.append(self.p_tp)
                  else:
                      templates.append(tf.keras.layers.ReLU()(X_relu[img,:,:,f] * self.t_n)) #x-->relu and then masked... investigate which is proper way
                      # p_T.append(self.p_tn)
                      
                  templates_2.append(x[img,:,:,f])
                  p_T_2.append(self.p_tp)
                  
                  templates_3.append(tf.keras.layers.ReLU()(X_relu[img,:,:,f] * self.t_n))
                  p_T_3.append(self.p_tn)
              #reshape templates
              #templates = tf.transpose(templates,[0,2,3,1])
              #t = tf.convert_to_tensor(templates)
              # trace_xT   += tf.math.exp(tf.reduce_mean(templates))
              # trace_xT_2 += tf.math.exp(tf.reduce_mean(templates_2))
              # trace_xT_3 += tf.math.exp(tf.reduce_mean(templates_3))
              trace_xT   += tf.reduce_mean(tf.math.exp(tf.reshape(tf.reduce_mean(tf.reduce_mean(templates,axis=1,keepdims=True),axis=2,keepdims=True),[x.shape[3]])))
              trace_xT_2   +=  tf.reduce_mean(tf.math.exp(tf.reshape(tf.reduce_mean(tf.reduce_mean(templates_2,axis=1,keepdims=True),axis=2,keepdims=True),[x.shape[3]])))
              trace_xT_3   +=  tf.reduce_mean(tf.math.exp(tf.reshape(tf.reduce_mean(tf.reduce_mean(templates_3,axis=1,keepdims=True),axis=2,keepdims=True),[x.shape[3]])))

          Z_t = trace_xT
          Z_t_2 = trace_xT_2
          Z_t_3 = trace_xT_3
          
          
          p_xT=0
          p_xT_2=0
          p_xT_3=0
          for f in range(x.shape[3]): #number of filters
              templates=[]
              p_T=[]
              
              templates_2=[]
              p_T_2=[]
              
              templates_3=[]
              p_T_3=[]

              filter_1 = x[:,:,:,f]
              f_class = filter_class[f]
              for img in range(x.shape[0]):
                  #for each filter, check if the batch_image and class are same, then stack positive template, else stack negative template
                  if f_class == gt_argmax[img]:
                      templates.append(x[img,:,:,f])#masked and then relu
                      p_T.append(self.p_tp)
                  else:
                      templates.append(tf.keras.layers.ReLU()(X_relu[img,:,:,f] * self.t_n)) #x-->relu and then masked... investigate which is proper way
                      p_T.append(self.p_tn)
                      
                  templates_2.append(x[img,:,:,f])
                  p_T_2.append(self.p_tp)
                  
                  templates_3.append(tf.keras.layers.ReLU()(X_relu[img,:,:,f] * self.t_n))
                  p_T_3.append(self.p_tn)
              #reshape templates
              #templates = tf.transpose(templates,[0,2,3,1])
              #t = tf.convert_to_tensor(templates)
             
              #tf.reshape(tf.reduce_mean(tf.reduce_mean(templates,axis=1,keepdims=True),axis=2,keepdims=True),[32])
              #p_xT   += tf.math.exp(tf.reduce_mean(templates))/Z_t
              #p_xT_2 += tf.math.exp(tf.reduce_mean(templates_2))/Z_t_2
              #p_xT_3 += tf.math.exp(tf.reduce_mean(templates_3))/Z_t_3
              
              p_xT   +=tf.math.exp(tf.reshape(tf.reduce_mean(tf.reduce_mean(templates,axis=1,keepdims=True),axis=2,keepdims=True),[x.shape[3]]))/Z_t
              p_xT_2   +=tf.math.exp(tf.reshape(tf.reduce_mean(tf.reduce_mean(templates_2,axis=1,keepdims=True),axis=2,keepdims=True),[x.shape[3]]))/Z_t_2
              p_xT_3   +=tf.math.exp(tf.reshape(tf.reduce_mean(tf.reduce_mean(templates_3,axis=1,keepdims=True),axis=2,keepdims=True),[x.shape[3]]))/Z_t_3
              
              p_x = tf.reduce_mean(tf.cast(p_T,tf.float32)*p_xT)
              loss += tf.reduce_mean((tf.cast(p_T,tf.float32)*p_xT*tf.math.log(p_xT/p_x)))
              
              p_x_2 = tf.reduce_mean(tf.cast(p_T_2,tf.float32)*p_xT_2)
              loss2 += tf.reduce_mean((tf.cast(p_T_2,tf.float32)*p_xT_2*tf.math.log(p_xT_2/p_x_2)))
              
              p_x_3 = tf.reduce_mean(tf.cast(p_T_3,tf.float32)*p_xT_3)
              loss3 += tf.reduce_mean((tf.cast(p_T_3,tf.float32)*p_xT_3*tf.math.log(p_xT_3/p_x_3)))
              
          #Z_t = tf.reduce_mean(exp_trace,axis=1, keepdims=True)
          #p_x_T = exp_trace/Z_t
          
          # p_x = tf.cast(tf.reduce_mean(p_T),tf.float32)*p_xT
          # loss = -(tf.cast(tf.reduce_mean(p_T),tf.float32)*p_xT*tf.math.log(p_xT/p_x))
          
          # p_x_2 = tf.cast(tf.reduce_mean(p_T_2),tf.float32)*p_xT_2
          # loss2= -(tf.cast(tf.reduce_mean(p_T_2),tf.float32)*p_xT_2*tf.math.log(p_xT_2/p_x_2))   
          
          # p_x_3 = tf.cast(tf.reduce_mean(p_T_3),tf.float32)*p_xT_3
          # loss3 = -(tf.cast(tf.reduce_mean(p_T_3),tf.float32)*p_xT_3*tf.math.log(p_xT_3/p_x_3))
              # treat trace_xT as the filter loss. If all filter and image class pairs are corresponding then trace value should be high. Else if most filter are nefative then trace value should be low.
              
              # create list of loss for all filter positive and negative
              # then check select form them corresponding to filter classes
          
          
          
          
          return loss, activation_sums, class_sums

      return 1
      
      
  # def get_masked_output_pooling(self,x):
  #   out = tf.nn.max_pool_with_argmax(x, ksize=(x.shape[1],x.shape[2]), strides=(x.shape[1],x.shape[2]), 
  #                          output_dtype=tf.dtypes.int64, include_batch_in_index=True,padding='SAME')
  #   indices = tf.unravel_index(tf.reshape(out[1],[-1]),x.shape)
  #   #indices returning indices of shape (4,1024) --> 4=(batch,x,y,filters); 1024 = 32*32 (batch*filters)
    
  #   #The indices in argmax are flattened, so that a maximum value at position [b, y, x, c] becomes 
  #   #flattened index: (y * width + x) * channels + c
  #   # 800 = (0,2,5,0) = (2*32+5)*32+0
    
  #   #selected_templates = t_p[indices]
  #   templates = []#faster
  #   for i in range(x.shape[0]):
  #       for j in range(x.shape[3]):
  #           st = indices[1:3,(i*x.shape[3] + j)]
  #           template = self.t_p[j,st[0],st[1],:,:]
  #           templates.append(template)
  #   templates = tf.convert_to_tensor(templates)
  #   templates = tf.stack([templates[i*x.shape[0]:i*x.shape[0]+x.shape[3],:,:] for i in range(x.shape[0])])
  #   templates = tf.transpose(templates,[0,2,3,1])
    
  #   # templates = []#slower
  #   # for i in range(x.shape[0]):
  #   #     st = indices[1:3,i*x.shape[0]:i*x.shape[0]+x.shape[3]]
  #   #     template = tf.stack([self.t_p[0,st[0,j],st[1,j],:,:] for j in range(x.shape[3])])
  #   #     templates.append(template)
  #   # templates = tf.convert_to_tensor(templates)
  #   # templates = tf.transpose(templates,[0,2,3,1])
    
  #   masked = x*templates
  #   masked = tf.keras.layers.ReLU()(masked)
    
  #   return masked

  def call(self, inputs, gt=None):
    # Define your forward pass here,
    # using layers you previously defined (in `__init__`).
    loss1=0
    loss2=0
    if self.args.interpretable:
        x = self.base_model(inputs)
        x1, X1_relu, positive_templates1 = self.mask1(x)
        x = self.conv1(x1)
        x2, X2_relu, positive_templates2 = self.mask2(x)
        x = self.maxpool(x2)
        x = self.flatten(x)
        #x = self.dense_1(x)
        x = self.dense_2(x)
        
        if gt is not None:
            loss1, activation_sums_1, class_sums_1 = self.compute_loss(x1,X1_relu, positive_templates1,self.activation_sums_1,self.class_sums_1,gt)
            loss2, activation_sums_2, class_sums_2 = self.compute_loss(x2,X2_relu, positive_templates2, self.activation_sums_2,self.class_sums_2,gt)
            
            self.activation_sums_1 = activation_sums_1
            self.activation_sums_2 = activation_sums_2
            self.class_sums_1 = class_sums_1
            self.class_sums_2 = class_sums_2
            
        
    else:
        x1 = self.base_model(inputs)
        #x1 = self.get_masked_output_pooling(x)
        x2 = self.conv1(x1)
        #x2 = self.get_masked_output_pooling(x)
        x = self.flatten(x2)
        #x = self.dense_1(x)
        x = self.dense_2(x)
        
    return x, x1, x2, loss1,loss2

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
    indices = tf.unravel_index(tf.reshape(out[1],[-1]),x.shape)
    #indices returning indices of shape (4,1024) --> 4=(batch,x,y,filters); 1024 = 32*32 (batch*filters)
    
    #The indices in argmax are flattened, so that a maximum value at position [b, y, x, c] becomes 
    #flattened index: (y * width + x) * channels + c
    # 800 = (0,2,5,0) = (2*32+5)*32+0
    
    #selected_templates = t_p[indices]
    templates = []#faster
    for i in range(x.shape[0]):
        for j in range(x.shape[3]):
            st = indices[1:3,(i*x.shape[3] + j)]
            template = self.t_p[j,st[0],st[1],:,:]
            templates.append(template)
    templates = tf.convert_to_tensor(templates)
    templates = tf.stack([templates[i*x.shape[0]:i*x.shape[0]+x.shape[3],:,:] for i in range(x.shape[0])])
    templates = tf.transpose(templates,[0,2,3,1])
    
    # templates = []#slower
    # for i in range(x.shape[0]):
    #     st = indices[1:3,i*x.shape[0]:i*x.shape[0]+x.shape[3]]
    #     template = tf.stack([self.t_p[0,st[0,j],st[1,j],:,:] for j in range(x.shape[3])])
    #     templates.append(template)
    # templates = tf.convert_to_tensor(templates)
    # templates = tf.transpose(templates,[0,2,3,1])
    
    masked = x*templates
    masked = tf.keras.layers.ReLU()(masked)
    #convolution????
    
    #X_relu = tf.keras.layers.ReLU()(x) #original x after relu (for loss computation)
    # x is already relud from previous layer
    
    return masked, x, templates
#%%
    
if __name__ == '__main__':
    model = MySubClassModel()