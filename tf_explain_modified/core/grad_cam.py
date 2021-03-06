"""
Core Module for Grad CAM Algorithm
"""
import cv2
import numpy as np
import tensorflow as tf

from tf_explain_modified.utils.display import grid_display, heatmap_display, heatmap_area_display, heatmap_cutout_display
from tf_explain_modified.utils.saver import save_rgb
import matplotlib.pyplot as plt


class GradCAM:

    """
    Perform Grad CAM algorithm for a given input

    Paper: [Grad-CAM: Visual Explanations from Deep Networks
            via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
    """

    def explain(
        self,
        validation_data,
        model,
        class_index,
        layer_name=None,
        colormap=cv2.COLORMAP_JET,#COLORMAP_VIRIDIS,COLORMAP_JET
        image_weight=0.7,
        image_nopreprocessed=None,# ali added,
        fmatrix = None,
        RF=False,
		heatmap_threshold=0.6,
        heatmap_cutout = False
    ):
        """
        Compute GradCAM for a specific class index.

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class
            layer_name (str): Targeted layer for GradCAM. If no layer is provided, it is
                automatically infered from the model architecture.
            colormap (int): OpenCV Colormap to use for heatmap visualization
            image_weight (float): An optional `float` value in range [0,1] indicating the weight of
                the input image to be overlaying the calculated attribution maps. Defaults to `0.7`.

        Returns:
            numpy.ndarray: Grid of all the GradCAM
        """
        images, _ = validation_data

        # if layer_name is None:
        #     layer_name = self.infer_grad_cam_target_layer(model)

        outputs, guided_grads = GradCAM.get_gradients_and_filters(
            model, images, class_index, fmatrix, RF
        )

        cams = GradCAM.generate_ponderated_output(outputs, guided_grads)


        #ali modified
        if image_nopreprocessed is not None:
            images = image_nopreprocessed
        
        if RF:
            if not heatmap_cutout:
                heatmaps = np.array(
                [
                    # not showing the actual image if image_weight=0
                    heatmap_area_display(cam.numpy(), image, colormap, image_weight,heat_threshold=heatmap_threshold)
                    for cam, image in zip(cams, images)
                ]        )
            else:
                heatmaps = np.array(
                [
                    # not showing the actual image if image_weight=0
                    heatmap_cutout_display(cam.numpy(), image, colormap, image_weight,heat_threshold=heatmap_threshold)
                    for cam, image in zip(cams, images)
                ]        )
                
        else:
            heatmaps = np.array(
            [
                # not showing the actual image if image_weight=0
                heatmap_display(cam.numpy(), image, colormap, image_weight)
                for cam, image in zip(cams, images)
            ]        )
        

        grid = grid_display(heatmaps)

        return grid, cams

    @staticmethod
    def infer_grad_cam_target_layer(model):
        """
        Search for the last convolutional layer to perform Grad CAM, as stated
        in the original paper.

        Args:
            model (tf.keras.Model): tf.keras model to inspect

        Returns:
            str: Name of the target layer
        """
        for layer in reversed(model.layers):
            # Select closest 4D layer to the end of the network.
            if len(layer.output_shape) == 4:
                return layer.name

        raise ValueError(
            "Model does not seem to contain 4D layer. Grad CAM cannot be applied."
        )

    #@staticmethod
    #@tf.function
    def get_gradients_and_filters(model, images, class_index,fmatrix, RF=False):
        """
        Generate guided gradients and convolutional outputs with an inference.

        Args:
            model (tf.keras.Model): tf.keras model to inspect
            images (numpy.ndarray): 4D-Tensor with shape (batch_size, H, W, 3)
            layer_name (str): Targeted layer for GradCAM
            class_index (int): Index of targeted class

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: (Target layer outputs, Guided gradients)
        """
        # grad_model = tf.keras.models.Model(
        #     [model.inputs], [model.get_layer(layer_name).output, model.output]
        # )

        with tf.GradientTape(persistent=True) as tape:
            inputs = tf.cast(images, tf.float32)
            #conv_outputs, predictions = grad_model(inputs)
            
            #predictions,conv_outputs_1,conv_outputs_2,target_1,target_2,raw_map,forward_1 = model(inputs)
            predictions,conv_outputs_2,mean_fmaps,_,pre_softmax = model([inputs,fmatrix])
            
            loss = predictions[:, class_index]
        outputs = conv_outputs_2
        grads = tape.gradient(loss, outputs)
        
        if RF or True:#to ensure heatmap is drawn even if the gradients are negatively associated.
            grads=abs(grads)

        guided_grads = (
            tf.cast(outputs > 0, "float32") * tf.cast(grads > 0, "float32") * grads
        )

        return outputs, guided_grads

    @staticmethod
    def generate_ponderated_output(outputs, grads):
        """
        Apply Grad CAM algorithm scheme.

        Inputs are the convolutional outputs (shape WxHxN) and gradients (shape WxHxN).
        From there:
            - we compute the spatial average of the gradients
            - we build a ponderated sum of the convolutional outputs based on those averaged weights

        Args:
            output (tf.Tensor): Target layer outputs, with shape (batch_size, Hl, Wl, Nf),
                where Hl and Wl are the target layer output height and width, and Nf the
                number of filters.
            grads (tf.Tensor): Guided gradients with shape (batch_size, Hl, Wl, Nf)

        Returns:
            List[tf.Tensor]: List of ponderated output of shape (batch_size, Hl, Wl, 1)
        """

        maps = [
            GradCAM.ponderate_output(output, grad)
            for output, grad in zip(outputs, grads)
        ]
        
        #in above method:
        #OperatorNotAllowedInGraphError: iterating over `tf.Tensor` is not allowed: AutoGraph did not convert this function. Try decorating it directly with @tf.function.
        # maps = [
        #     GradCAM.ponderate_output(outputs, grads)
        #     #for output, grad in zip(outputs, grads)
        # ]

        return maps

    @staticmethod
    def ponderate_output(output, grad):
        """
        Perform the ponderation of filters output with respect to average of gradients values.

        Args:
            output (tf.Tensor): Target layer outputs, with shape (Hl, Wl, Nf),
                where Hl and Wl are the target layer output height and width, and Nf the
                number of filters.
            grads (tf.Tensor): Guided gradients with shape (Hl, Wl, Nf)

        Returns:
            tf.Tensor: Ponderated output of shape (Hl, Wl, 1)
        """
        weights = tf.reduce_mean(grad, axis=(0, 1))

        # Perform ponderated sum : w_i * output[:, :, i]
        cams = tf.reduce_sum(tf.multiply(weights, output), axis=-1)
       
        # weights = tf.reduce_mean(grad, axis=(1, 2))
        # # Perform ponderated sum : w_i * output[:, :, i]
        # cams=[]
        # for i in range(len(grad)):
        #     cam = tf.reduce_sum(tf.multiply(weights[i], output[i]), axis=-1)
        #     cams.append(cam)
        return cams

    def save(self, grid, output_dir, output_name):
        """
        Save the output to a specific dir.

        Args:
            grid (numpy.ndarray): Grid of all the heatmaps
            output_dir (str): Output directory path
            output_name (str): Output name
        """
        save_rgb(grid, output_dir, output_name)
