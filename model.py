import os
import numpy as np
import cv2
import rasterio
from patchify import patchify
from math import ceil
from scipy.ndimage import label
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class Preprocess():
    """A class to handle preprocessing image
    
    This class is designed to patching, reconstruct and normalize image.
    
    Attribute:
        - image: The image content for preprocessing
        - patch_size=256 (int): The size of chucked image
        
    Methods:
        - normalize_image(image): Normalizing image using MinMaxScaller and Histogram Equalized
        - create_patches(image, patch_size=256, channel=12): Chunk image to smaller and same size
    """
    def __init__(self, image):
        self.image = image
        self.original_height = self.image.shape[1]
        self.original_width = self.image.shape[2]
        
    def normalize_image(self):
        """Normalize an image for model prediction using MinMaxScaller and Histogram Equalized
        
        Attribute:
            - image: An image content
        
        Returns: (numpy.array): Normalized image
        """
        equalized_bands = []
        for band in range(self.image.shape[2]):
            band_image = self.image[:,:,band]
            #min max scaller
            band_image = np.array((band_image - band_image.min()) / (band_image.max() - band_image.min()))
            band_image = np.nan_to_num(band_image, nan=0.0, posinf=255.0, neginf=0.0)
            band_image = (band_image * 255).astype(np.uint8)
            #his eq
            equalized_band = cv2.equalizeHist(band_image)
            equalized_bands.append(equalized_band)
        nimage = np.stack(equalized_bands, axis=-1)
        nimage = np.transpose(nimage, (1, 2, 0))
        self.image = nimage
        return nimage
    
    def create_image_patches(self, patch_size=256):
        """Chunk image to smaller and same size
        
        Attribute:
            - patch_size (int): Image size after patching
        
        Returns: (numpy.array): Chunked image list with same array size
        """
        patched_images = []
        pad_height = (patch_size - self.image.shape[0] % patch_size) % patch_size
        pad_width = (patch_size - self.image.shape[1] % patch_size) % patch_size
        padded_image = np.pad(self.image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
        
        patches = patchify(padded_image, (patch_size, patch_size, 12), step=patch_size)
        patched_images.extend(patches.reshape(-1, patch_size, patch_size, 12))
        
        return np.array(patched_images)

class Postprocess():
    """A class to handle Post Processing data after model predict. It provides
    method to remove small polygons, selecting rgb channel and reconstruc maks to original image.
    
    Methods:
        - remove_polygons(mask_array, min_size): Removing small polygon for better mask prediction
        - select_rgb_image(image): Selecting a RGB channel from 12 band satellite image
        - reconstruct_image(patched_image, original_height, original_width, patch_size): Reconstruct a chunked image to original size
    """
    def remove_polygons(self, mask_array, min_size=10):
        """Removing small polygon for better mask prediction

        Atrribute:
            - mask_array (numpy.array): A predicted mask array
            - min_size (int): Maximal polygon size to remove

        Returns:
            - (numpy.list) Clean predict mask from small polygon
        """
        result_array = np.zeros_like(mask_array)
        labeled_array, num_labels = label(mask_array)
        for label_id in range(1, num_labels + 1):
            component_size = np.sum(labeled_array == label_id)
            if component_size >= min_size:
                result_array[labeled_array == label_id] = 1
                
        return result_array
    
    def select_rgb_image(self, image):
        """Selecting a RGB channel from 12 band satellite image
        
        Attribute:
            - image: 12 band Sentinel 2 image
            
        Returns:
            - (numpy.list) array of RGB image channel
        """
        rgb_image = np.stack([image[:, :, 3], image[:, :, 2], image[:, :, 1]], axis=-1)
        rgb_image_normalized = cv2.normalize(rgb_image, None, 0, 255, cv2.NORM_MINMAX)
        return rgb_image_normalized
    
    def reconstruct_image(self, patched_image:list, original_height:int, original_width:int, patch_size:int=256):
        """Reconstruct a chunked image to original size
        
        Attribute:
            - patched_image (numpy.list): An array of patched images
            - original_height (int): Original height of the image
            - original_width (int): Original width of the image
            - patch_size (int): Image patched size
            
        Returns:
            - (numpy.list) List of reconstructed image
        """
        padded_height = ceil(original_height / patch_size) * patch_size
        padded_width = ceil(original_width / patch_size) * patch_size
        
        reconstructed_image = np.zeros((padded_height, padded_width, patched_image.shape[-1]), dtype=patched_image.dtype)
        
        index = 0
        for i in range(0, padded_height, patch_size):   
            for j in range(0, padded_width, patch_size):
                reconstructed_image[i:i + patch_size, j:j + patch_size, :] = patched_image[index]
                index += 1
        
        return reconstructed_image[:original_height, :original_width, :]

class Model():
    """A class to handle Machine Learning Models
    
    This class is designed to load pre-trained models and predict. It provides
    methods for loading models from file path, and predict using loaded model.
    
    Attribute:
        - file_path (str): The path to the file where the model is saved.
    
    Methods:
        - load_model(): Loads a model from the specified file path
        - predict(image_array): Predict an output from given image file     
    """
    def __init__(self, file_path:str):
        self.model_path:str = file_path
        self.model = None
        self.status = "not loaded"
        
    def load_model(self):
        """Loads a pre-trained model from specified file path using Tensorflow
        load_modes() function. It set the model status to "loaded" after model successfully
        loaded.
        """
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        self.status = "loaded"
    
    def predict(self, image):
        """Run preprocessing function and predict a field segmentation from given image
        using model.predict()
        
        Returns:
            - rgb_image (numpy.list): array of image
            - mask_array (numpy.list): array of mask
        """
        try:
            with rasterio.open(image) as src:
                loaded_image = src.read()
                postprocess = Postprocess()
                preprocessed_image = Preprocess(image=loaded_image)
                norm_image = preprocessed_image.normalize_image()
                patched_image = preprocessed_image.create_image_patches()
                patched_image = patched_image/255
                patch_mask_predict = self.model.predict(patched_image)
                mask_predict = postprocess.reconstruct_image(patch_mask_predict, preprocessed_image.original_height, preprocessed_image.original_width)
                mask_predict = np.argmax(mask_predict, axis=-1)
                mask_predict[mask_predict == 2] = 0
                mask_predict = postprocess.remove_polygons(mask_predict, 5)
                rgb_image = postprocess.select_rgb_image(norm_image)
                return rgb_image, mask_predict
        except Exception as e:
            print(e)
