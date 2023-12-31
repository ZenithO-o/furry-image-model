import os
import typing

import pathlib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer

file_path = pathlib.Path(__file__).parent.resolve()

class FurryImageModel:
    def __init__(self, model_path: typing.Union[str, os.PathLike], safe_threshold: float = 1/3, explicit_threshold: float = 2/3) -> None:
        """A wrapper class for using the tagger model.

        Args:
            model_path (str | os.PathLike): The path of the TensorFlow SavedModel model.
        """
        
        # load models for tagging and in
        self.full_model = tf.keras.models.load_model(model_path)
        input_layer = self.full_model.layers[0]
        hidden_layer = self.full_model.get_layer('feature_layer')
        self.feature_model = tf.keras.Model(inputs=input_layer.input, outputs=hidden_layer.output)

        # load tf config for metadata on model
        self.tf_config = self.full_model.get_config()
        self.img_size = self.tf_config['layers'][0]['config']['batch_input_shape'][1]
        self.channels = self.tf_config['layers'][0]['config']['batch_input_shape'][3]
        self.layer_order = [layer[0][:layer[0].find('_')] if layer[0].find('_') != -1 else layer[0] for layer in self.tf_config['output_layers']]
        
        # load multilabel binarizers for model
        self.mlbs = {}
        self.tags = {}
        
        category_dir = file_path / 'categories'
        category_files = os.listdir(category_dir)
        for category in category_files:
            with open(os.path.join(category_dir, category)) as f:
                tags = f.readlines()
            
            category_name = category[:category.find('_')]
            tags = [[tag.replace('\n', '')] for tag in tags]
            mlb = MultiLabelBinarizer()
            mlb.fit(tags)
            
            self.mlbs.update({category_name:mlb})
            self.tags.update({category_name:mlb.classes_})
        
        self.safe_threshold = safe_threshold
        self.explicit_threshold = explicit_threshold

         
    def _load_image(self, img_path: typing.Union[str, os.PathLike]) -> tf.Tensor:
        """Loads an image from the given path and applies the necessary modifications:
        converting to RGB, resizing to fit the model, and Normalizing the pixel values
        between [0.0, 1.0]
        
        Args:
            img_path (str | os.PathLike): The path for the image

        Returns:
            Tensor: Tensor representation of the Image
        """
        # Read an image from a file
        image_string = tf.io.read_file(str(img_path))
        # Decode it into a dense vector
        image_decoded = tf.image.decode_jpeg(image_string, channels=self.channels)
        # Resize it to fixed shape
        image_resized = tf.image.resize(image_decoded, [self.img_size, self.img_size])
        # Normalize it from [0, 255] to [0.0, 1.0]
        image_normalized = image_resized / 255.0
        
        return image_normalized

    def _get_rating(self, value: float):
        if value >= self.explicit_threshold:
            return 'explicit'
        elif value >= self.safe_threshold:
            return 'questionable'
        else:
            return 'safe'

    def _normalize_func(self, image, normalized):
        img_mean = normalized[0]
        img_stddev = normalized[1]

        offset = tf.constant(img_mean, shape=[1, 1, 3])
        image -= offset
        scale = tf.constant(img_stddev, shape=[1, 1, 3])
        image /= scale
        return image

        
    def predict_image_tags(self, *img_path: typing.Union[str, os.PathLike],
                           t: float = 0.5,
                           normalized: typing.List[tuple] = [(0.5, 0.5, 0.5), (0.225, 0.225, 0.225)],) -> typing.Any:
        """run full prediction model on image to predict tags

        Args:
            *img_path (str | os.PathLike): The path for each image
            
            t (float, optional): The probablility threshold at which to accept a tag as 
            valid. Defaults to 0.5.

        Returns:
            Any: Returns a list of all the tags predicted for each image.
        """

        if len(img_path) == 0:
            raise ValueError('There must be at least one image given.')

        if t < 0 or t > 1:
            raise ValueError('The threshold must be within [0.0, 1.0]')
        
        loaded_images = [self._load_image(img) for img in img_path]
        if normalized:
            loaded_images = [self._normalize_func(img, normalized) for img in loaded_images]
        
        num_images = len(loaded_images)
        images = tf.convert_to_tensor(loaded_images)
        x = self.full_model.predict(images)
        
        res = [[] for _ in range(num_images)]
        for layer, result in zip(self.layer_order, x):
            
            if layer == 'rating':
                out = [[{'tag': self._get_rating(r[0]), 'value': float(r[0]), 'category': layer}] for r in result]
                
            else:
                result = np.array(result)
                result_mask = np.where(result > t, 1, 0)
                mlb = self.mlbs.get(layer)
                
                mlb: MultiLabelBinarizer
                tags = mlb.inverse_transform(result_mask)
                values = [result[i, np.where(result_mask[i])][0] for i in range(num_images)]

                out = [[{'tag': t, 'value': float(v), 'category': layer} for t,v in zip(t_list, v_list)] for t_list, v_list in zip(tags, values)]
            

            for i in range(num_images):
                res[i].extend(out[i])

        return res
        
        
    def image_latent_vector(self, *img_path: typing.Union[str, os.PathLike]) -> np.ndarray:
        """Get the feature vector representation of an image.
        
        NOTE: The output array will be shaped (N, D), where N is the # of images, and D 
        is the dimension of the feature vector.

        Args:
            *img_path (str | os.PathLike): The path for each image
            
        Returns:
            np.ndarray: A dense vector representation of the images.
        """
        if len(img_path) == 0:
            raise ValueError('There must be at least one image given.')

        loaded_images = [self._load_image(img) for img in img_path]
        
        res = self.feature_model.predict(tf.convert_to_tensor(loaded_images))
        return np.array(res)