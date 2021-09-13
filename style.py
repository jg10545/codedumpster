import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import logging
import argparse

content_layers = ['block5_conv2'] 

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg =  vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                      outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

        content_dict = {content_name:value 
                    for content_name, value 
                    in zip(self.content_layers, content_outputs)}

        style_dict = {style_name:value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}
    
        return {'content':content_dict, 'style':style_dict}


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (600, 1024))
    img = img[tf.newaxis, :]
    return img

def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
  
    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def high_pass_x_y(image):
    x_var = image[:,:,1:,:] - image[:,:,:-1,:]
    y_var = image[:,1:,:,:] - image[:,:-1,:,:]
    return x_var, y_var

def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_mean(tf.abs(x_deltas)) + tf.reduce_mean(tf.abs(y_deltas))
    
    
def style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss
    
    
def _image_style_transfer(contentfile, stylefile, extractor, lr=1e-2, style_weight=1e-2,
                          content_weight=1e4, total_variation_weight=30, steps=1000):
    """
    
    """
    image = tf.Variable(load_img(contentfile))
    style_image = load_img(stylefile)
    opt = tf.optimizers.Adam(learning_rate=lr, beta_1=0.99, epsilon=1e-1)
    
    style_targets = extractor(style_image)['style']
    content_targets = extractor(image)['content']
    
    @tf.function()
    def train_step():
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight)
            loss += total_variation_weight*tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))
        
    for _ in range(steps):
        train_step()
        
    return tensor_to_image(image)


def main(contentdir, styledir, outputdir, lr=1e-2, style_weight=1e-2, content_weight=1e-4,
         total_variation_weight=30, steps=1000):
    """
    
    """
    # find all the images
    imfiles = [x for x in os.listdir(contentdir)]
    stylefiles = [os.path.join(styledir,x) for x in os.listdir(styledir)]
    logging.info("found %s content images and %s style images"%(len(imfiles),len(stylefiles)))
    # load the VGG19 model
    extractor = StyleContentModel(style_layers, content_layers)
    
    # iterate through each content image, modify it with a random style image, and
    # save to the output directory
    for i in tqdm(imfiles):
        s = np.random.choice(stylefiles)
        image = _image_style_transfer(os.path.join(contentdir,i), s, extractor, lr, style_weight, 
                                      content_weight, total_variation_weight, steps)

        image.save(os.path.join(outputdir, i))
        
    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("contentdir", help="path to directory containing content images", type=str)
    parser.add_argument("styledir", help="path to directory containing style images", type=str)
    parser.add_argument("outputdir", help="path to directory to save restyled images to", type=str)
    parser.add_argument("--lr", default=1e-2, help="learning rate", type=float)
    parser.add_argument("--steps", default=1000, help="number of training steps", type=int)
    parser.add_argument("--style_weight", default=1e-2, help="weight for style loss", type=float)
    parser.add_argument("--content_weight", default=1e-4, help="weight for content loss", type=float)
    parser.add_argument("--total_variation_weight", default=30., help="weight for total variation loss", type=float)
    
    args = parser.parse_args()
    
    main(args.contentdir, args.styledir, args.outputdir, args.lr, args.style_weight, args.content_weight,
         args.total_variation_weight, args.steps)











