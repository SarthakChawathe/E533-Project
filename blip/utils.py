import numpy as np
import tensorflow as tf
from transformers import AutoProcessor, TFCLIPModel
from constants import *



clip_model_name = "openai/clip-vit-base-patch32"
clip_processor = AutoProcessor.from_pretrained(clip_model_name, cache_dir=SCRATCH_DIR + "/processors")
clip_model = TFCLIPModel.from_pretrained(clip_model_name, cache_dir=SCRATCH_DIR + "/models")

def process_image(img):
  image = np.array(img)
  if len(image.shape) == 2:
    image = np.stack((image,)*3, axis=-1)
  image = tf.convert_to_tensor(tf.image.resize(image, (224, 224))) / 255.0
  return image

def get_context(sample_data, index, sentences_df):
  image = process_image(sample_data['image'])
  question = sample_data['question']
  image_input = clip_processor(images=image, return_tensors="tf")
  text_input = clip_processor(text=question, return_tensors="tf")
  image_features = clip_model.get_image_features(**image_input)
  text_features = clip_model.get_text_features(**text_input)
  output = image_features + text_features
  query = output.numpy()
  # normalize query
  query = query / np.linalg.norm(query)
  print('here here')
  # query = query.reshape(1, len(query))
  # taking top 5 results
  D, I = index.search(query, 10)
  res = sentences_df.iloc[I[0]]['sentence'].values.tolist()
  return res

