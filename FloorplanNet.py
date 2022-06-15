import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
import keras as keras
import random
import cv2 as cv
from keras.preprocessing.image import save_img
import math

# Global variables
# Can be change for better output 
TARGET_SIZE = 512
AUTO = tf.data.experimental.AUTOTUNE
SHARDS = 16
VALIDATION_SPLIT = 0.19
BATCH_SIZE = 6

# Read in a multi image, get the number of the image from the image path. This number is used to pair the Multi 
# image with the close wall image label for it's respective image  
def load_raw_images(img_path):
	bits = tf.io.read_file(img_path)
	#tf.print("image path",img_path)
	label = tf.strings.split(img_path, sep=os.path.sep)
	label = tf.strings.split(label[-1], sep=os.path.sep)
	label = tf.strings.split(label[0],sep='_')

	image = tf.io.decode_jpeg(bits)
	image = tf.image.resize(image,[TARGET_SIZE, TARGET_SIZE])
	image = tf.cast(image, tf.uint8)
	image = tf.io.encode_jpeg(image)
	return image,label[0]

def _bytestring_feature(list_of_bytestrings):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

# pass in the label, stitch together the path where the wall that pairs with the multi resides. 
# Load the wall label image, and write it to the TF record as a bytestring feature. 
def to_tfrecord(img_bytes,label):
	label=label.decode("utf-8")
	label=label.split(".")
	new_path_wall = "../dataset/jp/walls_inverted/"+str(label[0])+"_wall.png"
	
	bits = tf.io.read_file(new_path_wall)
	bits=bits.numpy()
	image = tf.image.decode_png(bits)
	image = tf.image.resize(image, [TARGET_SIZE, TARGET_SIZE])
	image = tf.cast(image, tf.uint8)
	image = tf.image.encode_png(image)
	image=image.numpy()

	new_path_close="../dataset/jp/closes_inverted/"+str(label[0])+"_close.png"

	# This code block was used to invert the training data label images so that the black wall 
	# white background is the same in both the base image and the label image.
	#image_inverted_first=Image.open(new_path_close)
	#inverted_image=PIL.ImageOps.grayscale(image_inverted_first)
	#inverted_image=PIL.ImageOps.invert(inverted_image)
	#inverted_image.save("../dataset/jp/closes_inverted/"+str(label[0])+"_close.png")
	#actual_path = "../dataset/jp/closes_inverted/"+str(label[0])+"_close.png"

	bits_close = tf.io.read_file(new_path_close)
	bits_close=bits_close.numpy()

	close_image = tf.image.decode_png(bits_close)
	close_image = tf.image.resize(close_image, [TARGET_SIZE, TARGET_SIZE])
	close_image = tf.cast(close_image, tf.uint8)
	close_image = tf.image.encode_png(close_image)
	close_image=close_image.numpy()


	feature = {
		"image": _bytestring_feature([img_bytes]),
		"wall_label": _bytestring_feature([image]),
		"close_label": _bytestring_feature([close_image]),
	}
	return tf.train.Example(features=tf.train.Features(feature=feature))

#read in the image and wall label and cast it back to float 32. 
def read_tfrecord(example):
	features = {
		"image": tf.io.FixedLenFeature([], tf.string),
		"wall_label": tf.io.FixedLenFeature([], tf.string),
		"close_label": tf.io.FixedLenFeature([],tf.string),

	}
	example = tf.io.parse_single_example(example, features)

	# These blocks are used for making sure that the images going into training are grayscaled and only 1 color bit is need for black/white images
	# Also important casting and  decoding is don here so that the data is in a form that is recognizable to the CNN
	image = tf.image.decode_jpeg(example['image'], channels=3)
	image=tf.image.rgb_to_grayscale(image)
	image = tf.cast(image, tf.float32) / 255.0
	image = tf.reshape(image, [TARGET_SIZE, TARGET_SIZE,1])
	
	wall_label= tf.image.decode_png(example['wall_label'], channels=3)
	wall_label=tf.image.rgb_to_grayscale(wall_label)
	wall_label= tf.cast(wall_label, tf.float32) / 255.0
	wall_label= tf.reshape(wall_label, [TARGET_SIZE, TARGET_SIZE,1])

	close_label= tf.image.decode_png(example['close_label'], channels=3)
	close_label=tf.image.rgb_to_grayscale(close_label)
	close_label= tf.cast(close_label, tf.float32) / 255.0
	close_label= tf.reshape(close_label, [TARGET_SIZE, TARGET_SIZE,1])
	#I was not able to create a multi label net so instead I just change the 2nd return fature to close_label when I want to train walls and windows 
	return image, close_label

def get_training_dataset(training_filenames):
  return get_batched_dataset(training_filenames)

def get_validation_dataset(validation_filenames):
  return get_batched_dataset(validation_filenames)

#prepare the TF records for learning. Create batches.
def get_batched_dataset(filenames):
	
	option_no_order = tf.data.Options()
	option_no_order.experimental_deterministic = False
	dataset = tf.data.Dataset.list_files(filenames)
	dataset = dataset.with_options(option_no_order)
	dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO)
	dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)

	#dataset = dataset.take(6)
	dataset = dataset.cache()  # This dataset fits in RAM
	#dataset = dataset.repeat() 
	dataset = dataset.shuffle(1024)
	dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
	dataset = dataset.prefetch(AUTO) 

	return dataset

def main():
		
		#fetching base images.
		nb_images = len(tf.io.gfile.glob('../dataset/jp/train/*.jpg'))
		shared_size = math.ceil(1.0 * nb_images / SHARDS)
		print("shared_size",shared_size)
		dataset = tf.data.Dataset.list_files('../dataset/jp/train/*.jpg',shuffle=True)
		dataset = dataset.map(load_raw_images)
		dataset = dataset.batch(shared_size)
		
		#for loop for creating TF records
		
		for shard, (image,label) in enumerate(dataset):
			shard_size = image.numpy().shape[0]
			filename = str(shard)+"_Black_and_White.tfrec"
			with tf.io.TFRecordWriter(filename) as out_file:
				for i in range(shard_size):
					example = to_tfrecord(image.numpy()[i],label.numpy()[i])
					out_file.write(example.SerializeToString())
				print("Wrote file {} containing {} records".format(filename, shard_size))
		
		
		
		filenames = tf.io.gfile.glob('*.tfrec')
		random.shuffle(filenames)
		split = int(len(filenames) * VALIDATION_SPLIT)

		training_filenames = filenames[split:]
		validation_filenames = filenames[:split]
		print(len(filenames), len(validation_filenames), BATCH_SIZE)

		#Mish activation function was used a bit. 
		def mish(x):
			return x* keras.backend.tanh(keras.backend.softplus(x))
		
		# Initially, model architecture was 100% based off of the architecture used in the paper. filters 64 ->512 then Upconv 512 ->32 and then a 3 filter at the end. 
		# After playing around with LeakyRelu, relu, linear, and mish activation functions and changing kernel sizes, I realized that I wasn't going to get a highly accurate model with
		# the paper network architecture. Since I didn't want to look into a totally different network architecutre I decided to make an ensemble model of all of the best models I had trained
		# when playing around with the network variables. I settled on a sigmoid on the last layer for activation because it created the least noise on the output prediction images.
		model = tf.keras.Sequential([


			tf.keras.layers.Conv2D(kernel_size=5, filters=64, padding='same', activation='LeakyReLU'),
			tf.keras.layers.MaxPooling2D(pool_size=2,strides=2,padding='valid'),
			#tf.keras.layers.Dropout(0.2,seed=2048),

			tf.keras.layers.Conv2D(kernel_size=5, filters=128, padding='same', activation='LeakyReLU'),
			tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'),
			#tf.keras.layers.Dropout(0.2,seed=2048),

			tf.keras.layers.Conv2D(kernel_size=3, filters=256, padding='same', activation='LeakyReLU'),
			tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'),
			#tf.keras.layers.Dropout(0.2,seed=2048),

			tf.keras.layers.Conv2D(kernel_size=3, filters=512, padding='same', activation='LeakyReLU'),
			tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'),

			tf.keras.layers.Conv2D(kernel_size=3, filters=512, padding='same', activation='LeakyReLU'),
			tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'),
			#tf.keras.layers.Dropout(0.2,seed=2048),

			tf.keras.layers.Conv2DTranspose(filters=256,kernel_size=4, strides=2, padding='same',activation='linear'),
			tf.keras.layers.Conv2DTranspose(filters=256,kernel_size=4, strides=2, padding='same',activation='linear'),
			tf.keras.layers.Conv2DTranspose(filters=128,kernel_size=4, strides=2, padding='same',activation='linear'),
			tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4,strides=2, padding='same',activation='linear'),
			tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=4,strides=2, padding='same',activation='linear'),


			tf.keras.layers.Conv2D(kernel_size=3, filters=1, padding='same', activation='sigmoid'),
			#do a 10,000 step mish model train 
		]) 
	
		# focal crossentropy is good for small features but not large ones. I used it in the loss for windows and door but used normal binary crossentropy for walls. 
		# The same metric is used for both walls and door&window
		"""
		model.compile(optimizer='Adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),metrics=tf.keras.metrics.BinaryAccuracy(threshold=0.5))
		model.fit(get_training_dataset(training_filenames),steps_per_epoch=350,validation_steps=75, epochs=11,validation_data=get_validation_dataset(validation_filenames))
		model.save("BESTWALLNETTHREE")
		"""

		# Load door and window models
		model_Walldoormemes=load_model("Wall_Net_ImprovedThree")
		model_Walldoormemes._name="WallDoorMemes"
		model_DOORWINDOWNETONE=load_model("Door_WindowNETONE")
		model_DOORWINDOWNETONE._name="DOORWINDOWNetONE"
		model_DOORWINDOWNETTWO=load_model("Door_WindowNETTWO")
		model_DOORWINDOWNETTWO._name="DOORWINDOWNetTWO"

		#load wall models
		model_seven=load_model("BESTWALLNETONE")
		model_seven._name="BESTWALLNETONE"
		model_nine=load_model("BESTWALLNETTWO")
		model_nine._name="BESTWALLNETTWO"
		model_eight=load_model("BESTWALLNETTHREE")
		model_eight._name="BESTWALLNETTHREE"
		
		#create an ensemble model from door models 
		Door_models= [model_Walldoormemes,model_DOORWINDOWNETONE,model_DOORWINDOWNETTWO]
		model_input= tf.keras.Input(shape=(512,512,1))
		model_outputs=[i(model_input)for i in Door_models]
		ensemble_output=tf.keras.layers.Average()(model_outputs)
		ensemble_model=tf.keras.Model(inputs=model_input,outputs=ensemble_output)
		
		#create an ensemble model from Wall models 
		Wall_models=[model_seven,model_nine,model_eight]
		Wall_model_input= tf.keras.Input(shape=(512,512,1))
		Wall_model_outputs=[i(Wall_model_input)for i in Wall_models] # work on taking the argmax instead.
		Wall_ensemble_output=tf.keras.layers.Average()(Wall_model_outputs)
		Wall_ensemble_model=tf.keras.Model(inputs=Wall_model_input,outputs=Wall_ensemble_output)

		#fetch tfrecord data and compile ensemble models
		ds = get_validation_dataset(validation_filenames)
		#model_Walldoormemes.evaluate(ds,steps=50)
		#model_DOORWINDOWNETONE.evaluate(ds,steps=50)
		#model_DOORWINDOWNETTWO.evaluate(ds,steps=50)
		#model_eight.evaluate(ds,steps=50)
		#model_seven.evaluate(ds,steps=50)
		#model_nine.evaluate(ds,steps=50)
		Wall_ensemble_model.compile(optimizer='Adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),metrics=['accuracy'])
		ensemble_model.compile(optimizer='Adam',loss=tf.keras.losses.BinaryFocalCrossentropy(from_logits=False),metrics=tf.keras.metrics.BinaryAccuracy(threshold=0.5))
		#ensemble_model.evaluate(ds,steps=50)
		
		# WALL ENSEMBLE MODEL AND THEN FIX THE OUTPUT FUNCTION 

		#FIGURE OUT HOW TO FIX THE OUTPUT IT'S ALL JACKED UP. TRAIN 2 MORE MODELS. 2 MORE DOOR/WINDOW and 3 MORE WALL MODELS

		#this for loop is for displaying the predictions.
		for images, labels in ds.take(2):
			fig = plt.figure(figsize=(200,200))
			predicted_classes = ensemble_model.predict(images)                 # get a pred for walls and Door&Window
			predicted_classes_two=Wall_ensemble_model.predict(images)
			cnt = 1
			for img, lbl, prediction ,prediction_two in zip(images, labels, predicted_classes,predicted_classes_two ):
			
				filename_wall=(str(cnt)+'predicted_multisssss.png')            
				save_img(filename_wall, prediction_two)
				bits = cv.imread(filename_wall)
				ret, thresh6 = cv.threshold(bits, 130, 255, cv.THRESH_BINARY)	# I want one uniform color to show wall pred so I hardcode some confidence so it looks good. It's still making the pred though.

				save_img(filename_wall, thresh6)
				bits = cv.imread(filename_wall)									# save the improved preds and convert it to RBG format.
				prediction_two = cv.cvtColor(bits, cv.COLOR_BGR2GRAY)
				plt.imsave(filename_wall,thresh6,cmap='binary')
				

				filename_green=str(cnt)+'predicted_multiss.png'
				save_img(filename_green, prediction)
				bits = cv.imread(filename_green)
				ret, thresh4 = cv.threshold(bits,175,255,cv.THRESH_TOZERO)		# same thing as above but this time for Door/Window preds
				thresh4 = cv.cvtColor(thresh4, cv.COLOR_BGR2GRAY)
				ret, thresh4 = cv.threshold(thresh4, 150, 255, cv.THRESH_BINARY)
				plt.imsave(filename_green,thresh4,cmap='winter')
				
				fig.add_subplot(3,4, cnt)
				plt.imshow(img, cmap='gray')

				Purple_file = cv.imread(filename_wall)
				Green_file = cv.imread(filename_green)
				dst = cv.addWeighted(Purple_file,0.5,Green_file,0.5,0)			# Read both pred files and combine them equal parts. 
				dst_filename=str(cnt)+'COMBINEDGIGACHADIMAGE.png'

				save_img(dst_filename, dst)
				bits = cv.imread(dst_filename)

				# pixel color manipulation from https://stackoverflow.com/questions/60018903/how-to-replace-all-pixels-of-a-certain-rgb-value-with-another-rgb-value-in-openc
				# this section is  manipulating the color of the preds so that they look good on a pyplot. When combing the 2 image preds the background colors often get very messed up 
				#so I just manually set the wall,backgrorund,door/window and both wall and door/window preds to the colors I want so that it looks better.
				
				bits[np.all(bits == (191,255,128), axis=-1)] = (255,255,255)  # change the background color to white
				bits[np.all(bits == (128,255,191), axis=-1)] = (255,255,255)
				
				bits[np.all(bits == (64,128,0), axis=-1)] = (0,0,0)   # change the Wall color to black
				bits[np.all(bits == (0,128,64), axis=-1)] = (0,0,0)

				bits[np.all(bits == (255,128,128), axis=-1)] = (248, 131, 121)  # change doorwindow color to pink coral
				bits[np.all(bits == (128,128,255), axis=-1)] = (248, 131, 121)

				bits[np.all(bits == (128,0,0), axis=-1)] = (255, 0, 255) # change overlap of wall and door/window color to pink 
				bits[np.all(bits == (0,0,128), axis=-1)] = (255, 0, 255)
			
				#save_img(dst_filename, bits)			
				cnt = cnt + 1
				fig.add_subplot(3, 4, cnt)
				plt.imshow(bits)			

				cnt = cnt + 1
				plt.subplots_adjust(top=1, bottom=-1)
				plt.subplots_adjust(left=0.074,bottom=0.06,right=0.952,top=1,wspace=0, hspace=0.138)
	
		#fig.tight_layout(pad=0)
		#plt.subplot_tool()
		plt.show()
			
		
if __name__ == "__main__":
		main()