import os
import cv2
import sys
import json
import math
import random
import logging
import numpy as np

logging.getLogger().setLevel(logging.INFO)
IMAGE_PATH = "./images/"
LABEL_PATH = "./labels/"
INPUT_PATH = "./input_images_labels/"
OUTPUT_PATH = "./output_images_labels/"

def random_rotate(img, rotate_img, rotate):
	img_width = img.shape[1]
	img_height = img.shape[0]

	for i in range(img_height):
		for j in range(img_width):
			if(rotate == 90):
				rotate_img[img_width-1-j, i] = img[i,j]
			elif(rotate == 180):
				rotate_img[img_height-1-i, img_width-1-j] = img[i,j]
			elif(rotate == 270):
				rotate_img[j,img_height-1-i] = img[i,j]
	#cv2.imwrite(output_path+"_3.jpg", rotate_img)

def save_label(label_output_name, location):
	output_string = ''
	for i in range(len(location)):
		output_string += str(location[i]) + " "
	file = open(label_output_name, 'w')
	file.write(output_string[:-1])
	file.close()

#??????
def add_gaussian_noise(noise_img):
	means = 0
	sigma = 25
	#r channel
	r = noise_img[:,:,0].flatten()
	#g channel
	g = noise_img[:,:,1].flatten()
	#b channel
	b = noise_img[:,:,2].flatten()

	for i in range(noise_img.shape[0] * noise_img.shape[1]):
		r[i] = r[i] + random.gauss(0, sigma)
		g[i] = g[i] + random.gauss(0, sigma)
		b[i] = b[i] + random.gauss(0, sigma)
	noise_img[:,:,0] = r.reshape(noise_img.shape[0], noise_img.shape[1])
	noise_img[:,:,1] = g.reshape(noise_img.shape[0], noise_img.shape[1])
	noise_img[:,:,2] = b.reshape(noise_img.shape[0], noise_img.shape[1])
	cv2.imwrite("result.jpg", noise_img);


#test gauss_size=105, gauss_kernel_size=4, ??????
def elasticDistort(img, elastic_img, gauss_size, gauss_kernel_size):
	x_mv = np.zeros((img.shape[0], img.shape[1]), np.float64)
	y_mv = np.zeros((img.shape[0], img.shape[1]), np.float64)
	for i in range(x_mv.shape[0]):
		for j in range(x_mv.shape[1]):
			x_mv[i][j] = random.uniform(-100,100)
	for i in range(y_mv.shape[0]):
		for j in range(y_mv.shape[1]):
			y_mv[i][j] = random.uniform(-100,100)

	#gaussian blur
	x_mv = cv2.GaussianBlur(x_mv,(gauss_size, gauss_size), gauss_kernel_size) 
	y_mv = cv2.GaussianBlur(y_mv,(gauss_size, gauss_size),gauss_kernel_size)

	#elastic_img = np.zeros(img.shape, np.uint8)
	sum = np.zeros((3,), np.double)
	for i in range(elastic_img.shape[0]):
		for j in range(elastic_img.shape[1]):
			elastic_img[i][j] = 255

	for i in range(elastic_img.shape[0]):
		for j in range(elastic_img.shape[1]):
			new_x_low = i + math.floor(x_mv[i][j])
			new_x_high = i + math.ceil(x_mv[i][j])
			new_y_low = j + math.floor(y_mv[i][j])
			new_y_high = j + math.ceil(y_mv[i][j])

			if (new_x_low < 0):    new_x_low = 0
			if (new_x_high < 0):   new_x_high = 0
			if (new_y_low < 0):    new_y_low = 0
			if (new_y_high < 0):   new_y_high = 0

			if(new_x_low >= img.shape[0]):new_x_low = img.shape[0]-1
			if(new_x_high >= img.shape[0]):new_x_high = img.shape[0]-1
			if(new_y_low >= img.shape[1]):new_y_low = img.shape[1]-1
			if(new_y_high >= img.shape[1]):new_y_high = img.shape[1]-1

			sum =   img[new_x_low][new_y_low].astype(float) + \
				img[new_x_low][new_y_high].astype(float) + \
				img[new_x_high][new_y_low].astype(float) + \
				img[new_x_high][new_y_high].astype(float)
			avg = sum*1.0*0.25
			elastic_img[i][j] = avg.astype(int)
	#cv2.imwrite("elastic.jpg", elastic_img);

#输入为需要做扩展的样本集，和python存在同一level, images/ labels/ 
def dirlist(path):
    filelist =  os.listdir(path)

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            new_path = os.path.join(OUTPUT_PATH,filepath)
            print new_path
            if(os.path.exists(new_path) == False):
                os.makedirs(new_path)
            dirlist(filepath, allfile)
        else:
            continue

def get_postfix(image_path):
	pos = image_path.rfind('.')
	return pos

def get_all_path(image_path, pos):
	postfix = line[:pos] + ".txt"
	#get input whole image path and label path
	file_name = IMAGE_PATH + image_path
	label_name = LABEL_PATH + postfix
	#get output folder path
	output_image_path = IMAGE_PATH + IMAGE_PATH + image_path[:pos]
	output_label_path = IMAGE_PATH + LABEL_PATH + image_path[:pos]
	


def data_augment(line):
	pos = get_postfix(line)
	file_name, label_name, output_image_path, output_label_path = get_all_path(line,pos)

	location = [0]*5
	img = cv2.imread(file_name, 1)
	img_width = img.shape[1]
	img_height = img.shape[0]

	#load labels
	f = open(label_name,"r")
        line = f.readline()
	n = 0
        while line:
		line = line.strip('\r\n')
		info = line.split(' ')

		location[0] = info[0]
		for i in range(1, len(info)):
			location[i] = float(info[i])
		n += 1
		line = f.readline()
	f.close()
	if(n > 1):
		print "there are more than 1 object in this image, return."
		return
	location_save = location
	print img.shape	
	#img.shape[1] is image width
	center_x = location[1] * img_width
	center_y = location[2] * img_height
	width = location[3] * img_width
	height = location[4] * img_height
	print center_x,center_y, width, height

	x1 = center_x - width/2
	y1 = center_y - height/2
	x2 = center_x + width/2
	y2 = center_y + height/2
	print x1,y1, x2, y2

	random_ratio = 0.2
	#get random location for 4 coordinate
	cut_x1 = random.randrange(0, int(x1*random_ratio)+1, 2)
	cut_y1 = random.randrange(0, int(y1*random_ratio)+1,2)
	cut_x2 = img_width - random.randrange(0, int((img_width-x2)*random_ratio)+1, 2)
	cut_y2 = img_height - random.randrange(0, int((img_height-y2)*random_ratio)+1, 2)
	print cut_x1, cut_y1, cut_x2, cut_y2

	new_center_x = center_x - cut_x1
	new_center_y = center_y - cut_y1
	new_width = cut_x2 - cut_x1
	new_height = cut_y2 - cut_y1
	print new_center_x/new_width, new_center_y/new_height, width/new_width, height/new_height

	crop_img = img[cut_y1:cut_y2, cut_x1: cut_x2]
	cv2.imwrite(output_image_path+"_1.jpg", crop_img);
	#save label crop
	location[1] = new_center_x/new_width
	location[2] = new_center_y/new_height
	location[3] = width/new_width
	location[4] = height/new_height
	save_label(output_label_path + "_1.txt", location)

	#2. mirror
	mirror_img = np.zeros(img.shape, np.uint8) 
	for i in range(img_height):
		for j in range(img_width):
			mirror_img[i, img_width-1-j] = img[i,j]  #left to right,correct.
	cv2.imwrite(output_image_path+"_2.jpg", mirror_img);
	#the points change to img_width-center_x, center_y, width, height
	print "mirror_img ", img_width-center_x, center_y, width, height
	print (img_width-center_x)/img_width, center_y/img_height, width/img_width, height/img_height
    #save label crop
	location[1] = (img_width-center_x)/img_width
	location[2] = center_y/img_height
	location[3] = width/img_width
	location[4] = height/img_height
	save_label(output_label_path + "_2.txt", location)

	#3. random rotate image (90/180/270 choose 1)
	rotate = random.choice([90,180,270])
	if(rotate == 90):
		rotate_img = np.zeros((img.shape[1], img.shape[0], img.shape[2]), np.uint8)
		print "90 ", center_y, img_width-center_x, height, width
		location[1] = center_y/img_height
		location[2] = (img_width-center_x)/img_width
		location[3] = height/img_height
		location[4] = width/img_width

	elif(rotate == 270):
		rotate_img = np.zeros((img.shape[1], img.shape[0], img.shape[2]), np.uint8)
		print "270 ", img_height-center_y, center_x, height, width
		location[1] = (img_height-center_y)/img_height
		location[2] = center_x/img_width
		location[3] = height/img_height
		location[4] = width/img_width
	elif(rotate == 180):
		rotate_img = np.zeros(img.shape, np.uint8)
		print "180 ", img_width-center_x, img_height-center_y, width, height
		location[1] = (img_width-center_x)/img_width
		location[2] = (img_height-center_y)/img_height
		location[3] = width/img_width
		location[4] = height/img_height
	random_rotate(img, rotate_img, rotate)
	cv2.imwrite(output_image_path+"_3.jpg", rotate_img)
	save_label(output_label_path + "_3.txt", location)

	elastic_img = np.zeros(img.shape, np.uint8)
	elasticDistort(img, elastic_img, int(sys.argv[2]),int(sys.argv[3]))
	cv2.imwrite(output_image_path+"_4.jpg", elastic_img)
	save_label(output_label_path + "_4.txt", location_save)


def main():
	f = open(sys.argv[1],"r")
	line = f.readline()
	# 创建输出目录
	dirlist(sys.argv[1])
	while line:
		line = line.strip('\r\n')
		data_augment(line)  #line保存图片的路径

		line = f.readline()
	f.close()

#输入原始图片列表，依次读取图片，并进行剪切，旋转，镜像和扭曲等操作。
if __name__ == '__main__':
	#python2 .py ./image_list.txt
	main()
