## Kongz Animation

import io
import os
import zipfile
import numpy as np
import random as rd
from PIL import Image, ImageOps
import skimage
import skimage.transform
import imageio

folder_name = "enter_folder_name_here"


# Layers loading

archive = zipfile.ZipFile(folder_name+'Kongz_movie_files.zip', 'r')

def get_file_from_zip(filename):
    image_data = archive.read(filename)
    fh = io.BytesIO(image_data)
    return Image.open(fh).convert("RGBA")


# Superposition

def superpose(image_over,image_under):
#    img = image_under.copy()
#    img.paste(image_over,(0,0),image_over)
    img = Image.alpha_composite(image_under, image_over)
    return img

# Mask

def mask(image,mask_image):
    return Image.composite(image,mask_image,mask_image)

def substract(image,mask_image):
    mask_image_negative = Image.fromarray(255-np.array(mask_image))
    return Image.composite(image,mask_image_negative,mask_image_negative)
    

# Transformation matrices

    
def trans_param(x0,y0):
        shift = skimage.transform.SimilarityTransform(
        translation = -np.array([x0,y0])
        )
        return (shift.params).flatten()[:6]


# Read image

input_img = Image.open(folder_name+"input_image.png", 'r').convert('RGBA')
empty_img = Image.fromarray(0*np.array(input_img))

### Background, head, body

# Background Extraction

background = np.array(input_img.copy())
for k in range(background.shape[0]):
    background[:,k,:] = background[:,-1,:]
background = Image.fromarray(background)

# Foreground Extraction

foreground = input_img.copy()
diff = np.array(foreground)-np.array(background)
diff[:,:,3] = 255*(np.sum(diff[:,:,:3],2)>0)
foreground = mask(input_img,Image.fromarray(diff))


# HeadBody Extraction

body_mask = get_file_from_zip('body_mask.png')
body = mask(foreground,body_mask)
head = mask(foreground,Image.fromarray(255-np.array(body_mask)))


# HeadBody Completion


def tint_image(src, color="RGB"): #colors b&w image with given rgb code
    src.load()
    r, g, b, alpha = src.split()
    gray = ImageOps.grayscale(src)
    result = ImageOps.colorize(gray, (0, 0, 0, 0), color) 
    result.putalpha(alpha)
    return result

hidden_body = get_file_from_zip("hidden_body.png")
tinted_hidden_body = tint_image(hidden_body, np.array(input_img)[500,200,:3])
body.paste(tinted_hidden_body,(0,0),tinted_hidden_body)

#Headphones

from PIL import ImageChops
    
headphones = get_file_from_zip("headphones.png")

diff = ImageChops.difference(superpose(headphones,input_img).convert('RGB'),input_img.convert('RGB'))
if not diff.getbbox(): #if headphones
    new_body = tint_image(body_mask, np.array(input_img)[500,200,:3])
    behind_headphones = mask(new_body,headphones)
    body = superpose(behind_headphones,body)
    head = superpose(headphones,head)

#Bowtie

bowtie = get_file_from_zip("bowtie.png")

diff = ImageChops.difference(superpose(bowtie,input_img).convert('RGB'),input_img.convert('RGB'))
if not diff.getbbox(): #if bowtie
    new_body = tint_image(body_mask, np.array(input_img)[500,200,:3])
    body = superpose(tinted_hidden_body,new_body)
    head = superpose(bowtie,head)
    

# Head Movement


def transition_head(t,head):
    xt = 10*(np.sin(2*np.pi*t))**2
    yt = 10*(np.sin(2*np.pi*t))**2
    im = head.transform(head.size, Image.AFFINE, trans_param(xt,yt))
    return im

# Body Movement

def transition_body(t):
    im = body.transform(body.size, Image.AFFINE, trans_param(0,10*abs(np.sin(2*np.pi*t))))
    return im

## Gif construction

import tqdm # Progression bar

#Save as gif

nb_images = 50

for i in tqdm.tqdm(range(nb_images)):
    t = i/nb_images
    imagei = empty_img

#    head
    headi = head.copy()
    headi = transition_head(t%1,headi)

#    body
    bodyi = transition_body(t%1)

#   superpose
    imagei = superpose(headi,imagei)
    imagei = superpose(imagei,bodyi)
    imagei = superpose(imagei,background)

    imagei.save(folder_name+"/working_dir/image_"+str(i)+".png", "png")

### Construction du fichier .gif

#Save as gif
            

filenames =[folder_name+"/working_dir/image_"+str(i)+".png" for i in range(nb_images)]
with imageio.get_writer(folder_name+"/resultat.gif", mode='I',duration=0.035) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

from pygifsicle import optimize
optimize(folder_name+"/resultat.gif")
