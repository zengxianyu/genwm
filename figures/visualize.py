import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import argparse

# configs
font = {'family': 'monospace', 'weight': 'normal', 'size': 18}
matplotlib.rc('font', **font)

# arguments
ag = argparse.ArgumentParser()
ag.add_argument('-s', '--save', type=str, default=None)
ag.add_argument('which', type=str, choices=['w', 'g'])
ag = ag.parse_args()

# draw
if ag.which == 'w':
	# stage 1
	fig = plt.figure(figsize=(14.4,20))

	ax = plt.subplot(4,3,1)
	im = Image.open('imgs/ffhq/01627_GT.png')
	ax.imshow(im)
	#ax.axis(False)
	ax.set_title('Ground Truth $x$')
	ax.set_ylabel('FFHQ')
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_xticks([])
	ax.set_yticks([])

	ax = plt.subplot(4,3,2)
	imw = Image.open('imgs/ffhq/01627_W.png')
	ax.imshow(im)
	ax.axis(False)
	ax.set_title('Watermarked $\\hat{x}$')

	ax = plt.subplot(4,3,3)
	kappa = np.linalg.norm(np.array(imw) - np.array(im), axis=2)
	kappa = kappa / kappa.max()
	plt.pcolormesh(np.flipud(kappa), cmap='plasma')
	#plt.pcolormesh(kappa, cmap='plasma')
	plt.colorbar()
	ax.axis(False)
	ax.axis('square')
	ax.set_title('Normalized $\\kappa$')

	ax = plt.subplot(4,3,4)
	im = Image.open('imgs/ffhq/28170_GT.png')
	ax.imshow(im)
	#ax.axis(False)
	ax.set_ylabel('FFHQ')
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_xticks([])
	ax.set_yticks([])

	ax = plt.subplot(4,3,5)
	imw = Image.open('imgs/ffhq/28170_W.png')
	ax.imshow(im)
	ax.axis(False)

	ax = plt.subplot(4,3,6)
	kappa = np.linalg.norm(np.array(imw) - np.array(im), axis=2)
	kappa = kappa / kappa.max()
	plt.pcolormesh(np.flipud(kappa), cmap='plasma')
	plt.colorbar()
	ax.axis(False)
	ax.axis('square')

	ax = plt.subplot(4,3,7)
	im = Image.open('imgs/imagenet/ILSVRC2012_val_00025488_GT.JPEG')
	ax.imshow(im)
	#ax.axis(False)
	ax.set_ylabel('ImageNet')
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_xticks([])
	ax.set_yticks([])

	ax = plt.subplot(4,3,8)
	imw = Image.open('imgs/imagenet/ILSVRC2012_val_00025488_W.JPEG')
	ax.imshow(im)
	ax.axis(False)

	ax = plt.subplot(4,3,9)
	kappa = np.linalg.norm(np.array(imw) - np.array(im), axis=2)
	kappa = kappa / kappa.max()
	plt.pcolormesh(np.flipud(kappa), cmap='plasma')
	#plt.pcolormesh(kappa, cmap='plasma')
	plt.colorbar()
	ax.axis(False)
	ax.axis('square')

	ax = plt.subplot(4,3,10)
	im = Image.open('imgs/imagenet/ILSVRC2012_val_00027351_GT.JPEG')
	ax.imshow(im)
	#ax.axis(False)
	ax.set_ylabel('ImageNet')
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_xticks([])
	ax.set_yticks([])

	ax = plt.subplot(4,3,11)
	imw = Image.open('imgs/imagenet/ILSVRC2012_val_00027351_W.JPEG')
	ax.imshow(im)
	ax.axis(False)

	ax = plt.subplot(4,3,12)
	kappa = np.linalg.norm(np.array(imw) - np.array(im), axis=2)
	kappa = kappa / kappa.max()
	#kim = (kappa*255).astype(np.uint8).reshape(256,256,1)
	#print(kim)
	#ax.imshow(kappa)
	plt.pcolormesh(np.flipud(kappa), cmap='plasma')
	#plt.pcolormesh(np.fliplr(kappa.T), cmap='plasma')
	plt.colorbar()
	ax.axis(False)
	ax.axis('square')

	#plt.figure()
	#plt.imshow(kappa)

elif ag.which == 'g':

	raise NotImplementedError

if ag.save is None:
	plt.show()
else:
	plt.savefig(ag.save)

#figures/imgs/ffhq/000057_stylegan2.png
#figures/imgs/ffhq/000266_stylegan2.png
#figures/imgs/ffhq/000932_stylegan2.png
#figures/imgs/ffhq/gdcat_1012_1_adm.png
#figures/imgs/ffhq/gdcat_1025_1_adm.png
#figures/imgs/ffhq/gdcat_1027_1_adm.png
#figures/imgs/ffhq/sample_000020_ldm.png
#figures/imgs/ffhq/sample_000109_ldm.png
#figures/imgs/imagenet/ILSVRC2012_val_00000262.JPEG_adm.png
#figures/imgs/imagenet/ILSVRC2012_val_00016962.JPEG_ldm.png
#figures/imgs/imagenet/ILSVRC2012_val_00018250.JPEG_adm.png
#figures/imgs/imagenet/ILSVRC2012_val_00023850.JPEG_ldm.png
#figures/imgs/imagenet/ILSVRC2012_val_00040606.JPEG_ldm.png
#figures/imgs/imagenet/ILSVRC2012_val_00041051.JPEG_adm.png

