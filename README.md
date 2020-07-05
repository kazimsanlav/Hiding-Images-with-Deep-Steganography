Hiding Images with Deep Steganography
=====================================

Steganography
-------------
[Steganography](https://en.wikipedia.org/wiki/Steganography) is the practice of concealing a file, message, image, or video within another file, message, image, or video.

Problem
-------
Hide a colorful image into another colorful image, then reveal the secret image as much as possible

Model
-----
[HidingNet](HidingNet.py) consists of hidingnet and revealing net.  
Hidingnet contains skip connections to preserve spatial features.
Revealnet consists of basic CNN layers and BatchNorm layers.

Dataset
-------
[MS COCO 2017](https://cocodataset.org/#download) is used.  
> I used testset(41K) for training and valset(5K) for testing.
> Original trainset(118K) should be preffered for training to get better performance. 

Should be placed in `data/raw/coco-test2017` & `data/raw/coco-val2017`  

Requirements
------------
Pytorch, Numpy, Pillow, skimage

Results
-------
### Training Data
Rows correspond to: Secret, Cover, Hidden and Revealed Images

![step0](reports/figures/step0.png)
<p style="text-align:center">Step 0</p>

![step0](reports/figures/step1000.png)
<p style="text-align:center">Epoch 1</p>

![step0](reports/figures/step20000.png)
<p style="text-align:center">Epoch 5</p>

![step0](reports/figures/step30000.png)
<p style="text-align:center">Epoch 10</p>

### Test Data
Rows correspond to Secret, Cover, Hidden and Revealed Images

![step0](reports/figures/testdata.png)

#### Hidden-Cover
Rows correspond to Hidden, Cover, Hidden-Cover, 10*(Hidden-Cover)

![hidden_cover](reports/figures/hidden_cover.png)

#### Hidden-Secret
Rows correspond to Hidden, Secret, Hidden-Secret, 10*(Hidden-Secret)

![hidden_cover](reports/figures/hidden_hide.png)

#### Reveal-Secret
Rows correspond to Reveal, Secret, Reveal-Secret, 10*(Reveal-Cover)

![hidden_cover](reports/figures/reveal_secret.png)

References
----------
[1] Baluja, Shumeet. "Hiding images in plain sight: Deep steganography." Advances in Neural Information Processing Systems. 2017.



