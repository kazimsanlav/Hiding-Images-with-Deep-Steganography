Hiding Images with Deep Steganography
=====================================

Steganography
-------------
[Steganography](https://en.wikipedia.org/wiki/Steganography) is the practice of concealing a file, message, image, or video within another file, message, image, or video.

Problem
-------
Hide a colorful image into another colorful image with minumum change, then reveal the secret image as much as possible

Model
-----
![Model Archtecture](reports/figures/model_diag.png)
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

### Test Results
**Rows correspond to Secret, Cover, Hidden and Revealed Images in order**

![step0](reports/figures/testdata.png)

#### 1) Hidden-Cover

**Rows correspond to Hidden, Cover, Hidden-Cover, 10\*(Hidden-Cover) in order**

![hidden_cover](reports/figures/hidden_cover.png)

#### 2) Hidden-Secret
**Rows correspond to Hidden, Secret, Hidden-Secret, 10\*(Hidden-Secret) in order**

![hidden_cover](reports/figures/hidden_hide.png)

#### 3) Reveal-Secret
**Rows correspond to Reveal, Secret, Reveal-Secret, 10\*(Reveal-Cover) in order**

![hidden_cover](reports/figures/reveal_secret.png)

### Training Results

**Rows correspond to: Secret, Cover, Hidden and Revealed Images in order**

![Untrained](reports/figures/step0.png)  
<p align="center">
  Before Training
</p>  

---   

![step0](reports/figures/step1000.png)  
<p align="center">
  Step 1000
</p>  


---   

![step0](reports/figures/step20000.png)  
<p align="center">
  Step 20000
</p>

---   

![step0](reports/figures/step30000.png)  
<p align="center">
  After Training
</p>  



References
----------
[1] Baluja, Shumeet. "Hiding images in plain sight: Deep steganography." Advances in Neural Information Processing Systems. 2017.



