# Goal
* Can we detect common objects in a variety of image domains without instance-level annotations?(**Method:** `cross-domain weakly supervised object detection`)
# Scene preparation
* In this paper, semi-supervised learning of `watercolor painting` is carried out through cross-domain operation of paper-cut painting and cartoon.
* Cartoons and paper-cut drawings have **instance-level annotations**, while watercolor paintings have only **image-level annotations**.
* In addition, the classes to be detected in the target domain are all or a `subset` of those in the source domain
# Conclusion
We test our methods on our newly collected datasets1 containing three image domains, and achieve an improvement of approximately 5 to 20 percentage points in terms of ***mean average precision (mAP)*** compared to the best-performing baselines.

# Preparations for the paper
```Datasets and codes```
https://naoto0804.github.io/cross_domain_detection

```The Best Performing Full Supervision Detector:```
* R. Girshick. Fast R-CNN. In ICCV, 2015. 1
* W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y.Fu, and A. C. Berg. ***SSD: Single shot multibox detector.*** In ECCV, 2016.
* Y. Li, K. He, J. Sun, et al. ***R-FCN: Object detection via region-based fully convolutional networks***. In NIPS, 2016
``` In this paper, the method of generating instance-level annotation image is presented```
* This generation is achieved by image-to-image translation methods from unpaired examples such as ***CycleGAN***
[<a>CycleGan](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
  
  ```Datasets```
  
|Name | sample number| 
|- | :-: |-: | 
|Clipart1k || 1000 |
|Watercolor2k|| 2000| 
|Comic2k ||2000|
