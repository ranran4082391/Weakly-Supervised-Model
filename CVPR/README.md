# Goal
* Can we detect common objects in a variety of image domains without instance-level annotations?(**Method:** `cross-domain weakly supervised object detection`)
# Scene preparation
* In this paper, semi-supervised learning of `watercolor painting` is carried out through cross-domain operation of paper-cut painting and cartoon.
* Cartoons and paper-cut drawings have **instance-level annotations**, while watercolor paintings have only **image-level annotations**.
* In addition, the classes to be detected in the target domain are all or a `subset` of those in the source domain
# Conclusion
* We test our methods on our newly collected datasets1 containing three image domains, and achieve an improvement of approximately 5 to 20 percentage points in terms of ***mean average precision (mAP)*** compared to the best-performing baselines.

# Preparations for the paper
## Datasets and codes
https://naoto0804.github.io/cross_domain_detection

## The Best Performing Full Supervision Detector:
* R. Girshick. Fast R-CNN. In ICCV, 2015. 1
* W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y.Fu, and A. C. Berg. ***SSD: Single shot multibox detector.*** In ECCV, 2016.
* Y. Li, K. He, J. Sun, et al. ***R-FCN: Object detection via region-based fully convolutional networks***. In NIPS, 2016.

## Weakly Supervised Detection 
* Many existing methods are built upon `region-of-interest (RoI)` extraction methods such as selective search [36]
* Feature extraction for each region, region selection,and classification of the selected region are performed through multiple instance learning (***MIL***), or two-stream CNN (V. Kantorov, M. Oquab, M. Cho, and I. Laptev. ***ContextLocNet***: Context-aware deep network models for weakly supervised localization. In ECCV, 2016.)

## Cross-domain Object Detection
* Using an object detector that is neither trained nor finetuned for the target domain causes a significant `drop` in performance as shown in [38]. (This is why this paper uses `DT`.)

```In this paper, the method of generating instance-level annotation image is presented```
* This generation is achieved by image-to-image translation methods from unpaired examples such as ***CycleGAN***
[CycleGan](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
  
## This paper support raw Datasets 
  
| Name | Classes | number | 
| - | :-: | -: | 
| clipart | 20| 1000 | 
| watercolor | 6 | 2000 | 
| comic | 6 | 2000 |

## Tip about dataset 
* This paper using Amazon Mechanical Turk, there are similar robotic service software in China, which is very efficient.



# Our main contributions are as follows

* We propose a framework for a novel task, cross-domain weakly supervised object detection. We achieve a two-step progressive domain adaptation by sequentially fine-tuning the FSD on the artificially generated samples by the proposed domain transfer and pseudo-labeling.

* We construct novel, fully instance-level annotated datasets with multiple instances of various object classes across three domains that are far from natural images

* Our experimental results show that our framework outperforms the best-performing baselines by approximately 5 to 20 percentage points in terms of ***mAP***.

# Experimental Procedures
**`First`, we pre-train it while using instancelevel annotations in the source domain. `Second`, we fine-tune it while using the images obtained by DT. `Lastly`, we finetune it while using the images obtained by PL. We would like to emphasize that the sequential execution of the two fine-tuning steps is critical as the performance of PL highly depends on the used FSD.**

# Experiments Result
* In Sec. <font color=red>1</font>, we explain the details of the implementations, the compared methods, and the evaluation metrics.
* In Sec. <font color=red>2</font>, we test our methods using Clipart1k and conduct error analysis and ablation studies on the FSDs. 
* In Sec. <font color=red>3</font>, we confirm that our framework is generalized for a variety of domains using Watercolor2k and Comic2k. 
* In Sec. <font color=red>4</font>, we show actual detection results and the generateddomain-transferred images for further discussion.

## Other
***VOC2007-trainval and VOC2012-trainval*** [6] were used as images in the <font face="黑体" color=green size=5>source domain</font>

