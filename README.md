# Awesome-XAD [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

Paper and dataset collection for the [paper](https://arxiv.org/pdf/2302.06670.pdf).

<div align="center">
  <img src="images/iad_updated.png" width="350px" height="350px"> ></a>
  <img src="images/vad_updated.jpg" width="350px" height="350px">
</div>


## Explainable 2D Anomaly Detection Methods

### [Image+Video]

#### - Attention-based
- **[AD-FactorVAE]** Towards visually explaining variational autoencoders. | **CVPR 2020** | [[pdf]](https://arxiv.org/pdf/1911.07389.pdf) [[code]](https://github.com/liuem607/expVAE) 
- **[CAVGA]** Attention guided anomaly localization in images. | **ECCV 2020** | [[pdf]](https://arxiv.org/pdf/1911.08616.pdf) 
- **[SSPCAB]** Self-Supervised Predictive Convolutional Attentive Block for Anomaly Detection. | **CVPR 2022** | [[pdf]](https://arxiv.org/pdf/2111.09099.pdf) [[code]](https://github.com/ristea/sspcab)
#### - Generative-model-based
- **[LSAND]** Latent space autoregression for novelty detection. | **CVPR 2019** | [[pdf]](https://arxiv.org/pdf/1807.01653.pdf) [[code]](https://github.com/aimagelab/novelty-detection)
- **[CFLOW-AD]** CFLOW-AD: Real-time unsupervised anomaly detection with localization via conditional normalizing flows. | **WACV 2022** | [[pdf]](https://arxiv.org/pdf/2107.12571.pdf) [[code]](https://github.com/gudovskiy/cflow-ad)

### [Image]

#### - Attention-based
- **[Gradcon]** Backpropagated gradient representations for anomaly detection. | **ECCV 2020** | [[pdf]](https://arxiv.org/pdf/2007.09507.pdf) [[code]](https://github.com/olivesgatech/gradcon-anomaly)
- **[FCCD]** Explainable deep one-class classification. | **ICLR 2021** | [[pdf]](https://arxiv.org/pdf/2007.01760.pdf) [[code]](https://github.com/liznerski/fcdd)


#### - Input-perturbation-based
- **[ODIN]** Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks. | **ICLR 2018** | [[pdf]](https://arxiv.org/pdf/1706.02690.pdf) [[code]](https://github.com/facebookresearch/odin) 
- **[Mahalanobis]** A Simple Unified Framework for Detecting
Out-of-Distribution Samples and Adversarial Attacks. | **NIPS 2018** | [[pdf]](https://arxiv.org/pdf/1807.03888.pdf) [[code]](https://github.com/pokaxpoka/deep_Mahalanobis_detector) 
- **[Generalized-ODIN]** Generalized ODIN: Detecting Out-of-distribution Image without Learning from Out-of-distribution Data. | **CVPR 2020** | [[pdf]](https://arxiv.org/pdf/2002.11297.pdf) [[code]](https://github.com/sayakpaul/Generalized-ODIN-TF)
- **[SLA2P]** Self-supervision meets adversarial perturbation: A novel framework for anomaly detection. | **CIKM 2022** | [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3511808.3557697) [[code]](https://github.com/wyzjack/SLA2P)
#### - Generative-model-based
- **[AnoGAN]** Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery. | **IPMI 2017** | [[pdf]](https://arxiv.org/pdf/1703.05921.pdf) [[code]](https://github.com/LeeDoYup/AnoGAN-tf) 
- **[ALAD]** Adversarially Learned Anomaly Detection. | **ICDM 2018** | [[pdf]](https://arxiv.org/pdf/1812.02288.pdf) [[code]](https://github.com/houssamzenati/Adversarially-Learned-Anomaly-Detection)
- **[f-AnoGAN]** f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks. | **Medical Image Analysis 2019** | [[pdf]](https://pdf.sciencedirectassets.com/272154/1-s2.0-S1361841519X00031/1-s2.0-S1361841518302640/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEDwaCXVzLWVhc3QtMSJIMEYCIQC%2FisjWoA72xskqwPoitP0ZaVBPEJsiLLwDoGKEpwkZmAIhAJ5XhRYoiigejzppOjtBOBXo%2FSGSxccJ65XD7GedzswxKrMFCCUQBRoMMDU5MDAzNTQ2ODY1Igx2MDPLo6QjvFXO2GQqkAX2xK%2BSvxJPtBqlo%2B1MmpiJSiQCtkfj6EkzRftxTACnO1KXA4%2FxB69Xwr3okwxDMcR8SKNAUXyIkCemTpfkJWn4D1Z6n6AGwG0xDhCguFFY%2BDfPvlDcZOSZYwJfBXm2gYYEKxO%2BJ8fLmeYZ3rYzK9nhHJiwvvtK6LFEC5uQD9eGrvUrm6V3AiKRcEclouUaxOmIgq2IngLrSjE%2BBXy8vySjMInPRNfXqnkR8nWZdEmD%2FmOBEJcWDHpz5ZAFZI%2FiFL1oumlqa5OEqpFKNAtfI1IRGrVoax83EOuHhy5NQN%2FujYT%2FR%2F9y%2BY%2FNHTFNe7EVbnfed4mYyNlslUXo9Wi5YBSQTHtOOCxStJeLxZD1p9sVukgxGD3J7Til848ZtqEE%2F0j6z9lfBOVlVPLvPwQE1eaMxehF%2BCTR5tK%2FXE%2FTjXF8uTPKKztBDNeEMJdTYL%2F%2BS1iNyEbGIeWSj0H7w78wD1ZGP2SKbgVEwEsvWdVrRUoIcq2xKbp9l30p8UYiS3EMi%2Fy1Ntn5hwZgh2InL32k%2F0Ow3enmDyOnaGXFMv4Qdz92%2BEdOD0wvbyrn%2BYAFdoB3YIIU4rTXAAD5jN6fDiqgJlKfunFjrPMh%2FWiHgRNX5B9aq60s0gTJEeME3aBCKBi1mN6QC15D078s%2BirlCo07EhoJWJ%2F7ymlQdgHZyM69tFS217lw5PyvHgsJ6TFy%2BgVxiZy35ZVO0vHoloAfpw%2BOYptMU%2FZ%2B4mSAszjgX%2FDvfHNoTq8HeSPMbrKpRCAImmKi4byUcmmecGxu7cwlDndd2eCKixcXkQa1Vf0nUBfqKDSa1%2FRzjWpuxAAlqE%2FxBj4ySIaTRgRYO2pGYa5CKXS%2B%2BgFKFqoNUzNQRdJ6Vy8rKNnFkjD7w9CuBjqwAWnn%2FjcBOyXnP4GP6NYLtHr0%2FBEAr0Yf94LBSM1qauItj2NAkwBlyNU0QFJ0h3vNlj19asTNeyXh2Qai%2FP3aY94JNV8TzHQb16peq26W8MxQiHsa6sOZ6k71kJY29C6h%2BNhg%2B32hyLEhJT8HreM4Tb1kQXnn%2BPm1gPXBF2sl3rN5bAN%2F835VE6g1rsCfzx6PdNMm0rUzVrIQ86w%2FB7VfG3E%2F%2BPpjE65Rwr1iP%2Fd%2FpET9&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240220T053214Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYZJ454XGU%2F20240220%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=92cf6bcc9892325e12943205cfd3f10d5a6f8a136123cd240adbe012bcd35184&hash=ce89a3f48c8aa22efef2511e79145d395bbd2f9a7742889524972bdba1ba0cac&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1361841518302640&tid=spdf-a0a09e10-c8c7-465d-98cb-2ed33ee82819&sid=b59e7eba6539554a7e8b1df4f38f615bc5d4gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=15115c5b5954530253&rr=85845e300e260ca2&cc=us) [[code]](https://github.com/tSchlegl/f-AnoGAN)
- **[Genomics-OOD]** Likelihood Ratios for Out-of-Distribution Detection. | **NeurIPS 2019** | [[pdf]](https://arxiv.org/pdf/1906.02845.pdf) [[code]](https://github.com/google-research/google-research/tree/master/genomics_ood)
- **[Likelihood-Regret]** Likelihood Regret: An Out-of-Distribution Detection Score For Variational Auto-encoder. | **NeurIPS 2020** | [[pdf]](https://arxiv.org/pdf/2003.02977.pdf) [[code]](https://github.com/XavierXiao/Likelihood-Regret)
- **[FastFlow]** FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows. | **arXiv 2021** | [[pdf]](https://arxiv.org/pdf/2111.07677.pdf) [[code]](https://github.com/gathierry/FastFlow)
- **[AnoDDPM]** AnoDDPM: Anomaly Detection with Denoising Diffusion Probabilistic Models using Simplex Noise. | **CVPRW 2022** | [[pdf]](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Wyatt_AnoDDPM_Anomaly_Detection_With_Denoising_Diffusion_Probabilistic_Models_Using_Simplex_CVPRW_2022_paper.pdf) [[code]](https://github.com/Julian-Wyatt/AnoDDPM)
- **[Diffusion-anomaly]** Diffusion Models for Medical Anomaly Detection. | **MICCAI 2022** | [[pdf]](https://arxiv.org/pdf/2203.04306.pdf) [[code]](https://github.com/JuliaWolleb/diffusion-anomaly)
- **[DDPM]** Fast unsupervised brain anomaly detection and segmentation with diffusion models. | **MICCAI 2022** | [[pdf]](https://arxiv.org/pdf/2206.03461.pdf) 
- **[DiAD]** DiAD: A Diffusion-based Framework for Multi-class Anomaly Detection. | **AAAI 2024** | [[pdf]](https://arxiv.org/pdf/2312.06607.pdf) [[code]](https://github.com/lewandofskee/DiAD)
- **[DeCo-Diff]** Correcting Deviations from Normality: A Reformulated Diffusion Model for Multi-Class Unsupervised Anomaly Detection. | **CVPR 2025** | [[pdf]](https://arxiv.org/pdf/2503.19357.pdf) [[code]](https://github.com/farzad-bz/DeCo-Diff)
#### - Foundation-model-based
- **[WinCLIP]** WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation. | **CVPR 2023** | [[pdf]](https://arxiv.org/pdf/2303.14814.pdf) [[code]](https://github.com/caoyunkang/WinClip)
- **[CLIP-AD]** CLIP-AD: A Language-Guided Staged Dual-Path Model for Zero-shot Anomaly Detection. | **arXiv 2023** | [[pdf]]() [[code]]()
- **[AnomalyCLIP]** AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection. | **ICLR 2024** | [[pdf]](https://arxiv.org/pdf/2310.18961.pdf) [[code]](https://github.com/zqhang/AnomalyCLIP)
- **[SAA+]** Segment Any Anomaly without Training via Hybrid Prompt Regularization. | **arXiv 2023** | [[pdf]](https://arxiv.org/pdf/2305.10724.pdf) [[code]](https://github.com/caoyunkang/Segment-Any-Anomaly)
- **[AnomalyGPT]** AnomalyGPT: Detecting Industrial Anomalies Using Large Vision-Language Models. | **AAAI 2024** | [[pdf]](https://arxiv.org/pdf/2308.15366.pdf) [[code]](https://github.com/CASIA-IVA-Lab/AnomalyGPT)
- **[Myriad]** Myriad: Large Multimodal Model by Applying Vision Experts for Industrial Anomaly Detection. | **arXiv 2023** | [[pdf]](https://arxiv.org/pdf/2310.19070.pdf) [[code]](https://github.com/tzjtatata/Myriad)
- **[GPT-4V]** Towards Generic Anomaly Detection and Understanding: Large-scale Visual-linguistic Model (GPT-4V) Takes the Lead. | **arXiv 2023** | [[pdf]](https://arxiv.org/pdf/2311.02782.pdf) [[code]](https://github.com/caoyunkang/GPT4V-for-Generic-Anomaly-Detection)
- **[GPT-4V-AD]** Exploring Grounding Potential of VQA-oriented GPT-4V for Zero-shot Anomaly Detection. | **arXiv 2023** | [[pdf]](https://arxiv.org/pdf/2311.02612.pdf) [[code]](https://github.com/zhangzjn/GPT-4V-AD)
- **[CLIP-SAM]** ClipSAM: CLIP and SAM Collaboration for Zero-Shot Anomaly Segmentation. | **arXiv 2024** | [[pdf]](https://arxiv.org/pdf/2401.12665.pdf) [[code]](https://github.com/Lszcoding/ClipSAM)
- **[AnomalyClip]** AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection. | **ICLR 2024** | [[pdf]](https://arxiv.org/pdf/2310.18961.pdf) [[code]](https://github.com/zqhang/AnomalyCLIP)
- **[AA-CLIP]** AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP. | **arXiv 2025** | [[pdf]](https://arxiv.org/pdf/2503.06661.pdf) [[code]](https://github.com/Mwxinnn/AA-CLIP)
- **[MVFA]** Adapting visual-language models for generalizable anomaly detection in medical images. | **CVPR 2024** | [[pdf]](https://arxiv.org/pdf/2311.14821.pdf) [[code]](https://github.com/MediaBrain-SJTU/MVFA-AD)
- **[InCTRL]** Toward generalist anomaly detection via in-context residual learning with few-shot sample prompts. | **CVPR 2024** | [[pdf]](https://arxiv.org/pdf/2311.16732.pdf) [[code]](https://github.com/mala-lab/InCTRL)
- **[Bayes-PFL]** Bayesian Prompt Flow Learning for Zero-Shot Anomaly Detection. | **arXiv 2025** | [[pdf]](https://arxiv.org/pdf/2503.10080.pdf) [[code]](https://github.com/xiaozhen228/Bayes-PFL)
- **[LogSAD]** Towards Training-free Anomaly Detection with Vision and Language Foundation Models. | **CVPR 2025** | [[pdf]](https://arxiv.org/pdf/2503.18325.pdf) [[code]](https://github.com/zhang0jhon/LogSAD)
- **[LogicAD]** LogicAD: Explainable Anomaly Detection via VLM-based Text Feature Extraction. | **AAAI 2025** | [[pdf]](https://arxiv.org/pdf/2501.01767.pdf) [[code]](https://arxiv.org/pdf/2501.01767)
- **[Anomaly-OV]** Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models. | **CVPR 2025** | [[pdf]](https://arxiv.org/pdf/2502.07601.pdf)  
- **[UniVAD]** UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection. | **CVPR 2025** | [[pdf]](https://arxiv.org/pdf/2412.03342.pdf) [[code]](https://github.com/FantasticGNU/UniVAD)

### [Video]

#### - Attention-based
- **[Self-trained-DOR]** Self-trained Deep Ordinal Regression for End-to-End Video Anomaly Detection. | **CVPR 2020** | [[pdf]](https://arxiv.org/pdf/2003.06780.pdf) 
- **[DSA]** Dance with self-attention: A new look of conditional random fields on anomaly detection in videos. | **ICCV 2021** | [[pdf]](https://openaccess.thecvf.com/content/ICCV2021/papers/Purwanto_Dance_With_Self-Attention_A_New_Look_of_Conditional_Random_Fields_ICCV_2021_paper.pdf) 
#### - Reasoning-based
- **[Scene-Graph]** Scene graphs for interpretable video anomaly classification. | **NIPSW 2018** | [[pdf]](https://nips2018vigil.github.io/static/papers/accepted/30.pdf) 
- **[CTR]** Learning causal temporal relation and feature discrimination for anomaly detection. | **TIP 2021** | [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9369126) 
- **[Interpretable]** Towards interpretable video anomaly detection. | **WACV 2023** | [[pdf]](https://openaccess.thecvf.com/content/WACV2023/papers/Doshi_Towards_Interpretable_Video_Anomaly_Detection_WACV_2023_paper.pdf) 
- **[VADor-w-LSTC]** Video Anomaly Detection and Explanation via Large Language Models. | **arXiv 2024** | [[pdf]](https://arxiv.org/pdf/2401.05702.pdf)
#### - Intrinsic interpretable
- **[JDR]** Joint detection and recounting of abnormal events by learning deep generic knowledge. | **ICCV 2017** | [[pdf]](https://arxiv.org/pdf/1709.09121.pdf) 
- **[XMAN]** X-MAN: Explaining multiple sources of anomalies in video. | **CVPRW 2021** | [[pdf]](https://arxiv.org/pdf/2106.08856.pdf) 
- **[VQU-Net]** Discrete neural representations for explainable anomaly detection. | **WACV 2022** | [[pdf]](https://arxiv.org/pdf/2112.05585.pdf)
- **[AI-VAD]** Attribute-based Representations for Accurate and Interpretable Video Anomaly Detection. | **arXiv 2022** | [[pdf]](https://arxiv.org/pdf/2212.00789.pdf) [[code]](https://github.com/talreiss/Accurate-Interpretable-VAD)
- **[EVAL]** Eval: Explainable video anomaly localization. | **arXiv 2022** | [[pdf]](https://arxiv.org/pdf/2212.07900.pdf)
#### - Memory-based
- **[Memory AD]** Learning memory-guided normality for anomaly detection. | **CVPR 2020** | [[pdf]](https://arxiv.org/pdf/2004.02232.pdf) [[code]](https://github.com/cvlab-yonsei/MNAD)
- **[DLAN-AC]** Dynamic local aggregation network with adaptive clusterer for anomaly detection. | **ECCV 2022** | [[pdf]](https://arxiv.org/pdf/2207.07824.pdf) [[code]](https://github.com/Beyond-Zw/DLAN-AC)
  

## Datasets

### [Image]

#### - Classification datasets for semantic anomaly detection and OOD detection
[2009] [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [2009] [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), [2011] [Texture](https://kylberg.org/kylberg-texture-dataset-v-1-0/), [2011] [SVHN](http://ufldl.stanford.edu/housenumbers/), [2012] [MNIST](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6296535), [2003] [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02), [2015] [LSUN](https://paperswithcode.com/dataset/lsun), [2015] [iSUN](https://turkergaze.cs.princeton.edu/), [2015] [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [2017] [Fashion-MNIST](https://arxiv.org/pdf/1708.07747.pdf), [2017] [CURE-TSR](https://github.com/olivesgatech/CURE-TSD)

#### - Sensory anomaly detection datasets
[2007] [DAGM](https://hci.iwr.uni-heidelberg.de/content/weakly-supervised-learning-industrial-optical-inspection), [2012] [BRATS](https://www.med.upenn.edu/cbica/brats/), [2015] [BrainMRI](https://www.researchgate.net/profile/Muhammad-Usman-209/publication/320472617_Brain_tumour_detection_using_MRI/links/59e784a40f7e9ba6e3048cd1/Brain-tumour-detection-using-MRI.pdf), [2015] [UK-Biobank](https://www.ukbiobank.ac.uk/enable-your-research/about-our-data), [2017] [MSLUB](https://link.springer.com/article/10.1007/s12021-017-9348-7), [2019] [WMH](https://github.com/hjkuijf/wmhchallenge), [2019] [Fishyscapes](https://fishyscapes.com/), [2019] [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/), [2019] [LAG](https://github.com/smilell/AG-CNN), [2019] [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), [2020] [MTD](https://github.com/abin24/Magnetic-tile-defect-datasets.), [2020] [SDD](https://www.vicos.si/resources/kolektorsdd/), [2021] [BTAD](https://paperswithcode.com/dataset/btad),  [2021] [MPDD](https://vutbr-my.sharepoint.com/:f:/g/personal/xjezek16_vutbr_cz/EhHS_ufVigxDo3MC6Lweau0BVMuoCmhMZj6ddamiQ7-FnA?e=oHKCxI), [2021] [MedMNIST](https://medmnist.com/), [2021] [KSSD2](https://www.vicos.si/resources/kolektorsdd2/), [2021] [RoadAnomaly21](https://segmentmeifyoucan.com/datasets), [2022] [VisA](https://paperswithcode.com/dataset/visa), [2022] [MVTec LOCO AD](https://www.mvtec.com/company/research/datasets/mvtec-loco?gad_source=1&gclid=CjwKCAiAuNGuBhAkEiwAGId4apFQMGEJODgQZRX3Tg37hVLBLEDY3808RhHBrNy3OM_nIqZcC7qmAhoCJF0QAvD_BwE)

### [Video]
[2008] [Subway](https://www.researchgate.net/figure/Subway-dataset-exit-gate-three-abnormal-events-and-their-corresponding-detection-maps_fig2_329353016), [2009] [UMN](https://www.crcv.ucf.edu/projects/Abnormal_Crowd/#UMN), [2010] [UCSD-Ped](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm), [2013] [CUHK-Avenue](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html), [2018] [UCF-Crime](https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa?e=1&dl=0), [2018] [ShanghaiTech-Campus](https://svip-lab.github.io/dataset/campus_dataset.html), [2019] [Street-Scene](https://www.merl.com/research/highlights/video-anomaly-detection), [2020] [XD-Violence](https://roc-ng.github.io/XD-Violence/), [2021] [X-MAN](https://arxiv.org/pdf/2106.08856.pdf), [2021] [TAD](https://github.com/ktr-hubrt/WSAL), [2022] [UBnormal](https://github.com/lilygeorgescu/UBnormal/), [2023] [NWPU-Campus](https://campusvad.github.io/)







  

