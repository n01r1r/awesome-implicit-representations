# Awesome Implicit Neural Representations [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
A curated list of resources on implicit neural representations, originally forked from [vsitzmann/awesome-implicit-representations](https://github.com/vsitzmann/awesome-implicit-representations).

> [!NOTE]
> This is a forked repository that includes additional search results and papers curated by the current maintainer.
> Please note that these curations are primarily focused on **image-related tasks**. Other categories from the original list may be intentionally not updated or removed to maintain this specific focus.
>
> For non-image INR applications (audio, PDEs, generic signals), see [Non-Image INRs](#implicit-neural-representations-beyond-images). For methods that are not strictly INRs but commonly used alongside them, see [Related Non-INR Works](#related-but-non-inr-works).

## What counts as an INR in this list?

We consider a method an Implicit Neural Representation (INR) if it:
- Represents a continuous signal $x \mapsto f_\theta(x)$ with a coordinate-based neural network (typically an MLP), rather than a discrete grid.
- Uses this network as the primary representation of the signal (image, shape, field, etc.), not merely as an auxiliary module.
- Falls into the broader family of neural fields (NeRF, SDF fields, signed distance functions, occupancy fields, etc.).

We exclude works that:
- Use the word “implicit” only conceptually, while the representation itself is voxel- or grid-based.
- Use an MLP only as a classifier, without directly modeling a continuous signal over coordinates.

## Table of contents
- [Surveys & Reviews](#surveys--reviews)
- [Computational Imaging, ISP & Color](#computational-imaging-isp--color)
- [Inverse Rendering & 3D Reconstruction](#inverse-rendering--3d-reconstruction)
- [Generative Visual Models](#generative-visual-models)
- [Dynamics & Video](#dynamics--video)
- [Semantics & Visual Representation](#semantics--visual-representation)
- [Foundations & Theory](#foundations--theory)
- [Colabs](#colabs)
- [Talks](#talks)

---

### Surveys & Reviews
* [Where Do We Stand with Implicit Neural Representations? A Technical and Performance Survey](https://arxiv.org/abs/2411.03688) (Essakine et al. 2024) - Classifies INR methods and compares performance across multiple tasks.
* [Implicit Neural Representation in Medical Imaging: A Comparative Study](https://openaccess.thecvf.com/content/ICCV2023W/CVAMD/papers/Molaei_Implicit_Neural_Representation_in_Medical_Imaging_A_Comparative_Survey_ICCVW_2023_paper.pdf) (Molaei et al. ICCV 2023 Workshop) - Systematic study comparing INR-based methods across various medical imaging tasks.

### Computational Imaging, ISP & Color
* [GamutMLP: A Lightweight MLP for Color Loss Recovery](https://arxiv.org/abs/2304.11743) (Le & Brown, CVPR 2023) - Optimizes a lightweight MLP during gamut reduction to predict clipped color values.
* [NILUT: Conditional Neural Implicit 3D Lookup Tables for Image Enhancement](https://arxiv.org/abs/2306.11920) (Zhang et al. AAAI 2024) - Implicitly defined continuous 3D color transformations for memory-efficient and controllable image enhancement.
* [Signal Processing for Implicit Neural Representations](https://arxiv.org/abs/2210.12648) (Xu et al. NeurIPS 2022) - Performs classical signal processing operations (denoising, smoothing, filtering) directly on INR-parameterized signals.

### Inverse Rendering & 3D Reconstruction
* [PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields](https://arxiv.org/abs/2306.02708) (2023/2024) - Jointly estimates geometry, materials, and lighting using physics-based priors for realistic relighting.
* [Benchmarking Implicit Neural Representation and Geometric Estimation for SLAM](https://arxiv.org/abs/2403.19473) (Hua & Wang, CVPR 2024) - Comparative analysis of INR/Geometric representations in SLAM.
* [Rethinking Implicit Neural Representations for Vision Learners](https://arxiv.org/abs/2211.12352) (Song et al. 2022) - Reinterprets INR learning/usage from a vision learner perspective.
* [Light Field Networks: Neural Scene Representations with Single-Evaluation Rendering](https://vsitzmann.github.io/lfns/) (Sitzmann et al. 2021) - Represents 3D scenes via their 360-degree light field.
* [Neural Radiance Fields (NeRF)](https://www.matthewtancik.com/nerf) (Mildenhall et al. 2020) - The foundational work on volumetric rendering for novel view synthesis.
* [Pixel-NERF](https://alexyu.net/pixelnerf/) (Yu et al. 2020) - Conditions a NeRF on local features lying on camera rays.
* [Multiview neural surface reconstruction by disentangling geometry and appearance](https://lioryariv.github.io/idr/) (Yariv et al. 2020) - Sphere-tracing with positional encodings for complex 3D scenes.
* [Neural Unsigned Distance Fields for Implicit Function Learning](https://arxiv.org/pdf/2010.13938.pdf) (Chibane et al. 2020) - Learning unsigned distance fields from raw point clouds.
* [Scene Representation Networks](https://vsitzmann.github.io/srns/) (Sitzmann et al. 2019) - Continuous 3D-structure-aware neural scene representations.
* [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://arxiv.org/abs/1901.05103) (Park et al. 2019)
* [Occupancy Networks: Learning 3D Reconstruction in Function Space](https://arxiv.org/abs/1812.03828) (Mescheder et al. 2019)
* [Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization](https://shunsukesaito.github.io/PIFu/) (Saito et al. 2019)
* [Implicit Geometric Regularization for Learning Shapes](https://github.com/amosgropp/IGR) (Gropp et al. 2020) - Learns signed distance fields (SDFs) from raw 3D data using an Eikonal regularization for smooth implicit surfaces.
* [AutoInt: Automatic Integration for Fast Neural Volume Rendering](https://davidlindell.com/publications/autoint) (Lindell et al. 2020) - Accelerates neural volume rendering by learning closed-form integral approximations along rays in neural fields.

### Generative Visual Models
* [Alias-Free Generative Adversarial Networks (StyleGAN3)](https://nvlabs.github.io/stylegan3/) (Karras et al. 2021) - Alias-free image GAN architecture; included as an INR-adjacent generative model often used in conjunction with neural fields (see also [Related Non-INR Works](#related-but-non-inr-works)).
* [GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields](https://arxiv.org/abs/2011.12100) (Niemeyer et al. 2021)
* [Unsupervised Discovery of Object Radiance Fields](https://kovenyu.com/uorf/) (Yu et al. 2021)
* [pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis](https://arxiv.org/abs/2012.00926) (Chan et al. 2020)
* [Generative Radiance Fields for 3D-Aware Image Synthesis (GRAF)](https://autonomousvision.github.io/graf/) (Schwarz et al. 2020)
* [Learning Continuous Image Representation with Local Implicit Image Function (LIIF)](https://github.com/yinboc/liif) (Chen et al. 2020) - Continuous image representation for super-resolution.

### Dynamics & Video
* [Neural Radiance Flow for 4D View Synthesis](https://yilundu.github.io/nerflow/) (2020/2021)
* [Space-time Neural Irradiance Fields for Free-Viewpoint Video](https://video-nerf.github.io/) (2020/2021)
* [Non-Rigid Neural Radiance Fields](https://gvv.mpi-inf.mpg.de/projects/nonrigid_nerf/) (2020/2021)
* [Nerfies: Deformable Neural Radiance Fields](https://nerfies.github.io/) (2020/2021)
* [D-NeRF: Neural Radiance Fields for Dynamic Scenes](https://arxiv.org/abs/2011.13961) (2020/2021)
* [X-Fields: Implicit Neural View-, Light- and Time-Image Interpolation](https://xfields.mpi-inf.mpg.de/) (Bemana et al. 2020)

### Semantics & Visual Representation
Methods that use implicit neural fields primarily as representational substrates for classification, segmentation, or generic vision encoding, rather than for direct image synthesis or 3D reconstruction.

* [End-to-End Implicit Neural Representations for Classification](https://arxiv.org/abs/2503.18123) (Gielisse & van Gemert, CVPR 2025) - *Note: Pre-print/accepted paper.*
* [Implicit Neural Representation Facilitates Unified Universal Vision Encoding](https://arxiv.org/abs/2601.14256) (Hu et al. 2026/2025) - *Note: "HUVR" paper, arxiv ID placeholder pending final pub.*
* [Inferring Semantic Information with 3D Neural Scene Representations](https://www.computationalimaging.org/publications/semantic-srn/) (2020) - Weakly supervised semantic segmentation.

### Foundations & Theory
* [H-SIREN: Improving implicit neural representations with hyperbolic periodic functions](https://arxiv.org/abs/2410.04716) (Gao & Jaiman 2024) - Uses hyperbolic periodic activation functions to improve INR performance and convergence.
* [Improved Implicit Neural Representation with Fourier Reparameterized Training](https://openaccess.thecvf.com/content/CVPR2024/papers/Shi_Improved_Implicit_Neural_Representation_with_Fourier_Reparameterized_Training_CVPR_2024_paper.pdf) (Shi et al. CVPR 2024)
* [Fourier features let networks learn high frequency functions](https://people.eecs.berkeley.edu/~bmild/fourfeat/) (Tancik et al. 2020)
* [Multiresolution Neural Networks for Imaging](https://visteam.isr.uc.pt/wp-content/uploads/Paper_069_2022_Paz.pdf) (Paz et al. 2022) - Proposes multiresolution coordinate-based networks that are continuous in space and scale, with applications to continuous image representation and multilevel reconstruction.
* [Implicit Neural Representations with Periodic Activation Functions (SIREN)](https://vsitzmann.github.io/siren/) (Sitzmann et al. 2020) - The canonical foundational INR architecture; shows sinusoidal activations with principled initialization enable fitting high-frequency signals such as images and 3D scenes with coordinate-based MLPs.

### Colabs
* [Implicit Neural Representations with Periodic Activation Functions](https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb)
* [Neural Radiance Fields (NeRF)](https://colab.research.google.com/github/bmild/nerf/blob/master/tiny_nerf.ipynb)
* [MetaSDF & MetaSiren](https://colab.research.google.com/github/vsitzmann/metasdf/blob/master/MetaSDF.ipynb)
* [Neural Descriptor Fields](https://colab.research.google.com/drive/16bFIFq_E8mnAVwZ_V2qQiKp4x4D0n1sG?usp=sharing)

### Talks
* [Vincent Sitzmann: Implicit Neural Scene Representations](https://www.youtube.com/watch?v=__F9CCqbWQk&amp;t=1s)
* [Andreas Geiger: Neural Implicit Representations for 3D Vision](https://www.youtube.com/watch?v=F9mRv4v80w0)
* [Gerard Pons-Moll: Shape Representations: Parametric Meshes vs Implicit Functions](https://www.youtube.com/watch?v=_4E2iEmJXW8)
* [Yaron Lipman: Implicit Neural Representations](https://www.youtube.com/watch?v=rUd6qiSNwHs&list=PLat4GgaVK09e7aBNVlZelWWZIUzdq0RQ2&index=11) 

## Links
* [awesome-NeRF](https://github.com/yenchenlin/awesome-NeRF) - List of implicit representations specifically on neural radiance fields (NeRF)

---

### Implicit Neural Representations Beyond Images
Classic INR papers for non-image signals (audio, PDEs, generic fields) that are outside the main image-focused scope of this list.

* [Implicit Neural Representations with Periodic Activation Functions (SIREN)](https://vsitzmann.github.io/siren/) (Sitzmann et al. 2020) – Also used for audio signals and simple PDEs.
* [Signal Processing for Implicit Neural Representations](https://arxiv.org/abs/2210.12648) (Xu et al. NeurIPS 2022) – General INR-based signal processing beyond images (denoising, smoothing, filtering).
* [AutoInt: Automatic Integration for Fast Neural Volume Rendering](https://davidlindell.com/publications/autoint) (Lindell et al. 2020) – General volume rendering acceleration for neural fields.

### Related but Non-INR Works
Methods that are not strictly coordinate-based neural fields, but are commonly used together with INRs or inspire INR architectures.

* [Alias-Free GAN (StyleGAN3)](https://nvlabs.github.io/stylegan3/) (Karras et al. 2021) – Alias-free convolutional GAN, often used as a backbone in 3D-aware generative pipelines.

## License
License: MIT
