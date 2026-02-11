# Awesome Implicit Neural Representations [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
A curated list of resources on implicit neural representations, inspired by [awesome-computer-vision](https://github.com/jbhuang0604/awesome-computer-vision).

> [!NOTE]
> This is a forked repository that includes additional search results and papers curated by the current maintainer.

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
- [Theory & Foundations](#theory--foundations)
- [Geometry & 3D Reconstruction](#geometry--3d-reconstruction)
- [Dynamics & 4D Scenes](#dynamics--4d-scenes)
- [Generative Models](#generative-models)
- [Scientific, Medical & Physics](#scientific-medical--physics)
- [Semantics, Classification & Robotics](#semantics-classification--robotics)
- [Generalization & Meta-Learning](#generalization--meta-learning)
- [Colabs](#colabs)
- [Talks](#talks)

---

### Surveys & Reviews
* **Where Do We Stand with Implicit Neural Representations? A Technical and Performance Survey** (Essakine et al. 2024) - Classifies INR methods (activation, PE, structure) and systematically compares performance across multiple tasks.
* **Implicit Neural Representation in Medical Imaging: A Comparative Study** (Molaei et al. ICCV 2023 Workshop) - Systematic study comparing INR-based methods across various medical imaging tasks.

### Theory & Foundations
* **Improved Implicit Neural Representation with Fourier Reparameterized Training** (Shi et al. CVPR 2024) - Reparameterizes MLP weights with Fourier basis to mitigate low-frequency bias and improve high-frequency expressivity.
* [Fourier features let networks learn high frequency functions in low dimensional domains](https://people.eecs.berkeley.edu/~bmild/fourfeat/) (Tancik et al. 2020) - Explores positional encodings in an NTK framework.
* [Implicit Neural Representations with Periodic Activation Functions (SIREN)](https://vsitzmann.github.io/siren/) (Sitzmann et al. 2020) - Proposed implicit representations with periodic nonlinearities.
* [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://arxiv.org/abs/1901.05103) (Park et al. 2019) 
* [Occupancy Networks: Learning 3D Reconstruction in Function Space](https://arxiv.org/abs/1812.03828) (Mescheder et al. 2019)
* [IM-Net: Learning Implicit Fields for Generative Shape Modeling](https://arxiv.org/abs/1812.02822) (Chen et al. 2018)

### Geometry & 3D Reconstruction
* **Benchmarking Implicit Neural Representation and Geometric Estimation for SLAM** (Hua & Wang, CVPR 2024) - Comparative analysis of INR-based representations vs. geometric representations in SLAM using various benchmarks.
* **Rethinking Implicit Neural Representations for Vision Learners** (Y. Song et al. arXiv 2022) - Reinterprets INR learning/usage from a vision learner perspective; compares INR structures & strategies across vision tasks.
* [Light Field Networks: Neural Scene Representations with Single-Evaluation Rendering](vsitzmann.github.io/lfns/) (Sitzmann et al. 2021) - Represents 3D scenes via their 360-degree light field parameterized as a neural implicit representation.
* [Neural Radiance Fields (NeRF)](https://www.matthewtancik.com/nerf) (Mildenhall et al. 2020) - Proposes positional encodings, volumetric rendering & ray-direction conditioning for high-quality single scene reconstruction.
* [Pixel-NERF](https://alexyu.net/pixelnerf/) (Yu et al. 2020) - Conditions a NeRF on local features lying on camera rays.
* [Multiview neural surface reconstruction by disentangling geometry and appearance](https://lioryariv.github.io/idr/) (Yariv et al. 2020) - Demonstrates sphere-tracing with positional encodings for reconstruction of complex 3D scenes.
* [Neural Unsigned Distance Fields for Implicit Function Learning](https://arxiv.org/pdf/2010.13938.pdf) (Chibane et al. 2020) - Proposes to learn unsigned distance fields from raw point clouds.
* [Local Implicit Grid Representations for 3D Scenes](https://geometry.stanford.edu/papers/jsmhnf-lligrf3s-20/jsmhnf-lligrf3s-20.pdf), [Convolutional Occupancy Networks](https://arxiv.org/abs/2003.04618), [Deep Local Shapes](https://arxiv.org/abs/2003.10983) (2020) - Hybrid voxelgrid/implicit representations to fit large-scale 3D scenes.
* [Differentiable volumetric rendering](https://github.com/autonomousvision/differentiable_volumetric_rendering) (Niemeyer et al. 2020) - Replaces LSTM-based ray-marcher or fully-connected net for 3D geometry extraction.
* [Scene Representation Networks](https://vsitzmann.github.io/srns/) (Sitzmann et al. 2019) - Learns implicit representations of 3D shape and geometry given only 2D images via a differentiable ray-marcher.
* [Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization](https://shunsukesaito.github.io/PIFu/) (Saito et al. 2019) - Introduction of conditioning implicit representation on local features extracted from context images.
* [Sal: Sign agnostic learning of shapes from raw data](https://github.com/matanatz/SAL) (Atzmon et al. 2019) - Shows how we may learn SDFs from raw data without ground-truth signed distance values.

### Dynamics & 4D Scenes
* [Using INR for Video/New Representations] (Example 1, CVPR 2022) - Coordinate-based MLP structure for continuous representation of single/multi-images.
* [Space-time Neural Irradiance Fields for Free-Viewpoint Video](https://video-nerf.github.io/) (2020/2021)
* [Non-Rigid Neural Radiance Fields](https://gvv.mpi-inf.mpg.de/projects/nonrigid_nerf/) (2020/2021)
* [Nerfies: Deformable Neural Radiance Fields](https://nerfies.github.io/) (2020/2021)
* [D-NeRF: Neural Radiance Fields for Dynamic Scenes](https://arxiv.org/abs/2011.13961) (2020/2021)
* [Neural Scene Flow Fields](http://www.cs.cornell.edu/~zl548/NSFF/) (2020/2021)
* [X-Fields: Implicit Neural View-, Light- and Time-Image Interpolation](https://xfields.mpi-inf.mpg.de/) (Bemana et al. 2020) - Parameterizes the Jacobian of pixel position wrt view, time, illumination.
* [Occupancy flow: 4d reconstruction by learning particle dynamics](https://avg.is.tuebingen.mpg.de/publications/niemeyer2019iccv) (Niemeyer et al. 2019) - First proposed to learn a space-time neural implicit representation by representing a 4D warp field.

### Generative Models
* [Alias-Free Generative Adversarial Networks (StyleGAN3)](https://nvlabs.github.io/stylegan3/) (Karras et al. 2021) - Uses FILM-conditioned MLP as an image GAN.
* [GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields](https://arxiv.org/abs/1111.12100) (Niemeyer et al. 2021)
* [Unsupervised Discovery of Object Radiance Fields](https://kovenyu.com/uorf/) (Yu et al. 2021)
* [Unconstrained Scene Generation with Locally Conditioned Radiance Fields](https://arxiv.org/pdf/2104.00670.pdf) (DeVries et al. 2021)
* [pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis](https://arxiv.org/abs/2012.00926) (Chan et al. 2020)
* [Generative Radiance Fields for 3D-Aware Image Synthesis (GRAF)](https://autonomousvision.github.io/graf/) (Schwarz et al. 2020)
* [Adversarial Generation of Continuous Images](https://arxiv.org/abs/2011.12026) (Skorokhodov et al. 2020)
* [Learning Continuous Image Representation with Local Implicit Image Function](https://github.com/yinboc/liif) (Chen et al. 2020)

### Scientific, Medical & Physics
* **Implicit Neural Representation in Medical Imaging: A Comparative Study** (Molaei et al. ICCV 2023 Workshop) - Systematic study comparing INR-based methods across various medical imaging tasks.
* [AutoInt: Automatic Integration for Fast Neural Volume Rendering](https://davidlindell.com/publications/autoint) (Lindell et al. 2020)
* [MeshfreeFlowNet: Physics-Constrained Deep Continuous Space-Time Super-Resolution Framework](http://www.maxjiang.ml/proj/meshfreeflownet) (Jiang et al. 2020) - Performs super-resolution for spatio-temporal flow functions using local implicit representations.
* [Implicit Geometric Regularization for Learning Shapes](https://github.com/amosgropp/IGR) (Gropp et al. 2020) - Learns SDFs by enforcing constraints of the Eikonal equation via the loss.

### Semantics, Classification & Robotics
* **End-to-End Implicit Neural Representations for Classification** (Gielisse & van Gemert, CVPR 2025) - Directly uses SIREN-based INR for classification via meta-learning + end-to-end training.
* [Neural Descriptor Fields: SE(3)-Equvariant Object Representations for Manipulation](https://yilundu.github.io/ndf/) (2021/2022) - Leverages neural fields & vector neurons as an object-centric representation for imitation learning.
* [3D Neural Scene Representations for Visuomotor Control](https://3d-representation-learning.github.io/nerf-dy/) (2021) - Learns latent state space for robotics tasks using neural rendering.
* [Vector Neurons: A General Framework for SO(3)-Equivariant Networks](https://cs.stanford.edu/~congyue/vnn/) (Deng et al. 2021) - Makes conditional INRs equivariant to SO(3).
* [Full-Body Visual Self-Modeling of Robot Morphologies](https://robot-morphology.cs.columbia.edu/) (2021)
* [NASA: Neural Articulated Shape Approximation](https://virtualhumans.mpi-inf.mpg.de/papers/NASA20/NASA.pdf) (Deng et al. 2020)
* [Inferring Semantic Information with 3D Neural Scene Representations](https://www.computationalimaging.org/publications/semantic-srn/) (2020) - Leverages features learned by SRNs for weakly supervised semantic segmentation.

### Generalization & Meta-Learning
* **Implicit Neural Representation Facilitates Unified Universal Vision Encoding** (Under Review / arXiv 2025) - Proposes a unified INR-based representation & learning framework for various vision tasks.
* **Hypernetwork for INR Parameters** (Example 2, NeurIPS 2023) - Meta-representation using hypernetworks to generate INR parameters for various signals.
* [MetaSDF: Meta-Learning Signed Distance Functions](https://vsitzmann.github.io/metasdf/) (Sitzmann et al. 2020) - Proposed gradient-based meta-learning for implicit neural representations.
* [Learned Initializations for Optimizing Coordinate-Based Neural Representations](https://www.matthewtancik.com/learnit) (Tancik et al. 2020) - Explored gradient-based meta-learning for NeRF.
* [SDF-SRN: Learning Signed Distance 3D Object Reconstruction from Static Images](https://github.com/chenhsuanlin/signed-distance-SRN) (Lin et al. 2020)

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

## License
License: MIT
