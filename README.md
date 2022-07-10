# High-throughput molecular imaging via deep learning enabled Raman spectroscopy
This repository is for DeepeR, introduced in the following paper:

[Conor C. Horgan](https://www.kcl.ac.uk/people/conor-horgan), [Magnus Jensen](https://www.kcl.ac.uk/people/magnus-jensen), [Anika Nagelkerke](https://www.rug.nl/staff/a.p.nagelkerke/), [Jean-Phillipe St-Pierre](https://engineering.uottawa.ca/people/st-pierre-jean-philippe), [Tom Vercauteren](https://www.kcl.ac.uk/people/tom-vercauteren), [Molly M. Stevens](http://www.stevensgroup.org/), and [Mads S. Bergholt](http://www.bergholtlab.com/), "High-throughput molecular imaging via deep learning enabled Raman spectroscopy", [Analytical Chemistry 2021, 93, 48, 15850-15860](https://doi.org/10.1021/acs.analchem.1c02178).

The code was implemented in Python 3.7.3 using PyTorch 1.4.0 on a desktop computer with a Core i7-8700 CPU at 3.2 GHz (Intel), 32 GB of RAM, and a Titan V GPU (NVIDIA), running Windows 10 (Microsoft).

## Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Training](#training)
4. [Testing](#testing)
5. [Results](#results)
6. [Citation](#citation)
7. [Acknowledgements](#acknowledgements)

## Introduction
Raman spectroscopy enables non-destructive, label-free imaging with unprecedented molecular contrast but is limited by slow data acquisition, largely preventing high-throughput imaging applications. Here, we present a comprehensive framework for higher-throughput molecular imaging via deep learning enabled Raman spectroscopy, termed DeepeR, trained on a large dataset of hyperspectral Raman images, with over 1.5 million spectra (400 hours of acquisition) in total. We firstly perform denoising and re-construction of low signal-to-noise ratio Raman molecular signatures via deep learning, with a 10x improvement in mean squared error over common Raman filtering methods. Next, we develop a neural network for robust 2–4x spatial super-resolution of hyperspectral Raman images that preserves molecular cellular information. Combining these approaches, we achieve Raman imaging speed-ups of up to 40-90x, enabling good quality cellular imaging with high resolution, high signal-to-noise ratio in under one minute. We further demonstrate Raman imaging speed-up of 160x, useful for lower resolution imaging applications such as the rapid screening of large areas or for spectral pathology. Finally, transfer learning is applied to extend DeepeR from cell to tissue-scale imaging. DeepeR provides a foundation that will enable a host of higher-throughput Raman spectroscopy and molecular imaging applications across biomedicine.

DeepeR is designed to operate on hyperspectral Raman images, where high information-content Raman spectra at each pixel provide detailed insight into the molecular composition of cells/tissues. To improve the speed of Raman spectroscopic imaging and enable high-throughput applications, we first (i) train a 1D ResUNet neural network for Raman spectral denoising to effectively reconstruct a high SNR Raman spectrum (long acquisition time) from a corresponding low SNR input spectrum (short acquisition time). Next, we (ii) train a hyperspectral residual channel attention neural network to accurately reconstruct high spatial resolution hyperspectral Raman images from corresponding low spatial resolution hyper-spectral Raman images to significantly reduce imaging times. Then, by combining (i) and (ii) we achieve extreme speed-ups of up to 160x in Raman imaging time while maintaining high reconstruction fidelity. Finally, we (iii) demonstrate that transfer learning can be used to take our pre-trained neural networks (trained on large datasets) to operate on an entirely unrelated hyperspectral data domain for which there is only a limited dataset (insufficient to effectively train a neural network from scratch).

![Figure_1](/Figures/Figure_1.png)
DeepeR Overview.

## Dataset
DeepeR was trained on a collection of hyperspectral Raman images, consisting of over 1.5 million Raman spectra (400 hours of acquisition) in total. The dataset will be made publicly available soon.

## Training
### Raman spectral denoising
1. Download [dataset](https://drive.google.com/drive/folders/1590Zqr56txK5_hVlrfe7oEIdcKoUTEIH?usp=sharing).
2. Edit /Raman Spectral Denoising/train.py to point towards the dataset path.
3. In your python environment, run /Raman Spectral Denoising/train.py with your chosen options, e.g.:
    ```bash
    python train.py --epochs 500 --batch-size 256 --optimizer adam --lr 5e-4 --scheduler one-cycle-lr --batch-norm 
    ```
### Hyperspectral super-resolution
1. Download [dataset](https://drive.google.com/drive/folders/1W9vUVUCW21A4vw_KBjuMj3O5qlAzjdvi?usp=sharing).
2. Edit /Hyperspectral Super-Resolution/train.py to point towards the dataset path.
3. In your python environment, run /Hyperspectral Super-Resolution/train.py with your chosen options, e.g.:
    ```bash
    python train.py --epochs 600 --batch-size 2 --optimizer adam --lr 1e-5 --lr-image-size 16 --hr-image-size 64 
    ```

## Testing
### Raman spectral denoising
1. (Optionally) download [pre-trained model](https://drive.google.com/drive/folders/1ISE5yxZZcOYLZntN7L-knwOFrRtxpj42?usp=sharing).
2. Edit /Raman Spectral Denoising/test.py to point towards the dataset path.
3. In your python environment, run /Raman Spectral Denoising/test.py:
    ```bash
    python test.py 
    ```
### Hyperspectral super-resolution
1. (Optionally) download [pre-trained models](https://drive.google.com/drive/folders/1o8P3ztMcggd7-iQo8ohEEOYagJ6SBTpq?usp=sharing)
2. Edit /Hyperspectral Super-Resolution/test.py to point towards the dataset path.
3. In your python environment, run /Hyperspectral Super-Resolution/test.py:
    ```bash
    python test.py 
    ```
## Results
### Raman spectral denoising
![Figure_2](/Figures/Figure_2.png)
Deep Learning Enabled Raman Denoising. (a) Exemplar test set pair of low SNR input Raman spectrum (light grey) and corresponding high SNR target Raman spectrum (dark grey) as well as the Savitzky-Golay (light blue), wavelet denoising (purple), PCA denoising (dark blue), and neural network (red) outputs for the given input spectrum (normalised to maximum peak intensity). (b) Mean squared error (performed across all spectral channels and all image pixels) across all test set hyperspectral Raman cell images for raw input spectra, 1D ResUNet output spectra, PCA denoising output spectra, wavelet denoising output spectra, and Savitzky-Golay output spectra (order x, frame width y) output spectra with respect to corresponding target spectra (n = 11) (error bars: mean ± STD) (one-way ANOVA with Dunnett’s multiple comparisons test against raw input spectra, *** P < 0.005). (c) Exemplar 1450 cm-1 peak intensity heatmaps for low SNR input hyperspectral Raman image, PCA denoising of input hyperspectral Raman image, 1D ResUNet output, and target high SNR hyperspectral Raman image with corresponding imaging times shown in white (min:sec) (scale bar = 10 µm). (d) Exemplar vertex component analysis (VCA) performed on target high SNR hyperspectral Raman image identifies 5 key components (proteins/lipids (red), nucleic acids (blue), proteins (green), lipids (yellow), and background (black)) which are applied to low SNR input, PCA denoising output, and 1D ResUNet output images via non-negatively constrained least-squares regression demonstrating that low SNR input and PCA denoising output data do not effectively identify different cell components. (e-f) Exemplar Raman spectra (white arrows in (c)) corresponding to (e) a lipid-rich cytoplasmic region and (f) the nucleus.

### Hyperspectral super-resolution
![Figure_3](/Figures/Figure_3.png)
Deep learning enabled hyperspectral image super-resolution. (a) 2x, 3x, and 4x super-resolution of example test set hyperspectral Raman image enables a significant reduction in imaging times (shown in white) while recovering important spatial and spectral information (scale bars = 10 µm). Images shown are the result of a VCA performed on the target HR hyperspectral Raman image, which identified 4 key components (nucleic acids (blue), proteins (green), lipids (yellow), and background (black)). VCA components were applied to the nearest neighbour output, bicubic output, and HyRISR output images via non-negatively constrained least-squares regression. (b) Exemplar Raman spectrum at white arrow in (a) demonstrating that the neural network output (red) is more closely aligned to the target (ground truth) spectrum (dark grey). (c-d) Mean test set (c) PSNR, (d) SSIM, (e) MSE values for nearest neighbour upsampling, bicubic upsampling, and HyRISR output for 2x, 3x, and 4x super-resolution (n = 9) (error bars: mean ± STD) (One-way paired analysis of variance (ANOVA) with Geisser-Greenhouse correction and Tukey’s multiple comparisons test, * P < 0.05, ** P < 0.01, *** P < 0.001).

### Combined Raman spectral denoising and hyperspectral super-resolution
![Figure_4](/Figures/Figure_4.png)
Combined Raman spectral denoising and hyperspectral image super-resolution enables extreme speed-ups in Raman imaging time. (a) Sequential application of Raman spectral denoising followed by hyperspectral image super-resolution enables extreme speed-ups in imaging time (shown in white) from 68:15 (min:sec) to 01:42 for 2x super-resolution, 00:44 for 3x super-resolution, and 00:26 for 4x super-resolution while largely preserving molecular information (scale bars = 10 µm). Images shown are the result of a VCA performed on the target HR, high SNR hyperspectral Raman image, which identified 4 key components (nucleic acids (blue), proteins (green), lipids (yellow), and background (black)). VCA compo-nents were applied to input, Savitky-Golay pluc bicubic upsampling, PCA plus bicubic upsampling, and neural network out-put images via non-negatively constrained least-squares regression. (b) Pixel classification accuracy for input, Savitzky-Golay filtering plus bicubic upsampling output, PCA denoising plus bicubic upsampling output, and neural network output images as compared to VCA pixel classification of target HR, high SNR hyperspectral Raman image.

### Transfer learning
![Figure_5](/Figures/Figure_5.png)
Transfer learning enables effective super-resolution for a small dataset of tissue-engineered cartilage hyperspectral Raman images. (a) Transfer learning of our HISR neural network, trained only on MDA-MB-231 breast cancer cell imag-es, enabled effective cross-domain 4x super-resolution of hyperspectral Raman images despite having only a very small dataset of tissue-engineered cartilage for training. For each condition, images shown on the left are the result of a VCA performed on the target HR, high SNR hyperspectral Raman image, which identified 5 key components (substrate (blue), dense ECM/cells (green), sparse ECM (yellow), cells (red), and background (black)). VCA components were applied to nearest neighbour upsampling, bicubic upsampling, tissue model (from scratch), and cell model (transfer learning) images via non-negatively constrained least-squares regression. Images shown on the right for each condition are 1450 cm-1 peak intensity heatmaps. All images formed as composition of overlapping 64x64 pixel image patches (scale bars = 10 µm). (b) Exemplar Raman spectrum (white arrow in (a)) demonstrating that transfer learning achieves high accuracy reconstruc-tion of the target spectra for each pixel. (c-d) Mean test set (c) PSNR and (d) SSIM values for nearest neighbour upsam-pling, bicubic upsampling, and neural network outputs for 4x super-resolution, calculated on a per-image patch basis (n = 12 patches) (Error bars: mean ± STD) (One-way paired analysis of variance (ANOVA) with Geisser-Greenhouse correction and Tukey’s multiple comparisons test, * P < 0.05, ** P < 0.01, *** P < 0.001).

## Citation
If you find this code helpful in your work, please cite the following [paper](https://doi.org/10.1021/acs.analchem.1c02178):

Conor C. Horgan, Magnus Jensen, Anika Nagelkerke, Jean-Phillipe St-Pierre, Tom Vercauteren, Molly M. Stevens, and Mads S. Bergholt, "High-throughput molecular imaging via deep learning enabled Raman spectroscopy", Analytical Chemistry 2021, 93, 48, 15850-15860.

arXiv link: [arXiv:2009.13318](https://arxiv.org/abs/2009.13318)

## Acknowledgements
Parts of DeepeR were built on [RCAN](https://github.com/yulunzhang/RCAN). We thank the authors for making their code publicly available.
