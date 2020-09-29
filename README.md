# High-throughput molecular imaging via deep learning enabled Raman spectroscopy
This repository is for DeepeR, introduced in the following paper:

[Conor C. Horgan](https://www.kcl.ac.uk/people/conor-horgan), [Magnus Jensen](https://www.kcl.ac.uk/people/magnus-jensen), [Anika Nagelkerke](https://www.rug.nl/staff/a.p.nagelkerke/), [Jean-Phillipe St-Pierre](https://engineering.uottawa.ca/people/st-pierre-jean-philippe), [Tom Vercauteren](https://www.kcl.ac.uk/people/tom-vercauteren), [Molly M. Stevens](http://www.stevensgroup.org/), and [Mads S. Bergholt](http://www.bergholtlab.com/), "High-throughput molecular imaging via deep learning enabled Raman spectroscopy", arXiv 2020, [arXiv:2009.13318](https://arxiv.org/abs/2009.13318)

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
Raman spectroscopy enables non-destructive, label-free imaging with unprecedented molecular contrast but is limited by slow data acquisition, largely preventing high-throughput imaging applications. Here, we present a comprehensive framework for higher-throughput molecular imaging via deep learning enabled Raman spectroscopy, termed DeepeR, trained on a large dataset of hyperspectral Raman images, with over 1.5 million spectra (400 hours of acquisition) in total. We firstly perform denoising and reconstruction of low signal-to-noise ratio Raman molecular signatures via deep learning, with a 9x improvement in mean squared error over state-of-the-art Raman filtering methods. Next, we develop a neural network for robust 2–4x super-resolution of hyperspectral Raman images that preserves molecular cellular information. Combining these approaches, we achieve Raman imaging speed-ups of up to 160x, enabling high resolution, high signal-to-noise ratio cellular imaging in under one minute. Finally, transfer learning is applied to extend DeepeR from cell to tissue-scale imaging. DeepeR provides a foundation that will enable a host of higher-throughput Raman spectroscopy and molecular imaging applications across biomedicine.

DeepeR is designed to improve Raman spectroscopic acquisition times towards high-throughput Raman imaging applications. Working across hyperspectral Raman data, DeepeR performs i) Raman spectral denoising, ii) hyperspectral super-resolution, and iii) transfer learning (Figure 1). Raman spectral denoising is performed using a 1D residual UNet (ResUNet), which takes low SNR input spectra and reconstructs them to produce corresponding high SNR output spectra (i). Hyperspectral super-resolution is achieved using an adapted residual channel attention net-work (RCAN) to output a HR hyperspectral Raman image from a LR input (ii). The combination of i) Raman spectral denoising and ii) hyperspectral super-resolution then enables significant Raman imaging speed-ups for high-throughput applications. Finally, DeepeR can be generalised to a wide range of Raman imaging applications through transfer learning, where neural networks pre-trained on large hyperspectral datasets can be fine-tuned to operate effectively on small hyperspectral datasets (iii).

![Figure_1](/Figures/Figure_1.png)
DeepeR Overview.

## Dataset
DeepeR was trained on a collection of hyperspectral Raman images, consisting of over 1.5 million Raman spectra (400 hours of acquisition) in total. The dataset will be made publicly available soon.

## Training
### Raman spectral denoising
1. Download training data (available soon).
2. Edit /Raman Spectral Denoising/train.py to point towards the dataset path.
3. In your python environment, run /Raman Spectral Denoising/train.py with your chosen options, e.g.:
    ```bash
    python train.py --epochs 500 --batch-size 256 --optimizer adam --lr 5e-4 --scheduler one-cycle-lr --batch-norm 
    ```
### Hyperspectral super-resolution
1. Download training data (available soon).
2. Edit /Hyperspectral Super-Resolution/train.py to point towards the dataset path.
3. In your python environment, run /Hyperspectral Super-Resolution/train.py with your chosen options, e.g.:
    ```bash
    python train.py --epochs 600 --batch-size 2 --optimizer adam --lr 1e-5 --lr-image-size 16 --hr-image-size 64 
    ```

## Testing
### Raman spectral denoising
1. Download testing data (available soon).
2. Edit /Raman Spectral Denoising/test.py to point towards the dataset path.
3. In your python environment, run /Raman Spectral Denoising/test.py:
    ```bash
    python test.py 
    ```
### Hyperspectral super-resolution
1. Download testing data (available soon).
2. Edit /Hyperspectral Super-Resolution/test.py to point towards the dataset path.
3. In your python environment, run /Hyperspectral Super-Resolution/test.py:
    ```bash
    python test.py 
    ```
## Results
### Raman spectral denoising
![Denoising](/Figures/Figure_2.PNG)
Deep Learning Enabled Raman Denoising. (a) Exemplar test set pair of low SNR input Raman spectrum (light grey) and corresponding high SNR target Raman spectrum (dark grey) as well as the Savitzky-Golay (blue) and neural network (red) outputs for the given input spectrum (normalised to maximum peak intensity). (b) Mean squared error (performed across all spectral channels and all image pixels) across all test set hyperspectral Raman cell images for raw input spectra, 1D ResUNet output spectra, and Savitzky-Golay output spectra (order x, frame width y) with respect to corresponding target spectra (n = 11) (error bars: mean ± STD) (two-tailed Wilcoxon  paired signed rank test against best performing Savitzky-Golay filter, *** P < 0.005). (c) Exemplar 1450 cm-1 peak intensity heatmaps for low SNR input hyperspectral Raman image, Savitzky-Golay (1st order, frame length 9) filtering of input hyperspectral Raman image, 1D ResUNet output, and target high SNR hyperspectral Raman image with corresponding imaging times shown in white (min:sec) (scale bar = 10 µm). (d) Exemplar vertex component analysis (VCA) performed on target high SNR hyperspectral Raman image identifies 5 key components (proteins/lipids (red), nucleic acids (blue), proteins (green), lipids (yellow), and background (black)) which are applied to low SNR input, Savitzky-Golay output, and 1D ResUNet output images via non-negatively constrained least-squares regression demonstrating that low SNR input and Savitzky-Golay output data do not effectively identify different cell components. (e-f) Exemplar Raman spectra (white arrows in (c)) corresponding to (e) a lipid-rich cytoplasmic region and (f) the nucleus.

### Hyperspectral super-resolution
![SuperRes](/Figures/Figure_3.PNG)
Deep learning enabled hyperspectral image super-resolution. (a) 2x, 3x, and 4x super-resolution of example test set hyperspectral Raman image enables a significant reduction in imaging times (shown in white) while recovering important spatial and spectral information (scale bars = 10 µm). Images shown are the result of a VCA performed on the target HR hyperspectral Raman image, which identified 5 key components (lipid droplets (magenta), nucleic acids (blue), proteins (green), lipids (yellow), and background (black)). VCA components were applied to the nearest neighbour output, bicubic output, and HyRISR output images via non-negatively constrained least-squares regression. (b) Exemplar Raman spec-trum at white arrow in (a) demonstrating that the neural network output (red) is more closely aligned to the target (ground truth) spectrum (dark grey). (c-d) Mean test set (c) PSNR, (d) SSIM, (e) MSE values for nearest neighbour upsam-pling, bicubic upsampling, and HyRISR output for 2x, 3x, and 4x super-resolution (n = 9) (error bars: mean ± STD) (One-way paired analysis of variance (ANOVA) with Geisser-Greenhouse correction and Tukey’s multiple comparisons test, * P < 0.05, ** P < 0.01, *** P < 0.001).

### Combined Raman spectral denoising and hyperspectral super-resolution
![Combined](/Figures/Figure_4.PNG)
Combined Raman spectral denoising and hyperspectral image super-resolution enables extreme speed-ups in Raman imaging time. (a) Sequential application of Raman spectral denoising followed by hyperspectral image super-resolution enables extreme speed-ups in imaging time (shown in white) from 68:15 (min:sec) to 01:42 for 2x super-resolution, 00:44 for 3x super-resolution, and 00:26 for 4x super-resolution while largely preserving molecular information (scale bars = 10 µm). Images shown are the result of a VCA performed on the target HR, high SNR hyperspectral Raman image, which identified 4 key components (nucleic acids (blue), proteins (green), lipids (yellow), and background (black)). VCA components were applied to input, state-of-the-art output (SG filtering and bicubic upsampling), and neural network output images via non-negatively constrained least-squares regression. (b) Pixel classification accuracy for input, state-of-the-art output (SG filtering and bicubic upsampling), and neural network output images as compared to VCA pixel classification of target HR, high SNR hyperspectral Raman image.

### Transfer learning
![Transfer](/Figures/Figure_5.PNG)
Transfer learning enables effective super-resolution for a small dataset of tissue-engineered cartilage hyperspectral Raman images. (a) Transfer learning of our HISR neural network, trained only on MDA-MB-231 breast cancer cell images, enabled effective cross-domain 4x super-resolution of hyperspectral Raman images despite having only a very small dataset of tissue-engineered cartilage for training. For each condition, images shown on the left are the result of a VCA performed on the target HR, high SNR hyperspectral Raman image, which identified 5 key components (substrate (blue), dense ECM/cells (green), sparse ECM (yellow), cells (red), and background (black)). VCA components were applied to nearest neighbour upsampling, bicubic upsampling, tissue model (from scratch), and cell model (fine-tuning) images via non-negatively constrained least-squares regression. Images shown on the right for each condition are 1450 cm-1 peak intensity heatmaps. All images formed as composition of overlapping 64x64 pixel image patches (scale bars = 10 µm). (b) Exemplar Raman spectrum (white arrow in (a)) demonstrating that transfer learning achieves high accuracy reconstruction of the target spectra for each pixel. (c-d) Mean test set (c) PSNR and (d) SSIM values for nearest neighbour upsampling, bicubic upsampling, and neural network outputs for 4x super-resolution, calculated on a per-image patch basis (n = 12 patches) (Error bars: mean ± STD) (One-way paired analysis of variance (ANOVA) with Geisser-Greenhouse correction and Tukey’s multiple comparisons test, * P < 0.05, ** P < 0.01, *** P < 0.001).

## Citation
If you find this code helpful in your work, please cite the following [paper](https://arxiv.org/abs/2009.13318):

Conor C. Horgan, Magnus Jensen, Anika Nagelkerke, Jean-Phillipe St-Pierre, Tom Vercauteren, Molly M. Stevens, and Mads S. Bergholt, "High-throughput molecular imaging via deep learning enabled Raman spectroscopy", arXiv 2020, arXiv:2009.13318.

## Acknowledgements
Parts of DeepeR were built on [RCAN](https://github.com/yulunzhang/RCAN). We thank the authors for making their code publicly available.
