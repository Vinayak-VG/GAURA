# GAURA: Generalizable Approach for Unified Restoration and Rendering of Arbitrary Views (ECCV 2024)
[Vinayak Gupta](https://vinayak-vg.github.io/)<sup>1*</sup>,
[Rongali Girish](https://girish445ai.github.io/)<sup>1*</sup>,
[Mukund Varma T](https://mukundvarmat.github.io/)<sup>2*</sup>,
[Ayush Tewari](https://ayushtewari.com/)<sup>3</sup>,
[Kaushik Mitra](https://www.ee.iitm.ac.in/kmitra/)<sup>1</sup>,

<sup>1 </sup>Indian Institute of Technology Madras, <sup>2 </sup>University of California, San Diego, <sup>3 </sup>Massachusetts Institute of Technology, Cambridge 

<sup>\*</sup> Equal Contributions

[Project Page](https://vinayak-vg.github.io/GAURA/) | [Paper](https://arxiv.org/pdf/2402.04632.pdf)

This repository is built based on GNT's [offical repository](https://github.com/VITA-Group/GNT)

<ul>
  <li><span style="color: red">News!</span> GAURA is accepted at ECCV 2024 🎉. 
  <!-- Our updated cross-scene trained <a href="https://github.com/VITA-Group/GNT#pre-trained-models">checkpoint</a> should generalize to complex scenes, and even achieve comparable results to SOTA per-scene optimized methods without further tuning! -->
  </li>
  <!-- <li><span style="color: red">News!</span> Our work was presented by Prof. Atlas in his <a href="https://mit.zoom.us/rec/play/O-E4BZQZLc4km4Xd9EFXrMleMBPVoxK73HzZwo7iEmndSZb--QJXHoo4apFKWT_VEA09TQSO7p6CkIuw.q0ReKAVz5tfsS2Ye?continueMode=true&_x_zm_rtaid=GwwbZYSBSbqSZaZ-b10Qqw.1666125821172.50b38719911eea3b66d299aac233d421&_x_zm_rhtaid=94">talk</a> at the <a href="https://sites.google.com/view/visionseminar">MIT Vision and Graphics Seminar</a> on 10/17/22.</li> -->
</ul>

## Introduction

Neural rendering methods can achieve near-photorealistic image synthesis of scenes from posed input images. However, when the images are imperfect, e.g., captured in very low-light conditions, state- of-the-art methods fail to reconstruct high-quality 3D scenes. Recent approaches have tried to address this limitation by modeling various degradation processes in the image formation model; however, this limits them to specific image degradations. In this paper, we propose a general- izable neural rendering method that can perform high-fidelity novel view synthesis under several degradations. Our method, GAURA, is learning- based and does not require any test-time scene-specific optimization. It is trained on a synthetic dataset that includes several degradation types. GAURA outperforms state-of-the-art methods on several benchmarks for low-light enhancement, dehazing, deraining, and on-par for motion deblurring. Further, our model can be efficiently fine-tuned to any new incoming degradation using minimal data. We thus demonstrate adapta- tion results on two unseen degradations, desnowing and removing defocus blur. 

![teaser](assets/Teaser.pdf)

## Installation

Clone this repository:

```bash
git clone https://github.com/Vinayak-VG/GAURA.git
cd GAURA/
git submodule update --init --recursive
cd data_generation/MiDAS/weights
wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt
wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_large_384.pt
```

The code is tested with python 3.8, cuda == 11.1, pytorch == 1.10.1. Additionally dependencies include: 

```bash
torchvision
ConfigArgParse
imageio
matplotlib
numpy
opencv_contrib_python
Pillow
scipy
imageio-ffmpeg
lpips
scikit-image
```

## Datasets

### Training Dataset (Synthetic Scenes)
We reuse the training, evaluation datasets from [IBRNet](https://github.com/googleinterns/IBRNet). All datasets must be downloaded to a directory `data/` within the project folder and must follow the below organization. 
```bash
├──data/
    ├──ibrnet_collected_1/
    ├──ibrnet_collected_2/
    ├──real_iconic_noface/
    ├──nerf_llff_data/
```
We refer to [IBRNet's](https://github.com/googleinterns/IBRNet) repository to download and prepare data. For ease, we consolidate the instructions below:
```bash
mkdir data
cd data/

# IBRNet captures
gdown https://drive.google.com/uc?id=1rkzl3ecL3H0Xxf5WTyc2Swv30RIyr1R_
unzip ibrnet_collected.zip

# LLFF
gdown https://drive.google.com/uc?id=1ThgjloNt58ZdnEuiCeRf9tATJ-HI0b01
unzip real_iconic_noface.zip

## [IMPORTANT] remove scenes that appear in the test set
cd real_iconic_noface/
rm -rf data2_fernvlsb data2_hugetrike data2_trexsanta data3_orchid data5_leafscene data5_lotr data5_redflower data2_chesstable data2_colorfountain data4_shoerack data4_stove
cd ../
mkdir test_data/
mv real_iconic_noface/data2_chesstable test_data/
mv real_iconic_noface/data2_colorfountain test_data/
mv real_iconic_noface/data4_shoerack test_data/
mv real_iconic_noface/data4_stove test_data/

# LLFF dataset (eval)
gdown https://drive.google.com/uc?id=16VnMcF1KJYxN9QId6TClMsZRahHNMW5g
unzip nerf_llff_data.zip
```

### Evaluation Dataset (Real Scenes)
```bash
# Low-Light Enhancement
## Aleth-NeRF Dataset
gdown https://drive.google.com/file/d/1orgKEGApjwCm6G8xaupwHKxMbT2s9IAG
tar -xvzf LOM_full.tar.gz

## LLNeRF Dataset
gdown https://drive.google.com/file/d/1RfdBe7xJbNnyOvq_B6_cBBbvNRLumtFu
tar -xvzf normal-light-scenes.tar.gz

# Haze
## REVIDE-HAZE Dataset
gdown https://drive.google.com/file/d/1MYaVMUtcfqXeZpnbsfoJ2JBcpZUUlXGg
Note: We only pick these scenes - J005, L004, L008 and W002 for evaluation.

# Motion Blur
## Deblur-NeRF Dataset
https://drive.google.com/drive/folders/1X-NfxsZXWH5c4vjaKVjlFnEQU-l54ag_?usp=sharing

# Defocus Blur
## Deblur-NeRF Dataset
https://drive.google.com/drive/folders/1qXSgGWUbgIfKdNK16AytEHvxO0lRZ0K5?usp=drive_link

For Rain and Snow, we manually collected videos from Youtube and ran COLMAP to obtain the poses. These scenes don't have corresponding ground truth.
```

## Data Preparation

For Haze and Defocus degradations, we require depth maps that are precomputed and saved to save time while training. 
```bash
python3 data_generation/generate_depths.py --data_dir data/
```

## Usage

### Training

If you wish to start training from the pre-trained weights of GNT, then you can create a folder in out/ with the name of the experiment and then put the pre-trained weights in the folder. 

```bash
python3 -W ignore train.py --config configs/transibr_bigger_full.txt --expname generalise_expt --n_iters 400000 --N_rand 512 --i_img 10000 --i_weights 10000 --typeofmodel yesstrgth_dyndeg_emb_wgt_strenc -- pretrained_allweights --ft_corrup gen --train_dataset llff_dyn+ibrnet_collected_dyn --eval_dataset llff_test_dyn --sample_mode center
```

### Evaluation

You could also download our pre-train weights for direct model evaluation from [(google drive)](https://drive.google.com/file/d/1ShjmESBCGdmewwOtopBwOJ7hEqwYY4D0/view?usp=sharing)

To evaluate Low-Light Enhancement Results on Real Data
```bash
# On Aleth-NeRF Dataset
bash eval_scripts/eval_aleth.sh 

# On LLNeRF Dataset
bash eval_scripts/eval_llnerf.sh
```

To evaluate Motion Blur Removal Results on Deblur-NeRF Real Dataset
```bash
bash eval_scripts/eval_real_motion.sh
```

To evaluate Haze Removal Results on REVIDE-Haze Real Dataset
```bash
eval_scripts/eval_revidehaze.sh 
```

The code has been recently tidied up for release and could perhaps contain tiny bugs. Please feel free to open an issue.


## Cite this work

If you find our work/code implementation useful for your own research, please cite our paper.

```
@article{gupta2024gsn,
  title={GAURA: Generalizable Approach for Unified Restoration and Rendering of Arbitrary Views},
  author={Gupta, Vinayak and Girish, Rongali and Varma, Mukund T and Tewari, Ayush and Mitra, Kaushik},
  journal={arXiv preprint arXiv:2402.04632},
  year={2024}
}
```