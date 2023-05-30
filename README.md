# Diffusion-SDF: Text-to-Shape via Voxelized Diffusion

Created by [Muheng Li](https://ttlmh.github.io/), [Yueqi Duan](https://duanyueqi.github.io/), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), and [Jiwen Lu](https://scholar.google.com/citations?user=TN8uDQoAAAAJ&hl=en&authuser=1).

![intro](media/intro.png)

We propose a new generative 3D modeling framework called **Diffusion-SDF** for the challenging task of text-to-shape synthesis. Previous approaches lack flexibility in both 3D data representation and shape generation, thereby failing to generate highly diversified 3D shapes conforming to the given text descriptions. To address this, we propose a SDF autoencoder together with the Voxelized Diffusion model to learn and generate representations for voxelized signed distance fields (SDFs) of 3D shapes. Specifically, we design a novel UinU-Net architecture that implants a local-focused inner network inside the standard U-Net architecture, which enables better reconstruction of patch-independent SDF representations. We extend our approach to further text-to-shape tasks including text-conditioned shape completion and manipulation. Experimental results show that Diffusion-SDF is capable of generating both high-quality and highly diversified 3D shapes that conform well to the given text descriptions. Diffusion-SDF has demonstrated its superiority compared to previous state-of-the-art text-to-shape approaches.

![intro](media/pipeline.gif)

[[Project Page]](https://ttlmh.github.io/DiffusionSDF/) [[arXiv]](https://arxiv.org/abs/2212.03293)

## Code Demo
Please note that the current release of Diffusion-SDF repository includes the inference code specifically for text-to-shape generation. We are actively working on making the code for text-to-shape completions and manipulations available as well, and it will be added to the repository soon!

### Installation
To set up the Diffusion-SDF environment, you can use the provided `diffusionsdf.yml` file to create a Conda environment. Follow the steps below:
1. Clone the repository:
```
git clone https://github.com/ttlmh/Diffusion-SDF.git
```
2. Create the Conda environment using the provided YAML file and activate:
```
conda env create -f diffusionsdf.yml 
conda activate diffusionsdf
```

### Download Pre-trained Models
Download the SDF auto-encoder model file (vae_epoch-120.pth: [Baidu Disk](https://pan.baidu.com/s/1z0eh8SXSrn8tbq5epo0r6Q?pwd=f1cv) / [Google Drive](https://drive.google.com/file/d/18MxWYf6IItYOxUzdM5LiWb8dr9zSwA-2/view?usp=sharing)) and the Voxelized Diffusion model file (voxdiff-uinu.ckpt: [Baidu Disk](https://pan.baidu.com/s/1Emu5kFVaYbuKIkdCKlghXQ?pwd=q1wv) / [Google Drive](https://drive.google.com/file/d/1Cno18LFR_V24oCLxwmTJttdBu7AyP1aa/view?usp=sharing))) from the above links. Place the downloaded model files in the directory ```./ckpt``` .


### Demo for Text-to-Shape Generation
To generate 3D shapes from text descriptions using Diffusion-SDF, you can run the following command:

```
python txt2sdf.py --prompt "a revolving chair" --save_obj
```
This command will initiate the text-to-shape generation process with the provided prompt, in this case, "a revolving chair". The generated 3D shape will be saved as GIF renderings and OBJ files.

Feel free to modify the prompt or adjust the command parameters as needed to explore different text descriptions.

Please ensure that you have set up the environment, downloaded the pre-trained models, and placed them in the correct directory before running the demo command.

## Acknowledgement
Our code is based on [Stable-Diffusion](https://github.com/CompVis/stable-diffusion), and [AutoSDF](https://github.com/yccyenchicheng/AutoSDF).

## Citation
If you find our work useful in your research, please consider citing:

```
@inproceedings{li2023diffusionsdf,
  author={Li, Muheng and Duan, Yueqi and Zhou, Jie and Lu, Jiwen},
  title={Diffusion-SDF: Text-to-Shape via Voxelized Diffusion},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```
