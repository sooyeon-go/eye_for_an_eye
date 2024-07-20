# Eye-for-an-eye: Appearance Transfer with Semantic Correspondence in Diffusion Models

[arXiv](https://arxiv.org/abs/2406.07008) | [Project Page](https://sooyeon-go.github.io/eye_for_an_eye/)

> **Eye-for-an-eye: Appearance Transfer with Semantic Correspondence in Diffusion Models**<br>
> [Sooyeon Go](https://sooyeon-go.github.io/), [Kyungmook Choi](https://chkmook.github.io/), [Minjung Shin](https://minjung-s.github.io/), [Youngjung Uh](https://vilab.yonsei.ac.kr/member/professor)<br>
> 
>**Abstract**: <br>
As pretrained text-to-image diffusion models have become a useful tool for image synthesis, people want to specify the results in various ways. In this paper, we introduce a method to produce results with the same structure of a target image but painted with colors from a reference image, especially following the semantic correspondence between the result and the reference. E.g., the result wing takes color from the reference wing, not the reference head. Existing methods rely on the query-key similarity within self-attention layer, usually producing defective results. To this end, we propose to find semantic correspondences and explicitly rearrange the features according to the semantic correspondences. Extensive experiments show the superiority of our method in various aspects: preserving the structure of the target and reflecting the color from the reference according to the semantic correspondences, even when the two images are not aligned.

![Teaser](./images/teaser_img.png)

Code coming Soon!

## Description  
Official implementation of our Eye-for-an-eye: Appearance Transfer with Semantic Correspondence in Diffusion Models paper.


## Environment
Our code builds on the requirement of the `diffusers` library. To set up their environment, please run:
```
git clone https://github.com/sooyeon-go/eye_for_an_eye.git
cd eye_for_an_eye
conda env create -f environment/environment.yaml
conda activate eye_for_eye
```

(Optional) You may also want to install [SAM-HQ](https://github.com/SysCV/sam-hq) to extract the instance masks:
pip install git+https://github.com/SysCV/sam-hq.git
Please download the ViT-L HQ-SAM model from the provided link.


## Citation
If you use this code for your research, please cite the following work: 
```
@misc{go2024eyeforaneye,
      title={Eye-for-an-eye: Appearance Transfer with Semantic Correspondence in Diffusion Models}, 
      author={Sooyeon Go and Kyungmook Choi and Minjung Shin and Youngjung Uh},
      year={2024},
      eprint={2406.07008},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```