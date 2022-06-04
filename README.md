## Update 2022.5.11
Our paper has been accepted by IEEE Transactions on Image Processing (TIP）！

# CMDSR
Conditional Hyper-Network for Blind Super-Resolution with Multiple Degradations.[[arxiv](https://arxiv.org/abs/2104.03926)][[IEEE TIP](https://ieeexplore.ieee.org/abstract/document/9785471)]

## Introduction
CMDSR is a novel conditional meta-network framework which helps the SR framework learn how to adapt to changes in the degradation distribution of input. The ConditionNet of our framework first learns the degradation prior from a support set, which is composed of a series of degraded image patches from the same task. Then the adaptive BaseNet rapidly shifts its parameters according to the conditional features. A task contrastive loss is also proposed to shorten the inner-task distance and enlarge the crosstask distance between task-level features. Without predefining degradation maps, our blind framework can conduct one single parameter update to yield considerable improvement in SR results. More details can be found in our [paper](https://arxiv.org/abs/2104.03926)

## Architecture
Conditional feature extraction at task-level.
<div align=center><img src="https://github.com/guanghaoyin/CMDSR/blob/main/figs/Task-level.png" alt="Task-level" align="middle" /></div>


Whole Architecture of CMDSR
<div align=center><img src="https://github.com/guanghaoyin/CMDSR/blob/main/figs/CMDSR.png" alt="CMDSR" align="middle" /></div>

## SR results for real images
<div align=center><img src="https://github.com/guanghaoyin/CMDSR/blob/main/figs/real_img1.png" alt="real_img1" align="middle" /></div>
<div align=center><img src="https://github.com/guanghaoyin/CMDSR/blob/main/figs/real_img2.png" alt="real_img2" align="middle" /></div>

## Trained model download
You can download our trained model from [Google Driver](https://drive.google.com/file/d/1431EsUDMRjmIhVGv4FwQQTkOQXehlPk5/view?usp=sharing) to test with your own LR image.

## Acknowledgement
Thanks for the support of Sea-NExT Joint Lab and ByteDance AI Lab.

## Citation

```
@article{yin2022conditional,
  title={Conditional Hyper-Network for Blind Super-Resolution with Multiple Degradations},
  author={Yin, Guanghao and Wang, Wei and Yuand, Zehuan and Ji, Wei and Yue, Dongdong and Sun, Shouqian and Chua, Tat-Seng and Wang, Changhu},
  journal={IEEE Transactions on Image Processing},
  year={2022},
  publisher={IEEE}
}
```
