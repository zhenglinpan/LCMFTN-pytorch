## LCMFTN(WACV 2020)

This is an unofficial implementation of [Line Art Correlation Matching Feature Transfer Network for Automatic Animation Colorization(WACV 2020)](https://arxiv.org/abs/2004.06718).

Before anything, you need to know although I've done most of the coding, I couldn't manage to gain the excellent results as presented in the paper. In fact, some issues remain unaddressed and I will discuss about this later. If you're good with this, you can proceed with

```bash
python lcmftn.py
```

to train and test your own model.

<p align="center">
  <img src="https://github.com/ZhenglinPan/LCMFTN-pytorch/blob/master/others/results.png" width="600" alt="accessibility text">
</p>

## Dataset
You need tp prepare your own dataset. In my dataset, 1k pairs were engaged, stride=6, width=12. RTX 3060 Laptop(6G) was used for experimental training. 

## Modification

### Using ResNet instead of ResNeXt
Although the researcher promoted the cons of resnext, I found that the network often leads to a model colapse. So I used a resnet instead. (I might hadn't implemented resnext as it should.)

### Absense of Illustration2Vec network
Illustration2Vec in the paper, denoted as $E_{I}$, was published in 2015 and uses caffe model, it is outdated. since $E_{I}$ plays as a feature extractor, so I replace it with a CNN model(not pretrained).

### Replacing pretrained LetNet work with ControlNet lineart processor
To generate lineart from color frames, LetNet was adopted by the researchers. Again, the model is washed-out a bit and is not easy to access. Therefore I replaced LetNet with [lineart processor](https://huggingface.co/ControlNet-1-1-preview/control_v11p_sd15_lineart) used in ControlNet project. It generates similar sketches.

## Discussion
### using '+' as concatenation in CMFT's input
The paper didn't mentioned but I noticed that the normal concatenation is not suitable for this project. Refer to figure 2 in the paper, you can tell CA should have the width $H*W$ the same as height of reshaped $F_{B}(y^{B})$, presented in tensors, it should be a matmul between `[N, C, H*W, H*W]` and `[N, C, H*W, 1]`. However, the normal concatenation adds up `C` and thus leaves us a CA of shape `[N, 2*C, H*W, H*W]` which is not available to use. You can see this from `CMFT0` block in Figure 4 and formula (6): $(E_{c}^{0}(C_{n}))_{e}=CMFT(ca(E_{s}^{0}(S_{n}), E_{I}(S_{n})), ca(E_{s}^{0}(S_{p}), E_{I}(S_{p})), E_{c}^{0}(C_{p}))$. The only specious explanation how the researchers carried out concatenation is they used `+` instead of `ca()`.

### Problematic CMFT
Yes, Correlation Matching Feature Transfer(CMFT) model might be wrong.
When I was checking the reasons for my bad outputs, I noticed CMFT model, the key idea of this paper, might be problematic. CMFT can not propagate the correlations between two sketches to inferring new color images. This can be tested with a simple example: set Sp = [[1, 2], [2, 1]], Sn = [[1, 2], [2, 1]], Cp = [[1, 3], [3, 1]] and Cn = [[1, 3], [3, 1]], after CMFT you'll notice the predicted Cn is deviated from real Cn atrociously. In ideal case, since Sp and Sn share the same spacial information, the Cp and Cn should be the same. But CMFT gives a `CA` that changes Cn greatly, which is incorrect. 

### GPU consumption
Follow with the parameters provided in the paper, there are some 20 million parameters in the model, which is acceptable, but CUDA OUT OF MEMORY will be triggered on RTX 3090 (24G) if you start training.

*@Implementation by [Zhenglin](https://github.com/ZhenglinPan)*

*@Paper Author: Qian Zhang, Bo Wang, Wei Wen, Hai Li, Junhui Liu*

*@Date: July 2, 2023*
