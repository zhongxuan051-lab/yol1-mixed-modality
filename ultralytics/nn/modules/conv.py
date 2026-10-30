# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "Conv",# 标准卷积
    "Conv2",
    "LightConv",
    "DWConv",# 逐深度卷积，每个输入通道单独用一个卷积核处理，通道之间在卷积过程中不混合信息。
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):# 继承自nn.Module
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    #                                  输入通道数，输出，卷积核大小，步长，填充，分组数，膨胀率（空洞卷积参数），是否用激活函数
    default_act = nn.SiLU()  # default activation默认使用 SiLU 激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)# 对输出通道 c2 批归一化
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):# 前向传播 ，数据流：输入 x → Conv2d → BN → 激活函数 → 输出
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):#融合前向传播：省略了 BN 层，用于推理加速
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):# 继承自父类 Conv。标准3x3卷积 + 并行1x1卷积->训练时：两条路相加-> 推理时：融合成一个3x3卷积.
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)# 调用父类的构造函数，让父类先完成它的初始化工作。父类执行完后，self.conv、self.bn、self.act 已经存在了。 然后再添加子类自己的 cv2
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv 添加的新分支功能

    def forward(self, x):# 两条路的输出相加，再统一过 BN 和激活函数。
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):# Conv2 在完成卷积融合之后使用的前向方法，只有主路self.conv（3x3卷积） 在工作，支路 self.cv2（1x1卷积）已经被删除（在 fuse_convs() 中通过 __delattr__ 移除）
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):# 把两条路合成一条路
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)# 创造一个全零的3*3卷积核
        i = [x // 2 for x in w.shape[2:]]# 找到中心点坐标
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()# 将 1×1 的权重放到 3×3 核的中心
        self.conv.weight.data += w # 将转换后的 1×1 核加到 3×3 核上
        self.__delattr__("cv2")#  删除 1×1 卷积分支
        self.forward = self.forward_fuse #之后前向只用主路


class LightConv(nn.Module):# 输入 x → 1x1 逐点卷积 (改变通道) → 深度卷积 (提取空间特征) → 输出
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv): # 逐深度卷积，每个输入通道单独用一个卷积核处理，通道之间在卷积过程中不混合信息。
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d): # 逐深度转置卷积，用于上采样操作。继承自 PyTorch 的 nn.ConvTranspose2d（转置卷积）
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):# 带批归一化和激活函数的转置卷积模块，是 Conv 类在转置卷积（上采样）中的对应版本。输入 → ConvTranspose2d → BN → 激活 → 输出
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation 默认使用 SiLU 激活函数

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):# 前向传播
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):# 融合前向传播，跳过 BN 层，用于推理加速
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):# 空间下采样技巧。将空间信息（宽、高）压缩，同时扩展通道数，下采样2倍但不丢失信息
    """将宽高信息集中到通道空间"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((
            x[..., ::2, ::2], # 左上角像素
            x[..., 1::2, ::2],# 右上角像素
            x[..., ::2, 1::2],# 左下角像素
            x[..., 1::2, 1::2]),# 右下角像素
            1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):# 幻影卷积。通过廉价线性变换（如深度卷积）从少量“内在特征”中“幻化”出更多特征图，避免昂贵卷积重复生成相似特征
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):# c1输入通道数，c2输出通道数
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        super().__init__()
        c_ = c2 // 2  # hidden channels这是GhostNet的核心参数。它决定用于“原始”特征图的通道数，这里设定为最终输出通道数 c2 的一半。另一半将由“廉价”操作生成。
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)# 标准卷积层，作用是将输入通道 c1 转换为 c_ 个通道，生成一小部分真正的、内在的特征图。这部分是主干，包含主要的语义信息
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)# 输入输出通道数不变。卷积核为5，步长为1，groups = 输入通道数，此时这个Conv就是一个在 c_ 个通道上独立进行的逐深度卷积DWConv。它的作用是通过一个非常廉价的线性变换（空间卷积），为 self.cv1 生成的每个内在特征图，生成一个“幽灵”（Ghost）特征图。

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)# 将原始的内在特征图 y 与通过廉价操作 self.cv2 生成的“幽灵”特征图在通道维度 (维度1) 上拼接起来。最终输出通道数为 c_ + c_ = c2。


class RepConv(nn.Module):# 重参数化卷积。
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1# 强制要求卷积核大小 k 必须等于 3，且填充 p 必须等于 1，否则程序报错并终止。
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None# 恒等连接分支（可选的BN层）。条件：bn=True 且输入输出通道相等且步长为1。满足则创建一个 BatchNorm2d，否则为 None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)#  3x3卷积分支
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)# 1x1卷积分支（通过padding调整尺寸，使其输出与3x3相同）

    def forward_fuse(self, x):# 推理融合后使用
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):# 训练时使用
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)# 若恒等分支存在则为 BN(x)，否则为 0
        return self.act(self.conv1(x) + self.conv2(x) + id_out) # 三个分支的输出逐元素相加，然后过激活函数。

    def get_equivalent_kernel_bias(self):# 这是整个模块的精髓。融合卷积。将三个分支（3×3、1×1、恒等）的卷积核与 BN 参数融合成一个等价的 3×3 卷积核和偏置。
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)# 3x3分支
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)# 1x1分支
        kernelid, biasid = self._fuse_bn_tensor(self.bn)# 恒等分支
        # 1×1 核需要填充成 3×3。 三个核相加，三个偏置相加。
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """将 1×1 卷积核（形状 [out_c, in_c, 1, 1]）上下左右各填充 1 行/列，变成 3×3，填充值为 0。"""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        # 它解决了一个工程问题：如何把"卷积层 + 它后面的 BN 层"合并成一个等价的卷积层。
        #在 RepConv 中，每个分支的结构都是：Conv2d → BatchNorm2d → (激活函数)
        #但 Conv2d 和 BatchNorm2d 是两个独立的层，推理时要分别计算：1、先做卷积.2、再做BN需要算均值、方差、γ、β）。这很慢
        #_fuse_bn_tensor 做的事情：把 Conv2d 的权重和 BN 的 γ、β、μ、σ 算在一起，输出一个新的 (kernel, bias)，这个新卷积层的计算结果 完全等价 于原来的 Conv2d + BN
        """通过融合神经网络的不同分支，生成适合用于卷积运算的卷积核和偏置项。"""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):# 如果是 Conv 模块
            kernel = branch.conv.weight # branch.conv.weight 是 卷积层的权重参数（即卷积核）。形状就是 [out_channels, in_channels, kernel_height, kernel_width]
            running_mean = branch.bn.running_mean# BN 层在训练过程中通过移动平均累积的输入数据的均值
            running_var = branch.bn.running_var#  BN 层累积的输入数据的方差
            gamma = branch.bn.weight# 缩放因子，BN 层的可学习参数，用于缩放归一化后的数据
            beta = branch.bn.bias# 平移因子，BN 层的可学习参数，用于平移归一化后的数据
            eps = branch.bn.eps# 小常数，防止除零错误的一个很小的正数
        elif isinstance(branch, nn.BatchNorm2d):# 如果是纯 BN 层（恒等分支）
            if not hasattr(self, "id_tensor"):# 创建单位卷积核（只执行一次）：检查是否已经创建过单位张量 id_tensor，避免重复创建。
                input_dim = self.c1 // self.g#  计算每个组的输入维度。在分组卷积中，每个组的输入通道数，例如：c1=64, g=4，则 input_dim=16
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)# 初始化全零的卷积核。形状为 [输出通道, 每组的输入通道, 3, 3]
                for i in range(self.c1):# 在卷积核的中心位置填 1
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)# 保存为单位张量，并移到正确的设备上
            kernel = self.id_tensor# 使用这个单位卷积核。这个单位卷积核的作用是：实现恒等映射（相当于一个"假"的卷积）
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()# 计算标准差
        t = (gamma / std).reshape(-1, 1, 1, 1)# 计算 BN 层对卷积核的缩放系数
        return kernel * t, beta - running_mean * gamma / std #融合并返回，返回两个值：新的卷积核 和 新的偏置

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class.将两个卷积层合并成一个单独的层，并从类中移除不再使用的属性"""
        if hasattr(self, "conv"):# 检查是否已经融合，如果已经存在 self.conv（融合后的卷积层），说明已经转换过了，直接返回。
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d( # 创建新的单卷积层
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,# 关键：启用偏置，因为融合后产生了偏置
        ).requires_grad_(False)# 冻结参数，推理时不需要梯度
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):# 通道注意力机制模块。核心思想是：让网络自己学会关注“哪些通道（特征图）更重要”，并自动增强重要特征、抑制不重要特征。
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1) # 全局平均池化，将每个通道的特征图压缩成 1 个值（全局信息）。输出形状从 (B, C, H, W) → (B, C, 1, 1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)# # 1x1卷积
        self.act = nn.Sigmoid()# 激活函数。将输出值压缩到 0~1 之间，作为每个通道的“注意力权重”

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x))) # 输入 → 全局池化 → 1x1卷积 → Sigmoid → 与原输入逐元素相乘
    # 增强有用特征：如果某个通道的“全局信息”经过卷积和 Sigmoid 后输出值接近 1，则该通道特征被保留。
    # 抑制无用特征：如果输出值接近 0，则该通道特征被削弱。
    # 自适应性：权重 (self.fc) 是可学习的，网络会根据任务目标自动调整

class SpatialAttention(nn.Module):# 是一个空间注意力模块，它与 ChannelAttention 是“兄弟模块”。ChannelAttention 关注“哪些通道重要”，而 SpatialAttention 关注“图片的哪些位置（像素/区域）重要”。
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"# 只支持 3 或 7 的卷积核大小
        padding = 3 if kernel_size == 7 else 1# 根据核大小计算填充（保持特征图尺寸不变）
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(
            self.cv1(
                torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)
            )
        )# 通道均值+最大值拼接 → 7x7卷积 → Sigmoid，生成空间权重图，让网络关注“哪里”有物体。


class CBAM(nn.Module):# 注意力模块，让网络既会“看哪里”（空间），也会“看什么”（通道），从而精准地关注重要特征。
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):# 张量拼接，将多个特征图在指定的维度上进行拼接。它本身就是对 PyTorch 原生 torch.cat 函数的一个轻量级封装，目的是为了方便地在网络模型（比如 YOLO 的颈部结构）中作为一个独立的层来使用。
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


if __name__ == "__main__":
    # 创建一个 Conv 模块
    conv = Conv(c1=3, c2=16, k=3, s=2)

    # 创建一个模拟输入 (batch=1, channels=3, height=640, width=640)
    x = torch.randn(1, 3, 640, 640)

    # 前向传播
    y = conv(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")  # 因为 stride=2，尺寸减半