# 机器学习概述

## 1.8 机器学习基础

### 1.8.2 参数过拟合、交叉验证与正则化
防止过拟合的方法：
1. 在训练过程中用测试集来检测模型性能，当模型性能在测试集上开始下降时，认为出现过拟合，停止训练
2. 在目标函数中加入正则项，控制参数的取值范围

### 1.8.3 结构过拟合与模型选择

设测试数据 $\mathbf x$ 的真实目标值为 $h(\mathbf x)$，观察到的目标值 $t$，模型预测值 $y(\mathbf x)$，记 $\mathbf x$ 和 $t$ 的联合分布为 $p(\mathbf x, t)$，那么目标值与预测值之间的误差（方差）为，
$$\begin{aligned}
&\int\int(y(\mathbf x)-t)^2 p(\mathbf x, t)d\mathbf x dt
\\ =& \int\int(y(\mathbf x)-h(\mathbf x)+h(\mathbf x)-t)^2 p(\mathbf x, t)d\mathbf x dt 
\\ =& \int\int (y(\mathbf x)-h(\mathbf x))^2p(\mathbf x, t)d\mathbf x dt +\int\int (h(\mathbf x)-t)^2p(\mathbf x, t)d\mathbf x dt
\\=&\int (y(\mathbf x)-h(\mathbf x))^2p(\mathbf x)d\mathbf x + \int\int (h(\mathbf x)-t)^2p(\mathbf x, t)d\mathbf x dt
\end{aligned} \qquad (1)$$

我们假设目标值 $t$ 符合以 $h(\mathbf x)$ 为中心的正态分布，根据

$$p(\mathbf x, t)=p(\mathbf x)p(t|\mathbf x)$$

于是，上式中第二个等式中交叉项
$$\begin{aligned}
&\int\int 2(y(\mathbf x)-h(\mathbf x))(h(\mathbf x)-t)p(\mathbf x, t)d\mathbf x dt
\\=&\int2(y(\mathbf x)-h(\mathbf x)) \left(\int(h(\mathbf x)-t)p(t|\mathbf x) dt \right) p(\mathbf x) d\mathbf x
\\=&\int2(y(\mathbf x)-h(\mathbf x)) \left(h(\mathbf x)-\int t \cdot p(t|\mathbf x) dt \right) p(\mathbf x) d\mathbf x
\\=&\int2(y(\mathbf x)-h(\mathbf x)) \left(h(\mathbf x)-h(\mathbf x) \right) p(\mathbf x) d\mathbf x
\\=& 0
\end{aligned} \qquad (2)$$

（2）式倒数第二个等式成立是因为随机变量 $t$ 符合正态分布，它的期望值等于中心值 $h(\mathbf x)$，于是（1）式的第二个等式成立。

根据（1）式可知，误差可分解为预测误差和噪声误差，前者与所选模型有关，后者与数据噪声有关。

__预测误差__

预测函数是通过某一数据集 $D$ 训练出来的故可写为 $y(\mathbf x;D)$，由于不同的数据集导致训练模型也会不同，考虑模型预测的期望值 $\mathbb E_D[y(\mathbf x; D)]$，那么预测误差如下，
$$\begin{aligned}
(y(\mathbf x;D)-h(\mathbf x))^2&=\{y(\mathbf x;D)-\mathbb E_D[y(\mathbf x; D)]+\mathbb E_D[y(\mathbf x; D)]-h(\mathbf x)\}^2
\\&=\{y(\mathbf x;D)-\mathbb E_D[y(\mathbf x; D)]\}^2+\{\mathbb E_D[y(\mathbf x; D)]-h(\mathbf x)\}^2
\\ & \quad + 2\{y(\mathbf x;D)-\mathbb E_D[y(\mathbf x; D)]\}\{\mathbb E_D[y(\mathbf x; D)]-h(\mathbf x)\}
\end{aligned} \quad(3)$$

对 $D$ 取期望，不难知道 $\mathbb E_D[y(\mathbf x; D)]-h(\mathbf x)$ 对 $D$ 而言是常数，且有
$$\mathbb E_D\{y(\mathbf x;D)-\mathbb E_D[y(\mathbf x; D)]\}=\mathbb E_D[y(\mathbf x; D)]-\mathbb E_D[y(\mathbf x; D)]=0$$

于是（3）式两边对 $D$ 取期望，得到
$$\mathbb E_D[y(\mathbf x; D)-h(\mathbf x)]^2=\{\mathbb E_D[y(\mathbf x; D)]-h(\mathbf x)\}^2+\mathbb E_D\{y(\mathbf x;D)-\mathbb E_D[y(\mathbf x; D)]\}^2 \quad(4)$$

（4）式右侧第一项是预测期望与真实值之间的差距，这个误差来源于预测模型和真实模型之间的偏差，第二项是不同训练集得到的模型产生的预测波动（方差），反映了模型对训练数据的敏感度。