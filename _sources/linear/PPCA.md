# PPCA

运行概率知识分析理解和解释 PCA。考虑如下线性模型（因为我们这里只讨论线性模型），

$$\mathbf t=\mathbf b+\mathbf {W x} + \boldsymbol \epsilon$$

其中 $\mathbf b$ 是固定偏移量，$\mathbf t \in R^D$ 是观察变量，$\mathbf W \in R^{R\times M}$ 是参数，$\mathbf x \in R^M$ 是符合正态分布的隐变量

$$p(\mathbf x)=N(\mathbf x| \mathbf 0, \mathbf I)$$

噪声 $\boldsymbol \epsilon$ 是高斯变量

$$p(\boldsymbol \epsilon)=N(\mathbf x|\mathbf 0, \sigma^2\mathbf I)$$

这里高斯随机噪声的假设很好理解。在没有其他关于 $\mathbf x$ 的已知知识时，将其假设为标准高斯随机变量是比较好的，满足了各向同性。


数据 $\mathbf t$ 的生成模型为：基于先验概率 $p(\mathbf x)$ （这个先验概率已知）生成隐变量的采样点（但是我们不可观察其具体值），通过线性变换得到 $\mathbf {W x}$，然后加入高斯噪声 $\boldsymbol \epsilon$，最后加入位移 $\mathbf b$。

现在基于最大似然估计法则估计 PPCA 的模型参数 $\mathbf W, \mathbf b, \sigma^2$。

观测变量 $\mathbf t$ 的期望和协方差为

$$\mathbb E[\mathbf t]=\mathbb E[\mathbf b+\mathbf {W x} + \boldsymbol \epsilon]=\mathbf b$$

$$\begin{aligned} \mathbf C&=\mathbb E \{(\mathbf b+\mathbf {W x} + \boldsymbol \epsilon-\mathbb E[\mathbf t])(\mathbf {W x} + \boldsymbol \epsilon-\mathbb E[\mathbf t])^{\top}\}
\\&=\mathbb E [(\mathbf {W x} + \boldsymbol \epsilon)(\mathbf {W x} + \boldsymbol \epsilon)^{\top}]
\\&=\mathbb E [\mathbf {Wxx}^{\top} \mathbf W^{\top}+ \boldsymbol {\epsilon \epsilon}^{\top}+\mathbf {Wx} \boldsymbol {\epsilon + \epsilon} (\mathbf {Wx})^{\top}]
\\ &= \mathbb E [\mathbf {Wxx}^{\top} \mathbf W^{\top}+ \boldsymbol {\epsilon \epsilon}^{\top}] + \mathbb E[\mathbf {Wx}] \mathbb E [\boldsymbol {\epsilon}] + \mathbb E[ \boldsymbol \epsilon]\mathbb E [ (\mathbf {Wx})^{\top}]
\\ &=\mathbb E [\mathbf {Wxx}^{\top} \mathbf W^{\top}+ \boldsymbol {\epsilon \epsilon}^{\top}]
\\ &= \mathbf {WW}^{\top} + \sigma^2 \mathbf I
\end{aligned}$$ (ppca1)

上式推导过程利用了 $\mathbf x$ 与噪声 $\boldsymbol \epsilon$ 相互独立，

且 $\mathbb E[\mathbf {xx}^{\top}]=\mathbf C(\mathbf x)+\mathbb E^2[\mathbf x]=I, \ \mathbb E[\boldsymbol {\epsilon \epsilon}^{\top}]=\mathbf C(\boldsymbol \epsilon)+\mathbb E^2[\boldsymbol \epsilon]=\sigma^2I$

于是

$$p(\mathbf t)=N(\mathbf t|\mathbf b, \mathbf C)=N(\mathbf t|\mathbf b,\mathbf {WW}^{\top} +\sigma^2 \mathbf I)$$

基于最大似然法则求模型参数，取对数似然函数

$$\begin{aligned} L &=\log p(D|\mathbf b, \mathbf W, \sigma^2)
\\ &=\sum_{n=1}^N \log p(\mathbf t_n | \mathbf b, \mathbf W, \sigma^2)
\\ &=-\frac {ND} 2 \log(2\pi)-\frac N 2 \log|\mathbf C|-\frac 1 2 \sum_{n=1}^N (\mathbf t_n- \mathbf b)^{\top}\mathbf C^{-1}(\mathbf t_n-\mathbf b)
\\ &=\frac N 2 \{D \log(2\pi)+\log|\mathbf C| +tr(\mathbf C^{-1}\mathbf S)\}
\end{aligned}$$

求对 $\mathbf b$ 的梯度并令其为零，

$$\nabla_{\mathbf b}L=\sum_{n=1}^N \mathbf C^{-1}(\mathbf t_n - \mathbf b)=\mathbf 0$$

两边同时左乘 $\mathbf C$，得到

$$\mathbf b_{ML}=\frac 1 N \sum_{n=1}^N \mathbf t_n$$ (ppca2)

对 $\mathbf C$ 求梯度，

$$\nabla_{\mathbf C} L = -\frac N 2 \mathbf C^{-\top} + \frac 1 2 \sum_{n=1}^N \mathbf C^{-\top}(\mathbf t_n-\mathbf b)(\mathbf t_n-\mathbf b)^{\top}\mathbf C^{-\top}=\frac N 2 \mathbf C^{-\top} \mathbf S \mathbf C^{-\top}-\frac N 2 \mathbf C^{-\top}$$

对 $\mathbf W$ 求梯度，

$$d\mathbf C=dW \cdot W^{\top} + W \cdot (dW)^{\top}=d \mathbf C^{\top}$$

于是

$$\begin{aligned} d L &=tr\left({\frac {\partial L} {\partial \mathbf C}}^{\top} d \mathbf C\right)
\\& =tr\left({\frac {\partial L} {\partial \mathbf C}}^{\top} (dW \cdot W^{\top} + W\cdot (dW)^{\top})\right)
\\ &=tr\left({\frac {\partial L} {\partial \mathbf C}}^{\top} dW \cdot W^{\top}\right) +tr\left( {\frac {\partial L} {\partial \mathbf C}}^{\top}W \cdot (dW)^{\top}\right)
\\ &=tr\left(W^{\top}  {\frac {\partial L} {\partial \mathbf C}}^{\top} dW\right)+tr\left( { (dW)^{\top} \frac {\partial L} {\partial \mathbf C}}^{\top}W \right)
\\ &=tr\left(W^{\top}  {\frac {\partial L} {\partial \mathbf C^{\top}}} dW\right)+tr\left( { W^{\top} \frac {\partial L} {\partial \mathbf C}} dW \right)
\\ &= tr\left(W^{\top}  {\frac {\partial L} {\partial \mathbf C}} dW\right)+tr\left( { W^{\top} \frac {\partial L} {\partial \mathbf C}} dW \right)
\\ &= tr\left(2({\frac {\partial L} {\partial \mathbf C}}^{\top}W)^{\top} dW\right)
\end{aligned}$$

所以

$$\nabla_{\mathbf W}L={\frac {\partial L} {\partial \mathbf C}}^{\top}W=N(\mathbf C^{-1} \mathbf S \mathbf C^{-1}-\frac N 2 \mathbf C^{-1})\mathbf W=\mathbf 0$$

两边左乘 $\mathbf C^{-1}$ 得

$$\mathbf {SC}^{-1} \mathbf W = \mathbf W$$ (ppca3)

{eq}`ppca3` 有三种可能性，我们分别讨论。

1) $\mathbf W=\mathbf 0$

此时 $\mathbf t$ 变成了以 $\mathbf b$ 为中心，$\mathbf I$ 为协方差的多维高斯分布，与输入 $\mathbf x$ 无关，显然不是我们想要的。

2) $\mathbf C=\mathbf S$

根据 {eq}`ppca1`，有

$$\mathbf {WW}^{\top}=\mathbf S - \sigma^2 \mathbf I$$

易知其解为 
$$\mathbf W=\mathbf U(\Lambda-\sigma^2\mathbf I)^{1/2}\mathbf R$$

其中 $\mathbf U$ 为 $\mathbf S$ 的特征向量组成的方阵，$\Lambda$ 为特征值组成的对角阵，$\mathbf R$ 为任意的正交矩阵。


3) $\mathbf W \neq \mathbf 0, \ \mathbf C \neq \mathbf S$

将参数矩阵 $\mathbf W$ 进行奇异值分解

$$\mathbf W = \mathbf {ULV}^{\top}$$

其中 $\mathbf U$ 和 $\mathbf V$ 维度分别为 $D \times D$ 和 $M \times M$ 的正交矩阵（酉矩阵），满足 $\mathbf U^{\top} \mathbf U=\mathbf {UU}^{\top}=\mathbf I_D$ 和 $\mathbf V^{\top} \mathbf V=\mathbf {VV}^{\top}=\mathbf I_M$， $\mathbf L$ 维度为 $D \times M$， 主对角线上元素为对应奇异值，其余元素为 0，由奇异值组成，根据 {eq}`ppca1` 有

$$\mathbf C=\mathbf {ULV}^{\top} \mathbf {VL}^{\top} \mathbf U^{\top}+\sigma^2 \mathbf I=\mathbf U \mathbf {LL}^{\top}\mathbf U^{\top}+\sigma^2 \mathbf U\mathbf U^{\top}=\mathbf U (\mathbf {LL}^{\top}+\sigma^2\mathbf I)\mathbf U^{\top}$$ (ppca4)

易得 $\mathbf C^{-1}=\mathbf U (\mathbf {LL}^{\top}+ \sigma^2\mathbf I)^{-1}\mathbf U^{\top}$

将上面 $\mathbf W$ 和 $\mathbf C^{-1}$ 代入 {eq}`ppca3` 得

$$\mathbf {SU}(\mathbf {LL}^{\top}+\sigma^2\mathbf I)^{-1}\mathbf U^{\top} \mathbf {ULV}^{\top}=\mathbf {ULV}^{\top}$$

化简得

$$\mathbf {SU}(\mathbf {LL}^{\top}+\sigma^2\mathbf I)^{-1} \mathbf L=\mathbf {UL}$$ (ppca5)

a. 如果 $l_j \neq 0$，{eq}`ppca5` 意味着 
$$\mathbf {S u}_j (l_j^2+\sigma^2)^{-1} l_j = \mathbf u_j l_j \Rightarrow \mathbf {S u}_j=(l_j^2+\sigma^2)\mathbf u_j$$

这表示 $\mathbf u_j$ 是 $\mathbf S$ 的特征向量，对应特征值为 $\lambda_j=l_j^2+\sigma^2$，故奇异值为

$$l_j=(\lambda_j-\sigma^2)^{1/2}$$

b. 如果 $l_j = 0$，那么 $\mathbf u_j$ 可以是任意的。


于是，$\mathbf W$ 可以写成以下形式，

$$\mathbf W = \mathbf {ULR}$$ 

其中，$\mathbf U$ 是由 $\mathbf S$ 的特征向量作为列向量组成的矩阵，$\mathbf R$ 为任意 $M \times M$ 的正交矩阵，$\mathbf L$ 为 $D \times M$ 的矩阵，主对角线上元素为 $l_j=(\lambda_j-\sigma^2)^{1/2}$，其余元素为 0。我们可以取 $\mathbf S$ 的前 M 个最大特征值来近似 $\mathbf W$，

$$\mathbf W_{ML} \approx  \mathbf U_M (\Lambda -\sigma^2 \mathbf I)^{1/2} \mathbf R$$ (ppca6)

此时 $\mathbf U_M$ 由 $\mathbf S$ 前 M 个最大特征值对应的特征向量组成，$\Lambda$ 是对应的最大 M 个特征值组成的对角阵，$\mathbf R$ 是任意正交矩阵。

根据 {eq}`ppca4` 和行列式的性质可知，

$$|\mathbf C| = \prod_j (l_j^2+\sigma^2)=\prod_j \lambda_j$$


$\mathbf S$ 根据特征值分解有 $\mathbf S = \mathbf U \Lambda \mathbf U^{\top}$，于是

$$\mathbf C^{-1} \mathbf S = \mathbf U (\mathbf {LL}^{\top}+ \sigma^2\mathbf I)^{-1}\mathbf U^{\top} \mathbf U \Lambda \mathbf U^{\top}=\mathbf U (\mathbf {LL}^{\top}+ \sigma^2\mathbf I)^{-1}\Lambda \mathbf U^{\top}$$



根据 trace 的性质 $tr(A)=\sum_i \lambda_i$，其中 $\lambda_i = eig(A)$ 可知

$$tr(\mathbf C^{-1} \mathbf S)=\sum_j (l_j^2 + \sigma^2)^{-1} \lambda_j$$

由于对 $\forall j=1,\cdots, M$，有 $l_j \neq 0$； 对 $\forall j > M$ 有 $l_j=0$， 所以有

$$\log |\mathbf C|=\sum_j \log \lambda_j = \sum_{j=1}^M \log\lambda_j+\sum_{j=M+1}^D \log \sigma^2=\sum_{j=1}^M \log\lambda_j+(D-M) \log \sigma^2$$

$$tr(\mathbf C^{-1} \mathbf S)=\sum_{j=1}^M (l_j^2 + \sigma^2)^{-1} \lambda_j + \sum_{j=M+1}^D \sigma^{-2} \lambda_j=M+\sigma^{-2}\sum_{j=M+1}^D \lambda_j$$

根据上面两式可知对数似然为

$$L=-\frac N 2 \{D \log(2\pi)+\sum_{j=1}^M \log\lambda_j+(D-M) \log \sigma^2+M +\sigma^{-2}\sum_{j=M+1}^D \lambda_j\}$$

对 $\sigma^2$ 取梯度，令其为0，得

$$\nabla_{\sigma^2}L=(D-M)/(\sigma^2) -\sum_{j=M+1}^D \lambda_j/(\sigma^2)^2=0$$

解得

$$\sigma_{ML}^2=\frac 1 {D-M}\sum_{j=M+1}^D \lambda_j$$ (ppca7)



## 后验概率

从观察变量 $\mathbf t$ 推理出隐变量 $\mathbf x$，需要求后验概率 $p(\mathbf {x|t})$，由于先验概率 $p(\mathbf x)$ 和条件概率 $p(\mathbf {t|x})$ 都是高斯分布，故后验概率也是高斯分布，

$$p(\mathbf {x|t})=\frac {p(\mathbf x) p(\mathbf {t|x})} {p(\mathbf t)}$$

归一化因子我们暂且忽略，仅看 $\exp$ 那一项，于是上式可写为

$$\begin{aligned}p(\mathbf {x|t}) &\propto \exp \{-\frac 1 2[\mathbf x^{\top} \mathbf x + \frac 1 {\sigma^2}(\mathbf {t- b - Wx})^{\top}(\mathbf {t- b - Wx})-(\mathbf {t-b})^{\top}\mathbf C^{-1}(\mathbf {t-b})]\}
\\& \propto \exp \{-\frac 1 {2\sigma^2} [\mathbf x^{\top}(\mathbf W^{\top}\mathbf W+\sigma^2 \mathbf I)\mathbf x-(\mathbf {t-b})^{\top}\mathbf {Wx}- \mathbf x^{\top}\mathbf W^{\top}(\mathbf {t-b})]\}
\\ & \propto \exp\{-\frac 1 2 [\mathbf x-\mathbf M^{-1}\mathbf W^{\top}(\mathbf {t-b})]^{\top} \frac {\mathbf M}{\sigma^2} [\mathbf x-\mathbf M^{-1}\mathbf W^{\top}(\mathbf {t-b})]\}
\\ &=N(\mathbf x|\mathbf M^{-1}\mathbf W^{\top}(\mathbf {t-b}), \sigma^2\mathbf M^{-1})
\end{aligned}$$

其中 $\mathbf M=\mathbf W^{\top}\mathbf W+\sigma^2 \mathbf I$，且由于 $\mathbf M^{\top}=\mathbf M$，故 $(\mathbf M^{-1})^{\top}=(\mathbf M^{\top})^{-1}=\mathbf M^{-1}$。

上式为后验概率，对给定一个观测变量 $\mathbf t$，基于最大后验（MAP）准则确定一个最能描述 $\mathbf t$ 的隐变量 $\mathbf x$，由于该后验概率是高斯分布，MAP 估计等同于均值，于是有

$$\mathbf x_{MAP}|\mathbf t=\mathbf M^{-1}\mathbf W^{\top}(\mathbf {t-b})=(\mathbf W^{\top}\mathbf W+\sigma^2 \mathbf I)^{-1}\mathbf W^{\top}(\mathbf {t-b})$$












