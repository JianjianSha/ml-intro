# Fisher准则

## 介绍

以二分类为例，训练集为 $D=\{(\boldsymbol \phi_n,t_n):n=1,2,...,N\}, \ t_n \in \{C_1, C_2\}$，通过线性映射投影到一维空间 $y$，

$$y=\mathbf w^{\top} \boldsymbol \phi$$(Fisher1)


Fisher 准则定义如下区分性度量：

$$L(\mathbf w)=\frac {(m_1-m_2)^2} {\sigma_1^2+\sigma_2^2}$$ (Fisher2)

上式中 $m_1, m_2$ 是 $C_1, C_2$ 的样本点在映射空间（这里是一维）的均值，$\sigma_1^2, \sigma_2^2$ 是映射空间的离散度，{eq}`Fisher2` 式表示类间距离越大，类内离散程度越小，则这两类区分性越强。

映射空间 $C_1$ 样本集合记为 $D_1$，数量记为 $N_1=|D_1|$，均值为

$$m_1=\frac 1 {N_1} \sum_{\boldsymbol \phi_i \in D_1} \mathbf w^{\top} \boldsymbol \phi_i=\mathbf w^{\top} \boldsymbol \mu_1$$

其中原始空间 $C_1$ 样本均值为

$$\boldsymbol \mu_1 = \frac 1 {N_1} \sum_{\boldsymbol \phi_i \in D_1} \boldsymbol \phi_i$$

于是

$$(m_2-m_1)^2=[\mathbf w^{\top}(\boldsymbol {\mu_2-\mu_1})]^2=\mathbf w^{\top} \mathbf S_{B} \mathbf w$$

其中

$$\mathbf S_{B}=(\boldsymbol {\mu_2-\mu_1})(\boldsymbol {\mu_2-\mu_1})^{\top}$$ (Fisher3)

另一方面，映射空间 $C_1$ 样本离散程度为

$$\sigma_1^2=\sum_{\boldsymbol \phi_i \in D_1} (y_i - m_1)^2=\sum_{\boldsymbol \phi_i \in D_1} (\mathbf w^{\top}(\boldsymbol {\phi_i - \mu_1}))^2 =\sum_{\boldsymbol \phi_i \in D_1} \mathbf w^{\top} (\boldsymbol {\phi_i-\mu_1})(\boldsymbol {\phi_i-\mu_1})^{\top} \mathbf w$$

于是

$$\sigma_1^2+\sigma_2^2=\mathbf w^{\top} \mathbf S_W \mathbf w$$

其中

$$\mathbf S_W=\sum_{\boldsymbol \phi_i \in D_1} (\boldsymbol {\phi_i-\mu_1})(\boldsymbol {\phi_i-\mu_1})^{\top} +  \sum_{\boldsymbol \phi_i \in D_2} (\boldsymbol {\phi_i-\mu_2})(\boldsymbol {\phi_i-\mu_2})^{\top} $$ (Fisher4)

```{note}
这里的样本离散程度没有取平均，所以这里的类内方差不是均方差。这是因为无论样本属于 $C_1$ 还是 $C_2$，它对类内离散程度的贡献应该是相同的权值，或者说跟类样本数量相关。

举个简单例子，假设 $N_1=10^4 N_2$， $D_1$ 和 $D_2$ 中的样本分为以 $\boldsymbol \phi^{(1)}$ 和 $\boldsymbol \phi^{(2)}$ 为圆心，半径为 $r_1$ 和 $r_2$ 的圆上均匀分布，且有 $r_1=10 r_2$，那么如果使用方差，$\sigma_1^2=\frac {r_1^2} {N_1} = \frac {100 r_2^2} {10^4 N_2}=\frac {\sigma_2^2} {100}$，这表明 $D_1$ 远远没有 $D_2$ 分散，然而 $D_1$ 分布更大的圆上，应该更分散才对，所以两类样本应该设置相同的离散贡献权值，相对 $\mathbf w$ 而言可视为常数，由于对求解最优 $\mathbf w$ 没有影响，故可直接设置为 1。
```

综上 {eq}`Fisher2` 式为

$$L(\mathbf w)=\frac {\mathbf w^{\top} \mathbf S_{B} \mathbf w} {\mathbf w^{\top} \mathbf S_W \mathbf w}$$ (Fisher5)

对 {eq}`Fisher5` 式中的 $\mathbf w$ 进行优化，取梯度并令其等于 0，得到

$$\nabla_{\mathbf w} L(\mathbf w)=\frac {\mathbf S_B \mathbf w(\mathbf w^{\top} \mathbf S_W \mathbf w)-(\mathbf w^{\top} \mathbf S_{B} \mathbf w)\mathbf S_W \mathbf w} {(\mathbf w^{\top} \mathbf S_W \mathbf w)^2}=0$$

整理得

$$\mathbf S_B \mathbf w(\mathbf w^{\top} \mathbf S_W \mathbf w)=(\mathbf w^{\top} \mathbf S_{B} \mathbf w)\mathbf S_W \mathbf w$$

注意到上式中括号部分为标量，且

$$\begin{aligned}\mathbf S_B \mathbf w &=\ (\boldsymbol {\mu_2-\mu_1})\{(\boldsymbol {\mu_2-\mu_1})^{\top} \mathbf w\}
\\ & \propto \ \boldsymbol {\mu_2-\mu_1}
\end{aligned}$$

（注：上式右侧大括号部分为标量）

所以

$$\mathbf S_W \mathbf w \propto \ \boldsymbol {\mu_2-\mu_1}$$

如果 $\mathbf S_W$ 满秩，则有

$$\mathbf w \propto \ \mathbf S_W^{-1} (\boldsymbol {\mu_2-\mu_1})$$ (Fisher6)

{eq}`Fisher6` 式表示最具有区分性的 $\mathbf w$ 应与这两类样本中心的连线 $\boldsymbol {\mu_2-\mu_1}$ 大致同向，但应基于类内方差矩阵 $\mathbf S_W$ 进行调整。

```{note}
在我们的直观理解上，判别超平面表示为
$y=\mathbf w \boldsymbol \phi$
，例如 $y \ge -w_0$ 表示 $C_1$ 类，$y<-w_0$ 表示 $C_2$ 类，显然判别超平面大致垂直于这两类样本中心连线，而超平面的法线方向就是 $\mathbf w$，所以 $\mathbf w$ 大致与中心连线 $\boldsymbol {\mu_2-\mu_1}$ 同向。
```

## 分析

为什么要用这个准则？

假设 $C_1$ 样本点的目标值 $t=N/N_1$，$C_2$ 样本点的目标值为 $t=-N/N_2$
```{note}
这种目标值的设定，使得整个数据集的样本在映射空间的中心为 0，这显然是合理的，两个类别的样本中心则位于 0 的两侧。但是在映射空间中 0 位置处作为分割位置不一定最优，还需要增加一个偏差 $w_0$
```

线性拟合函数为

$$y=\mathbf w^{\top} \mathbf x + w_0$$

拟合误差为

$$E(\mathbf w)=\frac 1 2 \sum_{i=1}^N (\mathbf w^{\top} \boldsymbol \phi_i+w_0-t_i)^2$$

根据误差对参数 $\mathbf w$ 的梯度为零对参数$\mathbf w$ 和 $w_0$ 进行优化，有

$$\sum_{i=1}^N (\mathbf w^{\top} \boldsymbol \phi_i + w_0 -t_i)=0$$ (Fisher7)

$$
\sum_{i=1}^N (\mathbf w^{\top} \boldsymbol \phi_i + w_0 -t_i)\boldsymbol \phi_i=0$$ (Fisher8)

由于 

$$\sum_{i=1}^N t_i = N_1 \frac N {N_1} - N_2 \frac N {N_2}=0$$

结合 {eq}`Fisher7` 得

$$w_0=-\mathbf w^{\top} \boldsymbol \mu=-\mathbf w^{\top} \frac 1 N (N_1 \boldsymbol \mu_1 + N_2 \boldsymbol \mu_2)$$

整理 {eq}`Fisher8` 式并代入上式，

$$\begin{aligned} \sum_{i=1}^N (\mathbf w^{\top} \boldsymbol \phi_i + w_0 -t_i)\boldsymbol \phi_i&=\sum_{i=1}^N \boldsymbol \phi_i(\boldsymbol \phi_i^{\top} \mathbf w + w_0 -t_i)
\\ &=\sum_{i=1}^N \boldsymbol \phi_i (\boldsymbol \phi_i^{\top} \mathbf w+w_0) - N(\boldsymbol \mu_1-\boldsymbol \mu_2)
\\ &= \sum_{i=1}^N \boldsymbol \phi_i \boldsymbol \phi_i^{\top} \mathbf w - N \boldsymbol \mu \boldsymbol \mu^{\top} \mathbf w - N(\boldsymbol \mu_1-\boldsymbol \mu_2)
\\ &= \left(\sum_{i=1}^N \boldsymbol \phi_i \boldsymbol \phi_i^{\top}  -\frac 1 N (N_1 \boldsymbol \mu_1 + N_2 \boldsymbol \mu_2) (N_1 \boldsymbol \mu_1 + N_2 \boldsymbol \mu_2) \right)\mathbf w - N(\boldsymbol \mu_1-\boldsymbol \mu_2)
\end{aligned}$$

另外根据 {eq}`Fisher3` 和 {eq}`Fisher4` 有

$$\begin{aligned}
\left(\mathbf S_W + \frac {N_1N_2} N\mathbf S_B \right)&=\sum_{i=1}^N \boldsymbol {\phi_i \phi_i^{\top}} -N_1 \boldsymbol {\mu_1 \mu_1^{\top}} -N_2 \boldsymbol {\mu_2 \mu_2^{\top}} + \frac 1 N N_1N_2(\boldsymbol {\mu_2- \mu_1}) (\boldsymbol {\mu_2- \mu_1}) ^{\top}
\\ &=\sum_{i=1}^N \boldsymbol {\phi_i \phi_i^{\top}} - \frac 1 N [(N_1+N_2)N_1 \boldsymbol {\mu_1 \mu_1^{\top}}+(N_1+N_2)N_2\boldsymbol {\mu_2 \mu_2^{\top}}-N_1N_2(\boldsymbol {\mu_2- \mu_1}) (\boldsymbol {\mu_2- \mu_1}) ^{\top}]
\\ &= \sum_{i=1}^N \boldsymbol {\phi_i \phi_i^{\top}} -\frac 1 N (N_1 \boldsymbol \mu_1 + N_2 \boldsymbol \mu_2) (N_1 \boldsymbol \mu_1 + N_2 \boldsymbol \mu_2)
\end{aligned}$$

结合以上两式以及 {eq}`Fisher8` 有

$$\left(\mathbf S_W + \frac {N_1N_2} N\mathbf S_B \right)\mathbf w = N(\boldsymbol \mu_1-\boldsymbol \mu_2)$$

上面分析到 $\mathbf S_B \mathbf w \propto \ \boldsymbol {\mu_2-\mu_1}$，所以当 $\mathbf S_W $ 满秩时，{eq}`Fisher6` 成立。

## 结论

依 Fisher 准则得到的映射函数和基于线性拟合得到的映射函数是等价的，但是线性拟合等价于一个线性回归模型，其基本假设 $p(t|\boldsymbol \phi)$ 是高斯分布，故 Fisher 方法事实上架设了任务中类别标记符合高斯分布，这对分类任务而言显示不合理，容易受到离群点的影响。


