# 线性模型

线性模型具有形式

$$\mathbf y = \mathbf {W x}$$

\## 多项式拟合

先考虑一维情况，

$$y=\sum_{j=0}^{M-1} w_j x^j=\mathbf w^{\top} \mathbf x$$

其中 $\mathbf x=[x_0, x_1, \dots , x_{M-1}]^{\top}$。

记目标值为 $t$，那么对于数量为 $N$ 的训练集 $D$，损失采用平方误差如下

$$L(\mathbf w)=\frac 1 2 \sum_{n=1}^N (y_n - t_n)^2=\frac 1 2 \sum_{n=1}^N (\mathbf w^{\top} \mathbf x_n - t_n)^2$$

计算梯度

$$\nabla_{\mathbf w}L = \sum_{n=1}^N(\mathbf w^{\top} \mathbf x_n - t_n) \mathbf x_n=\mathbf 0
\\ \Rightarrow \sum_{n=1}^N \mathbf x_n \mathbf x_n^{\top} \mathbf w = \sum_{n=1}^N t_n \mathbf x_n$$ (poly1)

记 

$$\mathbf t_{N \times 1} = [t_1, t_2, \dots, t_N]^{\top}$$

$$\mathbf X_{N \times M} = [\mathbf x_1^{\top}, \mathbf x_2^{\top}, \dots, \mathbf x_N^{\top}]^{\top}$$

那么 {eq}`poly1` 可写为

$$\mathbf {X^{\top} X w=X^{\top} t}$$

于是最优参数为

$$\mathbf {w = (X^{\top} X)^{-1} X^{\top} t}$$


\## 非线性问题的线性拟合
某些非线性问题也可以转化为线性拟合问题，通常将输入值经过某种变换

$$\boldsymbol \phi=[\phi_0(\mathbf x), \phi_1(\mathbf x), \cdots, \phi_{M-1}(\mathbf x)]^{\top}$$

然后将 $\boldsymbol \phi$ 替换上面的 $\mathbf x$。

以上是对回归问题的建模分析，我们还需要讨论分类问题。




