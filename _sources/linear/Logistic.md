# Logistic 回归

Logistic 回归是在线性函数的基础之上再套用一个非线性函数 

$$\sigma(x)=\frac 1 {1+ e^{-x}}$$

于是分类函数（这里以二分类为例）为

$$y=p(t|\mathbf x; \mathbf w)=\sigma(\mathbf {w^{\top} x})$$

目标函数为负对数似然函数，

$$L(\mathbf w)=-\log p(D;\mathbf w)=-\sum_{i=1}^N[t_i \log y_i+(1-t_i)\log(1-y_i)]$$ (Logis1)

{eq}`Logis1` 即损失函数，也成为交叉熵函数，其梯度为

$$\nabla_{\mathbf w}L=-\sum_{i=1}^N (\frac {t_i} {y_i} y_i' - \frac {1-t_i} {1-y_i} y_i')$$

其中

$$y'=\sigma'=\sigma(1-\sigma)$$

代入得

$$\nabla_{\mathbf w}L=-\sum_{i=1}^N \left[\frac {t_i} {y_i} y_i (1-y_i)\mathbf x - \frac {1-t_i} {1-y_i} y_i (1-y_i)\mathbf x\right]=\sum_{i=1}^N (y_i-t_i)\mathbf x$$ (Logis2)

根据上式采用梯度下降法进行数值求解。

## Softmax 回归

Softmax 回归适用于多分类问题。**预测** 目标值 $\mathbf t$ 是一个 one-hot 向量，故模型的输出 $\mathbf y$ 也是一个向量，模型参数为矩阵 $\mathbf W$，记输入向量 $\mathbf x$ 维度为 $M$，分类数量为 $C$，那么 $\mathbf W$ 维度为 $M\times C$，可写为 $\mathbf W=[\mathbf w_1, ..., \mathbf w_C]$

分类函数为

$$y_c=p(t_c=1|\mathbf x)=\frac {\exp (\mathbf w_c^{\top} \mathbf x)}{\sum_{j=1}^C \exp (\mathbf w_j^{\top} \mathbf x)}$$

注： $y_c$ 表示分类属于 c 的概率，即预测目标值 $t_c=1$ 的概率。

记线性变换的结果为 

$$a_c = \mathbf w_c^{\top} \mathbf x$$

梯度为

$$\begin{aligned}\frac {dy_c}{d a_i}&=\frac {\mathbb I(i=c) \exp(a_c) (\sum_{j=1}^C \exp(a_j))- \exp(a_c) \exp(a_i)} {(\sum_{j=1}^C \exp(a_j))^2}
\\ &= y_c \frac {\mathbb I(i=c) \sum_{j=1}^C \exp(a_j)- \exp(a_i)} {\sum_{j=1}^C \exp(a_j)}
\\ &= y_c [\mathbb I(i=c)-y_i]
\end{aligned}$$

记单位矩阵 $\mathbf I_{C \times C}$，那么梯度简化为

$$\frac {dy_c}{d a_i}=y_c (I_{ci}-y_i), \ i=1,2,...,C$$ (Logis3)

训练集 $D$，其中样本总数量为 $N$，**真实** 目标值记为 $\mathbf T_{N \times C}$，似然函数为

$$p(\mathbf T|\mathbf W)=\prod_{n=1}^N \prod_{c=1}^C p(t_c=1|\mathbf x_n)^{T_{nc}}=\prod_{n=1}^N \prod_{c=1}^C y_{nc}^{T_{nc}}$$

其中 $y_{nc}=p(t_c=1|\mathbf x_n)$，表示第 $n$ 个样本的预测目标值的 第 $c$ 个分量为 1 的概率。从这里再次可见，似然函数只考虑样本真实分类所对应的分类预测值，其他分类不考虑，例如第 $T_{nc}=1$，表示就第 $n$ 个样本而言，只考虑其预测分类为 $c$ 的概率 $p(t_c=1|\mathbf x_n)$。


损失采用负对数似然函数，

$$L(\mathbf W)=-\sum_{n=1}^N \sum_{c=1}^C T_{nc} \log y_{nc}$$

结合 {eq}`Logis3` ，注意由于 $T_{nc}$ 的作用，此时指示因子为 ，表示第 $n$ 个样本的预测梯度为

$$\begin{aligned} \nabla_{\mathbf w_j} L&=-\sum_{n=1}^N \sum_{c=1}^C T_{nc} \frac {y_{nc}(I_{cj}-y_{nj})} {y_{nc}}\mathbf x_n
\\ &=\sum_{n=1}^N \sum_{c=1}^C T_{nc}(y_{nj}-I_{cj})\mathbf x_n
\\ &=\sum_{n=1}^N \mathbf x_n \left(y_{nj}\sum_{c=1}^C T_{nc} -\sum_{c=1}^C T_{nc} I_{cj} \right)
\\ &=\sum_{n=1}^N \mathbf x_n (y_{nj}-T_{nj})
\end{aligned}$$

上式最后一个等式中，由于每个样本的目标值为 one-hot 向量，所以 $\sum_{c=1}^C T_{nc}=1$，另外 $\sum_{c=1}^C T_{nc} I_{cj}$ 的作用是筛选出 $T_{nj}$。
