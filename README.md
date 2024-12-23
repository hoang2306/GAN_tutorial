# GAN loss function
discriminator có thể coi là một mô hình phân loại (như logistic regression)
* $f[\mathbf{x}, \phi]$ trả về một giá trị vô hướng (xác suất input là real example)
loss function for classification task using cross-entropy

$$\hat{\phi}=\underset{\boldsymbol{\phi}}{\operatorname{argmin}}\left[\sum_i-\left(1-y_i\right) \log \left[1\operatorname{sig}\left[\mathrm{f}\left[\mathbf{x}_i, \boldsymbol{\phi}\right]\right]\right]-y_i \log\left[\operatorname{sig}\left[\mathrm{f}\left[\mathbf{x}_i, \boldsymbol{\phi}\right]\right]\right]\right]$$



* với $y_i \in \{0,1\}$ là label
* sig: sigmoid 
giả sử: real example $\mathbf{x}$ có nhãn là 1 và generate example $\mathbf{x}^*$ có nhãn là 0
$$
\hat{\boldsymbol{\phi}}=\underset{\boldsymbol{\phi}}{\operatorname{argmin}}\left[\sum_j-\log \left[1-\operatorname{sig}\left[\mathrm{f}\left[\mathbf{x}_j^*, \boldsymbol{\phi}\right]\right]\right]-\sum_i \log \left[\operatorname{sig}\left[\mathrm{f}\left[\mathbf{x}_i, \boldsymbol{\phi}\right]\right]\right]\right]
$$
* $i$: index real example
* $j$: index generate example 
subtitute generator $\mathbf{x}_j^* = g[\mathbf{z}_j, \theta]$ và chúng ta muốn generated sample bị phân loại sai 
$$
\hat{\boldsymbol{\theta}}=\underset{\boldsymbol{\theta}}{\operatorname{argmax}}\left[\min _{\boldsymbol{\phi}}\left[\sum_j-\log \left[1-\operatorname{sig}\left[\mathrm{f}\left[\mathbf{g}\left[\mathbf{z}_j, \boldsymbol{\theta}\right], \boldsymbol{\phi}\right]\right]\right]-\sum_i \log \left[\operatorname{sig}\left[\mathrm{f}\left[\mathbf{x}_i, \boldsymbol{\phi}\right]\right]\right]\right]\right]
$$
$$\hat{\boldsymbol{\theta}}=\underset{\boldsymbol{\theta}}{\operatorname{argmin}}\left[\max _{\boldsymbol{\phi}}\left[\sum_j\log \left[1-\operatorname{sig}\left[\mathrm{f}\left[\mathbf{g}\left[\mathbf{z}_j, \boldsymbol{\theta}\right], \boldsymbol{\phi}\right]\right]\right] +\underbrace{(\sum_i \log \left[\operatorname{sig}\left[\mathrm{f}\left[\mathbf{x}_i, \boldsymbol{\phi}\right]\right]\right])}_{\mathbb{E}_{x\sim p_{data}(x)[\log D(x)]}} \right]\right]$$
chuẩn hóa cho giống công thức trong: https://github.com/tomsercu/gan-tutorial-pytorch/blob/master/2019-04-23%20GAN%20Tutorial%20with%20outputs.ipynb
* chỉ cần đổi dấu là ta có 

$$
\min _G \max _D V(D, G)=\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}(\boldsymbol{x})}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]
$$
# Trainining GAN
![[Pasted image 20241223163450.png]]
training GAN là min-max game 
$\rightarrow$ solution: cân bằng Nash (Nash equilibrium)
để train GAN, ta có thể tách thành 2 hàm loss 
$$
L[\boldsymbol{\phi}]=\sum_j-\log \left[1-\operatorname{sig}\left[\mathrm{f}\left[\mathbf{g}\left[\mathbf{z}_j, \boldsymbol{\theta}\right], \boldsymbol{\phi}\right]\right]\right]-\sum_i \log \left[\operatorname{sig}\left[\mathrm{f}\left[\mathbf{x}_i, \boldsymbol{\phi}\right]\right]\right]
$$
$$
L[\boldsymbol{\theta}]=\sum_j \log \left[1-\operatorname{sig}\left[\mathrm{f}\left[\mathbf{g}\left[\mathbf{z}_j, \boldsymbol{\theta}\right], \boldsymbol{\phi}\right]\right]\right]
$$
