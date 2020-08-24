Solution
===================


[Multi-Class(Label) Classification](https://wordbe.tistory.com/entry/ML-Cross-entropyCategorical-Binary%EC%9D%98-%EC%9D%B4%ED%95%B4)   
[AutoEncoder](https://wjddyd66.github.io/pytorch/Pytorch-AutoEncoder/)    
[AutoEncoder](https://debuggercafe.com/autoencoder-neural-network-application-to-image-denoising/)
[wikidocs](https://wikidocs.net/3413) : Deep Learning 이론과 실습    
[Convolutional Autoencoders for Image Noise Reduction](https://towardsdatascience.com/convolutional-autoencoders-for-image-noise-reduction-32fce9fc1763)   
[NoiseReduction4DeepLearning](https://github.com/hashnut/NoiseReduction4DeepLearning)    
[Unsupervised](https://ebbnflow.tistory.com/165)     
[SegNet 1](https://github.com/say4n/pytorch-segnet/blob/master/src/model.py)    
[SegNet 2](https://medium.com/hyunjulie/1%ED%8E%B8-semantic-segmentation-%EC%B2%AB%EA%B1%B8%EC%9D%8C-4180367ec9cb)     
[FCN](https://medium.com/@msmapark2/fcn-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-fully-convolutional-networks-for-semantic-segmentation-81f016d76204)    
[FCN code](https://github.com/pochih/FCN-pytorch)    
[Semantic Segmentation](https://kuklife.tistory.com/117?category=872135)   
[Labeling tools](https://hoya012.github.io/blog/Tutorials-of-Object-Detection-Using-Deep-Learning-labeling/)     
[Labeling tools2](https://honeycomb-makers.tistory.com/14)   
[Unet&Segnet](https://github.com/trypag/pytorch-unet-segnet)   
[SOTA?](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes)    

[GAN](https://dreamgonfly.github.io/blog/gan-explained/) : Generative Adversarial Network (Make a fake image)



[more...](https://kuklife.tistory.com/121?category=872135)




Keyword : Reduce noise CNN    
Paper 1 : 영상에서 패치기반 CNN 모형을 이용한 잡음 제거    
Paper 2 : 완전 합성곱 신경망을 활용한 드론 비행음 및 바람소리 제거 [YouTube](https://www.youtube.com/watch?v=4LYmovbp5vA)    

-------------------------------------
[모두의 연구소](https://modulabs-biomedical.github.io/)   
[라온피플](http://blog.naver.com/PostView.nhn?blogId=laonple&logNo=220478250374&parentCategoryNo=&categoryNo=16&viewDate=&isShowPopularPosts=false&from=postView)   
[IntFlow](https://github.com/intflow)   


```python
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
```


```python
PATH = '../compu/data/'
filename = 'dataset_1.mat'
mat_file = io.loadmat(PATH + filename)
filename = 'angle.mat'
angle_file = io.loadmat(PATH + filename)
```


```python
S_left = mat_file['S_left']
S_right = mat_file['S_right']
clean_left = mat_file['clean_left']
clean_right = mat_file['clean_right']

phi = angle_file['phi'][0:1000]
```


```python
S_left = np.transpose(S_left, (2, 0, 1))
S_right = np.transpose(S_right, (2, 0, 1))
clean_left = np.transpose(clean_left, (2, 0, 1))
clean_right = np.transpose(clean_right, (2, 0, 1))
```


```python
S_left = np.reshape(S_left, (1000, 1, 257, 382))
S_right = np.reshape(S_right, (1000, 1, 257, 382))
clean_left = np.reshape(clean_left, (1000, 1, 257, 382))
clean_right = np.reshape(clean_right, (1000, 1, 257, 382))
```


```python
S_left = np.log10(S_left + 1)
S_right = np.log10(S_right + 1)
clean_left = np.log10(clean_left + 1)
clean_right = np.log10(clean_right + 1)
```


```python

```


```python

```


```python
index = np.random.randint(0, 1000)
print(index)
plt.imshow(S_left[index,0,])
```


```python
plt.imshow(clean_left[index,0,])
```


```python
k = clean_left[index,0,].copy()
```


```python
plt.imshow(clean_left[index,0,])
```


```python
k.mean() * 1.2
```


```python
k[k >= (k.mean()* 2)] = 1
k[k < (k.mean()* 2)] = 0
```


```python
plt.imshow(k)
```


```python
(k[k!=0] == 1.).all()
```


```python

```
