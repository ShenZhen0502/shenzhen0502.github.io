---
title: Canny边缘检测
date: 2024-02-22 18:11:07
categories: 传统图像处理
tags:
---

# Canny边缘检测原理和逐行代码详解
## 理论基础
我们希望最后的图像是一个二值图，就是说只有两种情况，要么是边缘要么不是边缘。并且最后的边缘是一个很细的线，甚至只有一个像素点。

### 卷积算子
我们在进行边缘检测前需要执行一个去噪的步骤，去噪我们一般使用高斯核。

```python
def im2col(image, filter_h, filter_w, padding, stride):
    """
    只支持二维图像，传统方法一般都是用灰度图或二值图
    :param image: 要转换的图像
    :param filter_h: 滤波的长
    :param filter_w: 滤波的宽
    :param padding: padding的宽度
    :param stride: 卷积的步长
    :return: 要转换的图像
    """

    matrix = []  # 保存最后的输出结果
    img_shape = image.shape  # 读取输入图像形状
    image = np.pad(image, ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')  # 对输入图像进行pad
    for j in range(0, img_shape[0], stride):  # 遍历行
        for k in range(0, img_shape[1], stride):  # 遍历列
            col = image[j:j + filter_w, k:k + filter_h].reshape(1, -1)  # 找到对应位置的数据（二维的），将其reshape成一行数据
            matrix.append(col)  # 将reshape的数据添加到结果列表中

    return np.array(matrix)
```
![](./images/a.png)
