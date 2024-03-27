---

title: DiffusionModel
date: 2024-03-07 22:59:49
tags:
categories: AIGC
---

本文将从直观解释，代码实现，数学公式推导来全面阐述扩散模型。

# 扩散模型直观理解

扩散模型的目的是学习从纯噪声生成图片的模型。

$$
N(x_t;\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)I)
$$
