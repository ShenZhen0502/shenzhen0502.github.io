---
title: OLMo
date: 2024-03-13 10:52:16
tags:
categories: 自然语言处理
description: 一个基于pytorch的大语言模型训练，推理库。仓库地址为https://github.com/allenai/OLMo，感谢作者为开源所做的贡献。

---

# 结构简介

# 数据预处理

在深度学习中，数据的读取与预处理是在进行训练之前的一个重要步骤。我从olmo使用的数据集dolma中随便取了一行内容，如下所示：

```json
{"text":"Fasnet 2015: Spindrift & Prince de Bretagne , \"Neck to neck\"\nSpindrift reports that After 48 hours of sailing, Spindrift 2 is neck and neck with @trimaranpdb Prince de Bretagne Tri.\nBoth Tris followed by Phaedo, Oman & Concise MOD70s. On corrected time Phaedo 3 leads.","url":"https://www.catsailingnews.com/2015/08/fasnet-2015-spindrift-prince-de.html","date":"2019-04-18T20:34:13Z"}
```

该数据集是json文本，上面是从dolma中随机选择的一行，文本中每行都是和上面一样格式的内容。其中的url和date称为metadata(元数据)，text就是我们训练时要用的文本数据。

假设我们整个数据集就只有上面text中的数据。接下来的操作就是为每个单词赋一个索引，也就是说每个单词都要对应一个唯一的整数，这个整数有什么用，可以看我的博客word2vec。下面我使用代码来对这个操作进行讲解。

```python
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# 假设我们有一些文本数据
texts = ["Fasnet 2015: Spindrift & Prince de Bretagne , \"Neck to neck\"\nSpindrift reports that After 48 hours of sailing, Spindrift 2 is neck and neck with @trimaranpdb Prince de Bretagne Tri.\nBoth Tris followed by Phaedo, Oman & Concise MOD70s. On corrected time Phaedo 3 leads."]

# 步骤1: 分词
# 使用torchtext的get_tokenizer获取基本的英文分词器
tokenizer = get_tokenizer("basic_english")

# 步骤2: 构建词汇表
# 首先，我们需要一个生成器，它会遍历数据集中的所有文本，并为每个文本返回一个分词后的列表
token_stream = map(tokenizer, texts)

# 使用build_vocab_from_iterator构建词汇表
# build_vocab_from_iterator接受一个迭代器，迭代器提供了分词后的单词序列
vocab = build_vocab_from_iterator(token_stream, specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])  # 设置默认索引为未知词标记
print(vocab.vocab.itos_)
# 打印结果为：['<unk>', 'neck', ',', '.', 'spindrift', '&', 'bretagne', 'de', 'phaedo', 'prince', '2', '2015', '3', '48', '@trimaranpdb', 'after', 'and', 'both', 'by', 'concise', 'corrected', 'fasnet', 'followed', 'hours', 'is', 'leads', 'mod70s', 'of', 'oman', 'on', 'reports', 'sailing', 'that', 'time', 'to', 'tri', 'tris', 'with']

# 步骤3: 将单词转换为索引
# 假设我们要转换的文本
text_to_convert = "Prince reports that 3 beautiful leads."
tokenized_text = tokenizer(text_to_convert)
print(tokenized_text)
# 打印结果为：['prince', 'reports', 'that', '3', 'beautiful', 'leads', '.']

# 使用词汇表将单词转换为索引
# 输出类似于 [2, 3, 0, 4]，具体数字取决于词汇表的构建方式
indexed_tokens = [vocab[token] for token in tokenized_text]
print(indexed_tokens)
# 打印结果为：[9, 30, 32, 12, 0, 25, 3]
```

上面代码首先将我们的数据集做成了一个词典，即代码变量中的vocab，这个词典中保存了所有在数据集中出现的单词，标点符号，而且还有一个叫做<unk>的，unk的意思就是unkown，这个unk的作用是：如果想查找某个单词的索引，但是这个单词不在这个词典中，那么我们就认为这个单词为unk，就是说这个单词的索引为0，因为unk在词典中的最前面。单词在字典中的位置就是这个单词的索引。例如上面所示我们想查找索引的句子是"Prince reports that 3 beautiful leads."其中除了beautiful，所有的单词都在词典中，那么我们可以找到索引，对于beautiful这个单词的索引就是0。

找到单词的索引后，下面的操作就是执行词嵌入，词嵌入就是使用一个密集向量来表示一个单词，这样我们才能将单词输入网络中进行训练，因为你总不能直接把一个单词文本输入到网络中吧。当然pytorch也内置了预训练词向量，这样就不要我们训练了，直接输入单词就可以得到对应的向量表示，如下所示：

```python
from torchtext.vocab import GloVe

# 加载GloVe词向量，这里以'6B'维度的100维版本为例
glove = GloVe(name='6B', dim=100)

# 你可以直接通过单词获取其向量
vector = glove['python']
print(vector)
```

但是从头开始训练大型语言模型（如GPT、BERT等）时，通常不会使用预训练的词向量。这些大型语言模型采用了不同的方法和架构，主要有以下几个理由：

### 1. 嵌入层学习

大型语言模型通常在模型的一开始就包括一个嵌入层（embedding layer），这个层负责将输入的单词（或更常见的是，单词的子分割，如Byte-Pair Encoding（BPE）或WordPiece）转换为稠密的向量表示。这个嵌入层是模型训练过程中学习的一部分，使得模型能够学到针对特定任务优化的词嵌入。

### 2. 端到端训练

这些模型通过大规模的数据集进行端到端的训练，意味着从输入的文本到输出的预测，模型的所有参数（包括嵌入层）都是一起学习的。这样做可以确保学到的词嵌入能够与模型的其他部分（如注意力机制、Transformer层等）协同工作，以最大化整体性能。

### 3. 上下文相关的嵌入

与传统的预训练词向量（如Word2Vec、GloVe等）不同，大型语言模型通常产生的是上下文相关的嵌入。这意味着同一个单词在不同的句子或上下文中可能会有不同的向量表示。这种能力是通过模型的整体结构（特别是在基于Transformer的模型中）而非仅仅通过嵌入层来实现的，因此预训练的静态词向量对于这类模型来说价值有限。

### 4. 词汇表和分词方法

大型语言模型通常使用特定的分词方法（如BPE、SentencePiece或WordPiece）来处理文本，这些方法生成的词汇表可能与预训练词向量使用的词汇表不同。这意味着即使想要使用预训练词向量，也可能会因为词汇表的不匹配而遇到困难。

所以我们如果想从头开始训练一个大语言模型，我们词嵌入矩阵也需要自己训练。由于训练大语言模型的数据集都是非常大的，可能有好几个T，我们不可能直接把所有的数据全部加载到内存中，所以我们需要使用内存映射。在OLMO库中的scripts文件夹下有一个prepare_memmap_dataset.py文件，它的目的是将一个或多个原始的JSON格式的数据集文件（通常是大型文本数据集，比如C4）转换成NumPy的内存映射（memory-mapped）文件，便于高效地进行大规模的语言模型训练。内存映射文件允许内容存储在磁盘上，同时表现得就像是存储在RAM中一样，这样可以处理比可用RAM更大的数据集。这个脚本文件会将数据集中的每行数据的text部分的内容的每个单词转化成索引，然后存入.npy文件，最后输出一个.npy文件，用于数据的读取。

# 数据读取

数据读取代码在OLMo仓库olmo文件夹内的data文件夹内的所有.py文件。

# 模型搭建



# 模型训练

# 模型推理
