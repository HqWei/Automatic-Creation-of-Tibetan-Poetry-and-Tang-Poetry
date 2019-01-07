# Automatic-creation-of-Tibetan-poetry-and-Tang-poetry
## Introduction

Automatic creation of Tibetan poetry and Tang poetry based on LSTM in tensorflow.

## Result:

### 藏头诗生成效果:

武汉别名江城，这哥别名源于李白的《与史郎中钦听黄鹤楼上吹笛》，一为迁客去长沙，西望长安不见家。黄鹤楼中吹玉笛，江城五月落梅花。镌刻于如今重建的黄鹤楼。下面以江城美景为头：

#### 五言诗：
    江焰红花里，风经雨起烟。
    城西深夜后，叶满不胜经。
    美洁漏将受，出门临碧池。
    景间陪待罢，佳景尽依依。
#### 七言诗：
    江边树绿半堪山，八月西洲伴春色。
    城郭花开雪满地，满江杨柳归何处。
    美女尘中心自偶，远平一是子为珍。
    景阳幽色最临声，此意浮风坐绕禅。
### 唐诗生成效果：

#### 五言诗：

    花偏君亦长，一别少看花。

    项小黔州路，天边山已深。

    渡长淮河上，月夜南山分。

    情世不可识，歌枝一少愁。

#### 七言诗：

    彼处闻寒溜泉频，竹峰蛛网木浓阴。

    先生成性思成远，白发幽人事肯醒。

    惟有月圆心便寝，起经徒到意无身。

    跂襟藉笋丛青菊，声价同亲奈欲何。

## Usage：

* 1. The code is based on tensorflow and python2.7

* 2. train_model.py Training  the LSTM mdoel
    gene_head_poetry.py Generating a Tibetan poetry.
    gene_poetry.py Generating a Tang poetry.
