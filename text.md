
we can find the source code here: https://gitee.com/xiongjiajun0315/clip
and here: https://zhuanlan.zhihu.com/p/521151393
https://blog.csdn.net/h661975/article/details/135116957

# vit

原先的image，4维：
batch chanel H(224) W
经过vit变成了3维（flatten + linear projection到固定维度，一般是768）
batch  #patch(196) 16*16*3

文本经过embedding
batch token 嵌入维度 也是3维
维度对应上了

通过对比学习来训练参数，来保证相似度矩阵(注意力分数)的对角线上值是最大的,是一种无监督学习，不需要有label
预训练出来一个text encoder和image encoder

之后，适用于下游任务：比如，将所有的词放入一个句子，再放入text encoder，再放入一个图片，
做zero-shot，找到和图片匹配度最高的token（训练的话，需要告诉系统正确的token是什么）
