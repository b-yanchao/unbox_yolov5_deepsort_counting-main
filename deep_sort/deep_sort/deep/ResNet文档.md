# RestNet

<img src="https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9NR01jMlpCZjluNDJPT2ljSjhneExrdE5WZXlsSDdERnlkaFJ0WlFaZkpUUGZxejBVV3pLb2M2eHhCdll6ODgzblVyc0xpY2ljcXd1cjhNd0prclpDaFlWdy82NDA?x-oss-process=image/format,png" alt="img" style="zoom:67%;" />

结构图：

![img](https://note.youdao.com/yws/public/resource/5a7dbe1a71713c317062ddeedd97d98e/xmlnote/WEBRESOURCE422f1039d96b8bd68a80758d37d71378/4454)

## Residual net(残差网络)

将靠前若到干层的某一层数据输出直接跳过多层引入到后面数据层的输入部分。意味着后面的特征层的内容会有一部分由其前面的某一层线性贡献。

结构图：

![img](https://note.youdao.com/yws/public/resource/5a7dbe1a71713c317062ddeedd97d98e/xmlnote/2CE5A563BE904A768D4B940819240992/4458)

功能：为了克服由于网络深度加深而产生的学习效率变低与准确率无法有效提升的问题。

## 基本构成

### Conv Block

输入和输出维度（通道数和size）不一样的，所以不能连续串联，它的作用是改变网络的维度。

![img](https://note.youdao.com/yws/public/resource/5a7dbe1a71713c317062ddeedd97d98e/xmlnote/8246B59C888746A8916AAA69E992A50E/4466)

### Identity Block

输入维度和输出维度（通道数和size）相同，可以串联，用于加深网络。

![img](https://note.youdao.com/yws/public/resource/5a7dbe1a71713c317062ddeedd97d98e/xmlnote/EA4492FF979649D095C02C17A23FC35E/4468)

### 总体的网络结构

![img](https://note.youdao.com/yws/public/resource/5a7dbe1a71713c317062ddeedd97d98e/xmlnote/91C43146DDE646BD9AD37E3E5F818A3D/4471)