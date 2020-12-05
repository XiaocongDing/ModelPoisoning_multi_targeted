从keras中导入mnist会导致报如下的错：

tensorflow.python.pywrap_tensorflow' has no attribute 'TFE_DEVICE_PLACEMENT_EXPLICIT

解决方法，提前把数据集下好，然后把相关的依赖都删掉。

但是细读代码，发现模型中的卷积层都没有实现。modelA~H，代码离能跑距离有点远。

细读算法：



### 搭建环境遇到的问题

作者写的代码是用的python2写的，从python2转成python3，费了不少时间。主要print加括号，以及包的相对路径需要更改。

开始tensorflow版本太新，有些方法已经淘汰了。

对tensorflow进行版本回退。回退至作者说的版本1.8

回退过后，会遇到如下问题：

![1606737033987](Adversarial Len源码阅读.assets/1606737033987.png)

原因是keras与tensorflow版本不兼容导致的，卸载keras，重新下一个低版本的keras解决。别听这个回答下的瞎扯。

windows下python路径“\”转义问题，两个\就可以了。

读完路径正确，能够跑了，但是又有如下问题：

![1606737236130](Adversarial Len源码阅读.assets/1606737236130.png)



python 全局变量的调用问题：

global 关键字不能初始化变量

global 函数内部定义的全局变量，必须要引用该函数一次才能被其它模块所调用。







