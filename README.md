# 树莓派口罩检测

![](https://github.com/17513146506/Mask-detection/blob/main/label_img/2.png)



### 说明：项目提供的模型基于互联网收集的图片训练（每类2000张，共6000张），样本及场景过少

## 一、硬件：

* PC端运行：Windows10或11（无需GPU，有最好）或MacOS 都测试可行
* 树莓派运行：树莓派 4B model B 8G 版
* USB RGB 摄像头

## 二、软件：

* Python==3.8
* 电脑需要：TensorFlow，树莓派需要：TensorFlow lite
* opencv 
* pixellib


### 3.3、自己训练模型

* 采集照片：放到`images`文件夹下，`1.yes、2.no、3.nose`分别代表`正常佩戴、未佩戴、漏出鼻子`
* 数据预处理：运行`1.images_preprocess.ipynb`将图片数据预处理为`numpy`文件，存在`data`目录下；
* 训练模型：运行`2.model_train.ipynb`训练模型，模型文件在`data`目录下；
* 压缩模型：运行`4.tflite.ipynb`压缩模型，压缩后的模型在`data`目录下，需要手动搬到树莓派`rasp_lite/data`目录下；
