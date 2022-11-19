# 树莓派口罩检测

![]([https://image.baidu.com/search/detail?ct=503316480&z=0&ipn=d&word=%E5%8F%A3%E7%BD%A9%E5%9B%BE%E7%89%87&hs=0&pn=6&spn=0&di=7146857200093233153&pi=0&rn=1&tn=baiduimagedetail&is=0%2C0&ie=utf-8&oe=utf-8&cl=2&lm=-1&cs=874045201%2C3791794877&os=606042473%2C374080397&simid=4213354516%2C959896423&adpicid=0&lpn=0&ln=30&fr=ala&fm=&sme=&cg=&bdtype=0&oriquery=%E5%8F%A3%E7%BD%A9%E5%9B%BE%E7%89%87&objurl=https%3A%2F%2Fgimg2.baidu.com%2Fimage_search%2Fsrc%3Dhttp%3A%2F%2Fimgservice.suning.cn%2Fuimg1%2Fb2c%2Fimage%2F0AO3aAccchoI6DTOUSwBmA.jpg_800w_800h_4e%26refer%3Dhttp%3A%2F%2Fimgservice.suning.cn%26app%3D2002%26size%3Df9999%2C10000%26q%3Da80%26n%3D0%26g%3D0n%26fmt%3Dauto%3Fsec%3D1671421402%26t%3D6c79eff521fd28afb7b866566bd1ad89&fromurl=ippr_z2C%24qAzdH3FAzdH3Fooo_z%26e3Bf7gtg2_z%26e3Bv54AzdH3Ftpj4et1j5AzdH3Faa089mnaldAzdH3F8dna8898bln_z%26e3Bip4s&gsm=&islist=&querylist=&dyTabStr=MCwzLDIsMSw1LDYsNCw4LDcsOQ%3D%3D](https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fimg13.360buyimg.com%2Fn1%2Fjfs%2Ft1%2F195768%2F5%2F14006%2F279510%2F60f7b8c5Ef14466a5%2F0141d5dbe5622b4a.jpg&refer=http%3A%2F%2Fimg13.360buyimg.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1671421454&t=a613f618c068fb14b299ec3bdc26c574))



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
