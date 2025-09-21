# 创建一个虚拟机
### 1.下载一个VMmare和镜像
![alt text](e2a25140ef03a46ff78917f9db2fed16.png)
去清华大学镜像站，下载一个Ubuntu
mirrors.tuna.tsinghua.edu.cn
![alt text](9db89d3b09cbe905493bf75842d541dc-1.png)
* 一定要下ubuntu桌面版
* 一定不可以下成centos  
* 
  惨痛教训在此

  ---

### 2.在VMmare上创建一个虚拟机
![alt text](58bc8bc2c1c6025fa0f748ea1e361e56.png)

**网络选用NAT模式！！**

惨痛教训+1

![alt text](7ffcac0a5f3458e34641d2e26ba52989-1.png)

选则刚刚下载好的镜像文件

然后开始**痛苦**的设置网络

![alt text](27691f3450f78280196b8b011c609872.png)

首先我们应该有VMnet1和VMnet8

把VMnet设置为NAT

**注意子网！！记住他的前三段**，子网掩码随便输一个就行

![alt text](3e684d2808efdf8c5a03d72d33262094.png)

**网关必须前三段和子网一致**

![](cf04cca0193f104d04b8668016caf04b.png)

**同时在电脑的网络设置里，需要手动把IPv4地址设置与子网前三段一致**

惨痛教训+2

不设置上不了网你就知道什么叫崩溃了~~亲身经历~~



ok得到了一个虚拟机，只需要随便登录账号设置密码就可以使用了

![alt text](ed8660fb5563ee31f243a38e6b04e0ac.png)
*可以看到我走的弯路，还下载了一个centos（哭泣）*


![alt text](9c2e107b925fd56f6ea62287f5aa5aca.png)

像这样就是连接好网络的一个完整的虚拟机了

---
### 3.通过ssh连接主机和虚拟机

首先需要下载一个xshell

![alt text](0afb557fb3a55b6a04f24b5c2f6f184f.png)

ok可以看到我是连接好了补的截图（bushi）

我们新建一个会话

![alt text](0a5e4929daaefcb7ce9af91182ac110f.png)

在主机那个地方填上我们虚拟机的网络ip

这个在哪里查看呢

就是刚刚我们设置网络那里

然后我们可以先ping看看是否两边接通

![alt text](873adf659c028b8993c99114a4693e3f.png)
![alt text](a5e40682defd246efd805a17297acbad.png)

这样就是ok的

然后在网上找个教程一步一步输入指令

只要**亦步亦趋**就会成功的！

#### 我的参考文献：[SSH远程连接详细步骤](https://www.bilibili.com/video/BV1RF11YzEV9/?spm_id_from=333.337.search-card.all.click&vd_source=1cf9ae4d63ff94aa45638c152e755eab)

然后我就没截上图了，，，，

不过值得注意的是这一步

![alt text](f5c3473ec6cdf0c769284f1a04b42053.jpg)
**我们需要解决掉我们的防火墙**

惨痛教训+3

OK呀我们就完成了虚拟机的创建

---
### 4.ideas

* 创建虚拟机是个痛苦的过程，只能跟着教程一步一步来，网络设置的教程，ssh的教程，各种突发问题还需要ai帮我们解答      
* 但是创建虚拟机很有用，linux上运行vscode简单多了，这个ubuntu上还可以下任何软件，解放主机，避免主机盘里全是杂七杂八的东西