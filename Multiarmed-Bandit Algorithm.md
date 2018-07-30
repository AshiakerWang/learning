## Thompson Sampling
### 1. 算法概述
老虎机的每一个臂都有一个产生收益的概率P，我们用一个 $\beta（win,lose）$ 分布来模拟。其中：

- $win$ 为选取该臂后有收益的次数。（对于推荐系统可以看做推荐后用户的点击次数）

- $lose$ 为选取该臂后没有获得收益的次数。（对于推荐系统可以看做推荐后用户未点击的次数）

对于每次实验：

- 根据每个臂对应的参数 $win$ 和 $lose$ 利用 $\beta$ 分布生成随机数。

- 选取生成的随机数最大的那个臂，如果该臂产生回报则该臂对应的 $win$ 加一，否则，该臂对应的 $lose$ 加一。

### 2. 代码实现
该算法的伪代码如下：

![](/users/shanks/desktop/imgs/thom.png)

Python 实现如下：

```
import  numpy as np

import  pymc

# wins 和 trials 都是一个 N 维向量，N 是臂的个数

# wins 表示所有臂的 α 参数，loses 表示所有臂的 β 参数

choice = np.argmax(pymc.rbeta(1 + wins, 1 + loses, len(wins)))

# wins[choice] += 1

# loses[choice] += 1
```
### 3.  $\beta$ 分布
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;在概率论中，Β分布也称贝塔分布，是指一组定义在  (0,1)区间的连续概率分布，有两个参数 $ \alpha ,\beta >0$。
$\beta$分布的概率密度函数如下：
<div align = center><img src = "/users/shanks/desktop/imgs/beta.svg"></div>
下图为不同取值的$\alpha,\beta$对应的分布：
<div align = center><img src = "/users/shanks/desktop/imgs/beta2.png"></div>

期望值和方差分别是：
<div align = center><img src = "/users/shanks/desktop/imgs/bea_e.svg"></div>
<div align = center><img src = "/users/shanks/desktop/imgs/bea_var.svg"></div>

直观的理解为什么汤普森采样算法有效：

- 当尝试的次数较多时，每个臂的 α + β 的值都很大，这时候每个臂对应的 beta 分布都会很窄，也就是说，生成的随机数都非常接近中心位置，每个臂的收益基本确定了。

- 当尝试的次数较少时，即每个臂的 α + β 的值都很小，这时候每个臂对应的 beta 分布都会很宽，生成的随机数有可能会比较大，增加被选中的机会。

- 当一个臂的 α + β 的值很大，并且 α/(α + β) 的值（均值）也很大，那么这个臂对应的 beta 分布会很窄，并且中心位置接近 1 ，那么这个臂每次选择时都很占优势。
