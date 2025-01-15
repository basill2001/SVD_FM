## 1. original, FM
Settings : `num_epochs = 100`, `data=ml100k`  
|metrics |value |
|---|---|
|precision|0.444954128440366 |
|recall   |0.079367341518164 |
|hit rate |0.853211009174311 |
|rec rank |0.679931192660550 |
|dcg      |1.395779875067928 |
|time     |108.8987262248992 |

## 2. original, DeepFM
Settings : `num_epochs = 100`, `data=ml100k`  
|metrics |value |
|---|---|
|precision|0.4330275229357798 |
|recall   |0.0772224272583683 |
|hit rate |0.8577981651376146 |
|rec rank |0.6479740061162079 |
|dcg      |1.3304601933264755 |
|time     |109.45841121673584 |

## 3. SVD, FM
Settings :`num_epochs = 100`, `data=ml100k`  
|metrics |value |
|---|---|
|precision|0.198165137614678 |
|recall   |0.026636492888923 |
|hit rate |0.550458715596330 |
|rec rank |0.371292048929663 |
|dcg      |0.631741929379093 |
|time     |104.9781436920166 |

## 3-1. NMF, FM
Settings : `num_epochs = 100`, `data=ml100k`
|metrics |value |value<sup>[1](#footnote_1)</sup> |
|---|---|---|
|precision |0.2628440366972477 |0.286238532110091 |
|recall    |0.0348235349105177 |0.038260829741647 |
|hit rate  |0.6261467889908257 |0.655963302752293 |
|rec rank  |0.4530581039755351 |0.475458715596330 |
|dcg       |0.8297901104417563 |0.896565102289302 |
|time      |95.44319891929626  |97.27948427200317 |
---
기존에는 `nndsvda`가 쓰임  
<a name='footnote_1'>1</a> : NMF의 `init`을 `nndsvd`로

## 4. SVD, DeepFM
Settings : `num_epochs = 100`, `data=ml100k`  
|metrics |value |
|---|---|
|precision|0.5279816513761467 |
|recall   |0.0967468529122231 |
|hit rate |0.9197247706422018 |
|rec rank |0.7345565749235473 |
|dcg      |1.6030452903462624 |
|time     |139.4771749973297  |
 
## 4-1. NMF, DeepFM
Settings : `num_epochs = 100`, `data=ml100k`
|metrics |value |value<sup>[1](#footnote_1)</sup> |value<sup>[2](#footnote_2)</sup> |
|---|---|---|---|
|precision |0.522477064220183 |0.501376146788990 |0.536697247706422 |
|recall    |0.095890446513980 |0.091670458356366 |0.097804399179589 |
|hit rate  |0.899082568807339 |0.899082568807339 |0.926605504587156 |
|rec rank  |0.715022935779816 |0.709441896024465 |0.749350152905198 |
|dcg       |1.589640648742865 |1.534121044400698 |1.645018191831604 |
|time      |105.3881254196167 |141.7736880779266 |133.9224553108215 |
---
<a name='footnote_1'>1</a> : sparse NMF 시행(0.1 이하인 embedding 값을 전부 0으로)  
<a name='footnote_2'>2</a> : sparse NMF 시행(0~0.01 값들을 전부 0으로)  
