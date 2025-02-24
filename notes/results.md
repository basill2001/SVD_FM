## 1. original, FM
Settings : `num_epochs = 100`, `data=ml100k`  
|metrics |value |value<sup>[1](#footnote_1)</sup> |
|---|---|---|
|precision|0.444954128440366 |0.450000000000000 |
|recall   |0.079367341518164 |0.080490733571922 |
|hit rate |0.853211009174311 |0.848623853211009 |
|rec rank |0.679931192660550 |0.655504587155963 |
|dcg      |1.395779875067928 |1.380704386660955 |
|time     |108.8987262248992 |94.04915118217468 |
---
<a name='footnote_1'>1</a> : `L2-norm`을 명시적으로 더하는 대신 `weight_decay` 사용

## 2. original, DeepFM
Settings : `num_epochs = 100`, `data=ml100k`  
|metrics |value |value<sup>[1](#footnote_1)</sup> |
|---|---|---|
|precision|0.435779816513761 |0.456880733944954 |
|recall   |0.082129396794385 |0.081870688544863 |
|hit rate |0.880733944954128 |0.869266055045871 |
|rec rank |0.664717125382263 |0.670451070336391 |
|dcg      |1.344460927879136 |1.398971377667273 |
|time     |164.4945077896118 |94.88665699958801 |
---
<a name='footnote_1'>1</a> : `L2-norm`을 명시적으로 더하는 대신 `weight_decay` 사용

## 3. SVD, FM
Settings :`num_epochs = 100`, `data=ml100k`  
|metrics |value |value<sup>[1](#footnote_1)</sup> |
|---|---|---|
|precision|0.198165137614678 |0.201376146788990 |
|recall   |0.026636492888923 |0.027525045805599 |
|hit rate |0.550458715596330 |0.548165137614678 |
|rec rank |0.371292048929663 |0.394571865443425 |
|dcg      |0.631741929379093 |0.663941525467433 |
|exp_var  |0.319258351509169 |0.319258351509169 | 
|const err|6.433065941228768 |6.433065941228768 |
|time     |104.9781436920166 |
---
<a name='footnote_1'>1</a> : `isuniform`을 `True`로 사용 i.e. complete random negative sample

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

Settings : `num_epochs = 100`, `data=ml1m`  
|metrics |value |
|---|---|
|precision|0.061092715231788 |
|recall   |0.006461771101007 |
|hit rate |0.198675496688741 |
|rec rank |0.123311258278145 |
|dcg      |0.192781487260310 |
|time     |991.7204689979553 |

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
