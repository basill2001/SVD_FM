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

Settings : `num_epochs = 100`, `data=goodbook`  
|metrics |value |
|---|---|
|precision|0.002744338749052 |
|recall   |0.000373540249956 |
|hit rate |0.013637511575048 |
|rec rank |0.006358559923674 |
|dcg      |0.008179955909675 |
|exp_var  |0.103176519270626 |
|time     |4482.333681344986 |


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

Settings : `num_epochs = 100`, `data=goodbook`  
|metrics |value |
|---|---|
|precision|0.393299099250778 |
|recall   |0.053248408478951 |
|hit rate |0.796026601565788 |
|rec rank |0.595497656929595 |
|dcg      |1.213845092514998 |
|exp_var  |0.103176519270626 |
|time     |3065.868867158889 |

## 3. SVD, FM
Settings :`num_epochs = 100`, `data=ml100k`  
|metrics |value |value<sup>[1](#footnote_1)</sup> |value<sup>[2](#footnote_2)</sup> |
|---|---|---|---|
|precision|0.198165137614678 |0.201376146788990 |0.2490825688073394 |
|recall   |0.026636492888923 |0.027525045805599 |0.0301064695525713 |
|hit rate |0.550458715596330 |0.548165137614678 |0.5825688073394495 |
|rec rank |0.371292048929663 |0.394571865443425 |0.4001911314984709 |
|dcg      |0.631741929379093 |0.663941525467433 |0.7704727808198837 |
|exp_var  |0.319258351509169 |0.319258351509169 |                   |
|const err|6.433065941228768 |6.433065941228768 |                   |
|time     |104.9781436920166 |                  |105.16335678100586 |
---
<a name='footnote_1'>1</a> : `isuniform`을 `True`로 사용 i.e. complete random negative sample  
<a name='footnote_2'>2</a> : `SparseSVD`를 사용, 이때 `sparsity` i.e. `alpha`는 1로 고정

Settings : `num_epochs = 100`, `data=goodbook`  
|metrics |value |
|---|---|
|precision|0.214883407694250 |
|recall   |0.029467402417454 |
|hit rate |0.593821028706120 |
|rec rank |0.340588433369812 |
|dcg      |0.627133449344527 |
|exp_var  |0.103176519270626 |
|time     |3452.502673149109 |


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
|metrics |value |value<sup>[1](#footnote_1)</sup> |value<sup>[2](#footnote_2)</sup> |
|---|---|---|---|
|precision|0.5279816513761467 |0.4857798165137615 |0.468349	|
|recall   |0.0967468529122231 |0.0838822768972341 |0.079619	|
|hit rate |0.9197247706422018 |0.8876146788990825 |0.874312 |
|rec rank |0.7345565749235473 |0.6909403669724771 |0.677080 |
|dcg      |1.6030452903462624 |1.4827439130712168 |         |
|time     |139.4771749973297  |106.86286878585815 |         |
---
<a name='footnote_1'>1</a> : `SparseSVD`를 사용, 이때 `sparsity` i.e. `alpha`는 1로 고정  

Settings : `num_epochs = 100`, `data=ml1m`  
|metrics |value |
|---|---|
|precision|0.18105960264900 |
|recall   |0.03187354130476 |
|hit rate |0.50711920529801 |
|rec rank |0.30680187637969 |
|dcg      |0.54466047580628 |
|time     |970.119738817215 |

Settings : `num_epochs = 100`, `data=goodbook`  
|metrics |value |
|---|---|
|precision|0.477767488845862 |
|recall   |0.064715489760303 |
|hit rate |0.880461318292785 |
|rec rank |0.708383141117378 |
|dcg      |1.490577470951982 |
|exp_var  |0.103176519270626 |
|const err|8.125652967151472 |
|time     |3519.840175390243 |

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
