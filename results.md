## 1. 
Settings : `embedding type = original`, `model_type = fm`, `num_epochs = 100`, `data=ml100k`  
|metrics| value| value<sup>[1](#footnote_1)</sup>|
|---|---|---|
|precision|0.295412844036697 | 0.44495412844036 |
|recall   |0.052328232882711 | 0.07936734151816 |
|hit rate |0.754587155963302 | 0.85321100917431 |
|rec rank |0.508103975535168 | 0.67993119266055 |
|dcg      |0.914304604489501 | 1.39577987506792 |
|time     |94.00355195999146 | 86.5315408706665 |
---
<a name='footnote_1'>1</a> : seed 설정  
## 2. 
Settings : `embedding type = original`, `model_type = deepfm`, `num_epochs = 100`, `data=ml100k`  
|metrics| value| value<sup>[1](#footnote_1)</sup>|
|---|---|---|
|precision|0.4380733944954128 |0.4330275229357798 |
|recall   |0.0789560314670283 |0.0772224272583683 |
|hit rate |0.8577981651376146 |0.8577981651376146 |
|rec rank |0.6675076452599388 |0.6479740061162079 |
|dcg      |1.3560263674407254 |1.3304601933264755 |
|time     |109.45841121673584 |103.32878303527832 |
---
<a name='footnote_1'>1</a> : seed 설정  
## 3.
Settings : `embedding type = SVD`, `model_type = fm`, `num_epochs = 100`, `data=ml100k`  
|metrics  | value|value<sup>[1](#footnote_1)</sup>|value<sup>[2](#footnote_2)</sup>|
|---|---|---|---|
|precision|0.2183486238532110 |0.1889908256880734 |0.198165137614678 |
|recall   |0.0291074901242037 |0.0256586943649130 |0.026636492888923 |
|hit rate |0.5688073394495413 |0.5619266055045872 |0.550458715596330 |
|rec rank |0.4031727828746177 |0.3949923547400611 |0.371292048929663 |
|dcg      |0.7048165887971195 |0.6305811680068265 |0.631741929379093 |
|time     |108.48559141159058 |146.82375121116638 |104.9781436920166 | 
---
<a name='footnote_1'>1</a> : preprocessor 일부 수정  
<a name='footnote_2'>2</a> : seed 설정
## 4.
Settings : `embedding type = SVD`, `model_type = deepfm`, `num_epochs = 100`, `data=ml100k`  
|metrics| value| value<sup>[1](#footnote_1)</sup>|value<sup>[2](#footnote_2)</sup>|
|---    |---   |---|---|
|precision|0.2724770642201835 |0.5114678899082569 |0.5279816513761467 |
|recall   |0.0352602390920223 |0.0917484469764606 |0.0967468529122231 |
|hit rate |0.6100917431192661 |0.9174311926605505 |0.9197247706422018 |
|rec rank |0.4261085626911315 |0.7372324159021407 |0.7345565749235473 |
|dcg      |0.8427053911790566 |1.5751689006246197 |1.6030452903462624 |
|time     |118.83392763137817 |119.92192196846008 |120.98636102676392 |
---
<a name='footnote_1'>1</a> : ui_matrix 순서 바꿈
<a name='footnote_2'>2</a> : seed 설정 

## 5.
Settings : `embedding type = SVD`, `model_type = fm`, `num_epochs = 100`, `data=frappe`  
|metrics  | value|
|---|---|
|precision| |
|recall   | |
|hit rate | |
|rec rank | |
|dcg      | |
|time     | |