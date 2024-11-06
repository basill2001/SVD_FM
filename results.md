## 1. 
Settings : `embedding type = original`, `model_type = fm`, `num_epochs = 100`, `data=ml100k`  
|metrics| value|
|---|---|
|precision|0.29541284403669726      |
|recall   |0.052328232882711226     |
|hit rate |0.7545871559633027       |
|rec rank |0.5081039755351682       |
|dcg      |0.914304604489501        |
|time     |94.00355195999146        |
## 2. 
Settings : `embedding type = original`, `model_type = deepfm`, `num_epochs = 100`, `data=ml100k`  
|metrics| value|
|---|---|
|precision|0.4380733944954128   | 
|recall   |0.0789560314670283   |
|hit rate |0.8577981651376146   |
|rec rank |0.6675076452599388   |
|dcg      |1.3560263674407254   |
|time     |109.45841121673584   |
## 3.
Settings : `embedding type = SVD`, `model_type = fm`, `num_epochs = 100`, `data=ml100k`  
|metrics| value|
|---|---|
|precision|0.21834862385321102   |
|recall   |0.029107490124203706  |
|hit rate |0.5688073394495413    |
|rec rank |0.4031727828746177    |
|dcg      |0.7048165887971195    |
|time     |108.48559141159058    |
## 4.
Settings : `embedding type = SVD`, `model_type = deepfm`, `num_epochs = 100`, `data=ml100k`  
|metrics| value|
|---|---|
|precision|0.2724770642201835   |
|recall   |0.03526023909202231  |
|hit rate |0.6100917431192661   |
|rec rank |0.4261085626911315   |
|dcg      |0.8427053911790566   |
|time     |118.83392763137817   |