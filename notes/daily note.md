##### 24.11.22
To Do
* 비교가 용이하도록 optuna를 implement할 것  
-> parameter로 SVD, NMF. sparse 여부 등등을 넣어서 비교  

Been Done
* SVD와 NMF의 결과값들 (user_embedding, item_embedding) 중 작은 값들을 0으로 처리 (if `sparse==True`)