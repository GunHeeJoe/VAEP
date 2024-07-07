# VAEP(Valuing Actions by Estimating Probabilities)

- 참고 github link : https://github.com/ML-KULeuven/socceraction

Introduction
----------------
-An indicator that uses data called StatsBomb to evaluate the value of all actions : VAEP<br/>
-VAEP is calculated as the difference between the change in scoring probability and the change in conceding probability after predicting the score probability and the concede probability of each action through machine learning.<br/>
-feature : action type, action result, position, dist to goal, angle to goal, time, etc<br/>
-label : score_label=1 if you score within 10 action after each action, and concede_label=1 if you concede within 10 action <br/><br/>

-Previously, machine learning techniques such as boosting, random-forest, and logistic were used to predict the score probability and the concede probability of each action<br/>
-In this study, Deep learning is used to predict the probability of scoring and conceding<br/><br/>
i) Before modeling, Data loading, extraction, and preprocessing are performed through 1, 2, and 3.<br/>
ii)Binary classification through existing machine learning uses a scoring probability model and a conceding probability model<br/>
iii) In deep learning, it should be implemented according to characteristics different from machine learning, not just changing the model.<br/>
-Embedding : There are numerical and categorical attributes in soccer data<br/>
-Data imbalance : Deep learning does not solve the class imbalance problem on its own<br/>
-Model : Various classification models<br/>

Function
----------------
<notebook>
- Due to GitHub's file size limits, we are unable to load the data directly on the repository. Therefore, I will follow the steps outlined in the notebook to obtain the data.
  
1. load-and-convert-data.ipynb : StatsBomb data loading<br/>
i) In this study, LaLiga data will be used<br/>
ii) game_id = 3,773,689 has data of score_label=1, concede_label=1. This is incorrect data and will remove the game data<br/><br/>

2. storing-features-and-labels.ipynb : Define SPADL, Feature, and Label for Train, Valid, and Test data<br/>
i) train : 2004/2005 ~ 2018/2019 season<br/>
ii) valid : 2019/2020 season<br/>
iii) test : 2020/2021 season<br/><br/>

3. data-preprocess.ipynb : soccer data preprocessing<br/>
i) Error data preproceing<br/>
ii) Create additional features<br/>
iii) Create labels for multi-classification<br/><br/>

Modeling & Analysis & Evaluation<br/>
----------------

<ML_VAEP>

- ML_BinaryClassification : Machine learning to perform binary-classification<br/>
i) Using the same dataset, CatBoost is used to create a scoring probability model and a concede probability model, respectively<br/>
ii) Calculate the VAEP using the probability of scoring and the conceding<br/>
iii) Quantitative & Qualitative Indicators<br/>
-Qualitative indicators will evaluate the play followed by Valverde's actions, Vasquez's cross, and finally Benzema's shot, starting with Tony Cross's pass at 18:36 of the link below.<br/>
link : https://www.youtube.com/watch?v=EhodpjwTtag&t=1986s<br/><br/>

#### Analysis 
- 골과 어시스트를 담당한 L. Vazquez와 K. Benzema선수의 이벤트(cross, shot)가 높은 VAEP 값을 보였습니다.
- 정말 그럴까요? 실제 영상을 보면, 이 골에 큰 기여를 한 선수는 F. Valverde입니다.
- 이는 ML(Catboost) 기반 알고리즘을 사용하여 VAEP를 추출할 때, 골의 마지막 이벤트인 슛팅에만 높은 가치를 부여하는 경향이 있음을 나타냅니다.
- 골을 기준으로 라벨링을 진행하다 보니, 골로 이어지는 이벤트의 기여도를 평가할 때 이러한 현상이 발생하는 것으로 보입니다.
![image](https://github.com/GunHeeJoe/VAEP/assets/112679136/b498d9b2-c257-4747-b4d7-1d30d3bdd977)

----------------
<DL_VAEP>
  
- DL_Classification<br/>
i) Deeplearning creates binary-classifications used in previous study and multi-classification proposed in this study<br/>
ii) Oversampling is performed to solve the class imbalance. The oversampling technique proceeds by extracting data equally at the ratio of each label for each batch<br/>
iii) Calculate the VAEP using the probability of scoring and the conceding<br/>
iv) Quantitative & Qualitative Indicators<br/>
vi) In this study, torchfm/model/Upgrade_Dcn.py & torchfm/layer.py was used. There are many other models and embedding techniques, so please refer to them<br/>
-https://github.com/kitayama1234/Pytorch-Entity-Embedding-DNN-Regressor/blob/master/model.py<br/>
-https://github.com/rixwew/pytorch-fm/tree/master/torchfm<br/><br/>

#### Analysis 
- 골과 어시스트를 담당한 L. Vazquez와 K. Benzema가 아니라 F. Valverde의 이벤트(돌파, 드리블, 패스)가 높은 VAEP 값을 보였습니다.
- ML이 골의 마지막 이벤트인 슛팅에 높은 가치를 부여한다면, DL은 슛팅보다는 공격 전개 과정에서 발생하는 중간 이벤트에 더 높은 가치를 부여하는 경향이 있습니다.
- 실제 경기를 관찰해 보면, 부족한 선수는 없었습니다. F. Valverde, L. Vazquez, K. Benzema 모두 훌륭한 활약을 했습니다.
- 하지만 이벤트 데이터를 기반으로 정밀한 분석을 수행한다면, DL이 제공하는 정성적 가치(VAEP)가 더 높다고 생각하지 않으십니까?
- 정성적 평가는 사람마다 해석이 다를 수 있지만, 이벤트 데이터를 기준으로 볼 때 DL이 플레이의 가치를 더 정확하게 분석하는 것으로 보입니다.
![image](https://github.com/GunHeeJoe/VAEP/assets/112679136/db9e638a-3d57-4c31-add6-15dae4d998d9)

Conclusion
----------------
i) Expressions for Deep Learning to Understand Soccer Data<br/>
ii) Multiple classifications and oversampling to solve the class imbalances<br/>
iii) Quantitatively, It not only showed performance improvement over existing boosting algorithms, but also verified that it is more convincing in qualitative indicators<br/>
