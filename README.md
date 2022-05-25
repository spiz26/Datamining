# Datamining Term Project

## Title. ‘한국교육종단연구’를 활용한 중학생 학습 성취도 예측 요인 분석 

### 1.Introduction

한국의 청소년들의 학업 부담은 상당한 수준이다. 청소년기에 받는 학업 스트레스는  
성장과정에서 강한 어려움을 겪게 하며 이에 따른 학교 생활 만족도도 OECD 평균보다 낮게 형성되고 있다.  
  
따라서 학생들의 교육 부담을 줄여 주기 위해 학생들의 인지적 요인에 많은 영향을 끼치는 요인을 분석하고 인과 관계를 규명하는 것은 매우 중요하다.  
  
복잡한 교육 현실의 인과관계 분석을 통해 현재 학교교육의 현상을 파악하고 동시에 앞으로의 교육 변화를 예측하여 그에 맞는 교육 계획을 수립에 도움을 주고자 한다.  
  
따라서 학생을 위한 효과적인 교육 정책 연구를 포함한 다양한 연구 진행 활성화를 위해 본 연구를 시작하였으며 학생의 지적,정의적, 사회-문화적 성장에 영향을 미치는 요인들을 분석하여 아동-청소년들을 위한 효과적인 교육정책을 제언하고자 한다.  

### 2. DATA  
#### 2-1. 데이터 출처  
* 한국교육개발원 - 한국교육종단연구2005(이하 KELS2005) 교육용 데이터 : 학생들의 교육적 경험과 성취에 대한 종단자료 구축  
<pre><code>
datasets
|----KELS2005 교육용 데이터
|     |--- Y1_SCH_EDU.csv
|     |___ Y1_STD_EDU.csv
|     |___ Y2_SCH_EDU.csv
|     |___ Y2_STD_EDU.csv
|     |___ Y3_SCH_EDU.csv
|     |___ Y3_STD_EDU.csv
|
|____|--- Y1_SCH_EDU.sav
      |___ Y1_STD_EDU.sav
      |___ Y2_STD_EDU.sav
      |___ Y2_STD_EDU.sav
      |___ Y3_SCH_EDU.sav
      |___ Y3_STD_EDU.sav
</pre></code>
  
#### 2-2. 데이터 표본  
	* 한국교육종단연구2005의 조사를 위하여 2005년 전국의 중학교 학생으로부터 표본 추출.  
	* 학생 표본은 체육 중학교와 분교를 제외한 전국의 2,929개 중학교에 재학하고 있는 1학년 학생 703,914명으로부터 6,908명 추출  
	* 주요 조사 내용 중 학생의 인지적 성취, 비인지적 성취를 활용  
- 인지적 성취 : 국어, 영어, 수학 성취도  
- 비인지적 성취 : 자기조절 학습능력, 자기개념, 가치관, 교육만족도 => 중학교 1학년부터 3학년까지 학년별 성취도 평가를 시행한 것을 바탕으로 활용  
	* 학교 데이터는 제외, 학생 데이터(Y_STD_EDU) 이용 - 학생이 응답한 설문지의 값(SID, SCHID, GENDER를 제외한 58개 변수)을 column으로 변환  
	* 3개년도 중 3학년을 제외한 1,2학년(Y1,Y2) 학생 데이터를 합쳐서 사용  
	* Y = 국어, 영어, 수학의 전체 평균성적  
  
#### 2-3. csv 데이터  
![image](https://user-images.githubusercontent.com/71483926/170243616-ee4e831a-78e4-4813-8508-80bc2b427b19.png)  
  
![image](https://user-images.githubusercontent.com/71483926/170243650-43447ce1-513f-474f-8359-021e0c6c2a5a.png)  

### 3. 데이터 전처리 및 데이터 탐색  
#### 3-1. 전처리  
 설문조사 문항 응답값의 결측치 제거  
 데이터 Scaling 기법인 StandardScaler와 MinMaxScaler 데이터에 적용해 보았을 때, 더 높은 성능을 가진 MinMaxScaler로 normalization.  

#### 3-2. 시각화(heatmap)  
1) 국영수 원점수, 척도점수와 변수 간의 상관계수  
![image](https://user-images.githubusercontent.com/71483926/170243989-8e8f0470-231a-4964-9374-ca9639d704b1.png)  

* 색이 연할수록 상관계수가 높은 것  
* 4-1,4-2,4-3측과 1-24,1-27측에서 목적과 상관관계가 상대적으로 높은 것으로 보인다.  
* 척도점수에서는 상관계수가 크게 작용하지 않는 것으로 확인 되었다.  
  
2) 평균점수와 변수 간의 상관계수  
![image](https://user-images.githubusercontent.com/71483926/170244184-29b6b3b3-7106-4558-a04d-91c0e272dc44.png)  
* 4-1,4-2,4-3측과 1-24,1-27측에서 상관관계가 상대적으로 높은 것으로 보인다.  
* 평균으로 보니 더욱 직관적으로 변한 것을 확인할 수 있다.  
  
### 4. Evaluation  
Training data를 이용하여 중학 학업 성취도에 영향을 미치는 요인을 선별해내는 것이 목표  
-> MAE, MSE, RMSE, RMSEL 중 제곱 된 에러를 다시 루트로 풀어, 데이터 셋이 큰 경우에 연산 속도가 느려지는 것을 방지하고 왜곡을 최소화하는 방식인 RMSE(평균 제곱근 오차)를 지표로 활용  
  
### 5. Modeling  
 4가지 분류기법(Linear Regression, 신경망, Decision Tree, Random Forest) 사용  
   
 #### 1) Decision Tree  
  
• Overfitting 방지 하기위해  hyperparameter : max_depth=6 설정  
  
• Decision Tree RMSE 결과  
- DT train RMSE : 15.3185  
- DT test RMSE : 15.9537  
    => 다른 알고리즘(LR, NN)과 비슷한 결과  
  
#### 2) Random Forest  
  
overfitting 방지 하기위해 hyperparameter 2종류를 tuning하여 최적의 hyperparameter 값을 찾아 제약한 상태에서 fitting  
 1. min_samples_split = 4  
 2. max_depth = 6  
Random Forest RMSE 결과  
RF train RMSE : 14.796034479862131  
RF test RMSE : 15.840798690462156  
    => 다른 알고리즘 모델 방법론과 비교했을 때, test set RMSE값이 가장 낮음  
  
#### 3) Linear Regression  
  
• label이 continuous하므로 제일 기초적인 linear regression모델 사용  
• Linear Regression RMSE 결과  
- LR train RMSE : 15.7946  
     => 국어, 영어, 수학 점수 나누기 3을 했을 때, 평균과 비교해서 약 15점 차이가 난다는 것을 확인할 수 있음.   
- LR test RMSE : 16.1208  
• Result : train loss와 test loss가 거의 같기 때문에 overfitting되지 않고 잘 나온 것을 확인.  
  
  
#### 4) Neural Network  
  
• 다른 종류의 머신러닝 알고리즘보다 DNN의 성능이 좋을 것으로 예상하고 학습 진행  
• hidden layer는 3층, activation은 ReLU, optimizer는 Adam을 사용  
• input layer의 node 수는 feature개수입니다.  
  
• 신경망 RMSE 결과  
- NN test RMSE : 16.437  
• Result : 예상과 다르게 다른 알고리즘(LR, NN)과 거의 비슷한 결과가 나온 것을 확인  
• layer의 수를 늘려 보고, Batch Normalization layer도 써봤으나 오히려 RMSE값이 증가  
  
#### Result  
Model |	RMSE  
--|--  
Decision Tree | 16.367  
Random Forest |	15.840  
Linear Regression	| 16.403  
Neural Network |	15.562  
DNN bagging | 15.412  

앙상블 기법을 활용한 Random Forest의 성능이 제일 높게 형성됨  







