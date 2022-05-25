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
  
* overfitting 방지 하기위해 hyperparameter 2종류를 tuning하여 최적의 hyperparameter 값을 찾아 제약한 상태에서 fitting  
 1) min_samples_split = 4  
 2) max_depth = 6  
* Random Forest RMSE 결과  
RF train RMSE : 14.796034479862131  
RF test RMSE : 15.840798690462156  
    => 다른 알고리즘 모델 방법론과 비교했을 때, test set RMSE값이 가장 낮음  
  
#### 3) Linear Regression  
  
* label이 continuous하므로 제일 기초적인 linear regression모델 사용  
* Linear Regression RMSE 결과  
LR train RMSE : 15.7946  
     => 국어, 영어, 수학 점수 나누기 3을 했을 때, 평균과 비교해서 약 15점 차이가 난다는 것을 확인할 수 있음.   
LR test RMSE : 16.1208  
* Result : train loss와 test loss가 거의 같기 때문에 overfitting되지 않고 잘 나온 것을 확인.  
  
  
#### 4) Neural Network  
  
* 다른 종류의 머신러닝 알고리즘보다 DNN의 성능이 좋을 것으로 예상하고 학습 진행  
* hidden layer는 3층, activation은 ReLU, optimizer는 Adam을 사용  
* input layer의 node 수는 feature개수
  
* 신경망 RMSE 결과  
- NN test RMSE : 16.437  
* Result : 예상과 다르게 다른 알고리즘(LR, NN)과 거의 비슷한 결과가 나온 것을 확인  
* layer의 수를 늘려 보고, Batch Normalization layer도 써봤으나 오히려 RMSE값이 증가  
  
### Result  
Model |	RMSE  
--|--  
Decision Tree | 16.367  
Random Forest |	15.840  
Linear Regression	| 16.403  
Neural Network |	15.562  
DNN bagging | 15.412  

앙상블 기법을 활용한 Random Forest의 성능이 제일 높게 형성됨  

### 6. 모델성능 향상 - Filter method, PCA and Bagging  
1.Feature Selection기법 중 하나인 Filter method를 활용을 활용하여 상관관계가 높은 변수 또는 성능이 높은 변수를 추출하여 모델 성능 개선시도  
2.데이터의 속성이 증가함에 따라 모델 성능에 나쁜 영향을 미쳐 학습에 정확도가 낮아 지는 것을 막기 위해   PCA (Principal component analysis)를 활용하여 feature를 30개로 축소 시도  
3.앙상블 기법 중 하나인 Bagging을 통해 Neural Network의 성능향상 시도  

#### 6-1. Retrain machine learning models with selected features  

• 52개의 속성이 너무 많은 것 같아 dimension reduction을 위해 filter method로 feature를 추출, filter method로 뽑힌 중요한 속성(20개)으로 위의 모델들을 다시 학습.
	
	
<20개 변수중 15개를 시각화한 표>  
  
![image](https://user-images.githubusercontent.com/71483926/170246773-38effeaa-9074-4c29-819f-2fe2190a52b0.png)  

### Result

LR train RMSE : 16.342650771262285
LR test RMSE : 16.858429581872567

DT train RMSE : 15.402194436195773
DT test RMSE : 16.324405928390007

RF train RMSE : 15.120813519808314
RF test RMSE : 16.026251972349637

#### 6-2. Retrain machine learning models with extracted features  
dimension reduction을 위해 PCA로 feature를 추출, PCA로 생성된 feature 30개로 위의 모델들을 다시 학습  

Result

LR train RMSE : 15.825035383328146
LR test RMSE : 16.458706870694744

DT train RMSE : 15.602696439106433
DT test RMSE : 17.15371808022978

RF train RMSE : 14.897265142093849
RF test RMSE : 16.26775673541663

DNN RMSE : 16.603415638975537

### 6-3. Retrain machine learning models with Bagging

* DNN의 성능을 강제로 끌어올리기 위해 bagging을 사용  
* 데이터셋을 10번 복원추출 후, 추출된 데이터셋으로 10개의 같은 DNN모델을 학습. 학습 완료 후, test mse값이 일정이상 넘지 못한 열등한 모델은 제외하고, threshold값을 넘은 우수한 모델들로만 bagging모델을 구성  
* 우수한 모델이 regression한 값의 평균으로 evaluation 진행  

### Result

DNN RMSE : 15.882743847097386  

### 7. 성능 향상 결과  
  
  Modeling | RMSE | RMSE 변화
Filter Decision Tree | 16.324 | -0.003
Filter Random Forest | 16.026 | +0.186
Filter Linear Regression | 16.848 | +0.445
PCA Decision Tree | 17.172 | +0.805
PCA Random Forest | 16.250 | +0.41
PCA Linear Regression | 16.455 | +0.025
PCA Neural Network | 16.603 | +0.166
Bagging Neural Network | 15.882 | -0.555

Filter method를 활용한 Decision Tree와 Bagging을 사용한 Neural Network 만성능이 향상되었으며 나머지 모델은 오히려 성능이 하락함  
앙상블 기법을 사용할 시 모델의 성능이 유의미 하게 향상 될 수 있음 을 발견  

### 8. 결과 시각화 및 Feature Importance 
  
#### 8-1. Heatmap
 ##### 1) Decision Tree
DT에서 나온 변수와 목적, 변수와 변수 서로의 상관계수  

![image](https://user-images.githubusercontent.com/71483926/170247464-a49329bf-a273-44c0-80cf-b3b01f694c70.png)  

  
* DT에서 구한 변수와 평균점수를 상관관계를 보니 4-2가 가장 목적과 상관관계가 존재  
* 변수간에는 4-2와 4-3간에 상관계수가 높음  
  
 ##### 2) Random Forest 
RF에서 나온 변수와 목적, 변수와 변수 서로의 상관계수  
  
![image](https://user-images.githubusercontent.com/71483926/170247661-0815f36d-b8df-4ada-98a6-4f3680e4c028.png)  
  
* RF에서도 구한 변수와 평균점수를 상관관계를 보니 4-2와 상관관계가 존재  
* 변수간에는 4-2와 4-3간에 상관계수가 높음  


 ##### 3) Filter method  
Filter에서 나온 변수와 목적, 변수와 변수 서로의 상관계수  

![image](https://user-images.githubusercontent.com/71483926/170247768-ff24a3bf-54bc-4cea-bac6-cfbc86a3828e.png)  
  
* Filter에서 구한 변수와 평균점수를 상관관계를 보니 4-2가 상관관계가 존재  
* 변수 간에는 앞의 숫자인 3끼리와 4끼리 상관계수가 높음  

#### 8-2. Feature Importance  
 ##### 1) Decision Tree  
 ![image](https://user-images.githubusercontent.com/71483926/170248022-5bcea531-1bb8-41ac-8a2f-e1ece3ddffaf.png)  

* 각 feature importance에 100을 곱하여 각 feature가 label에 얼만큼 기여하는지 바로 알 수 있게 그래프 생성  
* 기여도가 0.5%이하는 drop시키고 중요한 feature들만 남김  
* Result : Decision Tree에서는 4-2, 4-3, 1-27, 1-21, 1-24 등의 순서로 변수 중요도가 나타남  

 ##### 2) Random Forest  
 ![image](https://user-images.githubusercontent.com/71483926/170248196-51b26446-2996-48a0-be11-691e93fe5f31.png)  
  
* 기여도 2.5%이하는 drop시키고 중요한 feature들만 남김  
* Random Forest에서는 4-2, 4-3, 1-27 1-24 순서로 변수 중요도가 나타남  

 ##### 3) Filter method  
 ![image](https://user-images.githubusercontent.com/71483926/170248317-b6cef111-7b06-4fe5-8c7d-040f0613b98f.png)  
 
 * 머신러닝 알고리즘 Embedded  method가 아닌 filter method로 feature importance 추출  
 * 기여도가 3%이하인 feature들은 drop  
 * Regression이므로 f_regression을 활용.  
 * 변수 20개 적용, filter method에서 20개 중 15개를 시각화  
 * Linear Regression에서는 4-2, 4-3, 1-24, 1-27, 3-14, 3-11, 3-12 등의 순서로 변수 중요도가 나타남  

### 9. Descriptive Result

Model/feature | 1위 | 2위 | 3위 | 4위 | 5위
--|--|--|--|--|--  
Decision Tree | 수업이해정도_영어 | 수업이해정도_수학 | 사회 자기 개념_친구들은 나를 중요한 존재라고 생각함 | 신체 자기 개념_운동 신경 발달 | 학업 자기 개념_다른 친구들 보다 공부를 잘 할 수 있음
Random Forest | 수업이해정도_영어 | 수업이해정도_수학 | 학업 자기 개념_대학 과정을 마칠 능력 있음 | 학업 자기 개념_다른 친구들 보다 공부를 잘 할 수 있음 | 사회 자기 개념_친구들은 나를 중요한 존재라고 생각함
Filter Method | 수업이해정도_영어 | 수업이해정도_수학 | 학업 자기 개념_다른 친구들보다 공부 잘 할 수 있음 | 자기 효능감_수학(시험) | 자기 효능감_수학(교과서 어려운 내용)

3가지 모델링을 통해 총 7가지의 예측요인을 선택함   
  
  ① 수업이해정도_ 영어  
  ② 수업이해정도_수학  
  ③ 사회 자기 개념_친구들은 나를 중요한 존재라고 생각함  
  ④ 학업 자기 개념_대학 과정을 마칠 능력 있음  
  ⑤ 학업 자기 개념_다른 친구들보다 공부를 잘 할 수 있음  
  ⑥ 신체 자기 개념_운동 신경 발달  
  ⑦ 자기 효능감_ 수학(교과서 어려운 내용)
  
  #### Descriptive Result 해석 
* 학생 스스로가 영어, 수학에 대한 수업 이해도를 높다고 느끼는 경우 학업 성취도가 높은 경향이 있음을 확인
추가적으로 학생 스스로가 본인의 전반적인 학습 능력에 대한 자신감이 높고 사회 구성원으로부터 중요한 사람이라고 생각하는 경우 학업 성취도에 긍정적인 영향을 줄 수 있다는 결론을 낼 수 있음  

### 10. Discussion
#### 1) 객관적 지표 제공
	* 중등교육분석을 바탕으로 상급학교 진학에 대한 기초자료로 활용 가능 

	1. 효과적인 교육 정책 개선을 위한 동기 유발
	* 데이터 마이닝 알고리즘을 활용해 앞으로의 다양한 교육학 연구의 발전 및 활성화에 이바지
	* 더 효과적인 교육연구를 위한 교육 데이터 생태계 조성 재고 
	
	2. 향후 활용 방안
	* 한국 중등교육 실태 및 수준의 분석을 통한 앞으로의 교육정책 수립을 위한 지표로써 활용
	* 청소년기 학습에 대한 수업 이해정도를 꾸준하게 체크 할 수 있는 시스템 확충 재고
	* 학생들이 학습에 대한 자신감을 얻을 수 있는 교내 활동 프로젝트를 진행 해야함.  
	  
### 11. Limitation  
  
	• 데이터양의 한계
	- 더 많은 데이터들을 가지고 있었다면 효과적이고 정확한 분석을 진행 할 수 있었을 것 
	- 실제로 본 연구에서 활용한 2005년 종단연구는 중학생에 대한 데이터 뿐만 아니라 
    고교 및 대학생에 대한 데이터도 포함하고 있었으나 개방되어 있지 않아 활용할 수 없었음
	• 데이터 해석의 한계
	- 분석 결과에서 영어,수학 학습 성취도가 학업 성취도에 영향을 미치는 가장 큰 요인으로 확인 되었으나 국어 학습 성취도는 학업 성취도에 강한 영향을 주지 못했음
	- 이에 대한 원인을 파악해 보려 했으나 마땅한 추론 결과를 얻지 못함 

	• 모델의 정확성
	- RMSE를 줄이기 위해 다양한 method를 활용하였으나 오히려 RMSE가 증가함 
차후 더 다양한 분석을 통해 정확성이 높은 모델을 얻길 희망 ![image](https://user-images.githubusercontent.com/71483926/170249124-fe0dfe0f-5a45-4c45-aa08-79b8d552b8f0.png)
