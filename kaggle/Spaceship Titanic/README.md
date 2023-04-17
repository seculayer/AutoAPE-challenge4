# Spaceship Titanic
---
# 결과
---
### 요약 정보
* 도전기관 : 한양대학교
* 도전자 : 주조령
* 최종 스코어 : 0.80126
* 제출 일자 : 2023-03-27
* 총 참여 팀수 : 2355
* 순위 및 비율 : 601 (25.5%)

# 결과 화면
---
![final_rank_and_score](./img/Leaderboard_Score.JPG)

# 사용한 방법 & 알고리즘
---
* 1. 심층 신경망 모델:
    - CNN(Convolutional Neural Network)은 이미지 특징 추출을 위한 기본 모델로 사용된다.
    - 상대적으로 데이터 셋이 작기 때문에 convolutional layer 3개와 full connected layer 2개만 사용한다.

* 2. 데이터 전처리:
    - 데이터 다양성과 모델 견고성을 높이기 위해 이미지의 크기를 조정하고, 자르고, 뒤집다.
    - 이미지를 텐서로 변환하기 위해 torchvision.transforms를 사용하여 데이터를 사전 처리한다.

* 3. 손실 함수 및 옵티마이저:
    - 모델의 목적함수는 교차 엔트로피 손실 함수(Cross-Entropy Loss)를 사용한다.
    - 옵티마이저는 확률적 경사하강법 최적화기(Stochastic Gradient Descent, SGD)를 사용한다.

* 4. 교육 및 테스트:
    - 훈련 세트에서 모델을 훈련하고 테스트 세트에서 모델 성능을 평가했다.
    - Batch Training 및 Stochastic Gradient Descent(SGD) 및 기타 기술을 채택하여 모델의 학습 효율성과 정확도를 향상시킨다.

* 5. 모델 배포:
    - 생산용으로 TorchScript를 사용하여 모델을 컴파일했다.
    - 모델을 로드하고 예측하는 기능을 정의한다.

# 코드
---
[jupyter notebook code](Spaceship Titanic.ipynb)

# 참고자료
---
##### https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb
##### https://github.com/minsuk-heo/kaggle-titanic/blob/master/titanic-solution.ipynb
##### https://www.kaggle.com/code/odins0n/spaceship-titanic-eda-27-different-models
##### https://www.kaggle.com/code/eisgandar/spaceship-titanic-eda-xgboost-80
##### https://www.kaggle.com/code/georgyzubkov/spaceship-eda-catboost-with-optuna
```python

```