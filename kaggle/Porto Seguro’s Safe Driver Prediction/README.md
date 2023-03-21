# Porto Seguro’s Safe Driver Prediction
---
# 결과
---
### 요약 정보
* 도전기관 : 한양대학교
* 도전자 : 주조령
* 최종 스코어 : 0.28778
* 제출 일자 : 2023-03-21
* 총 참여 팀수 : 5156
* 순위 및 비율 : 1775 (34.4%)

# 결과 화면
---
![final_rank_and_score](./img/Leaderboard_Score.JPG)

# 사용한 방법 & 알고리즘
---
* 1. 무작위 부족 샘플링 방법을 사용하여 데이터를 균형 있게 처리하여 범주 불균형 문제가 모델 훈련에 미치는 영향을 방지합니다.

* 2. 데이터는 파이썬의 Pandas 및 NumPy 라이브러리를 사용하여 처리 및 변환되었고 데이터 세트는 sklearn 라이브러리의 train_test_split 방법을 사용하여 분할되었으며 XGBoost 라이브러리의 XGBClassifier 방법을 사용하여 모델링되었습니다.

* 3. max_depth, learning_rate, n_estimators, min_child_weight, 감마 및 기타 매개변수의 조정을 포함하여 XGBoost 모델의 매개변수를 최적화하여 더 나은 모델 성능을 얻습니다.

* 4. 교차 검증 방법을 사용하여 모델을 평가하고 모델의 정확도, 회수율 및 F1 점수와 같은 지표를 출력했습니다. 

# 코드
---
[jupyter notebook code](Porto Seguro’s Safe Driver Prediction.ipynb)

# 참고자료
---
### https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
### https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283
```python

```