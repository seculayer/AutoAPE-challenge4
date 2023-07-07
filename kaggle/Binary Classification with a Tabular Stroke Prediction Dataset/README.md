# Binary Classification with a Tabular Stroke Prediction Dataset
## 결과
### 요약 정보
- 도전기관 : SecuLayer
- 도전자 : 김소영
- 최종 스코어 : 0.89693
- 제출 일자 : 2023-07-05
- 총 참여 팀수 : 770
- 순위 및 비율 : 82(10.6%)

## 결과 화면
![score](./img/score.png)
![rank](./img/rank.png)

## 사용한 방법 & 알고리즘
- Step 1. 데이터 전처리
  - CatBoost 모델: 이상치 제거, 범주형변수(원-핫 인코딩)
  - NN 모델: 범주형변수(원-핫 인코딩), 수치형변수(StandardScaler)
- Step 2. CatBoost, NN - 앙상블

## 코드
- Binary_Classification_with_a_Tabular_Stroke_Prediction_Dataset.ipynb