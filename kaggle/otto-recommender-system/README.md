# OTTO - Multi-Objective Recommender System
---
# 결과
---
### 요약 정보
* 도전기관 : 시큐레이어
* 도전자 : 박상우
* 최종 스코어 : 0.57815
* 제출 일자 : 2023-01-30
* 총 참여 팀수 : 2575
* 순위 및 비율 : 372 (14.4%)

# 결과 화면
---
![final_rank_and_score](./img/rank_score.JPG)

# 사용한 방법 & 알고리즘
---
* Markov Chain, Matrix Factorization을 weighted하게 학습
* Handcraft Rule을 통해 Candidate를 추출하고 XGBoost Ranker 학습 
* 다양한 Feature를 Generation하여 모델 고도화

# 코드
---
[python code](main.ipynb)

# 참고자료
---
##### https://www.kaggle.com/competitions/otto-recommender-system/overview
##### https://medium.com/predictly-on-tech/learning-to-rank-using-xgboost-83de0166229d/
