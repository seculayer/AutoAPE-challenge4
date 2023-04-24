# Scrabble Player Rating
---
# 결과
---
### 요약 정보
* 도전기관 : 한양대학교
* 도전자 : 주조령
* 최종 스코어 : 104.61351
* 제출 일자 : 2023-04-24
* 총 참여 팀수 : 301
* 순위 및 비율 : 74 (29.8%)

# 결과 화면
---
![final_rank_and_score](./img/Leaderboard_Score.JPG)

# 사용한 방법 & 알고리즘
---
* 1. Q-러닝 알고리즘: Q-러닝 알고리즘은 AI 에이전트가 가장 높은 점수를 얻기 위해 최적의 알파벳 조합을 선택하도록 훈련하는 데 사용됩니다.

* 2. 가치 함수: Q-Learning 알고리즘에서 가치 함수는 가능한 각 행동의 가치를 평가하는 데 사용됩니다.이 프로젝트에서 가치 함수는 최적의 문자 조합을 선택하기 위해 가능한 각 문자 조합의 점수를 평가하는 데 사용됩니다.

* 3. 알파-베타 가지치기 알고리즘: 알파-베타 가지치기 알고리즘은 AI 에이전트가 최적의 결정을 내릴 수 있도록 가능한 모든 상대 행동을 검색하는 데 사용됩니다.

* 4. 이중 신경망: 이 프로젝트에서 가능한 각 문자 조합의 점수를 추정하기 위해 이중 신경망이 사용되었습니다.네트워크는 두 개의 레이어, 입력 레이어 및 출력 레이어를 포함하며, 여기서 입력 레이어는 문자 조합을 수락하고 이를 벡터 표현으로 변환하고 출력 레이어는 점수를 출력합니다.이 신경망은 역 전파 알고리즘을 통해 훈련됩니다.

* 5. 몬테카를로 트리 검색 알고리즘: 몬테카를로 트리 검색 알고리즘은 여러 게임을 시뮬레이션하여 가능한 각 행동의 점수를 평가하고 최적의 행동을 선택하는 게임 트리 검색을 위한 알고리즘입니다.이 프로젝트에서 몬테카를로 트리 검색 알고리즘은 상대방의 가능한 모든 행동을 검색하고 각 행동의 점수를 평가하는 데 사용됩니다.

* 6. 시뮬레이션 어닐링 알고리즘: 이 프로젝트에서 시뮬레이션 어닐링 알고리즘은 가장 높은 점수를 얻기 위해 문자 조합의 공간에서 최적의 문자 조합을 검색하는 데 사용됩니다.

* 7. 일반화 정책 반복 알고리즘: 일반화 정책 반복 알고리즘은 알려지지 않은 환경에서 최상의 전략을 학습하는 데 사용되는 강화 학습 알고리즘입니다.이 프로젝트에서 일반화된 정책 반복 알고리즘을 사용하여 AI 에이전트가 가장 높은 점수를 얻기 위해 최적의 문자 조합을 선택하도록 훈련했습니다.

* 8. Beam 검색 알고리즘: Beam 검색 알고리즘은 검색 공간에서 최적의 솔루션을 찾는 데 사용되는 휴리스틱 검색 알고리즘입니다.이 프로젝트에서 Beam 검색 알고리즘은 최적의 문자 조합을 선택하기 위해 가능한 모든 문자 조합을 검색하는 데 사용됩니다.


# 코드
---
[jupyter notebook code](Scrabble Player Rating.ipynb)

# 참고자료
---
##### https://www.kaggle.com/code/rtatman/data-cleaning-challenge-handling-missing-values/notebook
##### https://www.kaggle.com/code/ijcrook/full-walkthrough-eda-fe-model-tuning
##### https://www.kaggle.com/code/mtkc01245/scrabble-lightgbm-simple-tune-japanese
##### https://www.kaggle.com/code/hasanbasriakcay/scrabble-eda-fe-modeling
##### https://www.kaggle.com/code/venkatkumar001/scrabble-player-rating-baseline-xgb
```python

```