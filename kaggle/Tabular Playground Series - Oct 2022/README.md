# Tabular Playground Series - Oct 2022
---
# 결과
---
### 요약 정보
* 도전기관 : 한양대학교
* 도전자 : 주조령
* 최종 스코어 : 0.18513
* 제출 일자 : 2023-05-17
* 총 참여 팀수 : 463
* 순위 및 비율 : 83 (17.9%)

# 결과 화면
---
![final_rank_and_score](./img/Leaderboard_Score.JPG)

# 사용한 방법 & 알고리즘
---
* 1. Feather 형식으로 데이터 저장 및 읽기:
    (1) 데이터스토어: Pandas 라이브러리를 사용하여 데이터를 Feather 형식으로 저장하면 df.to _feather() 함수로 변환할 수 있다.

    (2) 데이터 읽기: Pandas 라이브러리를 사용하여 pd.read_feather() 함수를 통해 Feather 형식의 데이터를 DataFrame으로 읽다.
    
    (3) 데이터 압축: Feather 형식은 데이터를 효율적으로 압축하고 압축 해제할 수 있으며 데이터의 저장 공간과 전송 시간을 크게 줄일 수 있다.

* 2. XGBoost, LightGBM, CatBoost, 신경망 등 여러 모델을 훈련에 사용하고 교차 검증 및 모델 융합 기술을 통해 모델의 정확도를 향상시킨다.

* 3. 베이지안 최적화 및 그리드 검색과 같은 기술을 사용하여 모델을 최적화하고 최상의 하이퍼 파라미터 조합을 찾다.


# 코드
---
[jupyter notebook code](Tabular Playground Series - Oct 2022.ipynb)

# 참고자료
---
### https://www.kaggle.com/code/hsuyab/fast-loading-high-compression-with-feather
### https://www.kaggle.com/code/alexryzhkov/tps-2022-10-fastai-with-multistart-and-tta
```python

```