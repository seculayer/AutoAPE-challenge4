# The Nature Conservancy Fisheries Monitoring
---
# 결과
---
### 요약 정보
* 도전기관 : 한양대학교
* 도전자 : 주조령
* 최종 스코어 : 1.95975
* 제출 일자 : 2023-01-02
* 총 참여 팀수 : 2293
* 순위 및 비율 : 186 (8.1%)

# 결과 화면
---
![final_rank_and_score](./img/Leaderboard_Score.JPG)

# 사용한 방법 & 알고리즘
---
* NCFM 데이터셋에서 게시된 주석 json 파일을 다운로드하고 경계 상자 레이블을 생성한다.
* 제공된 데이터의 범주는 "ALB", "BET", "DOL", "LAG", "OTHER", "SHARK", "YFT"입니다.모델을 보다 정확하게 훈련하기 위해 '기타' 클래스를 삭제했다.
* 경계 상자의 숫자를 6가지로 분류 (0-5)
* VGG16 모델 사용

# 코드
---
[jupyter notebook code](The Nature Conservancy Fisheries Monitoring.ipynb)

# 참고자료
---
##### https://github.com/a4tunado/lectures/blob/master/007/007-detection.ipynb
##### https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/VGG16

```python

```
