# Santa 2022 - The Christmas Card Conundrum
---
# 결과
---
### 요약 정보
* 도전기관 : 한양대학교
* 도전자 : 주조령
* 최종 스코어 : 77248.9105127474
* 제출 일자 : 2023-04-10
* 총 참여 팀수 : 874
* 순위 및 비율 : 153 (17.5%)

# 결과 화면
---
![final_rank_and_score](./img/Leaderboard_Score.JPG)

# 사용한 방법 & 알고리즘
---
* 가능한 모든 선물 할당 프로토콜은 재귀 알고리즘을 사용하여 생성되었다.재귀 과정에서 알고리즘은 각 인원을 순환시키고 각 인원에 대해 실행 가능한 수신자에게 할당한다.각 재귀 계층이 끝날 때 알고리즘은 현재 선물 할당 방식을 기록한다.
* 생성된 모든 선물 할당 프로토콜에 대해 스크리닝되고 정렬된다. 1. 1인당 1개의 선물만 분배할 수 있고 2. 선물 받는 사람이 발송인의 가족이 될 수 없으며 3. 선물 받는 사람의 가족 수가 가능한 한 적어야 하며 4. 선물 받는 사람의 가족으로부터 가능한 한 멀리 떨어져 있어야 한다.

# 코드
---
[jupyter notebook code](santa2022.ipynb)

# 참고자료
---
##### https://github.com/mai174143/Santa_2022/blob/main/2-pixel-travel-map-removing-duplicates.ipynb
##### https://github.com/mai174143/Santa_2022/blob/main/baseline-but-faster%20(1).ipynb
##### https://bbs.huaweicloud.com/blogs/290255
```python

```