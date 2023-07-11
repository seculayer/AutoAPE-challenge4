# 2021 Ego-Vision 손동작 인식 AI 경진대회
---
# 결과
---
### 요약 정보
* 도전기관 : 시큐레이어
* 도전자 : 석민재
* 최종 스코어 : 0.0156
* 제출 일자 : 2023-07-04
* 총 참여 팀수 : 290
* 순위 및 비율 : 8 (2.78%)

# 결과 화면
---
<img width="800" alt="2" src="https://github.com/Jsonseok/SecuLayer/assets/112038669/d2e3a452-181d-4c65-bd4c-5c348c31b7fa">
<img width="800" alt="1" src="https://github.com/Jsonseok/SecuLayer/assets/112038669/42da86c1-8a9e-43c1-9824-f8ff91cfeba9">


# 사용한 방법 & 알고리즘
---
* Keypoint의 좌표이동을 기반으로 유사한 동작을 구별
* 잘못된 라벨링같은 학습을 방해하는 노이즈 이미지 제거
* Stratified KFold를 통해 데이터 부족 문제 해결

# 코드
---
[jupyter notebook code](main.ipynb)

# 참고자료
---
##### https://github.com/albumentations-team/albumentations
