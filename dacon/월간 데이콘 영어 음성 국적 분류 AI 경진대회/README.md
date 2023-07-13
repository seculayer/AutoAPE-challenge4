# 월간 데이콘 영어 음성 국적 분류 AI 경진대회
---
# 결과
---
### 요약 정보
* 도전기관 : 시큐레이어
* 도전자 : 석민재
* 최종 스코어 : 1.077
* 제출 일자 : 2023-07-12
* 총 참여 팀수 : 71
* 순위 및 비율 : 5 (7.04%)

# 결과 화면
---
<img width="800" alt="1" src="https://github.com/Jsonseok/SecuLayer/assets/112038669/2aeb9812-0b95-4e96-882e-b12ca8db68fe">
<img width="800" alt="2" src="https://github.com/Jsonseok/SecuLayer/assets/112038669/1dbf56ef-1e2d-42dc-9b7d-3185dd2b6c49">

# 사용한 방법 & 알고리즘
---
* win_length를 200, 400, 800, 1000으로 설정해 다양한 melspectrogram 생성
* melspectrogram에 정규화, min-max 스케일링 전처리 기법 사용
* Conv2d, BatchNorm2d, ReLU로 구성된 Sequential model 사용

# 코드
---
[jupyter notebook code](main.ipynb)

# 참고자료
---
##### https://github.com/librosa/librosa
