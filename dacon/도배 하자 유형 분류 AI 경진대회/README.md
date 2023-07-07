# 도배 하자 유형 분류 AI 경진대회
---
# 결과
---
### 요약 정보
* 도전기관 : 시큐레이어
* 도전자 : 석민재
* 최종 스코어 : 0.6256
* 제출 일자 : 2023-07-05
* 총 참여 팀수 : 1025
* 순위 및 비율 : 109 (10.63%)

# 결과 화면
---
<img width="800" alt="1" src="https://github.com/Jsonseok/SecuLayer/assets/112038669/5ccfd24c-8efa-4a25-9b55-a4873119733e">
<img width="800" alt="2" src="https://github.com/Jsonseok/SecuLayer/assets/112038669/6b3e4ec6-8d4d-4cca-aa59-92b3792a4d06">

# 사용한 방법 & 알고리즘
---
* Test-Time-Augumentation을 사용해, Test data에도 증강기법을 적용
* 적은 데이터 양을 보완하기 위해 Cross_validation 사용
* Vertical/Horizontal Flip, RandomBrightnessContrast와 같은 증강 기법을 통해 추가적인 데이터 생성

# 코드
---
[jupyter notebook code](main.ipynb)

# 참고자료
---
##### https://arxiv.org/pdf/1905.11946.pdf
##### https://github.com/qubvel/ttach
