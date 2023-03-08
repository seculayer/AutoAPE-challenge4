### 요약정보 
- 도전기관 : 한양대 
- 도전자 : 이자윤 
- 최종스코어 :  0.83671
- 제출일자 : 2023-03-08
- 총 참여 팀수 : 1049
- 순위 및 비율 : 161(15.35%)

### 결과화면 
![result](./img/1.PNG) 
![result](./img/2.PNG) 

### 사용한 방법 & 알고리즘 
- Method: 
  Transfer learning; base model is EfficientV2M in Keras

  - Pretrained weight: imagenet

  - Input size: 75x75 pixel

  - Train data augmentation: 

    ​	width_shift_range=0.2,
  
    ​    height_shift_range=0.2,
  
    ​    rotation_range=360,
  
    ​    shear_range=0.3,
  
    ​    zoom_range=0.2,
  
    ​    horizontal_flip=True,
  
    ​    vertical_flip=True,
  
    ​    fill_mode='constant',
  
    ​    cval=255,
  
    ​    validation_split=0.1 
  
   -  Loss function: SigmoidFocalCrossEntropy
  
     (It's a loss fuction suitabe for imbalanced training data.)

### 코드

[./NDSB.ipynb](./NDSB.ipynb)

### 참고자료

- https://keras.io/api/applications/efficientnet_v2/#efficientnetv2m-function
- https://www.tensorflow.org/addons/api_docs/python/tfa/losses/SigmoidFocalCrossEntropy
