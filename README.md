from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)
# 데이터 입력
#df = pd.read_csv('iris.csv', names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])

# 그래프로 확인
#sns.pairplot(df, hue='species');
#plt.show()

# 데이터 분류
#dataset = df.values
#X = dataset[:,0:4].astype(float)
#Y_obj = dataset[:,4]

# 문자열을 숫자로 변환
#e = LabelEncoder()
#e.fit(Y_obj)
#Y = e.transform(Y_obj)
#Y_encoded = np_utils.to_categorical(Y)
# 모델의 설정
model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 모델 컴파일 
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

# 모델 실행
model.fit(X, Y_encoded, epochs=50, batch_size=1)

# 결과 출력 
print("\n Accuracy: %.4f" % (model.evaluate(X, Y_encoded)[1]))
Epoch 1/50
150/150 [==============================] - 0s 3ms/step - loss: 1.6267 - acc: 0.3267
Epoch 2/50
150/150 [==============================] - 0s 1ms/step - loss: 1.1399 - acc: 0.4867
Epoch 3/50
150/150 [==============================] - 0s 1ms/step - loss: 0.9403 - acc: 0.4600
Epoch 4/50
150/150 [==============================] - 0s 1ms/step - loss: 0.7889 - acc: 0.7467
Epoch 5/50
150/150 [==============================] - 0s 1ms/step - loss: 0.6662 - acc: 0.7267
Epoch 6/50
150/150 [==============================] - 0s 1ms/step - loss: 0.5631 - acc: 0.8800
Epoch 7/50
150/150 [==============================] - 0s 1ms/step - loss: 0.5078 - acc: 0.8267
Epoch 8/50
150/150 [==============================] - 0s 1ms/step - loss: 0.4555 - acc: 0.9333
Epoch 9/50
150/150 [==============================] - 0s 1ms/step - loss: 0.4285 - acc: 0.8867
Epoch 10/50
150/150 [==============================] - 0s 1ms/step - loss: 0.4068 - acc: 0.8867
Epoch 11/50
150/150 [==============================] - 0s 1ms/step - loss: 0.3851 - acc: 0.9333
Epoch 12/50
150/150 [==============================] - 0s 1ms/step - loss: 0.3676 - acc: 0.9467
Epoch 13/50
150/150 [==============================] - 0s 1ms/step - loss: 0.3504 - acc: 0.9467
Epoch 14/50
150/150 [==============================] - 0s 1ms/step - loss: 0.3366 - acc: 0.9067
Epoch 15/50
150/150 [==============================] - 0s 1ms/step - loss: 0.3264 - acc: 0.9333
Epoch 16/50
150/150 [==============================] - 0s 1ms/step - loss: 0.3135 - acc: 0.9533
Epoch 17/50
150/150 [==============================] - 0s 1ms/step - loss: 0.3059 - acc: 0.9533
Epoch 18/50
150/150 [==============================] - 0s 1ms/step - loss: 0.2964 - acc: 0.9667
Epoch 19/50
150/150 [==============================] - 0s 1ms/step - loss: 0.2853 - acc: 0.9333
Epoch 20/50
150/150 [==============================] - 0s 1ms/step - loss: 0.2760 - acc: 0.9600
Epoch 21/50
150/150 [==============================] - 0s 1ms/step - loss: 0.2677 - acc: 0.9733
Epoch 22/50
150/150 [==============================] - 0s 1ms/step - loss: 0.2608 - acc: 0.9667
Epoch 23/50
150/150 [==============================] - 0s 1ms/step - loss: 0.2528 - acc: 0.9467
Epoch 24/50
150/150 [==============================] - 0s 1ms/step - loss: 0.2436 - acc: 0.9600
Epoch 25/50
150/150 [==============================] - 0s 1ms/step - loss: 0.2360 - acc: 0.9733
Epoch 26/50
150/150 [==============================] - 0s 1ms/step - loss: 0.2337 - acc: 0.9667
Epoch 27/50
150/150 [==============================] - 0s 1ms/step - loss: 0.2194 - acc: 0.9867
Epoch 28/50
150/150 [==============================] - 0s 1ms/step - loss: 0.2205 - acc: 0.9467
Epoch 29/50
150/150 [==============================] - 0s 1ms/step - loss: 0.2142 - acc: 0.9667
Epoch 30/50
150/150 [==============================] - 0s 1ms/step - loss: 0.1999 - acc: 0.9600
Epoch 31/50
150/150 [==============================] - 0s 1ms/step - loss: 0.1999 - acc: 0.9733
Epoch 32/50
150/150 [==============================] - 0s 1ms/step - loss: 0.1970 - acc: 0.9667
Epoch 33/50
150/150 [==============================] - 0s 1ms/step - loss: 0.1881 - acc: 0.9733
Epoch 34/50
150/150 [==============================] - 0s 1ms/step - loss: 0.1806 - acc: 0.9667
Epoch 35/50
150/150 [==============================] - 0s 2ms/step - loss: 0.1851 - acc: 0.9667
Epoch 36/50
150/150 [==============================] - 0s 1ms/step - loss: 0.1742 - acc: 0.9800
Epoch 37/50
150/150 [==============================] - 0s 1ms/step - loss: 0.1684 - acc: 0.9733
Epoch 38/50
150/150 [==============================] - 0s 1ms/step - loss: 0.1645 - acc: 0.9733
Epoch 39/50
150/150 [==============================] - 0s 2ms/step - loss: 0.1621 - acc: 0.9733
Epoch 40/50
150/150 [==============================] - 0s 1ms/step - loss: 0.1560 - acc: 0.9667
Epoch 41/50
150/150 [==============================] - 0s 1ms/step - loss: 0.1564 - acc: 0.9667
Epoch 42/50
150/150 [==============================] - 0s 1ms/step - loss: 0.1525 - acc: 0.9667
Epoch 43/50
150/150 [==============================] - 0s 1ms/step - loss: 0.1434 - acc: 0.9667
Epoch 44/50
150/150 [==============================] - 0s 2ms/step - loss: 0.1442 - acc: 0.9533
Epoch 45/50
150/150 [==============================] - 0s 1ms/step - loss: 0.1418 - acc: 0.9800
Epoch 46/50
150/150 [==============================] - 0s 1ms/step - loss: 0.1449 - acc: 0.9600
Epoch 47/50
150/150 [==============================] - 0s 1ms/step - loss: 0.1349 - acc: 0.9667
Epoch 48/50
150/150 [==============================] - 0s 991us/step - loss: 0.1351 - acc: 0.9800
Epoch 49/50
150/150 [==============================] - 0s 991us/step - loss: 0.1332 - acc: 0.9667
Epoch 50/50
150/150 [==============================] - 0s 1ms/step - loss: 0.1284 - acc: 0.9667
150/150 [==============================] - 0s 259us/step

 Accuracy: 0.9733
 
 
 프로젝트 설명

아이리스는 꽃잎의 모양과 길이에 따라 여러 가지 품종으로 나뉘는데 사진을 보면 품종마다 비슷해보이기 때문에 구별하기 위해 이 코드를 짜보았다.
아이리스의 속성 수는 꽃받침 길이, 꽃받침 넓이, 꽃입 길이, 꽃잎 넓이로 4가지다.
먼저 pairpot() 함수를 써서 데이터 전체를 한번에 보는 그래프를 출력하였다.
최종 출력 값이 3개 중 하나여야 하기 때문에 출력층에 해당하는 Dense 노드수를 3으로 설정하였다. 또한 활성화 함수로 앞서 나오지 않았던 소프트맥스를 사용하였다.
다중 분류에 적절한 오차 함수인 categorical_crossentropy를 사용하고, 최적하 함수로 adam을 사용하였다.
.
.
소감.
.
코딩을 잘하지 못하여서 시험도 잘 보지 못했다. 하지만 아이리스 꽃종 분류 코드를 짜기 위해 많은 노력을 하여서 코딩의 ㅋ자 정도는 이해할 수 있었다. 코딩을 하면서 중간중간에 계속 오류가 나는 부분을 해결하기 위해 계속 시도하였다. 이 과정을 통해 뭔가 나의 삶에 괜찮다고 하면서 살아왔지만 잘못된 부분을 지금이라도 늦이 않았고 그 부분을 고치고 해결한다면 좋은 결실을 맺을 수 있다고 생각했다.
.
