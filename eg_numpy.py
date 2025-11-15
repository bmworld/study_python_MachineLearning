# pip: Python 제공 Util
# -> pip install numpy
# conda: python 주요 Lib package
# -> conda install numpy



# ---------------------------------------------------------------------
# numpy: 동일 타입 데이터만 벡터연산가능 (타입다를 경우, 타입변환함)
import numpy as np

# Machine Learning 에서는 '속도/성능' 때문에 list 지원 X -> ndarray 만 지원

a = [1,2,3,4,5,6,7,8,9,10]
print("- type a1 = ", type(a))

a2 = np.array # list to ndarray (본래의 배열, 고속)
print("- type a2 = ", type(a2))


# ---------------------------------------------------------------------

x = [1,3,5,7,9]
y = [2,4,6,8,10]

print(x+y) # [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]

x1 = np.array(x)
y1 = np.array(y)
print(x1+y1) # 벡터연산: [ 3 7 11 15 19] (* numpy > ndarray 자료구조만 지원함)
