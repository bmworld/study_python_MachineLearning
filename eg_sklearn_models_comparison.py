# ---------------------------------------------------------
# [개요] Iris 데이터 분류 3가지 모델 비교 실습
# 1) 데이터 구조 확인
# 2) train/test 데이터 분할
# 3) K-NN / Decision Tree / Logistic Regression 학습
# 4) 각 모델별 정확도·리포트 출력
# 5) 샘플 1개 예측 결과 비교
# ---------------------------------------------------------

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------
# [1] 데이터 로드
# ---------------------------------------------------------
iris = load_iris()

print("🌱 [1] iris 객체 타입")
print(type(iris))  # <class 'sklearn.utils._bunch.Bunch'>

print("\n🗝️ [2] iris 키 목록")
print(iris.keys())  # dictionary와 유사한 Bunch 타입

# ---------------------------------------------------------
# [2] 데이터 구성 확인
# ---------------------------------------------------------
iris_data = iris["data"]  # X: 독립변수 (feature)
iris_target = iris["target"]  # y: 종(label)

print("\n📊 [3] iris data (X, feature) 예시 상위 5개")
print(iris_data[:5])

print("\n🏷️ [4] iris target (y, label) 예시 상위 10개")
print(iris_target[:10])

print("\n🏷️ [4-1] target_names (라벨 이름)")
print(iris["target_names"])  # 0=setosa, 1=versicolor, 2=virginica

print("\n전체 데이터 크기:", iris_data.shape)  # (150, 4)

# ---------------------------------------------------------
# [3] train(훈련) / test(테스트) 데이터 분할
# ---------------------------------------------------------
print("\n✂️ [5] train / test 분할")

X_train, X_test, y_train, y_test = train_test_split(
    iris_data,  # X: 입력 데이터
    iris_target,  # y: 정답 레이블
    test_size=0.3,  # train:70%, test:30%
    random_state=42,  # 재현성을 위한 시드 고정
    stratify=iris_target  # 클래스 비율 유지
)

print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train:", y_train.shape, "y_test:", y_test.shape)

# 공통 샘플 (모든 모델에서 같은 입력으로 예측)
sample = [[5.1, 3.5, 1.4, 0.2]]  # setosa 근처 값

# =========================================================
# [4-1] K-NN 분류 모델
# =========================================================
print("\n" + "=" * 60)
print("🤖 [6-1] K-NN 모델 (k=5)")
print("=" * 60)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print(f"\n📍K-NN 분류 모델 결과")
print(f"👉 모델 정확도(accuracy): {accuracy_knn:.4f}")
print(f"👉 훈련셋 정확도: {knn_model.score(X_train, y_train):.4f}")
print(f"👉 테스트셋 정확도: {knn_model.score(X_test, y_test):.4f}")

print("\n📄 분류 리포트 (K-NN)")
print(
    classification_report(y_test, y_pred_knn,
                          target_names=iris["target_names"]))

print("\n🔍 샘플 예측 (K-NN)")
pred_knn = knn_model.predict(sample)[0]
print("입력:", sample)
print("예측 label index:", pred_knn)
print("예측 품종:", iris["target_names"][pred_knn])

# =========================================================
# [4-2] 의사결정트리 (Decision Tree)
# =========================================================
print("\n" + "=" * 60)
print("🌳 [6-2] Decision Tree 모델")
print("=" * 60)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

print(f"\n📍Decision Tree 모델 결과")
print(f"👉 모델 정확도(accuracy): {accuracy_dt:.4f}")
print(f"👉 훈련셋 정확도: {dt_model.score(X_train, y_train):.4f}  (항상 과적합 1.0)"
      f"확인)")
print(f"👉 테스트셋 정확도: {dt_model.score(X_test, y_test):.4f}")

print("\n📄 분류 리포트 (Decision Tree)")
print(
    classification_report(y_test, y_pred_dt, target_names=iris["target_names"]))

print("\n🔍 샘플 예측 (Decision Tree)")
pred_dt = dt_model.predict(sample)[0]
print("입력:", sample)
print("예측 label index:", pred_dt)
print("예측 품종:", iris["target_names"][pred_dt])

# =========================================================
# [4-3] 로지스틱 회귀 (Logistic Regression)
# =========================================================
print("\n" + "=" * 60)
print("📈 [6-3] Logistic Regression 모델")
print("=" * 60)

lr_model = LogisticRegression(max_iter=200, random_state=42)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

print(f"\n📍Logistic Regression 모델 결과")
print(f"👉 모델 정확도(accuracy): {accuracy_lr:.4f}")
print(f"👉 훈련셋 정확도: {lr_model.score(X_train, y_train):.4f}")
print(f"👉 테스트셋 정확도: {lr_model.score(X_test, y_test):.4f}")

print("\n📄 분류 리포트 (Logistic Regression)")
print(
    classification_report(y_test, y_pred_lr, target_names=iris["target_names"]))

print("\n🔍 샘플 예측 (Logistic Regression)")
pred_lr = lr_model.predict(sample)[0]
print("입력:", sample)
print("예측 label index:", pred_lr)
print("예측 품종:", iris["target_names"][pred_lr])

# =========================================================
# [5] 세 모델 예측 결과 한눈에 비교
# =========================================================
print("\n" + "=" * 60)
print("📊 [7] 샘플 1개에 대한 세 모델 비교")
print("=" * 60)

print("입력 샘플:", sample)
print(
    f"K-NN                → {iris['target_names'][pred_knn]} (index={pred_knn})")
print(
    f"Decision Tree       → {iris['target_names'][pred_dt]} (index={pred_dt})")
print(
    f"Logistic Regression → {iris['target_names'][pred_lr]} (index={pred_lr})")
