# IRIS + KNN: CLI 입력 지원 + 예측 + 2D 분류 맵(입력점 강조)
# 실행 예:
#   1) 아규먼트
#      python iris_knn.py --input "5.1,3.5,1.4,0.2" --input "6.0,2.9,4.5,1.5" --input "6.9,3.1,5.4,2.1"
#   2) 인터랙티브
#      python iris_knn.py
#      > 5.1,3.5,1.4,0.2
#      > (빈 엔터로 종료)
# 옵션: --no-plot  (그래프 생략),  --k 7 (K 변경)

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FEATURE_NAMES = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
F1, F2 = 2, 3  # 시각화 축: petal length vs petal width

def parse_vec(s: str) -> List[float]:
  try:
    vals = [float(x.strip()) for x in s.split(",")]
    if len(vals) != 4:
      raise ValueError
    return vals
  except Exception:
    raise argparse.ArgumentTypeError("입력은 'a,b,c,d' 형태의 실수 4개여야 합니다.")

def read_interactive() -> List[List[float]]:
  print("입력해 주세요 (형식: sepal_len,sepal_wid,petal_len,petal_wid)")
  print("예: 5.1,3.5,1.4,0.2  (빈 엔터 입력 시 종료)\n")
  rows = []
  while True:
    try:
      line = input("> ").strip()
    except EOFError:
      break
    if not line:
      break
    try:
      rows.append(parse_vec(line))
    except argparse.ArgumentTypeError as e:
      print(f"⚠️ 형식 오류: {e}")
  return rows

def main():
  ap = argparse.ArgumentParser(description="Iris KNN 분류 (CLI 입력 지원)")
  ap.add_argument("--input", "-i", action="append", type=parse_vec,
                  help='예: --input "5.1,3.5,1.4,0.2" (여러 번 지정 가능)')
  ap.add_argument("--k", type=int, default=5, help="K값 (기본 5)")
  ap.add_argument("--no-plot", dest="no_plot", action="store_true", help="그래프 표시 생략")
  args = ap.parse_args()

  # 1) 데이터 로드
  iris = load_iris()
  X = pd.DataFrame(iris.data, columns=FEATURE_NAMES)
  y = pd.Series(iris.target, name="species")
  target_names = iris.target_names

  # 2) 4D 모델 학습
  X_tr, X_te, y_tr, y_te = train_test_split(
      X, y, test_size=0.3, random_state=42, stratify=y
  )
  model_4d = KNeighborsClassifier(n_neighbors=args.k)
  model_4d.fit(X_tr, y_tr)

  # 성능 리포트
  y_pred_te = model_4d.predict(X_te)
  print(f"\n✅ 정확도(Accuracy): {accuracy_score(y_te, y_pred_te):.4f}")
  print("\n[분류 보고서]\n", classification_report(y_te, y_pred_te, target_names=target_names))
  print("\n[혼동 행렬]\n", confusion_matrix(y_te, y_pred_te))

  # 3) 입력 수집 (인자 or 인터랙티브)
  if args.input and len(args.input) > 0:
    samples = args.input
  else:
    samples = read_interactive()

  if len(samples) == 0:
    print("\n(입력 샘플이 없어 예측 없이 종료합니다.)")
    return

  X_new_df = pd.DataFrame(samples, columns=FEATURE_NAMES)
  preds = model_4d.predict(X_new_df)

  print("\n입력 데이터:")
  print(X_new_df)
  print("\n예측 결과:")
  for idx, p in enumerate(preds, start=1):
    print(f"Sample {idx} → {target_names[p]}")

  # 4) 시각화 (2D 결정 경계 + 입력점 강조)
  if not args.no_plot:
    X_2d = X.iloc[:, [F1, F2]]
    model_2d = KNeighborsClassifier(n_neighbors=args.k)
    model_2d.fit(X_2d, y)

    # ---- 축 범위: 데이터 vs 사용자 입력 모두 포함하도록 자동 확장 ----
    base_min_x, base_max_x = X_2d.iloc[:, 0].min(), X_2d.iloc[:, 0].max()
    base_min_y, base_max_y = X_2d.iloc[:, 1].min(), X_2d.iloc[:, 1].max()

    X_new_2d = X_new_df.iloc[:, [F1, F2]].values
    if X_new_2d.size > 0:
      user_min_x, user_max_x = np.min(X_new_2d[:, 0]), np.max(X_new_2d[:, 0])
      user_min_y, user_max_y = np.min(X_new_2d[:, 1]), np.max(X_new_2d[:, 1])
    else:
      user_min_x = user_max_x = base_min_x
      user_min_y = user_max_y = base_min_y

    x_min = min(base_min_x, user_min_x)
    x_max = max(base_max_x, user_max_x)
    y_min = min(base_min_y, user_min_y)
    y_max = max(base_max_y, user_max_y)

    # 패딩은 범위의 5% (범위가 0이면 고정 패딩)
    def pad_range(lo, hi):
      rng = hi - lo
      pad = 0.05 * rng if rng > 0 else 0.5
      return lo - pad, hi + pad

    x_min, x_max = pad_range(x_min, x_max)
    y_min, y_max = pad_range(y_min, y_max)

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model_2d.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(7.5, 6.2))
    plt.contourf(xx, yy, Z, alpha=0.25, levels=np.arange(-0.5, 3, 1), cmap="viridis")

    # 원본 데이터 산점도
    for cls_idx, cls_name in enumerate(target_names):
      mask = (y.values == cls_idx)
      plt.scatter(
          X_2d.iloc[mask, 0], X_2d.iloc[mask, 1],
          label=cls_name, s=35, edgecolors="k", linewidths=0.2, alpha=0.9
      )

    # 사용자 입력점 크게 표시 + 번호
    for i, pt in enumerate(X_new_2d, start=1):
      plt.scatter(pt[0], pt[1], s=180, marker="*", edgecolors="black", linewidths=1.2)
      plt.text(pt[0] + 0.03, pt[1] + 0.03, f"#{i}", fontsize=9, weight="bold")

    plt.title(f"KNN (k={args.k}) Decision Map — {FEATURE_NAMES[F1]} vs {FEATURE_NAMES[F2]}")
    plt.xlabel(FEATURE_NAMES[F1]); plt.ylabel(FEATURE_NAMES[F2])
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
  sys.exit(main())
