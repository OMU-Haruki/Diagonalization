# 行列の対角化の基本実装とライブラリ実装の比較検証

知識科学概論の課題として作成した、実対称行列の対角化を行うPythonプロジェクトです。
基本的なプログラミング文法を使った実装と、数値計算のライブラリーを使った実装の2パターンを行い、実行速度と計算精度の比較検証を行いました。

## 概要

1.  **Basic Implementation:**
    * 標準ライブラリ（`math`）のみを使用。
    * ヤコビ法 (Jacobi Method) アルゴリズムを採用し、行列の回転によって非対角成分をゼロに収束させます。
2.  **Library Implementation:**
    * `numpy.linalg.eig` を使用。
3.  **Comparison & Analysis:**
    * 行列サイズ $N$ を変化させ（ $N=5 \sim 30$ ）、実行時間と計算誤差（フロベニウスノルム）を比較・グラフ化します。

## 環境 (Requirement)

* Python 3.x
* NumPy
* Matplotlib
