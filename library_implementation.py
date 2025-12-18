import numpy as np

def print_matrix_np(name, matrix):
    """NumPy配列を見やすく表示する関数"""
    print(f"{name}:")
    # 小数点以下4桁でフォーマットして表示
    with np.printoptions(precision=4, suppress=True):
        print(matrix)
    print()

# --- メイン実行部 ---

# 1. 行列の定義（NumPy配列として定義）
input_matrix = np.array([
    [5.0, 4.0, 1.0],
    [4.0, 5.0, 1.0],
    [1.0, 1.0, 4.0]
])

print("=== ライブラリを使用した実装（NumPy） ===")
print_matrix_np("入力行列 A", input_matrix)

# 2. 固有値と固有ベクトルの計算
# np.linalg.eig は固有値分解を行う関数です
# w: 固有値の配列 (1次元)
# v: 固有ベクトル行列 (各列が固有ベクトルに対応)
w, v = np.linalg.eig(input_matrix)

# 3. 固有値を対角行列形式に変換
D = np.diag(w)

print_matrix_np("対角化された行列 D (固有値)", D)
print_matrix_np("固有ベクトル行列 P", v)

# 4. 検算: P * D * P^-1 が元の行列に戻るか確認
# NumPyでは @ 演算子で行列積を計算できます
# np.linalg.inv(v) で逆行列を計算
A_rec = v @ D @ np.linalg.inv(v)

print_matrix_np("検算: P * D * P^-1 (元の行列に戻るか)", A_rec)

# 補足: 実対称行列の場合は np.linalg.inv(v) の代わりに 
# 転置 v.T を使っても元に戻ります（直交行列の性質）