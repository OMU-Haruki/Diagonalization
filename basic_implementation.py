import math

def print_matrix(name, matrix):
    """行列を見やすく表示する関数"""
    print(f"{name}:")
    for row in matrix:
        print("  [" + ", ".join(f"{x:7.4f}" for x in row) + "]")
    print()

def mat_mul(A, B):
    """行列の積 A * B を計算する関数"""
    rows_A = len(A)
    cols_A = len(A[0])
    cols_B = len(B[0])
    
    C = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return C

def transpose(A):
    """転置行列を計算する関数"""
    rows = len(A)
    cols = len(A[0])
    return [[A[j][i] for j in range(rows)] for i in range(cols)]

def jacobi_method(A, tol=1e-10, max_iter=100):
    """ヤコビ法による対角化の実装"""
    n = len(A)
    # 固有ベクトルを格納する行列 V を単位行列で初期化
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    
    # 元の行列を壊さないようにコピー
    D = [row[:] for row in A] 
    
    iter_count = 0
    while iter_count < max_iter:
        # 1. 非対角成分で絶対値が最大の要素 (p, q) を探す
        max_val = 0.0
        p, q = 0, 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(D[i][j]) > max_val:
                    max_val = abs(D[i][j])
                    p, q = i, j
        
        # 収束判定（最大値が十分小さければ終了）
        if max_val < tol:
            break
        
        # 2. 回転角 theta の計算
        # tan(2*theta) = 2 * D[p][q] / (D[p][p] - D[q][q])
        if D[p][p] == D[q][q]:
            theta = math.pi / 4
        else:
            theta = 0.5 * math.atan(2 * D[p][q] / (D[p][p] - D[q][q]))
        
        c = math.cos(theta)
        s = math.sin(theta)
        
        # 3. 回転行列 R の作成（必要な部分だけ計算して更新するのが高速だが、今回は分かりやすく行列積で行う）
        R = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        R[p][p] = c
        R[p][q] = -s
        R[q][p] = s
        R[q][q] = c
        
        # 4. 行列の更新 D = R^T * D * R
        #    固有ベクトルの更新 V = V * R
        RT = transpose(R)
        D = mat_mul(RT, mat_mul(D, R))
        V = mat_mul(V, R)
        
        iter_count += 1

    return D, V, iter_count

# --- メイン実行部 ---

# 対角化したい実対称行列（例: 3x3）
input_matrix = [
    [5.0, 4.0, 1.0],
    [4.0, 5.0, 1.0],
    [1.0, 1.0, 4.0]
]

print_matrix("入力行列 A", input_matrix)

# 計算実行
eigen_values_matrix, eigen_vectors_matrix, iterations = jacobi_method(input_matrix)

print(f"反復回数: {iterations} 回")
print_matrix("対角化された行列 D (対角成分が固有値)", eigen_values_matrix)
print_matrix("固有ベクトル行列 P (各列が固有ベクトル)", eigen_vectors_matrix)

# 検算: P * D * P^-1 (直交行列なので P^T) が元の行列に戻るか
# A_reconstructed = P * D * P^T
P = eigen_vectors_matrix
PT = transpose(P)
A_rec = mat_mul(P, mat_mul(eigen_values_matrix, PT))
print_matrix("検算: P * D * P^T (元の行列に戻るか)", A_rec)