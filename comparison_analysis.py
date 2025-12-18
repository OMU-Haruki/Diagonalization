import time
import math
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 基本的なプログラミング文法を使った実装（スクラッチ実装: ヤコビ法） ---
def mat_mul(A, B):
    """行列積"""
    rows_A = len(A)
    cols_A = len(A[0])
    cols_B = len(B[0])
    C = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            sum_val = 0.0
            for k in range(cols_A):
                sum_val += A[i][k] * B[k][j]
            C[i][j] = sum_val
    return C

def transpose(A):
    """転置"""
    rows = len(A)
    cols = len(A[0])
    return [[A[j][i] for j in range(rows)] for i in range(cols)]

def jacobi_method_scratch(A, tol=1e-10, max_iter=1000):
    n = len(A)
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    D = [row[:] for row in A]
    
    iter_count = 0
    while iter_count < max_iter:
        max_val = 0.0
        p, q = 0, 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(D[i][j]) > max_val:
                    max_val = abs(D[i][j])
                    p, q = i, j
        
        if max_val < tol:
            break
        
        if D[p][p] == D[q][q]:
            theta = math.pi / 4
        else:
            theta = 0.5 * math.atan(2 * D[p][q] / (D[p][p] - D[q][q]))
        
        c = math.cos(theta)
        s = math.sin(theta)
        
        # 行列の更新を効率化せずに、愚直に行列積で行う（教育的意図）
        R = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        R[p][p] = c
        R[p][q] = -s
        R[q][p] = s
        R[q][q] = c
        
        # D = R^T * D * R
        RT = transpose(R)
        D = mat_mul(RT, mat_mul(D, R))
        V = mat_mul(V, R)
        
        iter_count += 1
        
    # 固有値をリストとして抽出
    eigenvalues = [D[i][i] for i in range(n)]
    return eigenvalues, V, D

# --- 2. 数値計算ライブラリを使った実装（NumPy） ---
def numpy_eig_impl(A_np):
    # np.linalg.eigh は実対称行列に特化しており高速・高精度だが
    # 比較のため一般的な np.linalg.eig を使用してもよい。
    # ここではより一般的な np.linalg.eig を使う。
    w, v = np.linalg.eig(A_np)
    return w, v

# --- 3. 比較実験用関数 ---
def run_experiment(matrix_sizes):
    scratch_times = []
    numpy_times = []
    
    scratch_errors = []
    numpy_errors = []

    print(f"{'Size':>5} | {'Scratch Time (s)':>18} | {'NumPy Time (s)':>15} | {'Speedup':>10}")
    print("-" * 60)

    for n in matrix_sizes:
        # ランダムな実対称行列を生成
        # 対称行列にするために A + A.T を使う
        rand_mat = np.random.rand(n, n)
        sym_mat_np = rand_mat + rand_mat.T
        sym_mat_list = sym_mat_np.tolist()

        # --- スクラッチ実装の計測 ---
        start_time = time.time()
        ev_scratch, V_scratch, D_scratch_mat = jacobi_method_scratch(sym_mat_list)
        end_time = time.time()
        scratch_time = end_time - start_time
        scratch_times.append(scratch_time)
        
        # 誤差計算: || A - V * D * V^T ||
        # 計算はNumPyを使って楽をする（誤差評価自体の正当性のため）
        V_s_np = np.array(V_scratch)
        D_s_np = np.diag(ev_scratch)
        # 再構成
        A_rec_scratch = V_s_np @ D_s_np @ V_s_np.T
        error_scratch = np.linalg.norm(sym_mat_np - A_rec_scratch)
        scratch_errors.append(error_scratch)

        # --- NumPy実装の計測 ---
        start_time = time.time()
        w_np, v_np = numpy_eig_impl(sym_mat_np)
        end_time = time.time()
        numpy_time = end_time - start_time
        numpy_times.append(numpy_time)
        
        # 誤差計算
        D_n_np = np.diag(w_np)
        # NumPyのeigは V @ D @ inv(V) だが実対称なら直交
        # 念のため inv を使う
        A_rec_numpy = v_np @ D_n_np @ np.linalg.inv(v_np)
        error_numpy = np.linalg.norm(sym_mat_np - A_rec_numpy)
        numpy_errors.append(error_numpy)

        print(f"{n:5d} | {scratch_time:18.5f} | {numpy_time:15.5f} | {scratch_time/numpy_time:10.1f}x")

    return scratch_times, numpy_times, scratch_errors, numpy_errors

# 実験実行
# スクラッチ実装は遅いので、サイズは小さめに設定
sizes = [5, 10, 15, 20, 25, 30]
s_times, n_times, s_errors, n_errors = run_experiment(sizes)

# グラフ描画
plt.figure(figsize=(12, 5))

# 実行時間比較
plt.subplot(1, 2, 1)
plt.plot(sizes, s_times, 'o-', label='Scratch (Jacobi)')
plt.plot(sizes, n_times, 's-', label='NumPy (LAPACK)')
plt.xlabel('Matrix Size (N)')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time Comparison')
plt.legend()
plt.grid(True)

# 誤差比較 (対数スケール)
plt.subplot(1, 2, 2)
plt.plot(sizes, s_errors, 'o-', label='Scratch Error')
plt.plot(sizes, n_errors, 's-', label='NumPy Error')
plt.xlabel('Matrix Size (N)')
plt.ylabel('Reconstruction Error (Frobenius Norm)')
plt.yscale('log')
plt.title('Numerical Error Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('comparison_plot.png')
print("Comparison plot saved as 'comparison_plot.png'")