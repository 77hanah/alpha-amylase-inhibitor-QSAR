import os
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
import scienceplots
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from config import DATA_FOLDER, OUTPUTS_FOLDER, FIGURES_FOLDER, RANDOM_STATE
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def run_randomization_loop(X_train, y_train, X_test, y_test, model_parameters, n_randomizations=1000):

    print(f"开始Y-随机检验循环，次数: {n_randomizations}")
    r_random_list = []
    
    for i in tqdm(range(n_randomizations), desc="Y-Randomization"):
        # 只打乱训练集的标签
        y_train_scrambled = np.random.permutation(y_train)
        
        random_model = MLPRegressor(**model_parameters)
        
        # 使用打乱后的训练集进行训练
        random_model.fit(X_train, y_train_scrambled)
        
        # 在测试集上进行预测 
        y_pred_random = random_model.predict(X_test)
        
        r_random, _ = pearsonr(y_test, y_pred_random)
        r_random_list.append(r_random)
        
    return r_random_list

def plot_y_randomization_results(r_original, r_random_list, save_path=None):

    print("\n正在绘制Y-随机检验结果图...")
    
    plt.style.use(['science', 'seaborn-v0_8-talk'])
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.hist(r_random_list, bins=30, color='#1f77b4', edgecolor='white', density=False, label='Randomized Models')
    ax.axvline(x=r_original, color='#d62728', linestyle='--', linewidth=2.5, label=f'Optimal Model (R = {r_original:.4f})')
    ax.set_xlabel('Correlation Coefficient (R)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax.set_title('Y-Randomization Test Results', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='upper left', frameon=True, fontsize=12)
    
    # 确保X轴的刻度清晰
    ax.tick_params(axis='x', rotation=0)
    
    fig.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(FIGURES_FOLDER, 'y_randomization_histogram.png')
    plt.savefig(save_path, dpi=300) 
    
    print(f"Y-随机检验直方图已保存: {save_path}")


if __name__ == "__main__":

    X_train = pd.read_csv(os.path.join(DATA_FOLDER, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(DATA_FOLDER, 'y_train.csv')).squeeze("columns")
    X_test = pd.read_csv(os.path.join(DATA_FOLDER, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(DATA_FOLDER, 'y_test.csv')).squeeze("columns")
    
    # 合并测试集和训练集
    X_full = pd.concat([X_train, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)
    print(f"数据加载成功！总样本数: {len(y_full)}")

    # 加载模型
    results_path = os.path.join(OUTPUTS_FOLDER, 'prediction_results.pkl')
    try:
        results_dict = joblib.load(results_path)
        
        y_all_true = np.concatenate([
            results_dict['y_train_split'], 
            results_dict['y_val'], 
            results_dict['y_test']
        ])
        y_all_pred = np.concatenate([
            results_dict['y_train_pred'], 
            results_dict['y_val_pred'], 
            results_dict['y_test_pred']
        ])
        
        r_original, _ = pearsonr(y_all_true, y_all_pred)
        print(f"真实模型的基准性能: R = {r_original:.4f}")

    except FileNotFoundError:
        print(f"错误: 结果文件 'prediction_results.pkl' 未找到！")
        print("请先运行 model_results.py 来生成模型结果文件。")
        exit()

    model_parameters = {
        'hidden_layer_sizes': (4,),
        'activation': 'tanh',
        'solver': 'lbfgs',
        'alpha': 0.1,
        'max_iter': 2000,
        'early_stopping': True,
        'validation_fraction': 0.2,
        'n_iter_no_change': 10,
        'random_state': RANDOM_STATE,
        'tol': 1e-6
    }

    for key, value in model_parameters.items():
        print(f"  - {key}: {value}")
        
    r_random = run_randomization_loop(X_train, y_train, X_test, y_test, model_parameters, n_randomizations=1000)
    
    plot_y_randomization_results(r_original, r_random)

