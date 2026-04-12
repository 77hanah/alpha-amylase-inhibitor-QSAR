import os
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from config import DATA_FOLDER, OUTPUTS_FOLDER, FIGURES_FOLDER

def evaluate_applicability_domain(model, X_test, y_test, n_simulations=1000, noise_std=0.01, save_path=None):
    """蒙特卡洛方法"""

    print("开始适用域评估 (Monte Carlo Method)...")
    print(f"模拟次数: {n_simulations}, 噪声标准差: {noise_std}")
    print("="*50)

    y_pred_original = model.predict(X_test)
    n_compounds = len(y_test)
    all_errors = np.zeros((n_simulations, n_compounds))

    # 进行模拟
    for i in tqdm(range(n_simulations), desc="AD Simulation"):
        noise = np.random.normal(loc=0, scale=noise_std, size=n_compounds)
        y_true_noisy = y_test.values + noise
        errors_this_iteration = y_pred_original - y_true_noisy
        all_errors[i, :] = errors_this_iteration
        
    # 计算每平均误差和标准差
    mean_errors = np.mean(all_errors, axis=0)
    std_errors = np.std(all_errors, axis=0)

    # 生成表格
    ad_results_df = pd.DataFrame({
        'NO': range(1, n_compounds + 1),
        'Mean_Errors': mean_errors,
        'Std_Errors': std_errors
    })
    
    print("\n适用域评估结果如下:")
    print(ad_results_df.to_string(index=False, float_format="%.4f"))


    save_path = os.path.join(OUTPUTS_FOLDER, 'applicability_domain_results.csv')
    ad_results_df.to_csv(save_path, index=False, float_format='%.4f')
    print(f"\n适用域评估结果已保存: {save_path}")
    
    return ad_results_df


if __name__ == "__main__":

    model_path = os.path.join(OUTPUTS_FOLDER, 'trained_neural_network_model.pkl')
    x_test_path = os.path.join(DATA_FOLDER, 'X_test.csv')
    y_test_path = os.path.join(DATA_FOLDER, 'y_test.csv')


    trained_model = joblib.load(model_path)
    X_test_data = pd.read_csv(x_test_path)
    y_test_data = pd.read_csv(y_test_path).squeeze("columns")

    evaluate_applicability_domain(trained_model, X_test_data, y_test_data)
