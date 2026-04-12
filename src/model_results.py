import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from config import *
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV
import joblib  # 用于保存模型
import scienceplots

def optimize_hyperparameters(X_train, y_train, random_state=RANDOM_STATE):
    """寻找最优的隐藏层神经元数量和alpha值"""

    print("开始超参数调优...")

    param_grid = {
        'hidden_layer_sizes': [(3,), (4,), (5,), (6,), (8,)],  
        'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]     
    }

    mlp = MLPRegressor(
        activation='tanh',
        solver='lbfgs',
        max_iter=2000,          # 调优时给多一点迭代次数
        early_stopping=True,    
        validation_fraction=0.2,
        n_iter_no_change=10,
        random_state=random_state,
        tol=1e-6
    )

    # 创建网格搜索对象
    grid_search = GridSearchCV(
        estimator=mlp,
        param_grid=param_grid,
        cv=5,   #使用5折交叉验证
        scoring='neg_root_mean_squared_error',  # 以RMSE最小化为目标
        n_jobs=-1,  # 使用所有CPU核心加速
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("\n调优完成!")
    print(f"最佳参数组合: {grid_search.best_params_}")
    print(f"最佳验证集RMSE: {-grid_search.best_score_:.4f}")

    return grid_search.best_params_

def train_neural_network_model(X_train, y_train, X_test, y_test, hidden_layer_sizes=HIDDEN_LAYER_SIZES, alpha=0.5, max_iter=MAX_ITER,random_state=RANDOM_STATE):
    """
    训练前馈神经网络模型
    
    参数:
        X_train: 训练集特征
        y_train: 训练集目标值
        X_test: 测试集特征
        y_test: 测试集目标值
        hidden_layer_sizes: 隐藏层结构，例如 (10,) 表示1个隐藏层,10个神经元
        max_iter: 最大训练轮数
        random_state: 随机种子
    """

    print("开始训练神经网络模型...")

    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.2,           # 20%作为验证集
        random_state=random_state
    )
    
    print(f"\n数据集划分:")
    print(f"  - 训练集样本数: {len(X_train_split)}")
    print(f"  - 验证集样本数: {len(X_val)}")
    print(f"  - 测试集样本数: {len(X_test)}")
    
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,  # 隐藏层结构
        activation='tanh',          # 隐藏层激活函数：tanh（双曲正切）
        solver='lbfgs',             
        max_iter=max_iter,          # 最大迭代次数
        early_stopping=True,       # 早停机制
        validation_fraction=0.2,                                       
        n_iter_no_change=10,        # 最大验证失败次数
        random_state=random_state, 
        alpha=alpha,                 # L2正则化参数
        tol=1e-6,                   # 收敛容忍度
        verbose=True                # 显示训练过程
    )

    print("开始训练...")
    
    model.fit(X_train_split, y_train_split)
    
    print("训练完成！")
    print(f"实际训练轮数: {model.n_iter_}")
    
    y_train_pred = model.predict(X_train_split)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    y_all_true = np.concatenate([y_train_split, y_val, y_test])
    y_all_pred = np.concatenate([y_train_pred, y_val_pred, y_test_pred])
    
    def calculate_metrics(y_true, y_pred, dataset_name):
        """计算评估指标"""

        r2 = r2_score(y_true, y_pred)
        r, _ = pearsonr(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
        print(f"\n{dataset_name}集性能:")
        print(f"  - R² (决定系数):        {r2:.4f}")
        print(f"  - R (相关系数):         {r:.4f}")
        print(f"  - RMSE (均方根误差):    {rmse:.4f}")
        print(f"  - MAE (平均绝对误差):   {mae:.4f}")
        print(f"  - MAPE (平均绝对百分比误差): {mape:.3f}%")
        
        return r2, r, rmse, mae, mape
    
    print("模型性能评估:")
    
    train_metrics = calculate_metrics(y_train_split, y_train_pred, "训练集")
    val_metrics = calculate_metrics(y_val, y_val_pred, "验证集")
    test_metrics = calculate_metrics(y_test, y_test_pred, "测试集")
    all_metrics = calculate_metrics(y_all_true, y_all_pred, "全部数据")

    joblib.dump(model, os.path.join(OUTPUTS_FOLDER, 'trained_neural_network_model.pkl'))
    print("模型已保存为 'trained_neural_network_model.pkl'")
    
    y_train_full_pred = model.predict(X_train)
    
    pd.DataFrame(y_train_full_pred, columns=['pIC50_predicted']).to_csv(
        os.path.join(OUTPUTS_FOLDER, 'y_train_predictions.csv'), index=False)
    print(" 训练集预测结果已保存为 'y_train_predictions.csv'")
    
    pd.DataFrame(y_test_pred, columns=['pIC50_predicted']).to_csv(
        os.path.join(OUTPUTS_FOLDER, 'y_test_predictions.csv'), index=False)
    print(" 测试集预测结果已保存为 'y_test_predictions.csv'")
    
    # 同时保存用于绘图的详细数据
    results_dict = {
        'y_train_split': y_train_split,
        'y_train_pred': y_train_pred,
        'y_val': y_val,
        'y_val_pred': y_val_pred,
        'y_test': y_test,
        'y_test_pred': y_test_pred
    }
    joblib.dump(results_dict, os.path.join(OUTPUTS_FOLDER, 'prediction_results.pkl'))
    print("预测结果已保存为 'prediction_results.pkl'")

    return model, results_dict

plt.style.use(['science', 'seaborn-v0_8-talk'])

def plot_regression_performance(results_dict, save_path=None):

    y_train_split = results_dict['y_train_split']
    y_train_pred = results_dict['y_train_pred']
    y_val = results_dict['y_val']
    y_val_pred = results_dict['y_val_pred']
    y_test = results_dict['y_test']
    y_test_pred = results_dict['y_test_pred']
    
    y_all_true = np.concatenate([y_train_split, y_val, y_test])
    y_all_pred = np.concatenate([y_train_pred, y_val_pred, y_test_pred])
    
    # 计算R值
    r_train, _ = pearsonr(y_train_split, y_train_pred)
    r_val, _ = pearsonr(y_val, y_val_pred)
    r_test, _ = pearsonr(y_test, y_test_pred)
    r_all, _ = pearsonr(y_all_true, y_all_pred)
    

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    # fig.suptitle('Neural Network Model Performance', fontsize=16, fontweight='bold', y=0.995)
    
    # 定义每个子图的数据和样式
    datasets = [
        (y_train_split, y_train_pred, f'Training: R={r_train:.5f}', '#1f77b4', axes[0, 0]),
        (y_val, y_val_pred, f'Validation: R={r_val:.5f}', '#2ca02c', axes[0, 1]),
        (y_test, y_test_pred, f'Test: R={r_test:.5f}', '#d62728', axes[1, 0]),
        (y_all_true, y_all_pred, f'All: R={r_all:.5f}', '#9467bd', axes[1, 1])
    ]
    
    for y_true, y_pred, title, color, ax in datasets:
        ax.scatter(y_true, y_pred,facecolors='none',
                   edgecolors='gray',
                   s=80,
                   linewidths=1.5,
                   alpha=0.7,
                   label='Data')
        
        # 拟合线
        coefficients = np.polyfit(y_true, y_pred, 1)  # 一次多项式拟合
        poly_func = np.poly1d(coefficients)
        x_fit = np.linspace(y_true.min(), y_true.max(), 100)
        y_fit = poly_func(x_fit)
        
        ax.plot(x_fit, y_fit, color=color, linewidth=2, label='Fitting')
        
        # 理想对角线 
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.3, label='Y = T')
        
        ax.set_xlabel('Objectives', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Output = {coefficients[0]:.2f}*X + {coefficients[1]:.2f}', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        
        margin = (max_val - min_val) * 0.05
        ax.set_xlim(min_val - margin, max_val + margin)
        ax.set_ylim(min_val - margin, max_val + margin)
        
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(loc='upper left', frameon=True, fontsize=10)
        
        ax.tick_params(labelsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path is None:
        save_path = os.path.join(FIGURES_FOLDER, 'regression_performance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n回归性能图已保存!")

def create_prediction_table(results_dict, save_path=None):
 
    y_test = results_dict['y_test']
    y_test_pred = results_dict['y_test_pred']
    
    table_data = pd.DataFrame({
        'NO': range(1, len(y_test) + 1),
        'pIC50': y_test.values if hasattr(y_test, 'values') else y_test,
        'Pred': y_test_pred,
        'Error': y_test.values - y_test_pred if hasattr(y_test, 'values') else y_test - y_test_pred
    })
    
    # 保存为CSV
    if save_path is None:
        save_path = os.path.join(OUTPUTS_FOLDER, 'test_predictions_table.csv')
    table_data.to_csv(save_path, index=False, float_format='%.2f')
    print(f"预测结果表格已保存!")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格
    table = ax.table(cellText=table_data.round(2).values,
                    colLabels=table_data.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15, 0.25, 0.25, 0.25])
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(len(table_data.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # 设置交替行颜色
    for i in range(1, len(table_data) + 1):
        for j in range(len(table_data.columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
    
    # plt.title('Table 1. Test set neural network model prediction results', fontsize=12, fontweight='bold', pad=20)
    
    table_img_path = os.path.join(FIGURES_FOLDER, 'test_predictions_table.png')
    plt.savefig(table_img_path, dpi=300, bbox_inches='tight')
    print(f"预测结果表格图片已保存!")
    
    return table_data

def plot_actual_vs_predicted(results_dict, save_path=None):
   
    y_test = results_dict['y_test']
    y_test_pred = results_dict['y_test_pred']
    samples = np.arange(1, len(y_test) + 1)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(samples, y_test, 
            marker='o', 
            color='#1f77b4', 
            linewidth=2, 
            markersize=5,
            label='Actual',
            alpha=0.7)
    
    # 绘制预测值（红色线）
    plt.plot(samples, y_test_pred, 
            marker='s', 
            color='#ff7f0e', 
            linewidth=2, 
            markersize=5,
            label='Predicted',
            alpha=0.7)
    
    plt.xlabel('Sample', fontsize=12, fontweight='bold')
    plt.ylabel('pIC50', fontsize=12, fontweight='bold')
    # plt.title('Actual vs Predicted pIC50 Values (Test Set)', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc='best', frameon=True, fontsize=11, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.xticks(samples)
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(FIGURES_FOLDER, 'actual_vs_predicted.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"实际值vs预测值折线图已保存!")
    

if __name__ == "__main__":

    X_train = pd.read_csv(os.path.join(DATA_FOLDER, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(DATA_FOLDER, 'y_train.csv')).squeeze("columns")
    X_test = pd.read_csv(os.path.join(DATA_FOLDER, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(DATA_FOLDER, 'y_test.csv')).squeeze("columns")

    best_params = optimize_hyperparameters(X_train, y_train)

    trained_model, prediction_results = train_neural_network_model(
        X_train, y_train, 
        X_test, y_test,
        hidden_layer_sizes=best_params['hidden_layer_sizes'], # 使用调优得到的神经元数
        alpha=best_params['alpha'],
        random_state=RANDOM_STATE
    )
    
    plot_regression_performance(prediction_results)
    prediction_table = create_prediction_table(prediction_results)
    plot_actual_vs_predicted(prediction_results)
    




  





