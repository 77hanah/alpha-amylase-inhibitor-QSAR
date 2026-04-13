import os
import pandas as pd
import joblib
import numpy as np
from scipy.stats import pearsonr
from config import *
from calculate_descriptors import calculate_descriptors
from data_process import data_processor
from PCA import pca, visualize_pca
from model_results import optimize_hyperparameters, train_neural_network_model, plot_regression_performance, create_prediction_table, plot_actual_vs_predicted
from applicability_domain_evaluation import evaluate_applicability_domain
from y_randomation_evaluate import run_randomization_loop, plot_y_randomization_results


if __name__ == "__main__":

    # ===计算分子描述符===
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    file_path = os.path.join(DATA_FOLDER, SMILES_FILE)
    df_input = pd.read_csv(file_path)
    
    df_calculated_descriptors = calculate_descriptors(df_input[SMILES_COLUMN])
    df_final = pd.concat([df_input['NO'],df_input['pic50'], df_calculated_descriptors], axis=1)

    output_file = 'descriptors.csv'
    df_final.to_csv(os.path.join(DATA_FOLDER, output_file), index=False)
    
    print(f"\n分子描述符已保存到 '{output_file}'")  

    # ===数据处理===
    X_train, X_test, y_train, y_test = data_processor()

    # ===PCA===
    X_train = pd.read_csv(os.path.join(DATA_FOLDER,'X_train.csv'))
    y_train = pd.read_csv(os.path.join(DATA_FOLDER, 'y_train.csv')).squeeze("columns")

    pca_model, pca_results_dict = pca(X_train)
    visualize_pca(pca_model, X_train, y_train)

    # ===训练神经网络模型===
    X_train = pd.read_csv(os.path.join(DATA_FOLDER, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(DATA_FOLDER, 'y_train.csv')).squeeze("columns")
    X_test = pd.read_csv(os.path.join(DATA_FOLDER, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(DATA_FOLDER, 'y_test.csv')).squeeze("columns")

    scaler_y = joblib.load(os.path.join(OUTPUTS_FOLDER, 'scaler_y.pkl'))
    print("已加载 scaler_y.pkl")

    best_params = optimize_hyperparameters(X_train, y_train)

    trained_model, prediction_results = train_neural_network_model(
        X_train, y_train, 
        X_test, y_test,
        scaler_y=scaler_y,  
        hidden_layer_sizes=best_params['hidden_layer_sizes'],
        random_state=RANDOM_STATE
    )
    
    plot_regression_performance(prediction_results)
    prediction_table = create_prediction_table(prediction_results)
    plot_actual_vs_predicted(prediction_results)

    # ===适用域评估===

    model_path = os.path.join(OUTPUTS_FOLDER, 'trained_neural_network_model.pkl')
    x_test_path = os.path.join(DATA_FOLDER, 'X_test.csv')
    y_test_path = os.path.join(DATA_FOLDER, 'y_test.csv')


    trained_model = joblib.load(model_path)
    X_test_data = pd.read_csv(x_test_path)
    y_test_data = pd.read_csv(y_test_path).squeeze("columns")

    evaluate_applicability_domain(trained_model, X_test_data, y_test_data)

    # ===Y-随机化测试===
    X_train = pd.read_csv(os.path.join(DATA_FOLDER, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(DATA_FOLDER, 'y_train.csv')).squeeze("columns")
    X_test = pd.read_csv(os.path.join(DATA_FOLDER, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(DATA_FOLDER, 'y_test.csv')).squeeze("columns")
    
    X_full = pd.concat([X_train, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)
    print(f"数据加载成功！总样本数: {len(y_full)}")

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