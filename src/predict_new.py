import pandas as pd
import numpy as np
import os
import joblib
from rdkit import Chem
from config import *
from calculate_descriptors import calculate_descriptors


def predict_new_compounds(smiles_csv_filename='new_compounds.csv'):

    # 读取新化合物的 SMILES 
    file_path = os.path.join(DATA_FOLDER, smiles_csv_filename)
    try:
        df_input = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
        return
    
    if SMILES_COLUMN not in df_input.columns:
        print(f"错误: 找不到列 '{SMILES_COLUMN}'，请检查 CSV 文件的列名。")
        return
    
    smiles_list = df_input[SMILES_COLUMN].tolist()
    print(f"共读取 {len(smiles_list)} 个新化合物。")
    
    # 检查 SMILES 有效性
    valid_mask = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        valid_mask.append(mol is not None)
    
    invalid_count = valid_mask.count(False)
    if invalid_count > 0:
        print(f"警告: 有 {invalid_count} 个无效 SMILES，这些行的预测结果将为 NaN。")


    # 计算描述符 
    print("\n--- Step 2: 计算分子描述符 ---")
    df_desc = calculate_descriptors(smiles_list)


    # 特征对齐
    print("\n--- Step 3: 特征对齐与预处理 ---")
    # 加载训练集的特征列名
    X_train_ref = pd.read_csv(os.path.join(DATA_FOLDER, 'X_train.csv'))
    trained_feature_cols = X_train_ref.columns.tolist()
    print(f"训练时使用的特征数量: {len(trained_feature_cols)}")
    # 与训练集一致的缺失值处理
    X_new = df_desc.copy()
    X_new = X_new.replace(0, pd.NA)
    # 只保留训练时筛选后的特征列
    missing_cols = set(trained_feature_cols) - set(X_new.columns)
    if missing_cols:
        print(f"警告: 新数据缺少 {len(missing_cols)} 个训练特征列，将用 0 补全。")
        for col in missing_cols:
            X_new[col] = 0 
    X_new = X_new[trained_feature_cols]   
    X_new = X_new.fillna(0)
    print(f"特征对齐完成，当前特征维度: {X_new.shape}")

    # 归一化
    print("\n--- Step 4: 特征归一化 ---")
    scaler_X = joblib.load(os.path.join(OUTPUTS_FOLDER, 'scaler_X.pkl'))
    X_new_scaled = scaler_X.transform(X_new)
    print("归一化完成。")

    # 模型预测
    print("\n--- Step 5: 模型预测 ---")
    model = joblib.load(os.path.join(OUTPUTS_FOLDER, 'trained_neural_network_model.pkl'))
    y_pred_scaled = model.predict(X_new_scaled)
    scaler_y = joblib.load(os.path.join(OUTPUTS_FOLDER, 'scaler_y.pkl'))
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    print("预测完成！")

    # 保存结果
    df_result = df_input.copy()
    df_result['pIC50_predicted'] = y_pred
    df_result.loc[[not v for v in valid_mask], 'pIC50_predicted'] = np.nan
    # 按预测活性从高到低排序
    df_result = df_result.sort_values('pIC50_predicted', ascending=False).reset_index(drop=True)
    output_path = os.path.join(OUTPUTS_FOLDER, 'new_compounds_predicted.csv')
    df_result.to_csv(output_path, index=False, float_format='%.4f')
    print(f"预测结果已保存至: '{output_path}'")

    print("\n预测结果预览 (前10条):")
    print(df_result.head(10).to_string(index=False))
    
    return df_result


if __name__ == "__main__":
    results = predict_new_compounds(smiles_csv_filename='new_compounds.csv')