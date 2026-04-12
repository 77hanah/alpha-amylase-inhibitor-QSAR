import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from config import *

def data_processor():

    df = pd.read_csv(os.path.join(DATA_FOLDER, 'descriptors.csv'))

    X = df.drop(columns=['NO', 'pic50'])
    y = df['pic50']

    # 处理缺失值
    X_clean = X.replace(0, pd.NA)
    missing_ratio = X_clean.isnull().sum() / len(X_clean)
    columns_to_keep = missing_ratio[missing_ratio <= 0.1].index     # 保留缺失比例不超过10%的列
    X_clean = X_clean[columns_to_keep]
    X = X_clean.fillna(0)  

    # 处理异常值
    selector = VarianceThreshold()  # 利用方差阈值选择器删除方差为0的特征
    X_selected = selector.fit_transform(X)
    selected_cols = X.columns[selector.get_support()]
    X = pd.DataFrame(X_selected, columns=selected_cols)

    # 数据集划分 (7:3) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    print(f"训练集: {X_train.shape[0]} 个样本")
    print(f"测试集: {X_test.shape[0]} 个样本")

    # 特征缩放 (Z-score标准化) 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    X_train.to_csv(os.path.join(DATA_FOLDER,'X_train.csv'), index=False, header=True)
    X_test.to_csv(os.path.join(DATA_FOLDER, 'X_test.csv'), index=False, header=True)
    y_train.to_csv(os.path.join(DATA_FOLDER, 'y_train.csv'), index=False, header=True)
    y_test.to_csv(os.path.join(DATA_FOLDER, 'y_test.csv'), index=False, header=True)

    print("数据完成预处理！")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = data_processor()