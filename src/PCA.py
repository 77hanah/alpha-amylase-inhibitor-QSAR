import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
from config import *
import os

def pca(X_scaled, n_components=10, random_state=RANDOM_STATE):

    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X_scaled)

    explained_variance = pca.explained_variance_ratio_      # 贡献率
    cumulative_variance = np.cumsum(explained_variance)     # 累计贡献率
    
    pca_results = {
        "explained_variance_ratio": explained_variance,
        "cumulative_variance_ratio": cumulative_variance
    }
    
    print("PCA计算完成。")
    print(f"贡献率分别为: {explained_variance[0]*100:.2f}%",
          f"{explained_variance[1]*100:.2f}%",
          f"{explained_variance[2]*100:.2f}%",
          f"{explained_variance[3]*100:.2f}%",
          f"{explained_variance[4]*100:.2f}%",
          f"{explained_variance[5]*100:.2f}%",
          f"{explained_variance[6]*100:.2f}%",
          f"{explained_variance[7]*100:.2f}%",
          f"{explained_variance[8]*100:.2f}%",
          f"{explained_variance[9]*100:.2f}%"
          )

    print(f"PC1+PC2 累计贡献率: {cumulative_variance[1]*100:.2f}%")

    return pca, pca_results

def visualize_pca(pca_model, X_scaled, y_target=None):

    plt.style.use('science') 

    # ---得分散点图---
    X_pca_scores = pca_model.transform(X_scaled)
    explained_variance = pca_model.explained_variance_ratio_

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
    x=X_pca_scores[:, 0],  # PC1为X轴
    y=X_pca_scores[:, 1],  # PC2为Y轴
    s=100,                 # 点大小
    alpha=0.8,             # 点透明度
    hue=y_target,          # 如果有类别标签，则进行颜色区分
    palette='viridis',     # 配色方案
    edgecolor='black',     # 点的边框颜色
    linewidth=0.5          # 点边框的粗细
    )   
    plt.title('PCA Score Plot', fontsize=16)
    plt.xlabel(f'PC1 ({explained_variance[0]*100:.2f}%)', fontsize=12)
    plt.ylabel(f'PC2 ({explained_variance[1]*100:.2f}%)', fontsize=12)
    plt.legend(title=y_target.name)
    plt.grid(True, alpha=0.3, linestyle='--')   # 网格线
    plt.axhline(0, color='grey', lw=0.5)    # 水平零线
    plt.axvline(0, color='grey', lw=0.5)    # 垂直零线
    plt.savefig(os.path.join(FIGURES_FOLDER, 'pca_score_plot.png'), dpi=300)
    print("'pca_score_plot.png'已保存")
    plt.close()

    # ---主成分贡献雷达图---
    labels = [f'PC{i+1}' for i in range(len(explained_variance))]
    values = explained_variance * 100  # 转换为百分比

    # 计算每个轴的角度
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()   # 将 NumPy 数组转为 Python 列表
    # 为了让雷达图闭合，需要将第一个值和角度追加到末尾
    values = np.concatenate((values,[values[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='#87ceeb', alpha=0.3)
    ax.plot(angles, values, color='#1f77b4', linewidth=2, marker='o', markersize=3)
    # 数值标签推荐手动加
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=14)     # 设置角度网格和标签
    ax.spines['polar'].set_color('grey')     # 设置极轴颜色
    ax.grid(color='grey')
    ax.set_title('Contribution of Top Principal Components', size=20, color='black', y=1.05)
    ax.set_rlabel_position(30)  # 设置径向标签位置
    max_val = np.ceil(values.max()) 
    yticks = np.arange(0, max_val + 10, 10)  
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{y}' for y in yticks], color='black', size=12)  
    ax.set_ylim(0, max_val) 

    plt.savefig(os.path.join(FIGURES_FOLDER, 'pca_radar_chart.png'), dpi=300)
    print("'pca_radar_chart.png'已保存")
    plt.close()

    # ---载荷图---
    loadings = pca_model.components_
    pc1_loadings = loadings[0]
    pc2_loadings = loadings[1]
    feature_indices = np.arange(len(pc1_loadings))
    feature_names = X_scaled.columns

    for i in range(2):
        pc_index = i + 1
        
        loadings_df = pd.DataFrame({
            'Feature': feature_names,
            'Loading': loadings[i]
        })
        
        positive_contributors = loadings_df[loadings_df['Loading'] > 0]
        sorted_pos = positive_contributors.sort_values(by='Loading', ascending=False)
        negative_contributors = loadings_df[loadings_df['Loading'] < 0]
        sorted_neg = negative_contributors.sort_values(by='Loading', ascending=True)

        save_path = os.path.join(OUTPUTS_FOLDER, f'PC{pc_index}_feature_contributions.txt')
        with open(save_path, 'w', encoding='utf-8', newline='') as f:
            f.write("正贡献特征:\n")
            f.write(sorted_pos.head(10).to_string(index=False) + "\n\n")  
            f.write("负贡献特征:\n")
            f.write(sorted_neg.head(10).to_string(index=False))
    print(f"\n特征贡献结果已成功保存!")


    fig, axes = plt.subplots(1, 2, figsize=(10, 8))

    # PC1s
    axes[0].bar(feature_indices, pc1_loadings, color='#1f77b4', alpha=0.8, width=0.5)
    axes[0].axhline(0, color='grey', linewidth=0.8)    # y=0的基准线
    axes[0].set_title('PC1', fontsize=18)
    axes[0].set_xlabel('Original', fontsize=14)
    axes[0].set_ylabel('Weight', fontsize=14)
    axes[0].set_xlim(-1, len(feature_indices))  # 去掉两边的多余空白

    # 绘制PC2的贡献图
    axes[1].bar(feature_indices, pc2_loadings, color='#1f77b4', alpha=0.8, width=0.5)
    axes[1].axhline(0, color='grey', linewidth=0.8)
    axes[1].set_title('PC2', fontsize=18)
    axes[1].set_xlabel('Original Feature Index', fontsize=14)
    axes[1].set_ylabel('Weight', fontsize=14) 
    axes[1].set_xlim(-1, len(feature_indices))
    
    # 调整子图间的间距，防止重叠
    plt.tight_layout(pad=3.0)
    
    plt.savefig(os.path.join(FIGURES_FOLDER, 'feature_contribution_plot.png'), dpi=300)
    print("'feature_contribution_plot.png'已保存")
    plt.close()

if __name__ == "__main__":

    try:
        X_train_scaled = pd.read_csv(os.path.join(DATA_FOLDER,'X_train.csv'))
        y_train = pd.read_csv(os.path.join(DATA_FOLDER, 'y_train.csv')).squeeze("columns")     # 使用 squeeze() 将其转换为 Series 
    except FileNotFoundError as e:
        print(f"错误: {e}. 请确保 'X_train_preprocessed.csv' 和 'y_train.csv' 文件都存在。")
        exit()

    pca_model, pca_results_dict = pca(X_train_scaled)
    visualize_pca(pca_model, X_train_scaled, y_train)


