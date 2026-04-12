import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
from config import *

def calculate_descriptors(smiles_list):
    """计算分子描述符 """
    
    # 描述符名称列表
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    
    results = []
    
    print(f"开始计算 {len(smiles_list)} 个分子的描述符...")
    
    for smiles in smiles_list:
        mol_descriptors = {}

        # 将SMILES转换为RDKit的分子对象
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is not None:
            for name, func in Descriptors._descList:
                try:
                    value = func(mol)
                    mol_descriptors[name] = value
                except Exception as e:
                    mol_descriptors[name] = None
                    print(f"分子 '{smiles}' 的描述符 '{name}' 计算失败: {e}")
        else:
            print(f"错误:无效的SMILES字符串 '{smiles}'。")
            mol_descriptors = {name: None for name in descriptor_names}
        
        results.append(mol_descriptors)
        
    print("描述符计算完成！")
    
    df_descriptors = pd.DataFrame(results)

    return df_descriptors


if __name__ == "__main__":

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    file_path = os.path.join(DATA_FOLDER, SMILES_FILE)
    try:
        df_input = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误: 输入文件 '{SMILES_FILE}' 未找到。")
        exit()

    if SMILES_COLUMN not in df_input.columns:
        print(f"错误: 在输入文件中找不到名为 '{SMILES_COLUMN}' 的列。")
        exit()
    
    df_calculated_descriptors = calculate_descriptors(df_input[SMILES_COLUMN])
    df_final = pd.concat([df_input['NO'],df_input['pic50'], df_calculated_descriptors], axis=1)

    output_file = 'descriptors.csv'
    df_final.to_csv(os.path.join(DATA_FOLDER, output_file), index=False)
    
    print(f"\n分子描述符已保存到 '{output_file}'")


