
-- --------------------------------------------------------------------------
-- 化合物主表
-- 存储 32 个吡啶酮衍生物的核心信息
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS compounds (
    compound_id        INTEGER PRIMARY KEY AUTOINCREMENT,  -- 化合物编号 (PK)
    compound_no        INTEGER NOT NULL,                   -- 原始编号 (NO)
    smiles             TEXT    NOT NULL,                   -- SMILES 结构式
    pic50              REAL,                               -- 活性值 (pIC50, 越大越强)

    -- ---- Lipinski 五规则相关字段（药物筛选核心指标）----
    mw                 REAL,   -- 分子量 Molecular Weight (Da), 规则: ≤500
    mol_log_p          REAL,   -- 脂水分配系数 LogP, 规则: ≤5
    h_bond_donors      INTEGER,-- 氢键供体数, 规则: ≤5
    h_bond_acceptors   INTEGER,-- 氢键受体数, 规则: ≤10

    -- ---- 其他重要的类药性指标 ----
    tpsa               REAL,   -- 拓扑极性表面积 (Å²), 反映口服吸收
    qed                REAL,   -- 定量类药性评估 (0-1, 越接近1越好)
    num_rotatable_bonds INTEGER,-- 可旋转键数, 反映分子柔性
    heavy_atom_count   INTEGER,-- 重原子数
    fraction_csp3      REAL,   -- sp3 碳比例, 反映分子饱和度 (三维性)
    ring_count         INTEGER,-- 环数

    -- ---- 元数据 ----
    created_at         TEXT    DEFAULT (datetime('now')),   -- 记录创建时间
    is_recommended     INTEGER DEFAULT 0                   -- 推荐候选标记 (0=否,1=是)
);

-- 为常用查询字段创建索引
CREATE INDEX idx_compounds_pic50     ON compounds(pic50);
CREATE INDEX idx_compounds_mw        ON compounds(mw);
CREATE INDEX idx_compounds_mol_log_p ON compounds(mol_log_p);


-- --------------------------------------------------------------------------
-- 分子描述符表
-- 存储 RDKit 计算的全部 200+ 个分子描述符
-- 与 compounds 表为 1:1 关系
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS molecular_descriptors (
    descriptor_id      INTEGER PRIMARY KEY AUTOINCREMENT,  -- 描述符记录ID (PK)
    compound_id        INTEGER NOT NULL,                   -- 关联化合物ID (FK)
    -- ---- 电子描述符 ----
    max_partial_charge     REAL,   -- 最大部分电荷
    min_partial_charge     REAL,   -- 最小部分电荷
    fp_density_morgan1     REAL,   -- Morgan指纹密度 (半径1)
    fp_density_morgan2     REAL,   -- Morgan指纹密度 (半径2)
    fp_density_morgan3     REAL,   -- Morgan指纹密度 (半径3)
    -- ---- 拓扑描述符 ----
    balaban_j              REAL,   -- Balaban J 指数
    bertz_ct               REAL,   -- Bertz 复杂度
    kappa1                 REAL,   -- Kappa1 形状指数
    kappa2                 REAL,   -- Kappa2 形状指数
    kappa3                 REAL,   -- Kappa3 形状指数
    chi0                   REAL,   -- Chi0 连接性指数
    chi1                   REAL,   -- Chi1 连接性指数
    -- ---- 分子表面性质 ----
    labute_asa             REAL,   -- Labute 近似表面积
    peoe_vsa1              REAL,   -- PEOE VSA 分箱1
    peoe_vsa2              REAL,   -- PEOE VSA 分箱2
    peoe_vsa3              REAL,   -- PEOE VSA 分箱3
    peoe_vsa4              REAL,   -- PEOE VSA 分箱4
    peoe_vsa5              REAL,   -- PEOE VSA 分箱5
    peoe_vsa6              REAL,   -- PEOE VSA 分箱6
    peoe_vsa7              REAL,   -- PEOE VSA 分箱7
    peoe_vsa8              REAL,   -- PEOE VSA 分箱8
    peoe_vsa9              REAL,   -- PEOE VSA 分箱9
    peoe_vsa10             REAL,   -- PEOE VSA 分箱10
    peoe_vsa11             REAL,   -- PEOE VSA 分箱11
    peoe_vsa12             REAL,   -- PEOE VSA 分箱12
    peoe_vsa13             REAL,   -- PEOE VSA 分箱13
    peoe_vsa14             REAL,   -- PEOE VSA 分箱14
    slogp_vsa1             REAL,   -- SlogP VSA 分箱1
    slogp_vsa2             REAL,   -- SlogP VSA 分箱2
    slogp_vsa3             REAL,   -- SlogP VSA 分箱3
    slogp_vsa4             REAL,   -- SlogP VSA 分箱4
    slogp_vsa5             REAL,   -- SlogP VSA 分箱5
    slogp_vsa6             REAL,   -- SlogP VSA 分箱6
    slogp_vsa7             REAL,   -- SlogP VSA 分箱7
    slogp_vsa8             REAL,   -- SlogP VSA 分箱8
    slogp_vsa9             REAL,   -- SlogP VSA 分箱9
    slogp_vsa10            REAL,   -- SlogP VSA 分箱10
    slogp_vsa11            REAL,   -- SlogP VSA 分箱11
    slogp_vsa12            REAL,   -- SlogP VSA 分箱12
    -- ---- BCUT 特征值 ----
    bcut2d_mwhi            REAL,   -- BCUT2D 分子量高
    bcut2d_mwlow           REAL,   -- BCUT2D 分子量低
    bcut2d_log_phi         REAL,   -- BCUT2D LogP 高
    bcut2d_log_plow        REAL,   -- BCUT2D LogP 低
    bcut2d_mrhi            REAL,   -- BCUT2D 摩尔折射率高
    bcut2d_mrlow           REAL,   -- BCUT2D 摩尔折射率低
    -- ---- 官能团计数 (fr_ 系列) ----
    fr_amide               INTEGER,-- 酰胺基团计数
    fr_ester               INTEGER,-- 酯基计数
    fr_ketone              INTEGER,-- 酮基计数
    fr_ether               INTEGER,-- 醚键计数
    fr_benzene             INTEGER,-- 苯环计数
    fr_pyridine            INTEGER,-- 吡啶环计数
    fr_nitrile             INTEGER,-- 氰基计数
    fr_halogen             INTEGER,-- 卤素计数
    fr_Ar_OH               INTEGER,-- 酚羟基计数
    fr_Al_OH               INTEGER,-- 脂肪醇羟基计数

    FOREIGN KEY (compound_id) REFERENCES compounds(compound_id)
);

-- 描述符表的索引
CREATE INDEX idx_desc_compound_id ON molecular_descriptors(compound_id);
CREATE INDEX idx_desc_balaban_j   ON molecular_descriptors(balaban_j);
CREATE INDEX idx_desc_chi0        ON molecular_descriptors(chi0);
