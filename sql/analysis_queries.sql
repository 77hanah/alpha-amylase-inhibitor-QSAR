
-- ============================================================================
-- Q1: 哪些化合物符合口服药物的 Lipinski 五规则？
--           Lipinski 规则: MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10
-- ============================================================================
SELECT '=== Q1: Lipinski 五规则筛选 ===' AS query;
SELECT
    compound_no,
    ROUND(pic50, 2) AS pic50,
    ROUND(mw, 1)    AS mw_da,
    ROUND(mol_log_p, 2) AS log_p,
    h_bond_donors,
    h_bond_acceptors,
    ROUND(tpsa, 1)  AS tpsa_a2,
    ROUND(qed, 3)   AS qed,
    CASE
        WHEN mw > 500 THEN 'MW超标'
        WHEN mol_log_p > 5 THEN 'LogP超标'
        WHEN h_bond_donors > 5 THEN 'HBD超标'
        WHEN h_bond_acceptors > 10 THEN 'HBA超标'
        ELSE '全部通过'
    END AS lipinski_status
FROM compounds
ORDER BY qed DESC;
/* 结论 :
   - 32 个化合物中有 30 个完全符合 Lipinski 五规则（通过率 93.75%）
   - 未通过者: 化合物 31 和 32，LogP=5.14 > 5（略超标，可优化）
   - QED 值最高的分子是化合物 7 (QED=0.786) 和 8 (0.786)
   - 整体成药性极佳，说明吡啶酮骨架本身具有良好类药性
*/


-- ============================================================================
-- Q2: 高活性分子集中在哪个分子量范围？
--           这对后续药物设计的分子量优化有指导意义
-- ============================================================================
SELECT '=== Q2: 分子量区间活性统计 ===' AS query;
SELECT
    CASE
        WHEN mw < 280 THEN '< 280 Da'
        WHEN mw BETWEEN 280 AND 300 THEN '280-300 Da'
        WHEN mw BETWEEN 301 AND 320 THEN '301-320 Da'
        WHEN mw BETWEEN 321 AND 350 THEN '321-350 Da'
        ELSE '> 350 Da'
    END AS mw_range,
    COUNT(*)          AS compound_count,
    ROUND(AVG(pic50), 3) AS avg_pic50,
    ROUND(MAX(pic50), 3) AS max_pic50,
    ROUND(MIN(pic50), 3) AS min_pic50,
    ROUND(MAX(pic50) - MIN(pic50), 3) AS activity_span
FROM compounds
GROUP BY mw_range
ORDER BY avg_pic50 DESC;
/* 结论 :
   - 活性与分子量呈正相关趋势：分子量越大，平均活性越高
   - >350 Da 组平均 pIC50 最高（4.611），<280 Da 组最低（4.254）
   - 但 >350 Da 组活性跨度最大（0.83），说明大分子活性差异大
   - 建议后续优化以 301-350 Da 为目标区间（活性和类药性平衡最优）
*/


-- ============================================================================
-- Q3: 哪些化合物活性最强？
-- ============================================================================
SELECT '=== Q3: 活性 Top 10 ===' AS query;
SELECT
    RANK() OVER (ORDER BY pic50 DESC) AS rank,
    compound_no,
    ROUND(pic50, 3) AS pic50,
    ROUND(mw, 1)    AS mw_da,
    ROUND(mol_log_p, 2) AS log_p,
    ROUND(qed, 3)   AS qed,
    ROUND(tpsa, 1)  AS tpsa
FROM compounds
ORDER BY pic50 DESC
LIMIT 10;
/* 结论 :
   - Top 1: 化合物 13 (pIC50=5.04) — 活性最强的分子
   - Top 3: 化合物 13, 12, 11 均为含硝基芳香环衍生物
   - 前 10 名中 QED 跨度大（0.516-0.786），说明活性和类药性不完全相关
   - 这些分子可作为后续结构优化的起点
*/


-- ============================================================================
-- Q4: 活性高的化合物在分子描述符上有什么共同特征？
-- ============================================================================
SELECT '=== Q4: 描述符-活性关联分析 ===' AS query;
WITH activity_groups AS (
    SELECT
        compound_id,
        pic50,
        CASE
            WHEN pic50 >= 4.5 THEN '高活性 (>=4.5)'
            WHEN pic50 >= 4.2 THEN '中活性 (4.2-4.49)'
            ELSE '低活性 (<4.2)'
        END AS activity_level
    FROM compounds
)
SELECT
    g.activity_level,
    COUNT(*)                          AS compound_count,
    ROUND(AVG(g.pic50), 3)            AS avg_pic50,
    ROUND(AVG(d.balaban_j), 3)        AS avg_balaban_j,
    ROUND(AVG(d.chi0), 3)             AS avg_chi0,
    ROUND(AVG(d.chi1), 3)             AS avg_chi1,
    ROUND(AVG(d.kappa1), 2)           AS avg_kappa1,
    ROUND(AVG(d.kappa2), 2)           AS avg_kappa2,
    ROUND(AVG(d.kappa3), 2)           AS avg_kappa3,
    ROUND(AVG(d.bertz_ct), 1)         AS avg_bertz_ct,
    ROUND(AVG(d.labute_asa), 1)       AS avg_labute_asa
FROM activity_groups g
JOIN molecular_descriptors d ON g.compound_id = d.compound_id
GROUP BY g.activity_level
ORDER BY g.activity_level DESC;
/* 结论 :
   - 高活性组拓扑描述符（Chi0=17.08, Chi1=11.49, Kappa2=6.31）显著高于低活性组
   - BertzCT（分子复杂度）在高活性组为 1011.4，低活性组仅 884.0
   - 说明分子复杂度与 α-淀粉酶抑制活性正相关
   - 这些拓扑描述符可作为 QSAR 模型的重要特征
*/


-- ============================================================================
-- Q5: 将化合物按活性分为高/中/低三档，并统计各档特征
-- ============================================================================
SELECT '=== Q5: CTE + 窗口函数活性分层 ===' AS query;
WITH ranked_compounds AS (
    SELECT
        compound_no,
        ROUND(pic50, 3) AS pic50,
        ROUND(mw, 1)    AS mw,
        ROUND(mol_log_p, 2) AS log_p,
        ROUND(qed, 3)   AS qed,
        RANK() OVER (ORDER BY pic50 DESC) AS activity_rank,
        NTILE(3) OVER (ORDER BY pic50 DESC) AS activity_tier
    FROM compounds
)
SELECT
    compound_no,
    pic50,
    mw,
    log_p,
    qed,
    activity_rank,
    CASE activity_tier
        WHEN 1 THEN '高活性'
        WHEN 2 THEN '中活性'
        WHEN 3 THEN '低活性'
    END AS tier_label
FROM ranked_compounds
ORDER BY activity_rank;
/* 结论 :
   - NTILE(3) 将 32 个化合物平分为高中低三档
   - 高活性组（11 个）pIC50 ≥ 4.65，低活性组 pIC50 ≤ 4.30
   - RANK() 显示化合物 14 和 15 并列第 9，化合物 11 和 12 前后相邻
   - 窗口函数在这里的价值：同时看到排名和档次，比单纯 ORDER BY 信息量更大
*/


-- ============================================================================
-- Q6: 哪些化合物既活性高又成药性好？
-- ============================================================================
SELECT '=== Q6: 高活性 + ADMET 合格候选 ===' AS query;
SELECT
    compound_no,
    ROUND(pic50, 3) AS pic50,
    ROUND(mw, 1)    AS mw,
    ROUND(mol_log_p, 2) AS log_p,
    ROUND(qed, 3)   AS qed,
    ROUND(tpsa, 1)  AS tpsa,
    '推荐候选' AS recommendation
FROM compounds
WHERE compound_no IN (
    SELECT compound_no
    FROM compounds
    WHERE mw <= 500 AND mol_log_p <= 5
      AND h_bond_donors <= 5 AND h_bond_acceptors <= 10
      AND tpsa <= 140
)
AND pic50 >= (SELECT AVG(pic50) FROM compounds)
ORDER BY pic50 DESC;
/* 结论 :
   - 16 个化合物被标记为推荐候选（活性高于均值 + 符合 Lipinski + TPSA合格）
   - 其中化合物 13 (pIC50=5.04) 为最优候选
   - 这些分子已标记入库，可直接导出用于分子对接验证
*/


-- ============================================================================
-- Q7: : 将筛选出的候选分子标记为推荐，方便后续分析引用
--           使用事务保证数据一致性
-- ============================================================================
BEGIN TRANSACTION;

UPDATE compounds
SET is_recommended = 1
WHERE compound_no IN (
    SELECT compound_no
    FROM compounds
    WHERE mw <= 500 AND mol_log_p <= 5
      AND h_bond_donors <= 5 AND h_bond_acceptors <= 10
      AND tpsa <= 140
      AND pic50 >= (SELECT AVG(pic50) FROM compounds)
);

SELECT '=== Q7: 标记推荐候选 ===' AS query;
SELECT
    '已标记 ' || changes() || ' 个推荐候选分子' AS update_result;

COMMIT;
-- 回滚注释: 如果只是分析不想改数据，把上面 COMMIT 改成 ROLLBACK
-- ROLLBACK;


-- ============================================================================
-- Q8 : 活性高的分子是否类药性一定好？pIC50 与 QED 的关系如何？
-- ============================================================================
SELECT '=== Q8 (Bonus): 活性 vs 类药性 ===' AS query;
SELECT
    CASE
        WHEN pic50 >= 4.5 THEN '高活性 (>=4.5)'
        WHEN pic50 >= 4.2 THEN '中活性 (4.2-4.49)'
        ELSE '低活性 (<4.2)'
    END AS activity_level,
    ROUND(AVG(qed), 4) AS avg_qed,
    ROUND(MIN(qed), 4) AS min_qed,
    ROUND(MAX(qed), 4) AS max_qed,
    ROUND(AVG(tpsa), 1) AS avg_tpsa,
    ROUND(AVG(mol_log_p), 3) AS avg_log_p,
    ROUND(AVG(mw), 1) AS avg_mw
FROM compounds
GROUP BY activity_level
ORDER BY activity_level DESC;
/* 结论 :
   - 关键发现: 低活性组的 QED（0.7816）反而高于高活性组（0.6479）
   - 说明类药性和活性之间存在权衡（trade-off），非正相关
   - 高活性组平均分子量 337.9 Da、LogP 4.01；低活性组 MW 280.8 Da、LogP 2.83
   - 启示: 先按活性筛选，再对候选分子做 ADMET 评估，而不是反过来
*/


-- ============================================================================
-- 汇总报告（最终输出）
-- ============================================================================
SELECT '=== 分析汇总 ===' AS report;
SELECT
    COUNT(*) AS total_compounds,
    ROUND(AVG(pic50), 3) AS mean_pic50,
    ROUND(MAX(pic50), 3) AS max_pic50,
    ROUND(MIN(pic50), 3) AS min_pic50,
    ROUND(AVG(mw), 1) AS mean_mw,
    ROUND(AVG(mol_log_p), 3) AS mean_logp,
    ROUND(AVG(qed), 4) AS mean_qed
FROM compounds;
