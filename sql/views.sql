
-- ============================================================
-- 视图 1: 符合 Lipinski 五规则的候选分子
-- ============================================================
CREATE VIEW IF NOT EXISTS v_lipinski_candidates AS
SELECT
    compound_no,
    ROUND(pic50, 2) AS pic50,
    ROUND(mw, 1)    AS mw_da,
    ROUND(mol_log_p, 2) AS log_p,
    h_bond_donors,
    h_bond_acceptors,
    ROUND(tpsa, 1)  AS tpsa,
    ROUND(qed, 3)   AS qed
FROM compounds
WHERE mw <= 500 AND mol_log_p <= 5
  AND h_bond_donors <= 5 AND h_bond_acceptors <= 10
ORDER BY qed DESC;

-- ============================================================
-- 视图 2: 活性 Top 10 化合物
-- ============================================================
CREATE VIEW IF NOT EXISTS v_top10_activity AS
SELECT
    RANK() OVER (ORDER BY pic50 DESC) AS rank,
    compound_no,
    ROUND(pic50, 3) AS pic50,
    ROUND(mw, 1)    AS mw,
    ROUND(mol_log_p, 2) AS log_p,
    ROUND(qed, 3)   AS qed
FROM compounds
ORDER BY pic50 DESC
LIMIT 10;

-- ============================================================
-- 视图 3: 推荐候选分子（高活性 + ADMET合格）
-- ============================================================
CREATE VIEW IF NOT EXISTS v_recommended_candidates AS
SELECT
    compound_no,
    ROUND(pic50, 3) AS pic50,
    ROUND(mw, 1)    AS mw,
    ROUND(mol_log_p, 2) AS log_p,
    ROUND(tpsa, 1)  AS tpsa,
    ROUND(qed, 3)   AS qed
FROM compounds
WHERE is_recommended = 1
ORDER BY pic50 DESC;

-- ============================================================
-- 视图 4: 按活性分层的化合物统计
-- ============================================================
CREATE VIEW IF NOT EXISTS v_activity_tiers AS
WITH ranked AS (
    SELECT compound_no, pic50, mw, mol_log_p, qed,
           NTILE(3) OVER (ORDER BY pic50 DESC) AS tier
    FROM compounds
)
SELECT compound_no,
       ROUND(pic50, 3) AS pic50,
       ROUND(mw, 1) AS mw,
       ROUND(mol_log_p, 2) AS log_p,
       ROUND(qed, 3) AS qed,
       CASE tier
           WHEN 1 THEN 'High'
           WHEN 2 THEN 'Medium'
           ELSE 'Low'
       END AS activity_tier
FROM ranked
ORDER BY pic50 DESC;
