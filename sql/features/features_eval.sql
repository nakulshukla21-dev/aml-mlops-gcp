-- Evaluation view: training features plus typology for per-pattern model analysis.
-- Do NOT feed typology into AutoML.

CREATE OR REPLACE VIEW `{{project_id}}.{{dataset}}.{{features_eval_view}}` AS
SELECT
  f.*,
  r.typology
FROM `{{project_id}}.{{dataset}}.{{features_view}}` AS f
JOIN `{{project_id}}.{{dataset}}.{{raw_table}}` AS r
  USING (transaction_id);
