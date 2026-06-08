-- Temporal train / validation / test views over features_training.
-- Default split: train Jan-Sep, val Oct, test Nov-Dec 2024.

CREATE OR REPLACE VIEW `{{project_id}}.{{dataset}}.{{features_train_view}}` AS
SELECT *
FROM `{{project_id}}.{{dataset}}.{{features_view}}`
WHERE txn_date <= DATE('{{train_end}}');

CREATE OR REPLACE VIEW `{{project_id}}.{{dataset}}.{{features_val_view}}` AS
SELECT *
FROM `{{project_id}}.{{dataset}}.{{features_view}}`
WHERE txn_date > DATE('{{train_end}}')
  AND txn_date <= DATE('{{val_end}}');

CREATE OR REPLACE VIEW `{{project_id}}.{{dataset}}.{{features_test_view}}` AS
SELECT *
FROM `{{project_id}}.{{dataset}}.{{features_view}}`
WHERE txn_date > DATE('{{val_end}}');
