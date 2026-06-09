-- AutoML training input: temporal splits with predefined split column.
-- Vertex AI requires TRAIN, VALIDATE, and TEST values (case-sensitive).

CREATE OR REPLACE VIEW `{{project_id}}.{{dataset}}.{{automl_input_view}}` AS
SELECT *, 'TRAIN' AS ml_split
FROM `{{project_id}}.{{dataset}}.{{features_train_view}}`
UNION ALL
SELECT *, 'VALIDATE' AS ml_split
FROM `{{project_id}}.{{dataset}}.{{features_val_view}}`
UNION ALL
SELECT *, 'TEST' AS ml_split
FROM `{{project_id}}.{{dataset}}.{{features_test_view}}`;
