-- Training-ready feature view: base + velocity + network + label (no typology).

CREATE OR REPLACE VIEW `{{project_id}}.{{dataset}}.{{features_view}}` AS
SELECT
  base.*,
  velocity.txn_count_24h,
  velocity.txn_count_7d,
  velocity.total_amount_24h,
  velocity.unique_receivers_24h,
  network.sender_fan_out_as_of,
  network.receiver_fan_in_as_of,
  network.both_high_risk,
  raw.is_fraud
FROM `{{project_id}}.{{dataset}}.{{features_base_view}}` AS base
JOIN `{{project_id}}.{{dataset}}.{{features_velocity_view}}` AS velocity
  USING (transaction_id)
JOIN `{{project_id}}.{{dataset}}.{{features_network_view}}` AS network
  USING (transaction_id)
JOIN `{{project_id}}.{{dataset}}.{{raw_table}}` AS raw
  USING (transaction_id);
