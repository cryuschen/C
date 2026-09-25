WITH d AS (
      SELECT dataset,
        MAX(CASE WHEN variant='original' THEN weighted_MSE END) AS original_fit_error,
        MAX(CASE WHEN variant='same_kernel_no_ridge' THEN weighted_MSE END) AS nr,
        MAX(CASE WHEN variant='same_kernel_plus_affine' THEN weighted_MSE END) AS af
      FROM descriptive_counterfactuals WHERE stage='v7' GROUP BY dataset),
    w AS (
      SELECT dataset,
        AVG(CASE WHEN variant='original' THEN weighted_MSE END) AS original_test,
        AVG(CASE WHEN variant='same_kernel_no_ridge' THEN weighted_MSE END) AS nr,
        AVG(CASE WHEN variant='same_kernel_plus_affine' THEN weighted_MSE END) AS af
      FROM heldout_waveforms WHERE split='test' GROUP BY dataset),
    b AS (
      SELECT dataset, SUM(total_error) AS total,
        SUM(coefficient_transfer_error) AS transfer, SUM(span_approximation_error) AS approximation
      FROM prediction_error_decomposition GROUP BY dataset)
    SELECT d.dataset,d.original_fit_error,
      1-d.nr/d.original_fit_error AS no_ridge_fit_reduction,
      1-d.af/d.original_fit_error AS affine_fit_reduction,
      w.nr/w.original_test-1 AS no_ridge_test_change,
      w.af/w.original_test-1 AS affine_test_change,
      b.transfer/b.total AS coefficient_transfer_share,
      b.approximation/b.total AS span_approximation_share
    FROM d JOIN w USING(dataset) JOIN b USING(dataset) ORDER BY d.dataset;

SELECT * FROM counterfactual_classification ORDER BY dataset,variant;
