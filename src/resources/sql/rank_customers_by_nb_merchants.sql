WITH
    --
    dataset AS (
        SELECT * FROM {table}
    ),
    --
    customer_merchant_map_0 AS (
        SELECT DISTINCT customer, merchant
        FROM dataset
        WHERE step < 30
    ),
    --
    customer_merchant_map_1 AS (
        SELECT customer,
               COUNT(*) AS nb_unq_merchants
        FROM customer_merchant_map_0
        GROUP BY customer
        ORDER BY nb_unq_merchants DESC,
                 customer ASC
    )
--
SELECT *
FROM customer_merchant_map_1
