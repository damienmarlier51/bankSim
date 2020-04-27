WITH
    --
    dataset AS (
        SELECT * FROM {table}
    ),
    --
    customer_category_map_0 AS (
        SELECT DISTINCT customer, category
        FROM dataset
    ),
    --
    customer_with_multiple_categories_0 AS (
        SELECT customer,
               COUNT(*) AS cnt
        FROM customer_category_map_0
        GROUP BY customer
        HAVING cnt > 1
    ),
    --
    customer_with_multiple_categories_1 AS (
        SELECT LSQ.customer,
               LSQ.category,
               LSQ.merchant
        FROM dataset AS LSQ
        INNER JOIN customer_with_multiple_categories_0 AS RSQ
        USING(customer)
    )
--
SELECT *
FROM customer_with_multiple_categories_1
ORDER BY customer,
         category,
         merchant
