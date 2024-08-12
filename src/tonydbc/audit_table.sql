-- This table is used if AUDIT_PATH = database
-- It will document the speed of each TonyDBC query

CREATE TABLE IF NOT EXISTS `tony` (
    `id`                 BIGINT(20) NOT NULL AUTO_INCREMENT,
    `table_name`         VARCHAR(255) DEFAULT NULL,
    `query`              TEXT DEFAULT NULL,
    `started_at`         TIMESTAMP DEFAULT NULL,
    `completed_at`       TIMESTAMP DEFAULT NULL,
    `duration_seconds`   DOUBLE DEFAULT NULL,
    `payload_size_bytes` BIGINT DEFAULT NULL,
    `num_rows`           INT DEFAULT NULL,
    `num_cols`           INT DEFAULT NULL,
    `method`             VARCHAR(50) DEFAULT NULL,
    `MBps`               DOUBLE DEFAULT NULL,
    `session_uuid`       VARCHAR(36) DEFAULT NULL,
    `host`               VARCHAR(255) DEFAULT NULL,
    `database_name`      VARCHAR(255) DEFAULT NULL,
    `timezone`           VARCHAR(255) DEFAULT NULL,
    `created_at`         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    `updated_at`         TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
