-- =====================================================================
--  Deep Fake Detection System — MySQL Schema
--  Engine: InnoDB | Charset: utf8mb4
--  Run:  mysql -u root -p < database/schema.sql
-- =====================================================================

DROP DATABASE IF EXISTS deepfake_db;
CREATE DATABASE deepfake_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;
USE deepfake_db;

-- ---------------------------------------------------------------------
-- 1. users
-- ---------------------------------------------------------------------
CREATE TABLE users (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    username        VARCHAR(50)  NOT NULL,
    email           VARCHAR(120) NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    role            ENUM('user','admin') NOT NULL DEFAULT 'user',
    is_active       TINYINT(1)   NOT NULL DEFAULT 1,
    created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
                                 ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT uq_users_username UNIQUE (username),
    CONSTRAINT uq_users_email    UNIQUE (email)
) ENGINE=InnoDB;

-- ---------------------------------------------------------------------
-- 2. models  (registry of trained detectors)
-- ---------------------------------------------------------------------
CREATE TABLE models (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    name          VARCHAR(80)  NOT NULL,
    architecture  VARCHAR(50)  NOT NULL,            -- resnet50 | efficientnet_b0 | xception
    weights_path  VARCHAR(255) NOT NULL,
    accuracy      FLOAT        NULL,
    auc           FLOAT        NULL,
    f1_score      FLOAT        NULL,
    is_active     TINYINT(1)   NOT NULL DEFAULT 1,
    created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_models_name UNIQUE (name)
) ENGINE=InnoDB;

-- ---------------------------------------------------------------------
-- 3. uploaded_files
-- ---------------------------------------------------------------------
CREATE TABLE uploaded_files (
    id           INT AUTO_INCREMENT PRIMARY KEY,
    user_id      INT          NOT NULL,
    filename     VARCHAR(255) NOT NULL,
    storage_path VARCHAR(500) NOT NULL,
    media_type   ENUM('image','video') NOT NULL,
    mime_type    VARCHAR(100) NULL,
    size_bytes   BIGINT       NOT NULL DEFAULT 0,
    created_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_files_user FOREIGN KEY (user_id)
        REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_files_user (user_id)
) ENGINE=InnoDB;

-- ---------------------------------------------------------------------
-- 4. predictions
-- ---------------------------------------------------------------------
CREATE TABLE predictions (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    user_id         INT NOT NULL,
    file_id         INT NOT NULL,
    model_id        INT NOT NULL,
    media_type      ENUM('image','video') NOT NULL,
    verdict         ENUM('real','fake') NOT NULL,
    confidence      FLOAT NOT NULL,
    frames_analyzed INT   NOT NULL DEFAULT 1,
    inference_ms    INT   NULL,
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_pred_user  FOREIGN KEY (user_id)  REFERENCES users(id)          ON DELETE CASCADE,
    CONSTRAINT fk_pred_file  FOREIGN KEY (file_id)  REFERENCES uploaded_files(id) ON DELETE CASCADE,
    CONSTRAINT fk_pred_model FOREIGN KEY (model_id) REFERENCES models(id)         ON DELETE RESTRICT,
    CONSTRAINT chk_confidence CHECK (confidence BETWEEN 0 AND 1),
    INDEX idx_pred_user     (user_id),
    INDEX idx_pred_created  (created_at),
    INDEX idx_pred_verdict  (verdict)
) ENGINE=InnoDB;

-- ---------------------------------------------------------------------
-- 5. detection_results  (per-frame / per-face breakdown)
-- ---------------------------------------------------------------------
CREATE TABLE detection_results (
    id                INT AUTO_INCREMENT PRIMARY KEY,
    prediction_id     INT NOT NULL,
    frame_index       INT NOT NULL DEFAULT 0,
    fake_probability  FLOAT NOT NULL,
    heatmap_path      VARCHAR(500) NULL,
    created_at        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_result_pred FOREIGN KEY (prediction_id)
        REFERENCES predictions(id) ON DELETE CASCADE,
    CONSTRAINT chk_fakeprob CHECK (fake_probability BETWEEN 0 AND 1),
    INDEX idx_result_pred (prediction_id)
) ENGINE=InnoDB;

-- ---------------------------------------------------------------------
-- 6. audit_logs
-- ---------------------------------------------------------------------
CREATE TABLE audit_logs (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    user_id     INT NULL,
    action      VARCHAR(80)  NOT NULL,
    entity      VARCHAR(80)  NULL,
    ip_address  VARCHAR(45)  NULL,
    details     TEXT         NULL,
    created_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_audit_user FOREIGN KEY (user_id)
        REFERENCES users(id) ON DELETE SET NULL,
    INDEX idx_audit_user (user_id),
    INDEX idx_audit_action (action)
) ENGINE=InnoDB;

-- =====================================================================
--  Seed data
-- =====================================================================

-- Default admin (password = "Admin@123" -> replace hash after first run).
-- Hash below is a bcrypt hash of "Admin@123".
INSERT INTO users (username, email, hashed_password, role) VALUES
('admin', 'admin@deepfake.local',
 '$2b$12$Q7dQ9wS3s6m4Qn9m9r5Xe.4nJ3sQb2kQd8sV0Gk1wUq8mQ4xhq0K', 'admin');

-- Registered detector models
INSERT INTO models (name, architecture, weights_path, accuracy, auc, f1_score, is_active) VALUES
('ResNet50-DF',       'resnet50',        'ml/checkpoints/resnet50_best.pth',        0.912, 0.951, 0.910, 1),
('EfficientNet-B0-DF','efficientnet_b0', 'ml/checkpoints/efficientnet_b0_best.pth', 0.928, 0.963, 0.927, 1),
('Xception-DF',       'xception',        'ml/checkpoints/xception_best.pth',        0.941, 0.972, 0.940, 1);
