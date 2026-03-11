# 完整版本 - 第4部分分析，包含ROC曲线
rm(list = ls())
setwd('/media/neurox/T7_Shield/PD_analyse')

# 加载包
library(readxl); library(caret); library(tidyverse); library(ggplot2); library(pROC); library(glmnet)

# 设置并行
library(doParallel)
cl <- makePSOCKcluster(6)
registerDoParallel(cl)

# 通用绘图函数：与 test_RBD_glmnet.R 保持一致风格
create_confusion_matrix_plot <- function(cm_table_df, title_text) {
  colnames(cm_table_df) <- c("Prediction", "Reference", "Freq")
  ggplot(data = cm_table_df, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile(color = "black") +
    geom_text(aes(label = Freq), color = "magenta", size = 5) +
    scale_fill_gradient(low = "white", high = "navyblue") +
    labs(title = title_text, x = "Predicted", y = "True") +
    theme_classic(base_size = 16) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", margin = margin(t = 8, b = 4)),
      plot.title.position = "plot",
      plot.margin = margin(t = 18, r = 12, b = 12, l = 12),
      axis.text = element_text(color = "black", size = 13),
      axis.title = element_text(color = "black", size = 14),
      legend.position = "none",
      panel.grid.major.x = element_line(color = "grey85", linewidth = 0.3),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_blank()
    )
}

# 输出目录
output_basedir <- "RBD_test_TOP20"
output_subdir_MSA_subtype <- file.path(output_basedir, "Part4_MSA_Subtype_Analysis")
dir.create(output_subdir_MSA_subtype, showWarnings = FALSE, recursive = TRUE)

cat("开始第4部分分析（包含ROC曲线）...\n")

# === 新增[目录与开关] ===
output_basedir <- "RBD_test_TOP20"  # 确保不改变现有变量
output_subdir_Strict <- file.path(output_basedir, "Part4_MSA_Subtype_Analysis_StrictNestedCV")
output_subdir_Perm   <- file.path(output_basedir, "Part4_MSA_Subtype_Analysis_Permutation")
output_subdir_LC     <- file.path(output_basedir, "Part4_MSA_Subtype_Analysis_LearningCurve")
output_subdir_Reps   <- file.path(output_basedir, "Part4_MSA_Subtype_Analysis_RepeatedSplits")
dir.create(output_subdir_Strict, showWarnings = FALSE, recursive = TRUE)
dir.create(output_subdir_Perm,   showWarnings = FALSE, recursive = TRUE)
dir.create(output_subdir_LC,     showWarnings = FALSE, recursive = TRUE)
dir.create(output_subdir_Reps,   showWarnings = FALSE, recursive = TRUE)

# 全特征（All-features）严格验证输出目录
output_subdir_Strict_All <- file.path(output_basedir, "Part4_MSA_Subtype_Analysis_StrictNestedCV_AllFeatures")
output_subdir_Perm_All   <- file.path(output_basedir, "Part4_MSA_Subtype_Analysis_Permutation_AllFeatures")
output_subdir_LC_All     <- file.path(output_basedir, "Part4_MSA_Subtype_Analysis_LearningCurve_AllFeatures")
dir.create(output_subdir_Strict_All, showWarnings = FALSE, recursive = TRUE)
dir.create(output_subdir_Perm_All,   showWarnings = FALSE, recursive = TRUE)
dir.create(output_subdir_LC_All,     showWarnings = FALSE, recursive = TRUE)

STRICT_MODE <- TRUE           # 严格嵌套CV（训练内特征选择）开关（发布主证据建议开启）
RUN_PERMUTATION <- TRUE       # 置换检验开关
RUN_LEARNING_CURVE <- TRUE    # 学习曲线开关
RUN_REPEATED_SPLITS <- TRUE   # 保存现有单次划分索引与扩展指标

# === 新增[工具函数：扩展指标、索引落盘] ===
balanced_acc <- function(cm) {
  # 对多分类，caret的byClass是每类一行，这里对Sensitivity与Specificity分别取均值再平均
  sens <- mean(cm$byClass[, "Sensitivity"], na.rm = TRUE)
  spec <- mean(cm$byClass[, "Specificity"], na.rm = TRUE)
  (sens + spec) / 2
}
macro_f1 <- function(cm) {
  # caret多分类byClass里通常包含F1列（不同版本可能为"F1"），若无可扩展计算
  if ("F1" %in% colnames(cm$byClass)) {
    mean(cm$byClass[, "F1"], na.rm = TRUE)
  } else {
    NA_real_
  }
}
f1_from_cm_binary <- function(cm) {
  bc <- cm$byClass
  ppv <- suppressWarnings(as.numeric(bc["Pos Pred Value"]))
  sens <- suppressWarnings(as.numeric(bc["Sensitivity"]))
  if (is.na(ppv) || is.na(sens) || (ppv + sens) == 0) return(NA_real_)
  2 * ppv * sens / (ppv + sens)
}
auc_ci95_delong <- function(roc_obj) {
  ci <- suppressWarnings(pROC::ci.auc(roc_obj, conf.level = 0.95, method = "delong"))
  as.numeric(ci)
}
save_split_indices <- function(idx_train, n_all, out_file_train, out_file_test) {
  write.csv(data.frame(train_idx = as.integer(idx_train)), out_file_train, row.names = FALSE)
  test_idx <- setdiff(seq_len(n_all), as.integer(idx_train))
  write.csv(data.frame(test_idx  = as.integer(test_idx)), out_file_test, row.names = FALSE)
}
save_session_info <- function(out_file) {
  writeLines(capture.output(sessionInfo()), out_file)
}

# === 新增[详细性能指标计算：含95% CI] ===
# 计算每个类别的指标（含95% CI）
calculate_class_metrics <- function(true_labels, pred_labels, class_name) {
  true_labels <- factor(true_labels, levels = c("HC", "MSA", "PD"))
  pred_labels <- factor(pred_labels, levels = c("HC", "MSA", "PD"))
  
  # 转换为二分类问题
  true_binary <- factor(ifelse(true_labels == class_name, class_name, "Other"),
                        levels = c("Other", class_name))
  pred_binary <- factor(ifelse(pred_labels == class_name, class_name, "Other"),
                        levels = c("Other", class_name))
  
  cm <- confusionMatrix(pred_binary, true_binary, positive = class_name)
  
  sensitivity <- cm$byClass["Sensitivity"]
  specificity <- cm$byClass["Specificity"]
  ppv <- cm$byClass["Pos Pred Value"]
  npv <- cm$byClass["Neg Pred Value"]
  
  # 计算95%置信区间（binomial方法）
  tp <- sum(true_binary == class_name & pred_binary == class_name)
  fn <- sum(true_binary == class_name & pred_binary == "Other")
  n_pos <- tp + fn
  sens_ci <- if (n_pos > 0) binom.test(tp, n_pos, conf.level = 0.95)$conf.int else c(NA, NA)
  
  tn <- sum(true_binary == "Other" & pred_binary == "Other")
  fp <- sum(true_binary == "Other" & pred_binary == class_name)
  n_neg <- tn + fp
  spec_ci <- if (n_neg > 0) binom.test(tn, n_neg, conf.level = 0.95)$conf.int else c(NA, NA)
  
  ppv_ci <- if ((tp + fp) > 0) binom.test(tp, tp + fp, conf.level = 0.95)$conf.int else c(NA, NA)
  npv_ci <- if ((tn + fn) > 0) binom.test(tn, tn + fn, conf.level = 0.95)$conf.int else c(NA, NA)
  
  data.frame(
    Class = class_name,
    Sensitivity = as.numeric(sensitivity),
    Sensitivity_CI95_Lower = sens_ci[1],
    Sensitivity_CI95_Upper = sens_ci[2],
    Specificity = as.numeric(specificity),
    Specificity_CI95_Lower = spec_ci[1],
    Specificity_CI95_Upper = spec_ci[2],
    PPV = as.numeric(ppv),
    PPV_CI95_Lower = ppv_ci[1],
    PPV_CI95_Upper = ppv_ci[2],
    NPV = as.numeric(npv),
    NPV_CI95_Lower = npv_ci[1],
    NPV_CI95_Upper = npv_ci[2],
    stringsAsFactors = FALSE
  )
}

# 计算AUC置信区间（Hanley & McNeil方法）
calculate_auc_ci_hanley <- function(auc_value, n_pos, n_neg) {
  Q1 <- auc_value / (2 - auc_value)
  Q2 <- (2 * auc_value^2) / (1 + auc_value)
  SE <- sqrt((auc_value * (1 - auc_value) + (n_pos - 1) * (Q1 - auc_value^2) + 
              (n_neg - 1) * (Q2 - auc_value^2)) / (n_pos * n_neg))
  z <- 1.96
  ci_lower <- max(0.5, auc_value - z * SE)
  ci_upper <- min(1.0, auc_value + z * SE)
  list(ci_lower = ci_lower, ci_upper = ci_upper, SE = SE)
}

# === 新增[训练内特征选择：从训练集重算PD/MSA哨兵→构建Top-N] ===
compute_topN_from_training <- function(X_all, y_all, idx_train, candidate_N = c(10, 12, 15), tuneLength = 20) {
  y_train <- droplevels(y_all[idx_train])
  X_train <- X_all[idx_train, , drop = FALSE]

  pd_binary <- factor(ifelse(y_train == "PD", "Disease_PD", "Other"), levels = c("Other","Disease_PD"))
  msa_binary <- factor(ifelse(y_train == "MSA", "Disease_MSA", "Other"), levels = c("Other","Disease_MSA"))

  train_ctrl_bin <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

  model_pd <- train(x = X_train, y = pd_binary, method = "glmnet", metric = "ROC",
                    trControl = train_ctrl_bin, tuneLength = tuneLength, preProc = c("center","scale"))
  model_msa <- train(x = X_train, y = msa_binary, method = "glmnet", metric = "ROC",
                     trControl = train_ctrl_bin, tuneLength = tuneLength, preProc = c("center","scale"))

  coef_pd  <- coef(model_pd$finalModel,  s = model_pd$bestTune$lambda)
  coef_msa <- coef(model_msa$finalModel, s = model_msa$bestTune$lambda)

  get_nonzero <- function(cmat) {
    fn <- rownames(cmat); cf <- as.numeric(cmat[,1])
    df <- data.frame(Feature = fn, Coef = cf, stringsAsFactors = FALSE)
    df <- df[df$Feature != "(Intercept)" & df$Coef != 0, ]
    df[order(-abs(df$Coef)), , drop = FALSE]
  }
  pd_rank  <- head(get_nonzero(coef_pd),  10)$Feature
  msa_rank <- head(get_nonzero(coef_msa), 10)$Feature
  pool <- unique(c(pd_rank, msa_rank))

  pick_bestN <- function(N) {
    feats <- pool[seq_len(min(N, length(pool)))]
    X_tr_sub <- X_train[, feats, drop = FALSE]
    ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = multiClassSummary)
    mdl <- train(x = X_tr_sub, y = y_train, method = "glmnet", metric = "Accuracy",
                 trControl = ctrl, tuneLength = tuneLength, preProc = c("center","scale"))
    best <- max(mdl$results$Accuracy, na.rm = TRUE)
    c(N = N, Acc = best)
  }
  resN <- do.call(rbind, lapply(candidate_N, pick_bestN))
  bestN <- as.integer(resN[which.max(resN[,"Acc"]), "N"])
  topN  <- pool[seq_len(min(bestN, length(pool)))]
  list(topN = topN, chosen_N = bestN, candidates = pool)
}

# === 新增[严格嵌套CV：外层Repeated-KFold，内层训练内特征选择与调参] ===
strict_nested_cv <- function(X_all, y_all, repeats = 5, folds = 5,
                             candidate_N = c(10,12,15),
                             tuneLength = 20,
                             outdir = output_subdir_Strict,
                             seed_base = 2025) {
  set.seed(seed_base)
  res_all <- list(); k <- 0
  for (rp in seq_len(repeats)) {
    idx_cv <- createFolds(y_all, k = folds, list = TRUE, returnTrain = TRUE)
    for (tr_name in names(idx_cv)) {
      k <- k + 1
      idx_train <- idx_cv[[tr_name]]
      idx_test  <- setdiff(seq_len(nrow(X_all)), idx_train)

      topN_obj <- compute_topN_from_training(X_all, y_all, idx_train, candidate_N, tuneLength)
      feats <- topN_obj$topN; chosen_N <- topN_obj$chosen_N

      ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = multiClassSummary)
      mdl <- train(x = X_all[idx_train, feats, drop = FALSE], y = y_all[idx_train],
                   method = "glmnet", metric = "Accuracy",
                   trControl = ctrl, tuneLength = tuneLength, preProc = c("center","scale"))
      pred <- predict(mdl, newdata = X_all[idx_test, feats, drop = FALSE])
      prb  <- predict(mdl, newdata = X_all[idx_test, feats, drop = FALSE], type = "prob")
      cm   <- confusionMatrix(pred, y_all[idx_test])
      auc_val <- as.numeric(pROC::auc(pROC::multiclass.roc(y_all[idx_test], prb)))

      # 保存外层折训练/测试索引（复现）
      try({
        save_split_indices(idx_train, nrow(X_all),
                           file.path(outdir, sprintf("fold_%03d_train_idx.csv", k)),
                           file.path(outdir, sprintf("fold_%03d_test_idx.csv",  k)))
      }, silent = TRUE)

      res_all[[k]] <- data.frame(
        rep_id = rp, fold = k, N = chosen_N,
        Accuracy = cm$overall["Accuracy"], Kappa = cm$overall["Kappa"],
        BalancedAccuracy = balanced_acc(cm), MacroF1 = macro_f1(cm), AUC = auc_val,
        stringsAsFactors = FALSE)
    }
  }
  df <- dplyr::bind_rows(res_all)
  n_obs <- nrow(df)
  agg <- data.frame(
    Accuracy_mean = mean(df$Accuracy, na.rm = TRUE),
    Accuracy_se   = sd(df$Accuracy, na.rm = TRUE) / sqrt(n_obs),
    Accuracy_ci   = 1.96 * sd(df$Accuracy, na.rm = TRUE) / sqrt(n_obs),
    Kappa_mean    = mean(df$Kappa, na.rm = TRUE),
    Kappa_se      = sd(df$Kappa, na.rm = TRUE) / sqrt(n_obs),
    Kappa_ci      = 1.96 * sd(df$Kappa, na.rm = TRUE) / sqrt(n_obs),
    BalancedAccuracy_mean = mean(df$BalancedAccuracy, na.rm = TRUE),
    BalancedAccuracy_se   = sd(df$BalancedAccuracy, na.rm = TRUE) / sqrt(n_obs),
    BalancedAccuracy_ci   = 1.96 * sd(df$BalancedAccuracy, na.rm = TRUE) / sqrt(n_obs),
    MacroF1_mean  = mean(df$MacroF1, na.rm = TRUE),
    MacroF1_se    = sd(df$MacroF1, na.rm = TRUE) / sqrt(n_obs),
    MacroF1_ci    = 1.96 * sd(df$MacroF1, na.rm = TRUE) / sqrt(n_obs),
    AUC_mean      = mean(df$AUC, na.rm = TRUE),
    AUC_se        = sd(df$AUC, na.rm = TRUE) / sqrt(n_obs),
    AUC_ci        = 1.96 * sd(df$AUC, na.rm = TRUE) / sqrt(n_obs)
  )

  write.csv(df,  file.path(outdir, "strict_nestedcv_fold_results.csv"), row.names = FALSE)
  write.csv(agg, file.path(outdir, "strict_nestedcv_aggregate_mean_se_ci.csv"), row.names = FALSE)
  df
}

# === 新增[置换检验：最简固定Top15] ===
permutation_test_auc <- function(X_all, y_all, feats_top15, reps = 200,
                                 outdir = output_subdir_Perm, seed = 3031) {
  set.seed(seed)
  ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = multiClassSummary)
  auc_null <- numeric(reps)
  for (i in seq_len(reps)) {
    yy <- sample(y_all)
    mdl <- train(x = X_all[, feats_top15, drop = FALSE], y = yy,
                 method = "glmnet", metric = "Accuracy",
                 trControl = ctrl, tuneLength = 10, preProc = c("center","scale"))
    pr <- predict(mdl, newdata = X_all[, feats_top15, drop = FALSE], type = "prob")
    auc_null[i] <- as.numeric(pROC::auc(pROC::multiclass.roc(yy, pr)))
  }
  write.csv(data.frame(AUC = auc_null), file.path(outdir, "permutation_auc_null.csv"), row.names = FALSE)
  auc_null
}

# === 新增[学习曲线：固定Top15示例] ===
learning_curve_top15 <- function(X_all, y_all, feats_top15,
                                 train_fracs = c(0.1,0.3,0.5,0.7,0.9),
                                 repeats = 5,
                                 outdir = output_subdir_LC,
                                 seed = 4041) {
  set.seed(seed)
  ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = multiClassSummary)
  res <- list()
  for (fr in train_fracs) {
    for (rp in seq_len(repeats)) {
      idx_tr <- createDataPartition(y_all, p = fr, list = FALSE)
      mdl <- train(x = X_all[idx_tr, feats_top15, drop = FALSE], y = y_all[idx_tr],
                   method = "glmnet", metric = "Accuracy",
                   trControl = ctrl, tuneLength = 10, preProc = c("center","scale"))
      idx_te <- setdiff(seq_len(nrow(X_all)), idx_tr)
      pr <- predict(mdl, newdata = X_all[idx_te, feats_top15, drop = FALSE], type = "prob")
      pd <- predict(mdl, newdata = X_all[idx_te, feats_top15, drop = FALSE])
      cm <- confusionMatrix(pd, y_all[idx_te])
      auc_val <- as.numeric(pROC::auc(pROC::multiclass.roc(y_all[idx_te], pr)))
      res[[length(res)+1]] <- data.frame(frac = fr, rep_id = rp,
                                         Accuracy = cm$overall["Accuracy"], Kappa = cm$overall["Kappa"],
                                         BalancedAccuracy = balanced_acc(cm), MacroF1 = macro_f1(cm), AUC = auc_val)
    }
  }
  df <- dplyr::bind_rows(res)
  write.csv(df, file.path(outdir, "learning_curve_results.csv"), row.names = FALSE)
  df
}

# === 新增[All-features：严格嵌套CV/置换/学习曲线] ===
strict_nested_cv_allfeatures <- function(X_all, y_all,
                                        repeats = 5, folds = 5,
                                        tuneLength = 20,
                                        outdir = output_subdir_Strict_All,
                                        seed_base = 2026) {
  set.seed(seed_base)
  res_all <- list(); k <- 0
  for (rp in seq_len(repeats)) {
    idx_cv <- createFolds(y_all, k = folds, list = TRUE, returnTrain = TRUE)
    for (tr_name in names(idx_cv)) {
      k <- k + 1
      idx_train <- idx_cv[[tr_name]]
      idx_test  <- setdiff(seq_len(nrow(X_all)), idx_train)

      ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = multiClassSummary)
      mdl <- train(x = X_all[idx_train, , drop = FALSE], y = y_all[idx_train],
                   method = "glmnet", metric = "Accuracy",
                   trControl = ctrl, tuneLength = tuneLength, preProc = c("center","scale"))
      pred <- predict(mdl, newdata = X_all[idx_test, , drop = FALSE])
      prb  <- predict(mdl, newdata = X_all[idx_test, , drop = FALSE], type = "prob")
      cm   <- confusionMatrix(pred, y_all[idx_test])
      auc_val <- as.numeric(pROC::auc(pROC::multiclass.roc(y_all[idx_test], prb)))

      # 保存外层折训练/测试索引（复现）
      try({
        save_split_indices(idx_train, nrow(X_all),
                           file.path(outdir, sprintf("fold_%03d_train_idx.csv", k)),
                           file.path(outdir, sprintf("fold_%03d_test_idx.csv",  k)))
      }, silent = TRUE)

      res_all[[k]] <- data.frame(
        rep_id = rp, fold = k,
        Accuracy = cm$overall["Accuracy"], Kappa = cm$overall["Kappa"],
        BalancedAccuracy = balanced_acc(cm), MacroF1 = macro_f1(cm), AUC = auc_val,
        stringsAsFactors = FALSE)
    }
  }
  df <- dplyr::bind_rows(res_all)
  n_obs <- nrow(df)
  agg <- data.frame(
    Accuracy_mean = mean(df$Accuracy, na.rm = TRUE),
    Accuracy_se   = sd(df$Accuracy, na.rm = TRUE) / sqrt(n_obs),
    Accuracy_ci   = 1.96 * sd(df$Accuracy, na.rm = TRUE) / sqrt(n_obs),
    Kappa_mean    = mean(df$Kappa, na.rm = TRUE),
    Kappa_se      = sd(df$Kappa, na.rm = TRUE) / sqrt(n_obs),
    Kappa_ci      = 1.96 * sd(df$Kappa, na.rm = TRUE) / sqrt(n_obs),
    BalancedAccuracy_mean = mean(df$BalancedAccuracy, na.rm = TRUE),
    BalancedAccuracy_se   = sd(df$BalancedAccuracy, na.rm = TRUE) / sqrt(n_obs),
    BalancedAccuracy_ci   = 1.96 * sd(df$BalancedAccuracy, na.rm = TRUE) / sqrt(n_obs),
    MacroF1_mean  = mean(df$MacroF1, na.rm = TRUE),
    MacroF1_se    = sd(df$MacroF1, na.rm = TRUE) / sqrt(n_obs),
    MacroF1_ci    = 1.96 * sd(df$MacroF1, na.rm = TRUE) / sqrt(n_obs),
    AUC_mean      = mean(df$AUC, na.rm = TRUE),
    AUC_se        = sd(df$AUC, na.rm = TRUE) / sqrt(n_obs),
    AUC_ci        = 1.96 * sd(df$AUC, na.rm = TRUE) / sqrt(n_obs)
  )

  write.csv(df,  file.path(outdir, "strict_nestedcv_allfeatures_fold_results.csv"), row.names = FALSE)
  write.csv(agg, file.path(outdir, "strict_nestedcv_allfeatures_aggregate_mean_se_ci.csv"), row.names = FALSE)
  df
}

permutation_test_auc_allfeatures <- function(X_all, y_all, reps = 200,
                                            outdir = output_subdir_Perm_All,
                                            seed = 3032) {
  set.seed(seed)
  ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = multiClassSummary)
  auc_null <- numeric(reps)
  for (i in seq_len(reps)) {
    yy <- sample(y_all)
    mdl <- train(x = X_all, y = yy,
                 method = "glmnet", metric = "Accuracy",
                 trControl = ctrl, tuneLength = 10, preProc = c("center","scale"))
    pr <- predict(mdl, newdata = X_all, type = "prob")
    auc_null[i] <- as.numeric(pROC::auc(pROC::multiclass.roc(yy, pr)))
  }
  write.csv(data.frame(AUC = auc_null), file.path(outdir, "permutation_auc_null_allfeatures.csv"), row.names = FALSE)
  auc_null
}

learning_curve_allfeatures <- function(X_all, y_all,
                                       train_fracs = c(0.1,0.3,0.5,0.7,0.9),
                                       repeats = 5,
                                       outdir = output_subdir_LC_All,
                                       seed = 4042) {
  set.seed(seed)
  ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = multiClassSummary)
  res <- list()
  for (fr in train_fracs) {
    for (rp in seq_len(repeats)) {
      idx_tr <- createDataPartition(y_all, p = fr, list = FALSE)
      mdl <- train(x = X_all[idx_tr, , drop = FALSE], y = y_all[idx_tr],
                   method = "glmnet", metric = "Accuracy",
                   trControl = ctrl, tuneLength = 10, preProc = c("center","scale"))
      idx_te <- setdiff(seq_len(nrow(X_all)), idx_tr)
      pr <- predict(mdl, newdata = X_all[idx_te, , drop = FALSE], type = "prob")
      pd <- predict(mdl, newdata = X_all[idx_te, , drop = FALSE])
      cm <- confusionMatrix(pd, y_all[idx_te])
      auc_val <- as.numeric(pROC::auc(pROC::multiclass.roc(y_all[idx_te], pr)))
      res[[length(res)+1]] <- data.frame(frac = fr, rep_id = rp,
                                         Accuracy = cm$overall["Accuracy"], Kappa = cm$overall["Kappa"],
                                         BalancedAccuracy = balanced_acc(cm), MacroF1 = macro_f1(cm), AUC = auc_val)
    }
  }
  df <- dplyr::bind_rows(res)
  write.csv(df, file.path(outdir, "learning_curve_results_allfeatures.csv"), row.names = FALSE)
  df
}

# 读取数据
demo_all <- read_excel('demo_all.xlsx')
demo_all$group <- factor(demo_all$group, levels = c('HC', 'MSA', 'PD', 'RBD'))

# 读取特征
library(reticulate)
use_python("/usr/local/fsl/bin/python", required = TRUE)
np <- import('numpy')
data_all_npy <- np$load('surface_metrics_RBD.npy')
struc_measures <- as.data.frame(array(data_all_npy, dim = c(nrow(demo_all), 800)))

# 列名
annot <- read.csv("/media/neurox/T7_Shield/R_done/analysis/surfaces/parcellations/lut/lut_schaefer-400_mics.csv")
annot <- annot[-c(1,202), ]
annot$label <- gsub("7Networks_", "", annot$label)
roi_labels <- annot$label
struc_labels <- rep(c('area', 'thickness'), each = 400)
combined_labels <- paste(roi_labels, struc_labels, sep = "_")
colnames(struc_measures) <- make.unique(combined_labels)

# 皮层下数据
asegstats <- read_excel("asegstats_all.xlsx", col_names = TRUE)
colnames(asegstats) <- paste0("subcort_", colnames(asegstats))
combined_measures <- cbind(struc_measures, asegstats)

# 读取TOP20统一特征清单
master_top20 <- file.path(output_basedir, "Part2_Disease_Sentinel_Definition", "top_features_master_TOP20.csv")
if (!file.exists(master_top20)) stop("top_features_master_TOP20.csv 不存在，请先运行 9.1test_RBD_glmnet_TOP20.R")
top20_features <- read.csv(master_top20, stringsAsFactors = FALSE)$Feature
cat("TOP20特征:", length(top20_features), "\n")

# 训练控制
train_ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, 
                          summaryFunction = multiClassSummary, allowParallel = TRUE,
                          verboseIter = FALSE)

# === 4.2: HC-PD-MSA (TOP20) ===
cat("\n=== HC-PD-MSA (TOP20特征) 分析 ===\n")
hc_pd_msa_idx <- which(demo_all$group %in% c('HC', 'PD', 'MSA'))
X_hc_pd_msa <- combined_measures[hc_pd_msa_idx, top20_features]
Y_hc_pd_msa <- droplevels(demo_all$group[hc_pd_msa_idx])

cat("各组样本数:", table(Y_hc_pd_msa), "\n")

set.seed(401)
train_idx <- createDataPartition(Y_hc_pd_msa, p = 0.8, list = FALSE)
X_train <- X_hc_pd_msa[train_idx, ]
Y_train <- Y_hc_pd_msa[train_idx]
X_test <- X_hc_pd_msa[-train_idx, ]
Y_test <- Y_hc_pd_msa[-train_idx]

cat("训练HC-PD-MSA TOP20模型...\n")
model1 <- train(x = X_train, y = Y_train, method = "glmnet", metric = "Accuracy", 
               trControl = train_ctrl, tuneLength = 20, preProc = c("center", "scale"))

pred1 <- predict(model1, X_test)
prob1 <- predict(model1, X_test, type = "prob")
cm1 <- confusionMatrix(pred1, Y_test)
cat("HC-PD-MSA TOP20准确率:", round(cm1$overall["Accuracy"], 4), "\n")

# 混淆矩阵图1
cm_df1 <- as.data.frame(cm1$table)
p1 <- create_confusion_matrix_plot(cm_df1, "Confusion Matrix - HC/PD/MSA (TOP20 Features)")
ggsave(file.path(output_subdir_MSA_subtype, "Confusion_Matrix_HPDMSA_TOP20.png"), p1, width = 7, height = 6, dpi = 300)

# 保存训练/测试详细分类结果到统一表（追加模式）
pred_detail_hpdmsa <- data.frame(
  Dataset = rep(c("Train","Test"), c(nrow(X_train), nrow(X_test))),
  True = c(as.character(Y_train), as.character(Y_test)),
  Pred = c(as.character(predict(model1, X_train)), as.character(pred1))
)
pred_detail_hpdmsa$Task <- "HC_PD_MSA_TOP20"
pred_detail_path <- file.path(output_subdir_MSA_subtype, "classification_details_all_models.csv")
try({
  if (file.exists(pred_detail_path)) {
    write.table(pred_detail_hpdmsa, pred_detail_path, sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE)
  } else {
    write.csv(pred_detail_hpdmsa, pred_detail_path, row.names = FALSE)
  }
}, silent = TRUE)

# ROC曲线1 - 多分类
auc1 <- auc(multiclass.roc(Y_test, prob1))
cat("HC-PD-MSA TOP20 多分类AUC:", round(auc1, 4), "\n")
cat("HC-PD-MSA TOP20 BalancedAccuracy:", round(balanced_acc(cm1), 4), "\n")
cat("HC-PD-MSA TOP20 MacroF1:", round(macro_f1(cm1), 4), "\n")

# 绘制多分类ROC（One-vs-Rest方法）
roc_list1 <- list()
for(class in levels(Y_test)) {
  binary_response <- ifelse(Y_test == class, 1, 0)
  binary_predictor <- prob1[, class]
  roc_obj <- roc(binary_response, binary_predictor, levels = c(0, 1), direction = "<")
  roc_list1[[class]] <- roc_obj
}

# 使用ggroc绘制多条ROC曲线
roc_data1 <- map_dfr(names(roc_list1), function(class_name) {
  roc_obj <- roc_list1[[class_name]]
  data.frame(
    specificity = roc_obj$specificities,
    sensitivity = roc_obj$sensitivities,
    class = class_name,
    auc = as.numeric(auc(roc_obj))
  )
})

roc_plot1 <- ggplot(roc_data1, aes(x = 1 - specificity, y = sensitivity, color = class)) +
  geom_line(size = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
  labs(
    title = paste0("ROC Curves - HC/PD/MSA (TOP20 Features)\nOverall AUC = ", round(auc1, 3)),
    x = "1 - Specificity",
    y = "Sensitivity"
  ) +
  scale_color_manual(values = c("HC" = "#E31A1C", "MSA" = "#1F78B4", "PD" = "#33A02C")) +
  theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.text = element_text(color = "black"),
    axis.title = element_text(color = "black"),
    legend.title = element_text(face = "bold")
  ) +
  guides(color = guide_legend(title = "Class"))

ggsave(file.path(output_subdir_MSA_subtype, "ROC_HPDMSA_TOP20.png"), roc_plot1, width = 7, height = 6, dpi = 300)

# === 新增：计算HC_PD_MSA_TOP20详细性能指标（含95% CI）===
cat("\n计算HC_PD_MSA_TOP20详细性能指标（含95% CI）...\n")

# 训练集预测
pred_train1 <- predict(model1, X_train)
prob_train1 <- predict(model1, X_train, type = "prob")
cm1_train <- confusionMatrix(pred_train1, Y_train)

# 计算训练集每个类别的指标
metrics_hc_train <- calculate_class_metrics(Y_train, pred_train1, "HC")
metrics_msa_train <- calculate_class_metrics(Y_train, pred_train1, "MSA")
metrics_pd_train <- calculate_class_metrics(Y_train, pred_train1, "PD")
metrics_train1 <- rbind(metrics_hc_train, metrics_msa_train, metrics_pd_train)
metrics_train1$Dataset <- "Train"

# 计算训练集AUC（使用与测试集相同的multiclass.roc方法）
auc_train1 <- as.numeric(auc(multiclass.roc(Y_train, prob_train1)))
cat("训练集多分类AUC:", round(auc_train1, 4), "\n")

# 计算训练集AUC置信区间
n_hc_train <- sum(Y_train == "HC")
n_msa_train <- sum(Y_train == "MSA")
n_pd_train <- sum(Y_train == "PD")
n_avg_pos_train <- mean(c(n_hc_train, n_msa_train, n_pd_train))
n_avg_neg_train <- mean(c(nrow(X_train) - n_hc_train, 
                          nrow(X_train) - n_msa_train, 
                          nrow(X_train) - n_pd_train))
auc_ci_train1 <- calculate_auc_ci_hanley(auc_train1, n_avg_pos_train, n_avg_neg_train)

metrics_train1$AUC_Overall <- auc_train1
metrics_train1$AUC_CI95_Lower <- auc_ci_train1$ci_lower
metrics_train1$AUC_CI95_Upper <- auc_ci_train1$ci_upper

# 计算测试集每个类别的指标
metrics_hc_test <- calculate_class_metrics(Y_test, pred1, "HC")
metrics_msa_test <- calculate_class_metrics(Y_test, pred1, "MSA")
metrics_pd_test <- calculate_class_metrics(Y_test, pred1, "PD")
metrics_test1 <- rbind(metrics_hc_test, metrics_msa_test, metrics_pd_test)
metrics_test1$Dataset <- "Test"

# 计算测试集AUC置信区间
n_hc_test <- sum(Y_test == "HC")
n_msa_test <- sum(Y_test == "MSA")
n_pd_test <- sum(Y_test == "PD")
n_avg_pos_test <- mean(c(n_hc_test, n_msa_test, n_pd_test))
n_avg_neg_test <- mean(c(nrow(X_test) - n_hc_test, 
                         nrow(X_test) - n_msa_test, 
                         nrow(X_test) - n_pd_test))
auc_ci_test1 <- calculate_auc_ci_hanley(auc1, n_avg_pos_test, n_avg_neg_test)

metrics_test1$AUC_Overall <- auc1
metrics_test1$AUC_CI95_Lower <- auc_ci_test1$ci_lower
metrics_test1$AUC_CI95_Upper <- auc_ci_test1$ci_upper

# 合并训练集和测试集结果
metrics_all1 <- rbind(metrics_train1, metrics_test1)
metrics_all1 <- metrics_all1[, c("Dataset", "Class", "AUC_Overall", "AUC_CI95_Lower", "AUC_CI95_Upper",
                                  "Sensitivity", "Sensitivity_CI95_Lower", "Sensitivity_CI95_Upper",
                                  "Specificity", "Specificity_CI95_Lower", "Specificity_CI95_Upper",
                                  "PPV", "PPV_CI95_Lower", "PPV_CI95_Upper",
                                  "NPV", "NPV_CI95_Lower", "NPV_CI95_Upper")]

# 保存详细指标
write.csv(metrics_all1, 
          file.path(output_subdir_MSA_subtype, "performance_HC_PD_MSA_TOP20_detailed_with_CI.csv"), 
          row.names = FALSE)

# 创建训练集总体摘要
summary_train1 <- data.frame(
  Model = "HC_PD_MSA_TOP20",
  Dataset = "Train",
  AUC_Overall = auc_train1,
  AUC_CI95_Lower = auc_ci_train1$ci_lower,
  AUC_CI95_Upper = auc_ci_train1$ci_upper,
  Sensitivity_Mean = mean(metrics_train1$Sensitivity),
  Sensitivity_CI95_Lower = mean(metrics_train1$Sensitivity_CI95_Lower),
  Sensitivity_CI95_Upper = mean(metrics_train1$Sensitivity_CI95_Upper),
  Specificity_Mean = mean(metrics_train1$Specificity),
  Specificity_CI95_Lower = mean(metrics_train1$Specificity_CI95_Lower),
  Specificity_CI95_Upper = mean(metrics_train1$Specificity_CI95_Upper),
  PPV_Mean = mean(metrics_train1$PPV),
  PPV_CI95_Lower = mean(metrics_train1$PPV_CI95_Lower),
  PPV_CI95_Upper = mean(metrics_train1$PPV_CI95_Upper),
  NPV_Mean = mean(metrics_train1$NPV),
  NPV_CI95_Lower = mean(metrics_train1$NPV_CI95_Lower),
  NPV_CI95_Upper = mean(metrics_train1$NPV_CI95_Upper),
  stringsAsFactors = FALSE
)

# 创建测试集总体摘要
summary_test1 <- data.frame(
  Model = "HC_PD_MSA_TOP20",
  Dataset = "Test",
  AUC_Overall = auc1,
  AUC_CI95_Lower = auc_ci_test1$ci_lower,
  AUC_CI95_Upper = auc_ci_test1$ci_upper,
  Sensitivity_Mean = mean(metrics_test1$Sensitivity),
  Sensitivity_CI95_Lower = mean(metrics_test1$Sensitivity_CI95_Lower),
  Sensitivity_CI95_Upper = mean(metrics_test1$Sensitivity_CI95_Upper),
  Specificity_Mean = mean(metrics_test1$Specificity),
  Specificity_CI95_Lower = mean(metrics_test1$Specificity_CI95_Lower),
  Specificity_CI95_Upper = mean(metrics_test1$Specificity_CI95_Upper),
  PPV_Mean = mean(metrics_test1$PPV),
  PPV_CI95_Lower = mean(metrics_test1$PPV_CI95_Lower),
  PPV_CI95_Upper = mean(metrics_test1$PPV_CI95_Upper),
  NPV_Mean = mean(metrics_test1$NPV),
  NPV_CI95_Lower = mean(metrics_test1$NPV_CI95_Lower),
  NPV_CI95_Upper = mean(metrics_test1$NPV_CI95_Upper),
  stringsAsFactors = FALSE
)

# 合并训练集和测试集总体摘要
summary_all1 <- rbind(summary_train1, summary_test1)

write.csv(summary_all1, 
          file.path(output_subdir_MSA_subtype, "performance_HC_PD_MSA_TOP20_summary_with_CI.csv"), 
          row.names = FALSE)

cat("HC_PD_MSA_TOP20详细指标已保存。\n")
cat(sprintf("训练集AUC: %.4f (95%% CI: %.4f - %.4f)\n", 
            auc_train1, auc_ci_train1$ci_lower, auc_ci_train1$ci_upper))
cat(sprintf("测试集AUC: %.4f (95%% CI: %.4f - %.4f)\n", 
            auc1, auc_ci_test1$ci_lower, auc_ci_test1$ci_upper))

# === 4.3: HC-PD-MSA-P (全特征) ===
cat("\n=== HC-PD-MSA-P (全特征) 分析 ===\n")
hc_pd_msap_idx <- which((demo_all$group == "HC") | (demo_all$group == "PD") | 
                        (demo_all$group == "MSA" & demo_all$MSA_group == "MSA-P"))

demo_hc_pd_msap <- demo_all[hc_pd_msap_idx, ] %>%
  mutate(group_new = case_when(
    group == "HC" ~ "HC",
    group == "PD" ~ "PD", 
    group == "MSA" & MSA_group == "MSA-P" ~ "MSA_P"
  )) %>%
  mutate(group_new = factor(group_new, levels = c("HC", "PD", "MSA_P")))

X_hc_pd_msap_full <- combined_measures[hc_pd_msap_idx, ]
Y_hc_pd_msap <- demo_hc_pd_msap$group_new

cat("HC-PD-MSA-P各组:", table(Y_hc_pd_msap), "\n")

set.seed(402)
train_idx2 <- createDataPartition(Y_hc_pd_msap, p = 0.8, list = FALSE)

cat("训练HC-PD-MSA-P 全特征模型...\n")
model2 <- train(x = X_hc_pd_msap_full[train_idx2, ], y = Y_hc_pd_msap[train_idx2], 
               method = "glmnet", metric = "Accuracy", trControl = train_ctrl, 
               tuneLength = 20, preProc = c("center", "scale"))

pred2 <- predict(model2, X_hc_pd_msap_full[-train_idx2, ])
prob2 <- predict(model2, X_hc_pd_msap_full[-train_idx2, ], type = "prob")
cm2 <- confusionMatrix(pred2, Y_hc_pd_msap[-train_idx2])
cat("HC-PD-MSA-P 全特征准确率:", round(cm2$overall["Accuracy"], 4), "\n")

# 混淆矩阵图2
cm_df2 <- as.data.frame(cm2$table)
# 重新映射标签显示
cm_df2$Prediction <- gsub("MSA_P", "MSA-P", cm_df2$Prediction)
cm_df2$Reference <- gsub("MSA_P", "MSA-P", cm_df2$Reference)
p2 <- create_confusion_matrix_plot(cm_df2, "Confusion Matrix - HC/PD/MSA-P (All Features)")
ggsave(file.path(output_subdir_MSA_subtype, "Confusion_Matrix_HC_PD_MSA-P_Full.png"), p2, width = 7, height = 6, dpi = 300)

pred_detail_msap_full <- data.frame(
  Dataset = rep(c("Train","Test"), c(nrow(X_hc_pd_msap_full[train_idx2, ]), nrow(X_hc_pd_msap_full[-train_idx2, ]))),
  True = c(as.character(Y_hc_pd_msap[train_idx2]), as.character(Y_hc_pd_msap[-train_idx2])),
  Pred = c(as.character(predict(model2, X_hc_pd_msap_full[train_idx2, ])), as.character(pred2))
)
pred_detail_msap_full$Task <- "HC_PD_MSA-P_Full"
try({
  write.table(pred_detail_msap_full, pred_detail_path, sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE)
}, silent = TRUE)

# ROC曲线2
auc2 <- auc(multiclass.roc(Y_hc_pd_msap[-train_idx2], prob2))
cat("HC-PD-MSA-P 全特征 多分类AUC:", round(auc2, 4), "\n")
cat("HC-PD-MSA-P 全特征 BalancedAccuracy:", round(balanced_acc(cm2), 4), "\n")
cat("HC-PD-MSA-P 全特征 MacroF1:", round(macro_f1(cm2), 4), "\n")

roc_list2 <- list()
for(class in levels(Y_hc_pd_msap[-train_idx2])) {
  binary_response <- ifelse(Y_hc_pd_msap[-train_idx2] == class, 1, 0)
  binary_predictor <- prob2[, class]
  roc_obj <- roc(binary_response, binary_predictor, levels = c(0, 1), direction = "<")
  roc_list2[[class]] <- roc_obj
}

roc_data2 <- map_dfr(names(roc_list2), function(class_name) {
  roc_obj <- roc_list2[[class_name]]
  data.frame(
    specificity = roc_obj$specificities,
    sensitivity = roc_obj$sensitivities,
    class = class_name,
    auc = as.numeric(auc(roc_obj))
  )
})

roc_plot2 <- ggplot(roc_data2, aes(x = 1 - specificity, y = sensitivity, color = class)) +
  geom_line(size = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
  labs(
    title = paste0("ROC Curves - HC/PD/MSA-P (All Features)\nOverall AUC = ", round(auc2, 3)),
    x = "1 - Specificity",
    y = "Sensitivity"
  ) +
  scale_color_manual(values = c("HC" = "#E31A1C", "PD" = "#33A02C", "MSA_P" = "#FF7F00"),
                     labels = c("HC" = "HC", "PD" = "PD", "MSA_P" = "MSA-P")) +
  theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.text = element_text(color = "black"),
    axis.title = element_text(color = "black"),
    legend.title = element_text(face = "bold")
  ) +
  guides(color = guide_legend(title = "Class"))

ggsave(file.path(output_subdir_MSA_subtype, "ROC_HC_PD_MSA-P_Full.png"), roc_plot2, width = 7, height = 6, dpi = 300)

# === 4.4: HC-PD-MSA-P (TOP20) ===
cat("\n=== HC-PD-MSA-P (TOP20特征) 分析 ===\n")
X_hc_pd_msap_top20 <- combined_measures[hc_pd_msap_idx, top20_features]

set.seed(403)
train_idx3 <- createDataPartition(Y_hc_pd_msap, p = 0.8, list = FALSE)

cat("训练HC-PD-MSA-P TOP20模型...\n")
model3 <- train(x = X_hc_pd_msap_top20[train_idx3, ], y = Y_hc_pd_msap[train_idx3], 
               method = "glmnet", metric = "Accuracy", trControl = train_ctrl, 
               tuneLength = 20, preProc = c("center", "scale"))

pred3 <- predict(model3, X_hc_pd_msap_top20[-train_idx3, ])
prob3 <- predict(model3, X_hc_pd_msap_top20[-train_idx3, ], type = "prob")
cm3 <- confusionMatrix(pred3, Y_hc_pd_msap[-train_idx3])
cat("HC-PD-MSA-P TOP20准确率:", round(cm3$overall["Accuracy"], 4), "\n")

# 混淆矩阵图3
cm_df3 <- as.data.frame(cm3$table)
# 重新映射标签显示
cm_df3$Prediction <- gsub("MSA_P", "MSA-P", cm_df3$Prediction)
cm_df3$Reference <- gsub("MSA_P", "MSA-P", cm_df3$Reference)
p3 <- create_confusion_matrix_plot(cm_df3, "Confusion Matrix - HC/PD/MSA-P (TOP20 Features)")
ggsave(file.path(output_subdir_MSA_subtype, "Confusion_Matrix_HC_PD_MSA-P_TOP20.png"), p3, width = 7, height = 6, dpi = 300)

pred_detail_msap_top20 <- data.frame(
  Dataset = rep(c("Train","Test"), c(nrow(X_hc_pd_msap_top20[train_idx3, ]), nrow(X_hc_pd_msap_top20[-train_idx3, ]))),
  True = c(as.character(Y_hc_pd_msap[train_idx3]), as.character(Y_hc_pd_msap[-train_idx3])),
  Pred = c(as.character(predict(model3, X_hc_pd_msap_top20[train_idx3, ])), as.character(pred3))
)
pred_detail_msap_top20$Task <- "HC_PD_MSA-P_TOP20"
try({
  write.table(pred_detail_msap_top20, pred_detail_path, sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE)
}, silent = TRUE)

# ROC曲线3
auc3 <- auc(multiclass.roc(Y_hc_pd_msap[-train_idx3], prob3))
cat("HC-PD-MSA-P TOP20 多分类AUC:", round(auc3, 4), "\n")
cat("HC-PD-MSA-P TOP20 BalancedAccuracy:", round(balanced_acc(cm3), 4), "\n")
cat("HC-PD-MSA-P TOP20 MacroF1:", round(macro_f1(cm3), 4), "\n")

roc_list3 <- list()
for(class in levels(Y_hc_pd_msap[-train_idx3])) {
  binary_response <- ifelse(Y_hc_pd_msap[-train_idx3] == class, 1, 0)
  binary_predictor <- prob3[, class]
  roc_obj <- roc(binary_response, binary_predictor, levels = c(0, 1), direction = "<")
  roc_list3[[class]] <- roc_obj
}

roc_data3 <- map_dfr(names(roc_list3), function(class_name) {
  roc_obj <- roc_list3[[class_name]]
  data.frame(
    specificity = roc_obj$specificities,
    sensitivity = roc_obj$sensitivities,
    class = class_name,
    auc = as.numeric(auc(roc_obj))
  )
})

roc_plot3 <- ggplot(roc_data3, aes(x = 1 - specificity, y = sensitivity, color = class)) +
  geom_line(size = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
  labs(
    title = paste0("ROC Curves - HC/PD/MSA-P (TOP20 Features)\nOverall AUC = ", round(auc3, 3)),
    x = "1 - Specificity",
    y = "Sensitivity"
  ) +
  scale_color_manual(values = c("HC" = "#E31A1C", "PD" = "#33A02C", "MSA_P" = "#FF7F00"),
                     labels = c("HC" = "HC", "PD" = "PD", "MSA_P" = "MSA-P")) +
  theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.text = element_text(color = "black"),
    axis.title = element_text(color = "black"),
    legend.title = element_text(face = "bold")
  ) +
  guides(color = guide_legend(title = "Class"))

ggsave(file.path(output_subdir_MSA_subtype, "ROC_HC_PD_MSA-P_TOP20.png"), roc_plot3, width = 7, height = 6, dpi = 300)

# === 结果总结 ===
results <- data.frame(
  Model = c("HC_PD_MSA_TOP20", "HC_PD_MSA-P_Full", "HC_PD_MSA-P_TOP20"),
  Accuracy = round(c(cm1$overall["Accuracy"], cm2$overall["Accuracy"], cm3$overall["Accuracy"]), 4),
  Kappa = round(c(cm1$overall["Kappa"], cm2$overall["Kappa"], cm3$overall["Kappa"]), 4),
  AUC = round(c(auc1, auc2, auc3), 4)
)

cat("\n=== 最终结果总结 ===\n")
print(results)

write.csv(results, file.path(output_subdir_MSA_subtype, "performance_summary_TOP20.csv"), row.names = FALSE)
results_ext <- data.frame(
  Model = results$Model,
  Accuracy = results$Accuracy,
  Kappa = results$Kappa,
  BalancedAccuracy = round(c(balanced_acc(cm1), balanced_acc(cm2), balanced_acc(cm3)), 4),
  MacroF1 = round(c(macro_f1(cm1), macro_f1(cm2), macro_f1(cm3)), 4),
  AUC = results$AUC
)
write.csv(results_ext, file.path(output_subdir_MSA_subtype, "performance_summary_TOP20_extended.csv"), row.names = FALSE)

# 新增：分别落盘三模型的训练/测试逐样本明细（已在上方追加到统一文件，这里再各自留档可查）
try({
  write.csv(pred_detail_hpdmsa, file.path(output_subdir_MSA_subtype, "classification_details_HC_PD_MSA_TOP20.csv"), row.names = FALSE)
  write.csv(pred_detail_msap_full, file.path(output_subdir_MSA_subtype, "classification_details_HC_PD_MSA-P_Full.csv"), row.names = FALSE)
  write.csv(pred_detail_msap_top20, file.path(output_subdir_MSA_subtype, "classification_details_HC_PD_MSA-P_TOP20.csv"), row.names = FALSE)
}, silent = TRUE)

# 新增：多分类 byClass 逐类指标落盘（HC/PD/MSA 与 HC/PD/MSA-P 两模型，测试/训练）
try({
  # HC/PD/MSA
  cm1_train <- confusionMatrix(predict(model1, X_train), Y_train)
  byclass1_test <- as.data.frame(cm1$byClass); byclass1_test$Class <- rownames(cm1$byClass); byclass1_test$Dataset <- "Test"
  byclass1_train <- as.data.frame(cm1_train$byClass); byclass1_train$Class <- rownames(cm1_train$byClass); byclass1_train$Dataset <- "Train"
  write.csv(rbind(byclass1_train, byclass1_test), file.path(output_subdir_MSA_subtype, "byclass_HC_PD_MSA_TOP20.csv"), row.names = FALSE)
  # HC/PD/MSA-P Full
  cm2_train <- confusionMatrix(predict(model2, X_hc_pd_msap_full[train_idx2, ]), Y_hc_pd_msap[train_idx2])
  byclass2_test <- as.data.frame(cm2$byClass); byclass2_test$Class <- rownames(cm2$byClass); byclass2_test$Dataset <- "Test"
  byclass2_train <- as.data.frame(cm2_train$byClass); byclass2_train$Class <- rownames(cm2_train$byClass); byclass2_train$Dataset <- "Train"
  write.csv(rbind(byclass2_train, byclass2_test), file.path(output_subdir_MSA_subtype, "byclass_HC_PD_MSA-P_Full.csv"), row.names = FALSE)
  # HC/PD/MSA-P TOP20
  cm3_train <- confusionMatrix(predict(model3, X_hc_pd_msap_top20[train_idx3, ]), Y_hc_pd_msap[train_idx3])
  byclass3_test <- as.data.frame(cm3$byClass); byclass3_test$Class <- rownames(cm3$byClass); byclass3_test$Dataset <- "Test"
  byclass3_train <- as.data.frame(cm3_train$byClass); byclass3_train$Class <- rownames(cm3_train$byClass); byclass3_train$Dataset <- "Train"
  write.csv(rbind(byclass3_train, byclass3_test), file.path(output_subdir_MSA_subtype, "byclass_HC_PD_MSA-P_TOP20.csv"), row.names = FALSE)
}, silent = TRUE)
write.csv(cm_df1, file.path(output_subdir_MSA_subtype, "confusion_matrix_HPDMSA_TOP20.csv"), row.names = FALSE)
write.csv(cm_df2, file.path(output_subdir_MSA_subtype, "confusion_matrix_HC_PD_MSA-P_Full.csv"), row.names = FALSE)
write.csv(cm_df3, file.path(output_subdir_MSA_subtype, "confusion_matrix_HC_PD_MSA-P_TOP20.csv"), row.names = FALSE)

# === 新增[严格嵌套CV / 置换检验 / 学习曲线 执行] ===
if (STRICT_MODE) {
  idx_hc_pd_msa_strict <- which(demo_all$group %in% c("HC","PD","MSA"))
  y_strict <- droplevels(demo_all$group[idx_hc_pd_msa_strict])
  y_strict <- factor(as.character(y_strict), levels = c("HC","PD","MSA"))
  X_strict <- combined_measures[idx_hc_pd_msa_strict, , drop = FALSE]
  strict_nested_cv(X_all = X_strict, y_all = y_strict,
                   repeats = 5, folds = 5,
                   candidate_N = c(10,12,15),
                   tuneLength = 20,
                   outdir = output_subdir_Strict,
                   seed_base = 2025)
}

if (RUN_PERMUTATION) {
  idx_perm <- which(demo_all$group %in% c("HC","PD","MSA"))
  y_perm <- droplevels(demo_all$group[idx_perm])
  y_perm <- factor(as.character(y_perm), levels = c("HC","PD","MSA"))
  X_perm <- combined_measures[idx_perm, , drop = FALSE]
  permutation_test_auc(X_perm, y_perm, feats_top15 = top20_features,
                       reps = 200, outdir = output_subdir_Perm, seed = 3031)
}

if (RUN_LEARNING_CURVE) {
  idx_lc <- which(demo_all$group %in% c("HC","PD","MSA"))
  y_lc <- droplevels(demo_all$group[idx_lc])
  y_lc <- factor(as.character(y_lc), levels = c("HC","PD","MSA"))
  X_lc <- combined_measures[idx_lc, , drop = FALSE]
  learning_curve_top15(X_lc, y_lc, feats_top15 = top20_features,
                       train_fracs = c(0.1,0.3,0.5,0.7,0.9), repeats = 5,
                       outdir = output_subdir_LC, seed = 4041)
}

# === 新增[执行：全特征严格验证与稳健性分析] ===
if (STRICT_MODE) {
  idx_all <- which(demo_all$group %in% c("HC","PD","MSA"))
  y_all <- droplevels(demo_all$group[idx_all])
  y_all <- factor(as.character(y_all), levels = c("HC","PD","MSA"))
  X_all <- combined_measures[idx_all, , drop = FALSE]
  strict_nested_cv_allfeatures(X_all = X_all, y_all = y_all,
                               repeats = 5, folds = 5,
                               tuneLength = 20,
                               outdir = output_subdir_Strict_All,
                               seed_base = 2026)
}

if (RUN_PERMUTATION) {
  idx_all <- which(demo_all$group %in% c("HC","PD","MSA"))
  y_all <- droplevels(demo_all$group[idx_all])
  y_all <- factor(as.character(y_all), levels = c("HC","PD","MSA"))
  X_all <- combined_measures[idx_all, , drop = FALSE]
  permutation_test_auc_allfeatures(X_all = X_all, y_all = y_all,
                                   reps = 200, outdir = output_subdir_Perm_All, seed = 3032)
}

if (RUN_LEARNING_CURVE) {
  idx_all <- which(demo_all$group %in% c("HC","PD","MSA"))
  y_all <- droplevels(demo_all$group[idx_all])
  y_all <- factor(as.character(y_all), levels = c("HC","PD","MSA"))
  X_all <- combined_measures[idx_all, , drop = FALSE]
  learning_curve_allfeatures(X_all = X_all, y_all = y_all,
                             train_fracs = c(0.1,0.3,0.5,0.7,0.9), repeats = 5,
                             outdir = output_subdir_LC_All, seed = 4042)
}

stopCluster(cl)
cat("\n第4部分分析完成！包含ROC曲线生成。\n")
