#%%
# 生成PD vs MSA二分类模型（用于外部验证）

rm(list = ls())
setwd('/media/neurox/T7_Shield/PD_analyse')

cat("\n═══════════════════════════════════════════════════════════════\n")
cat("  PD vs MSA 二分类模型生成（TOP20特征，供外部验证使用）\n")
cat("═══════════════════════════════════════════════════════════════\n\n")

# 输出目录
output_dir <- "features_glmnet/Final_PD_vs_MSA_Publication"
if (dir.exists(output_dir)) {
  try(unlink(output_dir, recursive = TRUE, force = TRUE), silent = TRUE)
}
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# 加载依赖包
suppressPackageStartupMessages({
  library(reticulate); library(readxl); library(caret)
  library(tidyverse); library(ggplot2); library(doParallel); library(pROC)
  library(neuroCombat)
})

registerDoParallel(cores = max(1, parallel::detectCores() - 2))
use_python("/usr/local/fsl/bin/python", required = TRUE)
np <- import('numpy')

# 工具函数
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
exact_ci <- function(x, n, conf.level = 0.95) {
  if (is.na(x) || is.na(n) || n <= 0) return(c(NA_real_, NA_real_))
  as.numeric(binom.test(x, n, conf.level = conf.level)$conf.int)
}
binary_exact_ci_from_cm <- function(cm) {
  tab <- as.matrix(cm$table)
  tp <- suppressWarnings(as.numeric(tab["PD", "PD"]))
  fn <- suppressWarnings(as.numeric(tab["MSA", "PD"]))
  tn <- suppressWarnings(as.numeric(tab["MSA", "MSA"]))
  fp <- suppressWarnings(as.numeric(tab["PD", "MSA"]))
  n_all <- tp + tn + fp + fn
  list(
    acc = exact_ci(tp + tn, n_all),
    sens = exact_ci(tp, tp + fn),
    spec = exact_ci(tn, tn + fp),
    ppv = exact_ci(tp, tp + fp),
    npv = exact_ci(tn, tn + fn)
  )
}

# 数据加载（仅PD和MSA）
cat("【步骤1】数据加载（PD vs MSA二分类）\n")
demo_all <- read_excel('demo_all.xlsx')
pd_msa_idx <- which(demo_all$group %in% c('PD', 'MSA'))
demo_pd_msa <- demo_all[pd_msa_idx, ] %>%
  mutate(site = as.factor(site), group = factor(group, levels = c("MSA", "PD")))

cat(sprintf("  PD=%d, MSA=%d, Total=%d\n",
            sum(demo_pd_msa$group == "PD"),
            sum(demo_pd_msa$group == "MSA"),
            nrow(demo_pd_msa)))

# 读取T1影像特征
data_all <- np$load('surface_ALL.npy')
arr <- as.array(data_all)
n_rois <- dim(arr)[1]; n_subs <- dim(arr)[2]
area_mat <- matrix(arr[,,2], nrow = n_rois, ncol = n_subs)
thick_mat <- matrix(arr[,,3], nrow = n_rois, ncol = n_subs)
struc_df <- as.data.frame(t(cbind(area_mat, thick_mat)))
aseg_all <- read_excel("asegstats_all.xlsx")

# 特征命名
annot <- read.csv('/media/neurox/T7_Shield/R_done/analysis/surfaces/parcellations/lut/lut_schaefer-400_mics.csv')
annot <- annot[-c(1,202), ]
annot$label <- gsub('7Networks_', '', annot$label)
roi_labels <- annot$label
combined_labels <- c(paste(roi_labels, 'area', sep = '_'), 
                     paste(roi_labels, 'thickness', sep = '_'))
colnames(struc_df) <- make.unique(combined_labels)
colnames(aseg_all) <- paste0('subcort_', colnames(aseg_all))
X_all <- cbind(struc_df, aseg_all)
X_pd_msa <- X_all[pd_msa_idx, ]
Y_pd_msa <- demo_pd_msa$group

# TOP20特征提取
cat("\n【步骤2】读取TOP20特征清单（来自9.1 Part2）\n")
top20_candidates <- c(
  "RBD_test_TOP20/Part2_Disease_Sentinel_Definition/top_features_master_TOP20.csv",
  "最终文章使用代码及原始文件/Part2_Disease_Sentinel_Definition/top_features_master_TOP20.csv"
)
top20_file <- top20_candidates[file.exists(top20_candidates)][1]
if (is.na(top20_file) || !nzchar(top20_file)) {
  stop("TOP20特征文件不存在，请先运行 05_03_discovery_top20_and_irbd_core.R")
}
top20_features_df <- read.csv(top20_file, stringsAsFactors = FALSE)
top20_names <- top20_features_df$Feature
available_top20 <- intersect(top20_names, colnames(X_pd_msa))
cat(sprintf("  可用TOP20: %d/%d\n", length(available_top20), length(top20_names)))
X_top20 <- X_pd_msa[, available_top20]

# 保存TOP20清单
write.csv(data.frame(Feature = available_top20, stringsAsFactors = FALSE),
          file.path(output_dir, "TOP20_features_PD_vs_MSA_specific.csv"),
          row.names = FALSE)

# 数据分割
cat("\n【步骤3】数据分割（80/20）\n")
set.seed(2026)
train_idx <- createDataPartition(Y_pd_msa, p = 0.8, list = FALSE)
X_train <- X_top20[train_idx, ]; X_test <- X_top20[-train_idx, ]
Y_train <- Y_pd_msa[train_idx]; Y_test <- Y_pd_msa[-train_idx]
demo_train <- demo_pd_msa[train_idx, ]
demo_test <- demo_pd_msa[-train_idx, ]
cat(sprintf("  训练集: %d (PD=%d, MSA=%d)\n", 
            length(Y_train), sum(Y_train=="PD"), sum(Y_train=="MSA")))
cat(sprintf("  测试集: %d (PD=%d, MSA=%d)\n", 
            length(Y_test), sum(Y_test=="PD"), sum(Y_test=="MSA")))

# 冻结ComBat：仅在训练集拟合，测试集仅from-training应用（防信息泄漏）
cat("\n【步骤3.5】冻结ComBat（fit-on-train, apply-to-test）\n")
combat_assets <- list(
  applied = FALSE,
  estimates = NULL,
  feature_names = colnames(X_train),
  training_sites = unique(as.character(demo_train$site)),
  formula = "~ age + sex"
)
if (length(unique(as.character(demo_train$site))) >= 2) {
  mod_train <- model.matrix(~ age + sex, data = demo_train)
  combat_fit <- neuroCombat::neuroCombat(
    dat = t(as.matrix(X_train)),
    batch = as.character(demo_train$site),
    mod = mod_train
  )
  X_train <- as.data.frame(t(combat_fit$dat.combat))
  colnames(X_train) <- combat_assets$feature_names

  known_test_idx <- which(as.character(demo_test$site) %in% combat_assets$training_sites)
  if (length(known_test_idx) > 0) {
    mod_test <- model.matrix(~ age + sex, data = demo_test[known_test_idx, , drop = FALSE])
    combat_test <- neuroCombat::neuroCombatFromTraining(
      dat = t(as.matrix(X_test[known_test_idx, , drop = FALSE])),
      batch = as.character(demo_test$site[known_test_idx]),
      mod = mod_test,
      estimates = combat_fit$estimates
    )
    X_test_h <- X_test
    X_test_h[known_test_idx, ] <- as.data.frame(t(combat_test$dat.combat))
    X_test <- X_test_h
  }

  combat_assets$applied <- TRUE
  combat_assets$estimates <- combat_fit$estimates
  cat(sprintf("  冻结ComBat已应用：训练site数=%d，测试集中可应用样本=%d\n",
              length(combat_assets$training_sites), length(known_test_idx)))
} else {
  cat("  训练集site不足2个，冻结ComBat不执行（保持原始值，避免伪校正）\n")
}
saveRDS(combat_assets, file.path(output_dir, "TOP20_frozen_combat_assets.rds"))

# Winsorize阈值
cat("\n【步骤4】计算Winsorize阈值\n")
compute_q <- function(v, p) quantile(as.numeric(v), p, na.rm=TRUE, type=7)
lq <- apply(as.matrix(X_train), 2, compute_q, p = 0.005)
uq <- apply(as.matrix(X_train), 2, compute_q, p = 0.995)
write.csv(data.frame(Feature=colnames(X_train), Lower=lq, Upper=uq),
          file.path(output_dir, "TOP20_winsorize_thresholds.csv"), row.names=FALSE)

# 训练集统计
train_means <- colMeans(as.matrix(X_train), na.rm=TRUE)
train_sds <- apply(as.matrix(X_train), 2, sd, na.rm=TRUE)
write.csv(data.frame(Feature=colnames(X_train), TrainMean=train_means, TrainSD=train_sds),
          file.path(output_dir, "TOP20_train_stats.csv"), row.names=FALSE)

# glmnet训练
cat("\n【步骤5】glmnet模型训练\n")
set.seed(2028)
model_top20 <- train(
  x = X_train, y = Y_train, method = "glmnet", metric = "ROC",
  trControl = trainControl(method="cv", number=10, classProbs=TRUE, 
                           summaryFunction=twoClassSummary, savePredictions="final"),
  tuneLength = 50,
  preProc = c("YeoJohnson", "center", "scale")
)
cat(sprintf("  最优: alpha=%.3f, lambda=%.6f, CV-AUC=%.4f\n",
            model_top20$bestTune$alpha, model_top20$bestTune$lambda,
            max(model_top20$results$ROC, na.rm=TRUE)))

# 测试集评估
cat("\n【步骤6】测试集评估\n")
pred_test_class <- predict(model_top20, X_test)
pred_test_prob <- predict(model_top20, X_test, type = "prob")
cm_test <- confusionMatrix(pred_test_class, Y_test, positive = "PD")
roc_test <- roc(Y_test, pred_test_prob$PD, levels = c("MSA", "PD"))
auc_test <- auc(roc_test); ci_test <- auc_ci95_delong(roc_test)
ci_bin <- binary_exact_ci_from_cm(cm_test)
cat(sprintf("  测试集AUC: %.4f (95%% CI: %.3f-%.3f)\n", auc_test, ci_test[1], ci_test[3]))

# 保存性能
performance_df <- data.frame(
  Model = "PD_vs_MSA_TOP20_FrozenComBat",
  Accuracy = cm_test$overall["Accuracy"],
  Accuracy_CI95_lower = ci_bin$acc[1],
  Accuracy_CI95_upper = ci_bin$acc[2],
  Sensitivity = cm_test$byClass["Sensitivity"],
  Sensitivity_CI95_lower = ci_bin$sens[1],
  Sensitivity_CI95_upper = ci_bin$sens[2],
  Specificity = cm_test$byClass["Specificity"],
  Specificity_CI95_lower = ci_bin$spec[1],
  Specificity_CI95_upper = ci_bin$spec[2],
  PPV = cm_test$byClass["Pos Pred Value"],
  PPV_CI95_lower = ci_bin$ppv[1],
  PPV_CI95_upper = ci_bin$ppv[2],
  NPV = cm_test$byClass["Neg Pred Value"],
  NPV_CI95_lower = ci_bin$npv[1],
  NPV_CI95_upper = ci_bin$npv[2],
  F1 = f1_from_cm_binary(cm_test),
  AUC = as.numeric(auc_test),
  AUC_CI_lower = ci_test[1], AUC_CI_upper = ci_test[3]
)
write.csv(performance_df, file.path(output_dir, "performance_PD_vs_MSA_TOP20_test.csv"), row.names=FALSE)

# 固定阈值（Youden）
cat("\n【步骤7】计算固定阈值（Youden）\n")
oof_pred <- model_top20$pred
if (is.data.frame(oof_pred) && all(c("obs", "PD") %in% colnames(oof_pred))) {
  roc_oof <- roc(oof_pred$obs, oof_pred$PD, levels = c("MSA", "PD"), quiet = TRUE)
  coords_y <- coords(roc_oof, "best", best.method = "youden", ret = c("threshold","sensitivity","specificity"))
  write.csv(data.frame(Method="Youden", Threshold=coords_y[1], 
                       Sensitivity=coords_y[2], Specificity=coords_y[3]),
            file.path(output_dir, "TOP20_fixed_threshold.csv"), row.names=FALSE)
  cat(sprintf("  固定阈值: %.4f\n", coords_y[1]))
}

# 保存模型（核心）
cat("\n【步骤8】保存最终模型\n")
saveRDS(model_top20, file.path(output_dir, "model_TOP20_FrozenComBat_glmnet.rds"))
cat(sprintf("  ✓ 模型已保存: model_TOP20_FrozenComBat_glmnet.rds\n"))

cat("\n═══════════════════════════════════════════════════════════════\n")
cat("  模型训练完成！\n")
cat("  后续: 14/15外部验证脚本将使用此模型\n")
cat("═══════════════════════════════════════════════════════════════\n\n")
