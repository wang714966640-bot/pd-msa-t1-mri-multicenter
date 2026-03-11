#%%
# T1结构特征（皮层+皮层下）的HC/MSA/PD三分类
# 去除所有图论指标，仅使用T1形态学数据
rm(list = ls())
setwd('/media/neurox/T7_Shield/PD_analyse')
run_tag <- format(Sys.time(), "%Y%m%d_%H%M%S")
output_subdir <- file.path("features_glmnet", paste0("T1_optimized_", run_tag))
dir.create(output_subdir, showWarnings = FALSE, recursive = TRUE)

# 加载必要包
library(reticulate)
use_python("/usr/local/fsl/bin/python", required = TRUE)
np <- import('numpy')

library(glmnet)
library(readxl)
library(caret)
library(Matrix)
library(tidyverse)
library(doParallel)
registerDoParallel(cores = parallel::detectCores())
library(pROC)

# 绘图工具函数
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

balanced_acc <- function(cm) {
  if (is.null(dim(cm$byClass))) return(NA_real_)
  sens <- mean(cm$byClass[, "Sensitivity"], na.rm = TRUE)
  spec <- mean(cm$byClass[, "Specificity"], na.rm = TRUE)
  (sens + spec) / 2
}

macro_f1 <- function(cm) {
  if (is.null(dim(cm$byClass))) return(NA_real_)
  if ("F1" %in% colnames(cm$byClass)) {
    mean(cm$byClass[, "F1"], na.rm = TRUE)
  } else {
    NA_real_
  }
}

acc_ci_from_cm <- function(cm) {
  tab <- as.matrix(cm$table)
  correct <- sum(diag(tab))
  n <- sum(tab)
  ci <- suppressWarnings(binom.test(correct, n, conf.level = 0.95)$conf.int)
  c(lower = ci[1], upper = ci[2])
}

auc_ci95_delong <- function(roc_obj) {
  ci <- suppressWarnings(pROC::ci.auc(roc_obj, conf.level = 0.95, method = "delong"))
  as.numeric(ci)
}
multiclass_auc_ci_bootstrap <- function(y_true, prob_df, n_boot = 1000, seed = 2026L) {
  set.seed(seed)
  n <- length(y_true)
  auc_boot <- numeric(n_boot)
  for (b in seq_len(n_boot)) {
    idx <- sample.int(n, n, replace = TRUE)
    auc_boot[b] <- tryCatch(
      as.numeric(pROC::auc(pROC::multiclass.roc(y_true[idx], prob_df[idx, , drop = FALSE]))),
      error = function(e) NA_real_
    )
  }
  auc_boot <- auc_boot[is.finite(auc_boot)]
  if (length(auc_boot) < 20) return(c(NA_real_, NA_real_))
  as.numeric(quantile(auc_boot, probs = c(0.025, 0.975), na.rm = TRUE))
}

#%% 数据加载（仅T1结构数据）
cat("\n加载T1结构数据...\n")
demo <- read_excel('demo_all.xlsx')
demo$group <- as.character(demo$group)
keep_idx <- which(demo$group %in% c("HC", "MSA", "PD"))

# 皮层形态学指标
data <- np$load('surface_ALL.npy')
struc_measures <- array(data, dim = c(nrow(demo), 800))
struc_measures <- as.data.frame(struc_measures)
struc_measures <- struc_measures[keep_idx, , drop = FALSE]
rm(data)

annot <- read.csv("/media/neurox/T7_Shield/R_done/analysis/surfaces/parcellations/lut/lut_schaefer-400_mics.csv")
annot <- annot[-c(1,202), ]
annot$label <- gsub("7Networks_", "", annot$label)
roi_labels <- annot$label

struc_labels <- rep(c('area', 'thickness'), each = 400)
combined_labels <- paste(roi_labels, struc_labels, sep = "_")
colnames(struc_measures) <- combined_labels

# 皮层下结构
asegstats_measures <- read_excel("asegstats_all.xlsx", col_names = TRUE)
asegstats_measures <- asegstats_measures[keep_idx, , drop = FALSE]

# 仅合并T1数据（无图论指标）
combined_measures <- cbind(struc_measures, asegstats_measures)

demo <- demo[keep_idx, , drop = FALSE]
features <- cbind(demo$group, combined_measures)
colnames(features)[1] <- 'group'
features <- features[features$group %in% c('HC','MSA','PD'), ]
colnames(features) <- make.unique(colnames(features))
group <- as.factor(features$group)

model_mat <- model.matrix(group ~ ., data = features)
model_mat <- model_mat[, -1]

cat(sprintf("T1特征数量: %d\n", ncol(model_mat)))

#%% 建模训练
set.seed(282828)
split_indices <- createDataPartition(group, p = 0.8, list = FALSE)
X_train <- model_mat[split_indices, ]
Y_train <- group[split_indices]
X_test <- model_mat[-split_indices, ]
Y_test <- group[-split_indices]

train_control <- trainControl(
  method = "cv",
  number = 10,
  search = "grid",
  allowParallel = TRUE,
  classProbs = TRUE,
  summaryFunction = multiClassSummary
)

tune_grid <- expand.grid(
  alpha = seq(0.165, 0.565, length = 15),
  lambda = exp(seq(-0.45, 0.85, length = 40))
)

cat("\n开始训练模型...\n")
model <- train(
  x = X_train,
  y = Y_train,
  family = "multinomial",
  type.multinomial = "ungrouped",
  metric = "Accuracy",
  method = "glmnet",
  trControl = train_control,
  tuneGrid = tune_grid,
  preProc = c("center", "scale")
)

print(model$bestTune)

# 测试集预测
predictions <- predict(model, X_test)
prediction_probabilities <- predict(model, X_test, type = "prob")
mc_auc_test <- as.numeric(pROC::auc(pROC::multiclass.roc(Y_test, prediction_probabilities)))

# 训练集预测
pred_train <- predict(model, X_train)
prob_train <- predict(model, X_train, type = "prob")
mc_auc_train <- as.numeric(pROC::auc(pROC::multiclass.roc(Y_train, prob_train)))

cat(sprintf("\n========== 初步结果 ==========\n"))
cat(sprintf("训练集AUC: %.4f\n", mc_auc_train))
cat(sprintf("测试集AUC: %.4f\n", mc_auc_test))
cat(sprintf("================================\n\n"))

# Confusion matrices
cm <- confusionMatrix(predictions, Y_test)
cm_train <- confusionMatrix(pred_train, Y_train)

# 保存混淆矩阵图
output_subdir_HC_PD_MSA <- output_subdir
cm_table_df <- as.data.frame(cm$table)
plot_cm <- create_confusion_matrix_plot(cm_table_df, "Confusion Matrix - HC/PD/MSA (Test)")
ggsave(filename = file.path(output_subdir_HC_PD_MSA, "Confusion_Matrix_HPDMSA_model_test.png"), plot = plot_cm, width = 7, height = 6, dpi = 300)

cm_table_df_tr <- as.data.frame(cm_train$table)
plot_cm_tr <- create_confusion_matrix_plot(cm_table_df_tr, "Confusion Matrix - HC/PD/MSA (Train)")
ggsave(filename = file.path(output_subdir_HC_PD_MSA, "Confusion_Matrix_HPDMSA_model_train.png"), plot = plot_cm_tr, width = 7, height = 6, dpi = 300)

write.csv(cm_table_df, file.path(output_subdir_HC_PD_MSA, "confusion_matrix_part1_test.csv"), row.names = FALSE)

# ROC曲线（600 DPI，24pt Bold，无边框）
roc_colors <- c("HC" = "#1F78B4", "MSA" = "#E31A1C", "PD" = "#33A02C")
classes <- colnames(prediction_probabilities)
prob_test <- prediction_probabilities

# 测试集ROC
roc_list_test <- list()
for (class_name in classes) {
  bin_true <- ifelse(Y_test == class_name, 1, 0)
  roc_obj <- pROC::roc(bin_true, prob_test[, class_name], levels = c(0, 1), direction = "<", quiet = TRUE)
  roc_list_test[[class_name]] <- roc_obj
}

roc_data_test <- do.call(rbind, lapply(names(roc_list_test), function(class_name) {
  roc_obj <- roc_list_test[[class_name]]
  data.frame(
    specificity = roc_obj$specificities,
    sensitivity = roc_obj$sensitivities,
    class = class_name,
    auc = as.numeric(pROC::auc(roc_obj))
  )
}))

roc_plot_test <- ggplot(roc_data_test, aes(x = 1 - specificity, y = sensitivity, color = class)) +
  geom_line(linewidth = 1.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60", linewidth = 0.8) +
  scale_color_manual(values = roc_colors, 
                    labels = sapply(classes, function(c) sprintf("%s (AUC=%.3f)", c, unique(roc_data_test$auc[roc_data_test$class == c])))) +
  labs(title = sprintf("One-vs-Rest ROC (Test)\nOverall AUC = %.3f", mc_auc_test),
       x = "1 - Specificity", y = "Sensitivity") +
  theme_classic(base_size = 14, base_family = "Arial") +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16, margin = margin(t = 5, b = 10)),
    axis.text = element_text(color = "black", face = "bold", size = 24),
    axis.title = element_text(color = "black", face = "bold", size = 24),
    legend.title = element_blank(),
    legend.text = element_text(face = "bold", size = 24),
    legend.position = c(0.70, 0.20),
    legend.background = element_rect(fill = "white", color = NA),
    legend.key.size = unit(1.2, "cm"),
    plot.margin = margin(t = 25, r = 15, b = 15, l = 15)
  ) +
  coord_fixed(ratio = 1) +
  scale_x_continuous(expand = c(0.02, 0.02), limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  scale_y_continuous(expand = c(0.02, 0.02), limits = c(0, 1), breaks = seq(0, 1, 0.2))

ggsave(file.path(output_subdir_HC_PD_MSA, "ROC_HPDMSA_Test_Part4Style.png"), 
       roc_plot_test, width = 8, height = 8, dpi = 600, bg = "white")

# 训练集ROC
roc_list_train <- list()
for (class_name in classes) {
  bin_true <- ifelse(Y_train == class_name, 1, 0)
  roc_obj <- pROC::roc(bin_true, prob_train[, class_name], levels = c(0, 1), direction = "<", quiet = TRUE)
  roc_list_train[[class_name]] <- roc_obj
}

roc_data_train <- do.call(rbind, lapply(names(roc_list_train), function(class_name) {
  roc_obj <- roc_list_train[[class_name]]
  data.frame(
    specificity = roc_obj$specificities,
    sensitivity = roc_obj$sensitivities,
    class = class_name,
    auc = as.numeric(pROC::auc(roc_obj))
  )
}))

roc_plot_train <- ggplot(roc_data_train, aes(x = 1 - specificity, y = sensitivity, color = class)) +
  geom_line(linewidth = 1.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60", linewidth = 0.8) +
  scale_color_manual(values = roc_colors,
                    labels = sapply(classes, function(c) sprintf("%s (AUC=%.3f)", c, unique(roc_data_train$auc[roc_data_train$class == c])))) +
  labs(title = sprintf("One-vs-Rest ROC (Train)\nOverall AUC = %.3f", mc_auc_train),
       x = "1 - Specificity", y = "Sensitivity") +
  theme_classic(base_size = 14, base_family = "Arial") +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16, margin = margin(t = 5, b = 10)),
    axis.text = element_text(color = "black", face = "bold", size = 24),
    axis.title = element_text(color = "black", face = "bold", size = 24),
    legend.title = element_blank(),
    legend.text = element_text(face = "bold", size = 24),
    legend.position = c(0.70, 0.20),
    legend.background = element_rect(fill = "white", color = NA),
    legend.key.size = unit(1.2, "cm"),
    plot.margin = margin(t = 25, r = 15, b = 15, l = 15)
  ) +
  coord_fixed(ratio = 1) +
  scale_x_continuous(expand = c(0.02, 0.02), limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  scale_y_continuous(expand = c(0.02, 0.02), limits = c(0, 1), breaks = seq(0, 1, 0.2))

ggsave(file.path(output_subdir_HC_PD_MSA, "ROC_HPDMSA_Train_Part4Style.png"), 
       roc_plot_train, width = 8, height = 8, dpi = 600, bg = "white")

# 性能指标计算（与all_optimized一致）
cat("\n计算详细性能指标...\n")
n_features <- ncol(model_mat)
acc_ci_te <- acc_ci_from_cm(cm)
acc_ci_tr <- acc_ci_from_cm(cm_train)

# All metrics with CI (exact/binomial for non-AUC)
all_metrics_list <- list()
idx <- 1

# Train Overall (multiclass AUC CI via bootstrap percentile)
n_train <- length(Y_train)
auc_ci_train <- multiclass_auc_ci_bootstrap(Y_train, prob_train, n_boot = 1000, seed = 2026L)
auc_ci_train_lower <- auc_ci_train[1]
auc_ci_train_upper <- auc_ci_train[2]

all_metrics_list[[idx]] <- data.frame(
  Set = "Train", Class = "Overall", FeatureCount = n_features,
  AUC = mc_auc_train, AUC_Low = auc_ci_train_lower, AUC_High = auc_ci_train_upper,
  Accuracy = unname(cm_train$overall["Accuracy"]),
  Acc_Low = acc_ci_tr["lower"], Acc_High = acc_ci_tr["upper"],
  Sensitivity = NA, Sens_Low = NA, Sens_High = NA,
  Specificity = NA, Spec_Low = NA, Spec_High = NA,
  PPV = NA, PPV_Low = NA, PPV_High = NA,
  NPV = NA, NPV_Low = NA, NPV_High = NA
)
idx <- idx + 1

# Train classes - Clopper-Pearson exact interval
for (i in seq_along(classes)) {
  bin_true <- factor(ifelse(Y_train == classes[i], classes[i], "other"), levels = c(classes[i], "other"))
  bin_pred <- factor(ifelse(pred_train == classes[i], classes[i], "other"), levels = c(classes[i], "other"))
  roc_obj <- pROC::roc(bin_true, prob_train[, i], levels = c(classes[i], "other"), quiet = TRUE)
  auc_ci <- auc_ci95_delong(roc_obj)
  
  cm_binary <- confusionMatrix(bin_pred, bin_true, positive = classes[i])
  tab_bin <- as.matrix(cm_binary$table)
  tp <- tab_bin[classes[i], classes[i]]
  fn <- tab_bin["other", classes[i]]
  tn <- tab_bin["other", "other"]
  fp <- tab_bin[classes[i], "other"]
  
  sens_ci <- suppressWarnings(binom.test(tp, tp + fn, conf.level = 0.95)$conf.int)
  spec_ci <- suppressWarnings(binom.test(tn, tn + fp, conf.level = 0.95)$conf.int)
  ppv_ci  <- suppressWarnings(binom.test(tp, tp + fp, conf.level = 0.95)$conf.int)
  npv_ci  <- suppressWarnings(binom.test(tn, tn + fn, conf.level = 0.95)$conf.int)
  
  all_metrics_list[[idx]] <- data.frame(
    Set = "Train", Class = classes[i], FeatureCount = n_features,
    AUC = as.numeric(pROC::auc(roc_obj)), AUC_Low = auc_ci[1], AUC_High = auc_ci[3],
    Accuracy = NA, Acc_Low = NA, Acc_High = NA,
    Sensitivity = cm_binary$byClass["Sensitivity"],
    Sens_Low = sens_ci[1], Sens_High = sens_ci[2],
    Specificity = cm_binary$byClass["Specificity"],
    Spec_Low = spec_ci[1], Spec_High = spec_ci[2],
    PPV = cm_binary$byClass["Pos Pred Value"],
    PPV_Low = ppv_ci[1], PPV_High = ppv_ci[2],
    NPV = cm_binary$byClass["Neg Pred Value"],
    NPV_Low = npv_ci[1], NPV_High = npv_ci[2]
  )
  idx <- idx + 1
}

# Test Overall (multiclass AUC CI via bootstrap percentile)
n_test <- length(Y_test)
auc_ci_test <- multiclass_auc_ci_bootstrap(Y_test, prob_test, n_boot = 1000, seed = 2027L)
auc_ci_test_lower <- auc_ci_test[1]
auc_ci_test_upper <- auc_ci_test[2]

all_metrics_list[[idx]] <- data.frame(
  Set = "Test", Class = "Overall", FeatureCount = n_features,
  AUC = mc_auc_test, AUC_Low = auc_ci_test_lower, AUC_High = auc_ci_test_upper,
  Accuracy = unname(cm$overall["Accuracy"]),
  Acc_Low = acc_ci_te["lower"], Acc_High = acc_ci_te["upper"],
  Sensitivity = NA, Sens_Low = NA, Sens_High = NA,
  Specificity = NA, Spec_Low = NA, Spec_High = NA,
  PPV = NA, PPV_Low = NA, PPV_High = NA,
  NPV = NA, NPV_Low = NA, NPV_High = NA
)
idx <- idx + 1

# Test classes - Clopper-Pearson exact interval
for (i in seq_along(classes)) {
  bin_true <- factor(ifelse(Y_test == classes[i], classes[i], "other"), levels = c(classes[i], "other"))
  bin_pred <- factor(ifelse(predictions == classes[i], classes[i], "other"), levels = c(classes[i], "other"))
  roc_obj <- pROC::roc(bin_true, prob_test[, i], levels = c(classes[i], "other"), quiet = TRUE)
  auc_ci <- auc_ci95_delong(roc_obj)
  
  cm_binary <- confusionMatrix(bin_pred, bin_true, positive = classes[i])
  tab_bin <- as.matrix(cm_binary$table)
  tp <- tab_bin[classes[i], classes[i]]
  fn <- tab_bin["other", classes[i]]
  tn <- tab_bin["other", "other"]
  fp <- tab_bin[classes[i], "other"]
  
  sens_ci <- suppressWarnings(binom.test(tp, tp + fn, conf.level = 0.95)$conf.int)
  spec_ci <- suppressWarnings(binom.test(tn, tn + fp, conf.level = 0.95)$conf.int)
  ppv_ci  <- suppressWarnings(binom.test(tp, tp + fp, conf.level = 0.95)$conf.int)
  npv_ci  <- suppressWarnings(binom.test(tn, tn + fn, conf.level = 0.95)$conf.int)
  
  all_metrics_list[[idx]] <- data.frame(
    Set = "Test", Class = classes[i], FeatureCount = n_features,
    AUC = as.numeric(pROC::auc(roc_obj)), AUC_Low = auc_ci[1], AUC_High = auc_ci[3],
    Accuracy = NA, Acc_Low = NA, Acc_High = NA,
    Sensitivity = cm_binary$byClass["Sensitivity"],
    Sens_Low = sens_ci[1], Sens_High = sens_ci[2],
    Specificity = cm_binary$byClass["Specificity"],
    Spec_Low = spec_ci[1], Spec_High = spec_ci[2],
    PPV = cm_binary$byClass["Pos Pred Value"],
    PPV_Low = ppv_ci[1], PPV_High = ppv_ci[2],
    NPV = cm_binary$byClass["Neg Pred Value"],
    NPV_Low = npv_ci[1], NPV_High = npv_ci[2]
  )
  idx <- idx + 1
}

write.csv(do.call(rbind, all_metrics_list),
          file.path(output_subdir_HC_PD_MSA, "performance_part1_all_metrics_with_ci.csv"), row.names = FALSE)

# 其他输出文件
ext_test <- data.frame(
  Model = "HC_PD_MSA_T1_Only",
  Dataset = "Test",
  Accuracy = unname(cm$overall["Accuracy"]),
  Accuracy_CI95_lower = acc_ci_te["lower"],
  Accuracy_CI95_upper = acc_ci_te["upper"],
  Kappa = unname(cm$overall["Kappa"]),
  BalancedAccuracy = balanced_acc(cm),
  MacroF1 = macro_f1(cm),
  MultiClass_AUC = mc_auc_test
)

ext_train <- data.frame(
  Model = "HC_PD_MSA_T1_Only",
  Dataset = "Train",
  Accuracy = unname(cm_train$overall["Accuracy"]),
  Accuracy_CI95_lower = acc_ci_tr["lower"],
  Accuracy_CI95_upper = acc_ci_tr["upper"],
  Kappa = unname(cm_train$overall["Kappa"]),
  BalancedAccuracy = balanced_acc(cm_train),
  MacroF1 = macro_f1(cm_train),
  MultiClass_AUC = mc_auc_train
)

write.csv(rbind(ext_train, ext_test),
          file.path(output_subdir_HC_PD_MSA, "performance_summary_HC_PD_MSA_T1_Only_extended.csv"), row.names = FALSE)

overall_summary <- data.frame(
  Set = c("Train", "Test"),
  Class = "Overall",
  FeatureCount = n_features,
  AUC = c(mc_auc_train, mc_auc_test),
  AUC_Low = c(auc_ci_train_lower, auc_ci_test_lower),
  AUC_High = c(auc_ci_train_upper, auc_ci_test_upper),
  Accuracy = c(unname(cm_train$overall["Accuracy"]), unname(cm$overall["Accuracy"])),
  Acc_Low = c(acc_ci_tr["lower"], acc_ci_te["lower"]),
  Acc_High = c(acc_ci_tr["upper"], acc_ci_te["upper"]),
  Sensitivity = NA, Sens_Low = NA, Sens_High = NA,
  Specificity = NA, Spec_Low = NA, Spec_High = NA,
  PPV = NA, PPV_Low = NA, PPV_High = NA,
  NPV = NA, NPV_Low = NA, NPV_High = NA
)
write.csv(overall_summary, file.path(output_subdir_HC_PD_MSA, "performance_part1_overall_summary.csv"), row.names = FALSE)

byclass_data <- rbind(
  data.frame(Set = "Train", Class = rownames(cm_train$byClass), cm_train$byClass),
  data.frame(Set = "Test", Class = rownames(cm$byClass), cm$byClass)
)
write.csv(byclass_data, file.path(output_subdir_HC_PD_MSA, "byclass_HC_PD_MSA_T1_Only.csv"), row.names = FALSE)

# 保存划分索引
write.csv(data.frame(train_idx = as.integer(split_indices)), file.path(output_subdir_HC_PD_MSA, "split_HC_PD_MSA_train_idx.csv"), row.names = FALSE)
write.csv(data.frame(test_idx  = setdiff(seq_len(nrow(model_mat)), as.integer(split_indices))), file.path(output_subdir_HC_PD_MSA, "split_HC_PD_MSA_test_idx.csv"), row.names = FALSE)

pred_detail <- data.frame(
  Dataset = c(rep("Train", length(Y_train)), rep("Test", length(Y_test))),
  True = c(as.character(Y_train), as.character(Y_test)),
  Pred = c(as.character(pred_train), as.character(predictions))
)
write.csv(pred_detail, file.path(output_subdir_HC_PD_MSA, "classification_details_HC_PD_MSA.csv"), row.names = FALSE)

write_excel_csv(combined_measures, file = file.path(output_subdir, "combined_measures_T1_only.csv"))

cat("\n========== 最终结果 ==========\n")
cat("训练集总AUC:", round(mc_auc_train, 4), "\n")
cat("测试集总AUC:", round(mc_auc_test, 4), "\n")
cat("输出文件夹:", output_subdir_HC_PD_MSA, "\n")
cat("==============================\n")
