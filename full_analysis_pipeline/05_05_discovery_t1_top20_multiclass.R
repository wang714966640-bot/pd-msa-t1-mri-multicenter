rm(list = ls())
setwd('/media/neurox/T7_Shield/PD_analyse')

# 输出目录（固定名称）
output_dir <- "features_glmnet/T1_TOP20"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
}

cat("\n═══════════════════════════════════════════\n")
cat("  T1 TOP20特征 HC-PD-MSA(P) 三分类模型\n")
cat("═══════════════════════════════════════════\n\n")

# 加载必要包
suppressPackageStartupMessages({
  library(reticulate)
  library(glmnet)
  library(readxl)
  library(caret)
  library(tidyverse)
  library(doParallel)
  library(pROC)
})

# Python环境
if (file.exists("/usr/local/fsl/bin/python")) {
  use_python("/usr/local/fsl/bin/python", required = TRUE)
} else {
  use_python(Sys.which("python3"), required = TRUE)
}
np <- import('numpy')
registerDoParallel(cores = parallel::detectCores())

# 工具函数
create_confusion_matrix_plot <- function(cm_table_df, title_text) {
  colnames(cm_table_df) <- c("Prediction", "Reference", "Freq")
  ggplot(data = cm_table_df, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile(color = "black") +
    geom_text(aes(label = Freq), color = "magenta", size = 5) +
    scale_fill_gradient(low = "white", high = "navyblue") +
    labs(title = title_text, x = "Predicted", y = "True") +
    theme_classic(base_size = 16) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.text = element_text(color = "black", size = 13),
      legend.position = "none"
    )
}

balanced_acc <- function(cm) {
  if (is.null(dim(cm$byClass))) return(NA_real_)
  (mean(cm$byClass[, "Sensitivity"], na.rm = TRUE) + 
   mean(cm$byClass[, "Specificity"], na.rm = TRUE)) / 2
}

format_metric_ci <- function(value, ci_lower, ci_upper) {
  if (is.na(value) || is.na(ci_lower) || is.na(ci_upper)) {
    return(sprintf("%.3f", value))
  }
  sprintf("%.3f (%.3f-%.3f)", value, ci_lower, ci_upper)
}

binom_ci <- function(x, n, conf.level = 0.95) {
  if (is.na(x) || is.na(n) || n <= 0) return(c(NA_real_, NA_real_))
  as.numeric(binom.test(x, n, conf.level = conf.level)$conf.int)
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

# ROC图绘制（T1样式：24pt粗体，600 DPI）
plot_multiclass_roc_T1style <- function(y_true, prob_mat, set_name, output_path) {
  classes <- colnames(prob_mat)
  n_classes <- length(classes)
  # 修改MSA-P颜色为桔红色 #FF6347
  color_map <- c("HC" = "#1F78B4", "MSA" = "#E31A1C", 
                 "PD" = "#33A02C", "MSA-P" = "#FF6347")
  
  roc_list <- list()
  auc_vals <- numeric(n_classes)
  
  for (i in 1:n_classes) {
    cls <- classes[i]
    y_binary <- ifelse(y_true == cls, 1, 0)
    roc_obj <- roc(y_binary, prob_mat[, cls], quiet = TRUE)
    roc_list[[cls]] <- roc_obj
    auc_vals[i] <- as.numeric(auc(roc_obj))
  }
  
  overall_auc <- multiclass.roc(y_true, prob_mat)$auc[1]
  
  png(output_path, width = 2400, height = 2400, res = 600)
  par(mar = c(5, 5, 4, 2) + 0.1)
  
  plot(NULL, xlim = c(0, 1), ylim = c(0, 1),
       xlab = "1 - Specificity", ylab = "Sensitivity",
       main = paste0("One-vs-Rest ROC (", set_name, ")\nOverall AUC = ", 
                     sprintf("%.3f", overall_auc)),
       las = 1, cex.main = 1.2, cex.lab = 1.8, cex.axis = 1.8)
  
  abline(a = 0, b = 1, lty = 2, col = "gray50", lwd = 2)
  
  for (i in 1:n_classes) {
    cls <- classes[i]
    roc_obj <- roc_list[[cls]]
    lines(1 - roc_obj$specificities, roc_obj$sensitivities,
          col = color_map[cls], lwd = 3)
  }
  
  legend_labels <- sprintf("%s (AUC=%.3f)", classes, auc_vals)
  legend_colors <- color_map[classes]
  
  legend("bottomright", legend = legend_labels, 
         col = legend_colors, lwd = 3, bty = "n",
         cex = 1.8, text.font = 2)
  
  dev.off()
  
  cat(sprintf("  ✓ ROC图已保存 (AUC=%.3f)\n", overall_auc))
  
  return(list(overall_auc = overall_auc, class_aucs = setNames(auc_vals, classes)))
}

# 读取数据
cat("【步骤1】读取数据...\n")
demo_all <- read_excel("demo_all.xlsx")
aseg_all <- read_excel("asegstats_all.xlsx")
surf_all_np <- np$load("surface_ALL.npy")

n_rois <- dim(surf_all_np)[1]
n_subs <- dim(surf_all_np)[2]

area_mat <- matrix(surf_all_np[,,2], nrow = n_rois, ncol = n_subs)
thickness_mat <- matrix(surf_all_np[,,3], nrow = n_rois, ncol = n_subs)

area_df <- as.data.frame(t(area_mat))
thickness_df <- as.data.frame(t(thickness_mat))

colnames(area_df) <- paste0("Area_", 1:n_rois)
colnames(thickness_df) <- paste0("Thickness_", 1:n_rois)

t1_features <- cbind(area_df, thickness_df, aseg_all[, -1])
cat(sprintf("  T1特征总数: %d\n", ncol(t1_features)))

# 读取TOP20特征
cat("\n【步骤2】读取TOP20特征...\n")
top20_candidates <- c(
  "RBD_test_TOP20/Part2_Disease_Sentinel_Definition/top_features_master_TOP20.csv",
  "最终文章使用代码及原始文件/Part2_Disease_Sentinel_Definition/top_features_master_TOP20.csv"
)
top20_file <- top20_candidates[file.exists(top20_candidates)][1]
if (is.na(top20_file) || !nzchar(top20_file)) {
  stop("TOP20特征文件不存在，请先运行 05_03_discovery_top20_and_irbd_core.R")
}
top20_features <- read.csv(top20_file, stringsAsFactors = FALSE)
top20_names <- top20_features$Feature
available_features <- intersect(top20_names, colnames(t1_features))

cat(sprintf("  可用特征数: %d / %d\n", length(available_features), length(top20_names)))

t1_top20 <- t1_features[, available_features]

# 准备HC-PD-MSA数据
cat("\n【步骤3】准备HC-PD-MSA数据...\n")
idx_hpdmsa <- which(demo_all$group %in% c("HC", "PD", "MSA"))
demo_hpdmsa <- demo_all[idx_hpdmsa, ]
t1_hpdmsa <- t1_top20[idx_hpdmsa, ]
y_hpdmsa <- factor(demo_hpdmsa$group, levels = c("HC", "PD", "MSA"))

cat(sprintf("  HC: %d, PD: %d, MSA: %d\n",
            sum(y_hpdmsa == "HC"), sum(y_hpdmsa == "PD"), sum(y_hpdmsa == "MSA")))

# 准备HC-PD-MSA-P数据
cat("\n【步骤4】准备HC-PD-MSA-P数据...\n")
idx_hpdmsap <- which(demo_all$group %in% c("HC", "PD") | 
                     (demo_all$group == "MSA" & demo_all$MSA_group == "MSA-P"))
demo_hpdmsap <- demo_all[idx_hpdmsap, ]
t1_hpdmsap <- t1_top20[idx_hpdmsap, ]
y_hpdmsap <- demo_hpdmsap$group
y_hpdmsap[demo_hpdmsap$MSA_group == "MSA-P"] <- "MSA-P"
y_hpdmsap <- factor(y_hpdmsap, levels = c("HC", "PD", "MSA-P"))

cat(sprintf("  HC: %d, PD: %d, MSA-P: %d\n",
            sum(y_hpdmsap == "HC"), sum(y_hpdmsap == "PD"), sum(y_hpdmsap == "MSA-P")))

# 训练与评估函数
train_and_evaluate <- function(X, y, task_name, seed_val, output_prefix) {
  cat(sprintf("\n【%s】开始训练...\n", task_name))
  
  set.seed(seed_val)
  train_idx <- createDataPartition(y, p = 0.8, list = FALSE)
  
  X_train <- X[train_idx, ]
  X_test <- X[-train_idx, ]
  y_train <- y[train_idx]
  y_test <- y[-train_idx]
  
  cat(sprintf("  训练集: %d, 测试集: %d\n", length(y_train), length(y_test)))
  
  # 训练模型
  set.seed(seed_val)
  model <- train(
    x = X_train, y = y_train,
    method = "glmnet",
    trControl = trainControl(method = "cv", number = 5, classProbs = TRUE,
                             summaryFunction = multiClassSummary),
    tuneLength = 20,
    preProc = c("center", "scale"),
    metric = "AUC"
  )
  
  # 训练集评估
  pred_train <- predict(model, X_train)
  prob_train <- predict(model, X_train, type = "prob")
  cm_train <- confusionMatrix(pred_train, y_train)
  
  roc_train <- plot_multiclass_roc_T1style(
    y_train, prob_train, "Train",
    file.path(output_dir, paste0("ROC_", output_prefix, "_Train_T1Style.png"))
  )
  
  cm_plot_train <- create_confusion_matrix_plot(
    as.data.frame(cm_train$table), paste(task_name, "- Train"))
  ggsave(file.path(output_dir, paste0("Confusion_Matrix_", output_prefix, "_train.png")),
         cm_plot_train, width = 8, height = 7, dpi = 300)
  
  # 测试集评估
  pred_test <- predict(model, X_test)
  prob_test <- predict(model, X_test, type = "prob")
  cm_test <- confusionMatrix(pred_test, y_test)
  
  roc_test <- plot_multiclass_roc_T1style(
    y_test, prob_test, "Test",
    file.path(output_dir, paste0("ROC_", output_prefix, "_Test_T1Style.png"))
  )
  
  cm_plot_test <- create_confusion_matrix_plot(
    as.data.frame(cm_test$table), paste(task_name, "- Test"))
  ggsave(file.path(output_dir, paste0("Confusion_Matrix_", output_prefix, "_test.png")),
         cm_plot_test, width = 8, height = 7, dpi = 300)
  
  # 保存详情
  details_train <- data.frame(
    Subject_Index = train_idx,
    True_Label = as.character(y_train),
    Predicted = as.character(pred_train),
    prob_train,
    Dataset = "Train"
  )
  
  details_test <- data.frame(
    Subject_Index = which(!(1:length(y) %in% train_idx)),
    True_Label = as.character(y_test),
    Predicted = as.character(pred_test),
    prob_test,
    Dataset = "Test"
  )
  
  write.csv(rbind(details_train, details_test),
            file.path(output_dir, paste0("classification_details_", output_prefix, ".csv")),
            row.names = FALSE)
  
  # 计算指标和CI
  n_train <- length(y_train)
  n_test <- length(y_test)
  
  auc_train <- roc_train$overall_auc
  auc_test <- roc_test$overall_auc
  
  auc_ci_train <- multiclass_auc_ci_bootstrap(y_train, prob_train, n_boot = 1000, seed = 2026L)
  auc_ci_test <- multiclass_auc_ci_bootstrap(y_test, prob_test, n_boot = 1000, seed = 2027L)
  
  acc_train <- cm_train$overall["Accuracy"]
  acc_test <- cm_test$overall["Accuracy"]
  acc_ci_train <- binom_ci(round(acc_train * n_train), n_train)
  acc_ci_test <- binom_ci(round(acc_test * n_test), n_test)
  
  # Macro指标
  get_macro <- function(cm, n) {
    if (is.null(dim(cm$byClass))) {
      return(list(sens = NA, spec = NA, ppv = NA, npv = NA, f1 = NA,
                  sens_ci = c(NA, NA), spec_ci = c(NA, NA),
                  ppv_ci = c(NA, NA), npv_ci = c(NA, NA)))
    }
    
    sens <- mean(cm$byClass[, "Sensitivity"], na.rm = TRUE)
    spec <- mean(cm$byClass[, "Specificity"], na.rm = TRUE)
    ppv <- mean(cm$byClass[, "Pos Pred Value"], na.rm = TRUE)
    npv <- mean(cm$byClass[, "Neg Pred Value"], na.rm = TRUE)
    f1 <- mean(cm$byClass[, "F1"], na.rm = TRUE)
    
    calc_ci <- function(val) {
      if (is.na(val)) return(c(NA, NA))
      binom_ci(round(val * n), n)
    }
    
    list(sens = sens, spec = spec, ppv = ppv, npv = npv, f1 = f1,
         sens_ci = calc_ci(sens), spec_ci = calc_ci(spec),
         ppv_ci = calc_ci(ppv), npv_ci = calc_ci(npv))
  }
  
  macro_train <- get_macro(cm_train, n_train)
  macro_test <- get_macro(cm_test, n_test)
  
  cat(sprintf("  训练集 AUC=%.3f, Acc=%.3f\n", auc_train, acc_train))
  cat(sprintf("  测试集 AUC=%.3f, Acc=%.3f\n", auc_test, acc_test))
  
  return(list(
    train = list(auc = auc_train, auc_ci = auc_ci_train,
                 acc = acc_train, acc_ci = acc_ci_train,
                 kappa = cm_train$overall["Kappa"],
                 bal_acc = balanced_acc(cm_train),
                 macro_sens = macro_train$sens, sens_ci = macro_train$sens_ci,
                 macro_spec = macro_train$spec, spec_ci = macro_train$spec_ci,
                 macro_ppv = macro_train$ppv, ppv_ci = macro_train$ppv_ci,
                 macro_npv = macro_train$npv, npv_ci = macro_train$npv_ci,
                 macro_f1 = macro_train$f1),
    test = list(auc = auc_test, auc_ci = auc_ci_test,
                acc = acc_test, acc_ci = acc_ci_test,
                kappa = cm_test$overall["Kappa"],
                bal_acc = balanced_acc(cm_test),
                macro_sens = macro_test$sens, sens_ci = macro_test$sens_ci,
                macro_spec = macro_test$spec, spec_ci = macro_test$spec_ci,
                macro_ppv = macro_test$ppv, ppv_ci = macro_test$ppv_ci,
                macro_npv = macro_test$npv, npv_ci = macro_test$npv_ci,
                macro_f1 = macro_test$f1)
  ))
}

# 运行两个分类
results_hpdmsa <- train_and_evaluate(
  X = t1_hpdmsa, y = y_hpdmsa,
  task_name = "HC-PD-MSA", seed_val = 401,
  output_prefix = "HPDMSA"
)

results_hpdmsap <- train_and_evaluate(
  X = t1_hpdmsap, y = y_hpdmsap,
  task_name = "HC-PD-MSA-P", seed_val = 402,
  output_prefix = "HPDMSAP"
)

# 生成汇总表
cat("\n【步骤5】生成汇总表...\n")

make_row <- function(res, task_name, dataset_cn, dataset_en) {
  data.frame(
    文件夹 = task_name,
    数据集 = dataset_cn,
    Dataset = dataset_en,
    AUC = format_metric_ci(res$auc, res$auc_ci[1], res$auc_ci[2]),
    准确率 = format_metric_ci(res$acc, res$acc_ci[1], res$acc_ci[2]),
    平衡准确率 = sprintf("%.3f", res$bal_acc),
    灵敏度 = format_metric_ci(res$macro_sens, res$sens_ci[1], res$sens_ci[2]),
    特异度 = format_metric_ci(res$macro_spec, res$spec_ci[1], res$spec_ci[2]),
    PPV = format_metric_ci(res$macro_ppv, res$ppv_ci[1], res$ppv_ci[2]),
    NPV = format_metric_ci(res$macro_npv, res$npv_ci[1], res$npv_ci[2]),
    F1分数 = sprintf("%.3f", res$macro_f1),
    `Macro.F1` = sprintf("%.3f", res$macro_f1),
    Kappa = sprintf("%.3f", res$kappa),
    stringsAsFactors = FALSE
  )
}

summary_df <- rbind(
  make_row(results_hpdmsa$train, "T1_TOP20_HPDMSA", "训练集", "Train"),
  make_row(results_hpdmsa$test, "T1_TOP20_HPDMSA", "测试集", "Test"),
  make_row(results_hpdmsap$train, "T1_TOP20_HPDMSAP", "训练集", "Train"),
  make_row(results_hpdmsap$test, "T1_TOP20_HPDMSAP", "测试集", "Test")
)

output_summary <- file.path(output_dir, "顶刊级别完整指标汇总_T1_TOP20_训练测试集.csv")
write.csv(summary_df, output_summary, row.names = FALSE, fileEncoding = "UTF-8")

cat("\n✓ 汇总表已保存:", output_summary, "\n")
print(summary_df[, c("文件夹", "数据集", "AUC", "准确率", "Kappa")])

cat("\n═══════════════════════════════════════════\n")
cat("  ✨ 所有任务完成！\n")
cat("═══════════════════════════════════════════\n")
cat("\n输出位置:", output_dir, "\n\n")

