# %%

# 形成包含所有结构和功能连接度量的大型数据集combined_measures数据框，用于进一步分析或机器学习

rm(list = ls())

setwd('/media/neurox/T7_Shield/PD_analyse') # 保持原始工作目录设置


# ==========================================================================

# 第0部分: 环境准备、包加载、数据加载和预处理

# ==========================================================================

cat("第0部分: 环境准备、包加载、数据加载和预处理...\n")


# --- 输出主目录 ---

output_basedir <- "RBD_test_TOP20"

dir.create(output_basedir, showWarnings = FALSE, recursive = TRUE)


# --- 加载必要包 ---

packages_to_load <- c(

"reticulate", "glmnet", "readxl", "caret", "Matrix", "tidyverse", "ggplot2",

"doParallel", "pROC", "RColorBrewer", "wordcloud", "gplots", "plotly",

"tidyr", "dplyr", "purrr", "writexl", "ggtern", "fmsb", "pheatmap"

)

# 检查并安装缺失的包 (不包括GitHub包)

for (pkg in packages_to_load) {

if (!requireNamespace(pkg, quietly = TRUE)) {

stop(paste0("缺少R包: ", pkg, "。请先在环境中安装后再运行该脚本。"))

}

library(pkg, character.only = TRUE)

}


# ggseg相关包（默认关闭以加速且当前脚本未使用脑表面图）
ENABLE_GGSEG <- FALSE
ggseg_available <- FALSE
if (ENABLE_GGSEG) {
if (!requireNamespace("remotes", quietly = TRUE)) stop("缺少R包: remotes。请先安装后再启用ggseg流程。")
if (!requireNamespace("ggseg", quietly = TRUE) || !requireNamespace("ggsegSchaefer", quietly = TRUE)) {
cat("尝试从GitHub安装ggseg和ggsegSchaefer...\n")
tryCatch({
stop("当前版本禁止运行时在线安装ggseg依赖。请预先安装ggseg与ggsegSchaefer。")
library(ggseg)
library(ggsegSchaefer)
ggseg_available <- TRUE
cat("ggseg和ggsegSchaefer已成功加载。\n")
}, error = function(e) {
cat("警告: ggseg或ggsegSchaefer安装/加载失败。脑表面图将受到影响。\n错误: ", conditionMessage(e), "\n")
ggseg_available <- FALSE
})
} else {
library(ggseg)
library(ggsegSchaefer)
ggseg_available <- TRUE
cat("ggseg和ggsegSchaefer已可用并已加载。\n")
}
} else {
cat("ggseg绘图被禁用（ENABLE_GGSEG=FALSE），不影响当前脚本的数值结果与图。\n")
}


# --- 指标工具函数（最小新增，不改变既有流程） ---
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

# 额外：二分类F1与AUC置信区间工具
f1_from_cm_binary <- function(cm) {
  bc <- cm$byClass
  ppv <- suppressWarnings(as.numeric(bc["Pos Pred Value"]))
  sens <- suppressWarnings(as.numeric(bc["Sensitivity"]))
  if (is.na(ppv) || is.na(sens) || (ppv + sens) == 0) return(NA_real_)
  2 * ppv * sens / (ppv + sens)
}
auc_ci95_delong <- function(roc_obj) {
  ci <- suppressWarnings(pROC::ci.auc(roc_obj, conf.level = 0.95, method = "delong"))
  # pROC::ci.auc 返回向量: c(lower, AUC, upper)
  as.numeric(ci)
}

# --- Python环境 (reticulate) ---

use_python("/usr/local/fsl/bin/python", required = TRUE) # 请确保此路径是您环境中FSL Python的正确路径

np <- import('numpy')


# --- 并行计算设置 ---

cores_to_use <- parallel::detectCores() - 2

if (cores_to_use < 1) cores_to_use <- 1

cl <- makePSOCKcluster(cores_to_use)

registerDoParallel(cl)

cat("已注册并行后端，使用核心数:", cores_to_use, "\n")

# --- 通用绘图工具函数（不改变样式，仅去重复） ---
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

create_binary_roc_plot <- function(roc_obj, auc_val_numeric, line_color = "#1F78B4") {
pROC::ggroc(roc_obj, colour = line_color, size = 1.2) +
ggtitle(paste0("ROC (Test) AUC=", format(as.numeric(auc_val_numeric), digits = 3))) +
xlab("1 - Specificity") +
ylab("Sensitivity") +
geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
theme_classic(base_size = 12) +
theme(plot.title = element_text(hjust = 0.5, face = "bold"), axis.text = element_text(color = "black"), axis.title = element_text(color = "black"))
}



# --- 数据加载与预处理 ---

cat("正在为所有受试者 (HC, PD, MSA, RBD) 加载和准备数据...\n")

demo_all <- read_excel('demo_all.xlsx')

demo_all$group <- as.character(demo_all$group)

demo_all$group <- factor(demo_all$group, levels = c('HC', 'MSA', 'PD', 'RBD'))

cat("完整数据集中各组分布:\n"); print(table(demo_all$group))

if (!"ID" %in% colnames(demo_all)) {

warning("在 demo_all 中未找到 'ID' 列。正在生成序列ID。")

demo_all$ID <- paste0("Subject_", seq_len(nrow(demo_all)))

}


expected_rows_all <- nrow(demo_all)

# 改为使用 surface_ALL.npy（三维 400×N×5），仅提取 area 与 thickness 两通道，展开为 N×800
data_all_npy <- np$load('surface_ALL.npy')
arr_all <- as.array(data_all_npy)
if (length(dim(arr_all)) == 3) {
  # 1=volume, 2=area, 3=thickness
  area_mat  <- t(arr_all[ , , 2, drop = TRUE])  # N×400
  thick_mat <- t(arr_all[ , , 3, drop = TRUE])  # N×400
  stopifnot(nrow(area_mat) == expected_rows_all)
  struc_measures_cortical_all <- as.data.frame(cbind(area_mat, thick_mat))
} else {
  # 兼容旧二维数组（N×800）
  struc_measures_cortical_all <- as.data.frame(array(arr_all, dim = c(expected_rows_all, 800)))
}
rm(data_all_npy, arr_all)


annot <- read.csv("/media/neurox/T7_Shield/R_done/analysis/surfaces/parcellations/lut/lut_schaefer-400_mics.csv")

annot <- annot[-c(1,202), ]

annot$label <- gsub("7Networks_", "", annot$label)

roi_labels <- annot$label

struc_labels <- rep(c('area', 'thickness'), each = 400)

combined_labels_cortical <- paste(roi_labels, struc_labels, sep = "_")

colnames(struc_measures_cortical_all) <- make.unique(combined_labels_cortical)


asegstats_path <- "/media/neurox/T7_Shield/PD_analyse/asegstats_all.xlsx"

asegstats_measures_all <- read_excel(asegstats_path, col_names = TRUE)

stopifnot(nrow(asegstats_measures_all) == expected_rows_all)

colnames(asegstats_measures_all) <- paste0("subcort_", colnames(asegstats_measures_all))


combined_measures_all <- cbind(struc_measures_cortical_all, asegstats_measures_all)

rm(struc_measures_cortical_all, asegstats_measures_all)


if (any(duplicated(colnames(combined_measures_all)))) {

warning("合并皮层和皮层下数据后发现重复列名，并已使其唯一。")

colnames(combined_measures_all) <- make.unique(colnames(combined_measures_all))

}

if (sum(is.na(combined_measures_all)) > 0) {

warning(paste("共发现NA值:", sum(is.na(combined_measures_all)), "。将使用列中位数进行插补。"))

for(i in 1:ncol(combined_measures_all)){

if(is.numeric(combined_measures_all[,i])){

combined_measures_all[is.na(combined_measures_all[,i]), i] <- median(combined_measures_all[,i], na.rm = TRUE)

}

}

}

if (sum(is.na(combined_measures_all)) > 0) stop("插补后仍存在NA值。请检查。")


# ==========================================================================

# 第1部分: HC, PD, MSA 三分类诊断模型

# ==========================================================================

cat("\n第1部分: HC, PD, MSA 三分类诊断模型...\n")

output_subdir_HC_PD_MSA <- file.path(output_basedir, "Part1_HPDMSA_Diagnosis")

dir.create(output_subdir_HC_PD_MSA, showWarnings = FALSE, recursive = TRUE)


hc_pd_msa_indices <- which(demo_all$group %in% c('HC', 'PD', 'MSA'))

demo_hc_pd_msa <- demo_all[hc_pd_msa_indices, ]

demo_hc_pd_msa$group <- droplevels(demo_hc_pd_msa$group)


X_hc_pd_msa <- combined_measures_all[hc_pd_msa_indices, ]

Y_hc_pd_msa <- demo_hc_pd_msa$group


set.seed(1)

split_indices_hc_pd_msa <- createDataPartition(Y_hc_pd_msa, p = 0.8, list = FALSE)

X_train_hc_pd_msa <- X_hc_pd_msa[split_indices_hc_pd_msa, ]

Y_train_hc_pd_msa <- Y_hc_pd_msa[split_indices_hc_pd_msa]

X_test_hc_pd_msa <- X_hc_pd_msa[-split_indices_hc_pd_msa, ]

Y_test_hc_pd_msa <- Y_hc_pd_msa[-split_indices_hc_pd_msa]


train_control_diag <- trainControl(

method = "cv", number = 10, search = "grid", allowParallel = TRUE,

classProbs = TRUE, summaryFunction = multiClassSummary

)

model_hc_pd_msa <- train(

x = X_train_hc_pd_msa, y = Y_train_hc_pd_msa, metric = "Accuracy", method = "glmnet",

trControl = train_control_diag, tuneLength = 50, preProc = c("center", "scale")

)

cat("\n=== HC, PD, MSA 模型性能 ===\n")

print(model_hc_pd_msa$bestTune)

predictions_hc_pd_msa_test <- predict(model_hc_pd_msa, newdata = X_test_hc_pd_msa)

cm_hc_pd_msa <- confusionMatrix(predictions_hc_pd_msa_test, Y_test_hc_pd_msa)

print(cm_hc_pd_msa)


cm_table_hc_pd_msa <- as.data.frame(cm_hc_pd_msa$table)
plot_cm_hc_pd_msa <- create_confusion_matrix_plot(cm_table_hc_pd_msa, "Confusion Matrix - HC/PD/MSA (Test)")

ggsave(filename = file.path(output_subdir_HC_PD_MSA, "Confusion_Matrix_HPDMSA_model.png"), plot = plot_cm_hc_pd_msa, width = 7, height = 6, dpi = 300)

cat("HPDMSA模型混淆矩阵图已保存。\n")

# 追加保存训练/测试详细分类结果（与Part4风格一致）
try({
pred_detail_path <- file.path(output_subdir_HC_PD_MSA, "classification_details_HC_PD_MSA.csv")
pred_detail_diag <- data.frame(
  Dataset = rep(c("Train","Test"), c(nrow(X_train_hc_pd_msa), nrow(X_test_hc_pd_msa))),
  True = c(as.character(Y_train_hc_pd_msa), as.character(Y_test_hc_pd_msa)),
  Pred = c(as.character(predict(model_hc_pd_msa, X_train_hc_pd_msa)), as.character(predictions_hc_pd_msa_test))
)
write.csv(pred_detail_diag, pred_detail_path, row.names = FALSE)
}, silent = TRUE)

# 新增：Part1 扩展指标与训练集性能落盘（最小改动）
try({
  # 训练集预测与混淆矩阵
  predictions_hc_pd_msa_train <- predict(model_hc_pd_msa, newdata = X_train_hc_pd_msa)
  cm_hc_pd_msa_train <- confusionMatrix(predictions_hc_pd_msa_train, Y_train_hc_pd_msa)
  # 测试集 mAUC
  test_prob <- predict(model_hc_pd_msa, newdata = X_test_hc_pd_msa, type = "prob")
  suppressPackageStartupMessages(require(pROC))
  mauc_test <- as.numeric(pROC::auc(pROC::multiclass.roc(Y_test_hc_pd_msa, test_prob)))
  # 组装扩展指标（测试集）
  acc_ci <- acc_ci_from_cm(cm_hc_pd_msa)
  ext_test <- data.frame(
    Model = "HC_PD_MSA_TOP20",
    Dataset = "Test",
    Accuracy = unname(cm_hc_pd_msa$overall["Accuracy"]),
    Accuracy_CI95_lower = acc_ci["lower"],
    Accuracy_CI95_upper = acc_ci["upper"],
    Kappa = unname(cm_hc_pd_msa$overall["Kappa"]),
    BalancedAccuracy = balanced_acc(cm_hc_pd_msa),
    MacroF1 = macro_f1(cm_hc_pd_msa),
    MultiClass_AUC = mauc_test,
    stringsAsFactors = FALSE
  )
  # 训练集 mAUC
  train_prob <- predict(model_hc_pd_msa, newdata = X_train_hc_pd_msa, type = "prob")
  mauc_train <- as.numeric(pROC::auc(pROC::multiclass.roc(Y_train_hc_pd_msa, train_prob)))
  acc_ci_tr <- acc_ci_from_cm(cm_hc_pd_msa_train)
  ext_train <- data.frame(
    Model = "HC_PD_MSA_TOP20",
    Dataset = "Train",
    Accuracy = unname(cm_hc_pd_msa_train$overall["Accuracy"]),
    Accuracy_CI95_lower = acc_ci_tr["lower"],
    Accuracy_CI95_upper = acc_ci_tr["upper"],
    Kappa = unname(cm_hc_pd_msa_train$overall["Kappa"]),
    BalancedAccuracy = balanced_acc(cm_hc_pd_msa_train),
    MacroF1 = macro_f1(cm_hc_pd_msa_train),
    MultiClass_AUC = mauc_train,
    stringsAsFactors = FALSE
  )
  ext_out <- rbind(ext_train, ext_test)
  write.csv(ext_out, file.path(output_subdir_HC_PD_MSA, "performance_summary_HC_PD_MSA_TOP20_extended.csv"), row.names = FALSE)
  # 多分类逐类byClass落盘（测试、训练）
  byclass_test <- as.data.frame(cm_hc_pd_msa$byClass)
  byclass_test$Class <- rownames(cm_hc_pd_msa$byClass)
  byclass_test$Dataset <- "Test"
  byclass_train <- as.data.frame(cm_hc_pd_msa_train$byClass)
  byclass_train$Class <- rownames(cm_hc_pd_msa_train$byClass)
  byclass_train$Dataset <- "Train"
  byclass_out <- rbind(byclass_train, byclass_test)
  write.csv(byclass_out, file.path(output_subdir_HC_PD_MSA, "byclass_HC_PD_MSA_TOP20.csv"), row.names = FALSE)
}, silent = TRUE)

# === 新增[保存Part1划分索引与sessionInfo，不改变结果] ===
try({
    write.csv(data.frame(train_idx = as.integer(split_indices_hc_pd_msa)),
              file.path(output_subdir_HC_PD_MSA, "split_HC_PD_MSA_train_idx.csv"), row.names = FALSE)
    write.csv(data.frame(test_idx  = setdiff(seq_len(nrow(X_hc_pd_msa)), as.integer(split_indices_hc_pd_msa))),
              file.path(output_subdir_HC_PD_MSA, "split_HC_PD_MSA_test_idx.csv"), row.names = FALSE)
}, silent = TRUE)



# ==========================================================================

# 第2部分: 定义疾病特异性哨兵特征候选池并评估二分类模型性能

# ==========================================================================

cat("\n第2部分: 定义疾病特异性哨兵特征候选池并评估二分类模型性能...\n")

output_subdir_sentinel_definition <- file.path(output_basedir, "Part2_Disease_Sentinel_Definition")

dir.create(output_subdir_sentinel_definition, showWarnings = FALSE, recursive = TRUE)


# pROC包已在第0部分加载，此处无需重复加载检查

# library(pROC)


train_control_binary <- trainControl(

method = "cv", number = 10, search = "grid", allowParallel = TRUE,

classProbs = TRUE, summaryFunction = twoClassSummary, verboseIter = FALSE

)


# --- 2.1: PD特异性哨兵特征定义与模型评估: PD vs. (HC+MSA) ---

cat("\n--- 2.1: PD特异性哨兵特征定义与模型评估: PD vs. (HC+MSA) ---\n")


# --- 2.1.1: PD vs. (HC+MSA) 数据准备 ---

demo_PD_vs_Others <- demo_all %>%

filter(group %in% c("HC", "PD", "MSA")) %>%

mutate(group_binary = factor(ifelse(group == "PD", "Disease_PD", "Other_HCMSA"), levels = c("Other_HCMSA", "Disease_PD")))


hc_pd_msa_indices_part2 <- which(demo_all$group %in% c('HC', 'PD', 'MSA'))

stopifnot(nrow(demo_PD_vs_Others) == length(hc_pd_msa_indices_part2))


X_hc_pd_msa_features_for_binary <- combined_measures_all[hc_pd_msa_indices_part2, ]

Y_pd_vs_others_labels <- demo_PD_vs_Others$group_binary


# --- 2.1.2: PD vs. (HC+MSA) 模型评估的训练/测试集划分 ---

set.seed(201)

split_pd_others_eval <- createDataPartition(Y_pd_vs_others_labels, p = 0.8, list = FALSE)

X_train_pd_eval <- X_hc_pd_msa_features_for_binary[split_pd_others_eval, ]

Y_train_pd_eval <- Y_pd_vs_others_labels[split_pd_others_eval]

X_test_pd_eval <- X_hc_pd_msa_features_for_binary[-split_pd_others_eval, ]

Y_test_pd_eval <- Y_pd_vs_others_labels[-split_pd_others_eval]


cat("正在训练用于评估的 PD vs (HC+MSA) 模型 (基于训练集划分)...\n")

model_pd_vs_hcmsa_eval <- train(

x = X_train_pd_eval, y = Y_train_pd_eval,

metric = "ROC", method = "glmnet", trControl = train_control_binary,

tuneLength = 50, preProc = c("center", "scale")

)


# --- 2.1.3: PD vs. (HC+MSA) 模型在测试集上的性能评估 ---

cat("\n=== PD vs. (HC+MSA) 模型性能 (测试集) ===\n")

print(model_pd_vs_hcmsa_eval$bestTune)

predictions_pd_others_test_prob <- predict(model_pd_vs_hcmsa_eval, newdata = X_test_pd_eval, type = "prob")

predictions_pd_others_test_class <- predict(model_pd_vs_hcmsa_eval, newdata = X_test_pd_eval, type = "raw")


cm_pd_vs_hcmsa <- confusionMatrix(predictions_pd_others_test_class, Y_test_pd_eval, positive = "Disease_PD")

print(cm_pd_vs_hcmsa)


# 混淆矩阵绘图

cm_table_pd_vs_hcmsa <- as.data.frame(cm_pd_vs_hcmsa$table)
plot_cm_pd_vs_hcmsa_eval <- create_confusion_matrix_plot(cm_table_pd_vs_hcmsa, "Confusion Matrix - PD vs (HC+MSA) (Test)")

ggsave(filename = file.path(output_subdir_sentinel_definition, "Confusion_Matrix_PD_vs_HCMSA_eval.png"), plot = plot_cm_pd_vs_hcmsa_eval, width = 7, height = 6, dpi = 300)

cat("PD vs (HC+MSA) 模型评估的混淆矩阵图已保存。\n")


# ROC 曲线和 AUC (明确使用pROC::)

roc_pd_vs_hcmsa_obj <- pROC::roc(response = Y_test_pd_eval, predictor = predictions_pd_others_test_prob$Disease_PD, levels = c("Other_HCMSA", "Disease_PD"))

auc_pd_vs_hcmsa_val <- pROC::auc(roc_pd_vs_hcmsa_obj)

cat("PD vs (HC+MSA) 模型 AUC (测试集):", as.numeric(auc_pd_vs_hcmsa_val), "\n")


plot_roc_pd_vs_hcmsa <- create_binary_roc_plot(roc_pd_vs_hcmsa_obj, auc_pd_vs_hcmsa_val, line_color = "#33A02C")

ggsave(filename = file.path(output_subdir_sentinel_definition, "ROC_PD_vs_HCMSA_model_eval.png"), plot = plot_roc_pd_vs_hcmsa, width = 7, height = 6, dpi = 300)

cat("PD vs (HC+MSA) 模型评估的ROC曲线图已保存。\n")


performance_summary_pd <- data.frame(

Model = "PD_vs_HCMSA_eval",

Accuracy = cm_pd_vs_hcmsa$overall["Accuracy"],

Sensitivity = cm_pd_vs_hcmsa$byClass["Sensitivity"],

Specificity = cm_pd_vs_hcmsa$byClass["Specificity"],

AUC = as.numeric(auc_pd_vs_hcmsa_val),

Kappa = cm_pd_vs_hcmsa$overall["Kappa"],

stringsAsFactors = FALSE

)

rownames(performance_summary_pd) <- NULL

write.csv(performance_summary_pd, file.path(output_subdir_sentinel_definition, "performance_summary_PD_vs_HCMSA_eval.csv"), row.names = FALSE)

cat("PD vs (HC+MSA) 模型评估的性能总结已保存。\n")

# === 新增[保存Part2 PD划分索引，不改变结果] ===
try({
    write.csv(data.frame(train_idx = as.integer(split_pd_others_eval)),
              file.path(output_subdir_sentinel_definition, "split_PD_vs_HCMSA_train_idx.csv"), row.names = FALSE)
    write.csv(data.frame(test_idx  = setdiff(seq_len(nrow(X_hc_pd_msa_features_for_binary)), as.integer(split_pd_others_eval))),
              file.path(output_subdir_sentinel_definition, "split_PD_vs_HCMSA_test_idx.csv"), row.names = FALSE)
}, silent = TRUE)


# --- 2.1.4: PD哨兵特征定义 (使用所有PD, HC, MSA数据) ---

cat("正在基于所有相关数据 (HC, PD, MSA) 训练 PD vs (HC+MSA) 模型以定义PD哨兵特征...\n")

model_pd_vs_hcmsa_sentinel <- train(

x = X_train_pd_eval, y = Y_train_pd_eval,

metric = "ROC", method = "glmnet", trControl = train_control_binary,

tuneLength = 50, preProc = c("center", "scale")

)

coef_pd_specific_raw <- coef(model_pd_vs_hcmsa_sentinel$finalModel, s = model_pd_vs_hcmsa_sentinel$bestTune$lambda)

pd_sentinel_candidates_df <- data.frame(

Feature = rownames(coef_pd_specific_raw),

Coefficient = as.numeric(coef_pd_specific_raw[,1])

) %>%

filter(Feature != "(Intercept)") %>% # 保留所有非零系数特征，包括正负

filter(Coefficient != 0) %>% # 确保只选取有贡献的特征

arrange(desc(abs(Coefficient))) # 按系数绝对值排序


write.csv(pd_sentinel_candidates_df, file.path(output_subdir_sentinel_definition, "sentinels_PD_vs_HCMSA_abs_sorted.csv"), row.names = FALSE)

cat("PD-specific sentinel candidates (all non-zero, sorted by abs(coeff)) saved. Count:", nrow(pd_sentinel_candidates_df), "\n")

print(head(pd_sentinel_candidates_df, 20)) # 打印绝对值最大的前20个



# --- 2.2: MSA特异性哨兵特征定义与模型评估: MSA vs. (HC+PD) ---

cat("\n--- 2.2: MSA特异性哨兵特征定义与模型评估: MSA vs. (HC+PD) ---\n")


# --- 2.2.1: MSA vs. (HC+PD) 数据准备 ---

demo_MSA_vs_Others <- demo_all %>%

filter(group %in% c("HC", "PD", "MSA")) %>%

mutate(group_binary = factor(ifelse(group == "MSA", "Disease_MSA", "Other_HCPD"), levels = c("Other_HCPD", "Disease_MSA")))


Y_msa_vs_others_labels <- demo_MSA_vs_Others$group_binary

stopifnot(nrow(X_hc_pd_msa_features_for_binary) == length(Y_msa_vs_others_labels))


# --- 2.2.2: MSA vs. (HC+PD) 模型评估的训练/测试集划分 ---

set.seed(202)

split_msa_others_eval <- createDataPartition(Y_msa_vs_others_labels, p = 0.8, list = FALSE)

X_train_msa_eval <- X_hc_pd_msa_features_for_binary[split_msa_others_eval, ]

Y_train_msa_eval <- Y_msa_vs_others_labels[split_msa_others_eval]

X_test_msa_eval <- X_hc_pd_msa_features_for_binary[-split_msa_others_eval, ]

Y_test_msa_eval <- Y_msa_vs_others_labels[-split_msa_others_eval]


cat("正在训练用于评估的 MSA vs (HC+PD) 模型 (基于训练集划分)...\n")

model_msa_vs_hcpd_eval <- train(

x = X_train_msa_eval, y = Y_train_msa_eval,

metric = "ROC", method = "glmnet", trControl = train_control_binary,

tuneLength = 50, preProc = c("center", "scale")

)


# --- 2.2.3: MSA vs. (HC+PD) 模型在测试集上的性能评估 ---

cat("\n=== MSA vs. (HC+PD) 模型性能 (测试集) ===\n")

print(model_msa_vs_hcpd_eval$bestTune)

predictions_msa_others_test_prob <- predict(model_msa_vs_hcpd_eval, newdata = X_test_msa_eval, type = "prob")

predictions_msa_others_test_class <- predict(model_msa_vs_hcpd_eval, newdata = X_test_msa_eval, type = "raw")


cm_msa_vs_hcpd <- confusionMatrix(predictions_msa_others_test_class, Y_test_msa_eval, positive = "Disease_MSA")

print(cm_msa_vs_hcpd)


# 混淆矩阵绘图

cm_table_msa_vs_hcpd <- as.data.frame(cm_msa_vs_hcpd$table)
plot_cm_msa_vs_hcpd_eval <- create_confusion_matrix_plot(cm_table_msa_vs_hcpd, "Confusion Matrix - MSA vs (HC+PD) (Test)")

ggsave(filename = file.path(output_subdir_sentinel_definition, "Confusion_Matrix_MSA_vs_HCPD_eval.png"), plot = plot_cm_msa_vs_hcpd_eval, width = 7, height = 6, dpi = 300)

cat("MSA vs (HC+PD) 模型评估的混淆矩阵图已保存。\n")


# ROC 曲线和 AUC (明确使用pROC::)

roc_msa_vs_hcpd_obj <- pROC::roc(response = Y_test_msa_eval, predictor = predictions_msa_others_test_prob$Disease_MSA, levels = c("Other_HCPD", "Disease_MSA"))

auc_msa_vs_hcpd_val <- pROC::auc(roc_msa_vs_hcpd_obj)

cat("MSA vs (HC+PD) 模型 AUC (测试集):", as.numeric(auc_msa_vs_hcpd_val), "\n")


plot_roc_msa_vs_hcpd <- create_binary_roc_plot(roc_msa_vs_hcpd_obj, auc_msa_vs_hcpd_val, line_color = "#1F78B4")

ggsave(filename = file.path(output_subdir_sentinel_definition, "ROC_MSA_vs_HCPD_model_eval.png"), plot = plot_roc_msa_vs_hcpd, width = 7, height = 6, dpi = 300)

cat("MSA vs (HC+PD) 模型评估的ROC曲线图已保存。\n")


performance_summary_msa <- data.frame(

Model = "MSA_vs_HCPD_eval",

Accuracy = cm_msa_vs_hcpd$overall["Accuracy"],

Sensitivity = cm_msa_vs_hcpd$byClass["Sensitivity"],

Specificity = cm_msa_vs_hcpd$byClass["Specificity"],

AUC = as.numeric(auc_msa_vs_hcpd_val),

Kappa = cm_msa_vs_hcpd$overall["Kappa"],

stringsAsFactors = FALSE

)

rownames(performance_summary_msa) <- NULL

write.csv(performance_summary_msa, file.path(output_subdir_sentinel_definition, "performance_summary_MSA_vs_HCPD_eval.csv"), row.names = FALSE)

cat("MSA vs (HC+PD) 模型评估的性能总结已保存。\n")

# === 新增[保存Part2 MSA划分索引，不改变结果] ===
try({
    write.csv(data.frame(train_idx = as.integer(split_msa_others_eval)),
              file.path(output_subdir_sentinel_definition, "split_MSA_vs_HCPD_train_idx.csv"), row.names = FALSE)
    write.csv(data.frame(test_idx  = setdiff(seq_len(nrow(X_hc_pd_msa_features_for_binary)), as.integer(split_msa_others_eval))),
              file.path(output_subdir_sentinel_definition, "split_MSA_vs_HCPD_test_idx.csv"), row.names = FALSE)
}, silent = TRUE)


# --- 2.2.4: MSA哨兵特征定义 (使用所有PD, HC, MSA数据) ---

cat("正在基于所有相关数据 (HC, PD, MSA) 训练 MSA vs (HC+PD) 模型以定义MSA哨兵特征...\n")

model_msa_vs_hcpd_sentinel <- train(

x = X_train_msa_eval, y = Y_train_msa_eval,

metric = "ROC", method = "glmnet", trControl = train_control_binary,

tuneLength = 50, preProc = c("center", "scale")

)

coef_msa_specific_raw <- coef(model_msa_vs_hcpd_sentinel$finalModel, s = model_msa_vs_hcpd_sentinel$bestTune$lambda)

msa_sentinel_candidates_df <- data.frame(

Feature = rownames(coef_msa_specific_raw),

Coefficient = as.numeric(coef_msa_specific_raw[,1])

) %>%

filter(Feature != "(Intercept)") %>%

filter(Coefficient != 0) %>%

arrange(desc(abs(Coefficient)))


write.csv(msa_sentinel_candidates_df, file.path(output_subdir_sentinel_definition, "sentinels_MSA_vs_HCPD_abs_sorted.csv"), row.names = FALSE)

cat("MSA-specific sentinel candidates (all non-zero, sorted by abs(coeff)) saved. Count:", nrow(msa_sentinel_candidates_df), "\n")

print(head(msa_sentinel_candidates_df, 20))


# --- 合并所有二分类评估模型的性能总结 ---

all_binary_eval_performance <- bind_rows(performance_summary_pd, performance_summary_msa)

write.csv(all_binary_eval_performance, file.path(output_subdir_sentinel_definition, "performance_summary_ALL_BINARY_MODELS_EVAL.csv"), row.names = FALSE)

cat("所有二分类评估模型的性能总结已保存。\n")

# =================================================================================================
# --- Part 2.3 (新增部分): 验证Top-N哨兵特征模型的性能 ---
# 目标：评估仅使用Top 10, 15, 20个特征时，模型的分类性能，以量化这些核心特征的代表性。
# =================================================================================================
cat("\n\n--- Part 2.3: 验证Top-N哨兵特征模型的性能 ---\n")

# 定义一个【完全修正后】的可重用函数
validate_top_n_features <- function(n_features_list,       # 一个包含要测试的N值的向量，例如 c(10, 15, 20)
                                    sentinel_df,           # 包含所有哨兵特征及其系数的数据框 (已按abs(coeff)排序)
                                    full_X_train,          # 完整的训练集特征
                                    full_X_test,           # 完整的测试集特征
                                    Y_train,               # 训练集标签
                                    Y_test,                # 测试集标签
                                    positive_class,        # "阳性"类别名称，用于pROC和confusionMatrix
                                    levels_ordered,        # 因子水平的顺序，用于pROC
                                    model_name_prefix,     # 模型名称前缀，用于命名和标题 (例如 "PD_vs_HCMSA")
                                    output_dir) {          # 输出目录

    top_n_performance_summary <- data.frame()

    for (n in n_features_list) {
        cat(paste0("\n--- 正在处理 ", model_name_prefix, " 的 Top-", n, " 特征模型 ---\n"))

        # --- 步骤1: 获取Top-N特征名称 ---
        top_n_features_df <- head(sentinel_df, n)
        top_n_features <- top_n_features_df$Feature

        # ##### 新增部分 1: 打印Top-N特征列表及其在【全特征模型】中的原始系数 #####
        cat(paste("--- 以下为选定的Top-", n, "特征及其在原始全特征模型中的系数：---\n"))
        print(top_n_features_df)
        cat("---------------------------------------------------------------------\n")

        # --- 步骤2: 创建只包含Top-N特征的新数据集 ---
        X_train_top_n <- full_X_train[, top_n_features]
        X_test_top_n <- full_X_test[, top_n_features]

        # --- 步骤3: 在简化的训练集上训练新模型 ---
        cat("正在训练新模型...\n")
        set.seed(400 + n)
        model_top_n <- train(
            x = X_train_top_n,
            y = Y_train,
            metric = "ROC",
            method = "glmnet",
            trControl = train_control_binary,
            tuneLength = 50,
            preProc = c("center", "scale")
        )

        # --- 步骤4 & 5: 评估新模型并存储性能 ---
        cat("正在评估新模型...\n")
        predictions_top_n_prob <- predict(model_top_n, newdata = X_test_top_n, type = "prob")
        predictions_top_n_class <- predict(model_top_n, newdata = X_test_top_n, type = "raw")
        cm_top_n <- confusionMatrix(predictions_top_n_class, Y_test, positive = positive_class)
        roc_top_n_obj <- pROC::roc(response = Y_test, predictor = predictions_top_n_prob[[positive_class]], levels = levels_ordered)
        auc_top_n_val <- pROC::auc(roc_top_n_obj)
        # 计算扩展指标（BA、F1、AUC CI）
        ba_top_n <- (as.numeric(cm_top_n$byClass["Sensitivity"]) + as.numeric(cm_top_n$byClass["Specificity"])) / 2
        f1_top_n <- f1_from_cm_binary(cm_top_n)
        ci_auc <- auc_ci95_delong(roc_top_n_obj)
        cat(paste0("Top-", n, " 模型性能 (测试集):\n"))
        print(cm_top_n$overall[c("Accuracy", "Kappa")])
        cat(paste0("AUC: ", as.numeric(auc_top_n_val), "\n"))
        current_performance <- data.frame(
            Model = paste0(model_name_prefix, "_Top", n),
            Accuracy = cm_top_n$overall["Accuracy"],
            Sensitivity = cm_top_n$byClass["Sensitivity"],
            Specificity = cm_top_n$byClass["Specificity"],
            AUC = as.numeric(auc_top_n_val), AUC_CI95_lower = ci_auc[1], AUC_CI95_upper = ci_auc[3],
            F1 = f1_top_n, BalancedAccuracy = ba_top_n,
            Kappa = cm_top_n$overall["Kappa"],
            stringsAsFactors = FALSE
        )
        rownames(current_performance) <- NULL
        top_n_performance_summary <- rbind(top_n_performance_summary, current_performance)

        # --- 步骤6: 生成与原始风格完全一致的图表 ---
        # 混淆矩阵图
        cm_table_top_n <- as.data.frame(cm_top_n$table)
        colnames(cm_table_top_n) <- c("Prediction", "Reference", "Freq")
        plot_cm_top_n <- ggplot(data = cm_table_top_n, aes(x = Prediction, y = Reference, fill = Freq)) +
            geom_tile(color = "black") + geom_text(aes(label = Freq), color = "magenta", size = 5) +
            scale_fill_gradient(low = "white", high = "navyblue") +
            labs(title = paste("Confusion Matrix -", model_name_prefix, "Top-", n, "(Test)"), x = "Predicted", y = "True") +
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
        ggsave(filename = file.path(output_dir, paste0("Confusion_Matrix_", model_name_prefix, "_Top", n, ".png")), plot = plot_cm_top_n, width = 7, height = 6, dpi = 300)

        # ROC曲线图（配色与主文一致：PD=绿色，MSA=蓝色）
        roc_color <- if (grepl("PD_vs_HCMSA", model_name_prefix)) "#33A02C" else "#1F78B4"
        plot_roc_top_n <- create_binary_roc_plot(roc_top_n_obj, auc_top_n_val, line_color = roc_color)
        ggsave(filename = file.path(output_dir, paste0("ROC_", model_name_prefix, "_Top", n, ".png")), plot = plot_roc_top_n, width = 7, height = 6, dpi = 300)

        # ##### 贡献度改为来自"全特征模型"的标准化系数（与Top-N的定义口径一致）#####
        cat(paste0("--- 计算Top-", n, "特征在全特征模型中的系数重要性... ---\n"))
        var_imp_df <- top_n_features_df %>%
            dplyr::transmute(
                Feature = Feature,
                Overall = 100 * abs(Coefficient) / max(abs(Coefficient))
            ) %>%
            dplyr::arrange(dplyr::desc(Overall))
        cat(paste0("Top-", n, "特征的相对重要性 (0-100):\n"))
        print(var_imp_df)
        # 基于全特征模型排序，计算"累计AUC"曲线（前k个特征逐步加入）
        ordered_features <- as.character(var_imp_df$Feature)
        auc_curve <- numeric(length(ordered_features))
        for (rk in seq_along(ordered_features)) {
            feats_k <- ordered_features[seq_len(rk)]
            Xtr_k <- full_X_train[, feats_k, drop = FALSE]
            Xte_k <- full_X_test[, feats_k, drop = FALSE]
            set.seed(700 + rk + n)
            if (ncol(Xtr_k) <= 1) {
                mdl_k <- train(
                    x = Xtr_k,
                    y = Y_train,
                    metric = "ROC",
                    method = "glm",
                    trControl = train_control_binary,
                    preProc = c("center", "scale")
                )
            } else {
                mdl_k <- train(
                    x = Xtr_k,
                    y = Y_train,
                    metric = "ROC",
                    method = "glmnet",
                    trControl = train_control_binary,
                    tuneLength = 50,
                    preProc = c("center", "scale")
                )
            }
            pr_k <- predict(mdl_k, newdata = Xte_k, type = "prob")
            auc_curve[rk] <- as.numeric(pROC::auc(pROC::roc(response = Y_test, predictor = pr_k[[positive_class]], levels = levels_ordered)))
        }
        df_auc <- data.frame(
            Feature = ordered_features,
            Rank = seq_along(ordered_features),
            Importance01 = var_imp_df$Overall / 100,
            AUC = auc_curve,
            stringsAsFactors = FALSE
        )
        # 仅使用单一暖色低饱和配色方案（与脑图风格一致）
        palette_schemes <- c("B")
        for (scheme in palette_schemes) {
            if (scheme == "A") {
                # 冷色单序列（蓝系，期刊稳妥）
                base_cols <- c("#E7ECF7", "#CFD9EE", "#B7C6E3", "#99AED3", "#798CBF")
            } else if (scheme == "B") {
                # 暖色单序列（橙棕系，低饱和）
                base_cols <- c("#F6E4DB", "#EED3C7", "#E6C2B1", "#D9A994", "#C9876F")
            } else if (scheme == "C") {
                # 近似色带（蓝青窄带，不跨到暖色）
                base_cols <- c("#BFD4F8", "#A8CFE3", "#94C8D5", "#86BECA", "#79B4BF")
            } else {
                # 中性基调+轻强调：同色更深（用于强调Top3时也适配单色）
                base_cols <- c("#E3E9F3", "#CFD9EE", "#BAC8E1", "#A6B8D5", "#8EA6C7")
            }
            pal_n <- grDevices::colorRampPalette(base_cols)(nrow(var_imp_df))
            # 设定因子水平为按重要度升序（顶端为最高重要度）
            levels_asc <- var_imp_df$Feature[order(var_imp_df$Overall)]
            var_imp_df$Feature <- factor(var_imp_df$Feature, levels = levels_asc)
            # 颜色映射：越重要越深（与levels_asc一致：最浅→最深）
            color_map <- setNames(pal_n, levels_asc)
            plot_var_imp <- ggplot(var_imp_df, aes(x = Feature, y = Overall, fill = Feature)) +
                geom_col(width = 0.7, color = NA) +
                coord_flip() +
                scale_fill_manual(values = color_map, guide = "none") +
                labs(
                    title = paste0(model_name_prefix, " Top-", n, " Feature Importance"),
                    x = "Feature (ROI)",
                    y = "Relative Importance (0–100)"
                ) +
                theme_classic(base_size = 12) +
                theme(
                    plot.title = element_text(hjust = 0.5, face = "bold"),
                    axis.text = element_text(color = "black"),
                    axis.title = element_text(color = "black"),
                    panel.grid.major.x = element_line(color = "grey85", linewidth = 0.3),
                    panel.grid.minor = element_blank(),
                    panel.grid.major.y = element_blank()
                )
            out_file <- file.path(output_dir, paste0("VarImp_", model_name_prefix, "_Top", n, ".png"))
            ggsave(filename = out_file, plot = plot_var_imp, width = 8, height = max(5, n * 0.25), dpi = 300)
            cat(paste0("Top-", n, "模型的特征重要性条形图已保存 (方案", scheme, ")。\n"))

            # 附加图1：棒棒糖图（凸显排序，顶刊常见）
            lolli <- ggplot(var_imp_df, aes(y = Feature, x = Overall)) +
                geom_segment(aes(x = 0, xend = Overall, y = Feature, yend = Feature), color = tail(pal_n, 1), linewidth = 0.7) +
                geom_point(color = tail(pal_n, 1), size = 3.0) +
                labs(title = paste0(model_name_prefix, " Top-", n, " Feature Importance (Lollipop)"), x = "Relative Importance (0–100)", y = "Feature (ROI)") +
                theme_classic(base_size = 12) +
                theme(plot.title = element_text(hjust = 0.5, face = "bold"), axis.text = element_text(color = "black"), axis.title = element_text(color = "black"))
            ggsave(filename = file.path(output_dir, paste0("VarImp_Lollipop_", model_name_prefix, "_Top", n, ".png")), plot = lolli, width = 8, height = max(5, n * 0.25), dpi = 300)

            # 附加图2：贡献度+累计AUC 双轴图（按需已移除）
        }
        cat("---------------------------------------------------------------------\n")

    } # <--- for循环的正确结束位置

    cat(paste0("\n", model_name_prefix, " 的Top-N模型验证完成。\n"))
    return(top_n_performance_summary)
}

# --- 为PD模型执行Top-N验证 ---
pd_top_n_results <- validate_top_n_features(
    n_features_list = c(10, 15, 20),
    sentinel_df = pd_sentinel_candidates_df,
    full_X_train = X_train_pd_eval,
    full_X_test = X_test_pd_eval,
    Y_train = Y_train_pd_eval,
    Y_test = Y_test_pd_eval,
    positive_class = "Disease_PD",
    levels_ordered = c("Other_HCMSA", "Disease_PD"),
    model_name_prefix = "PD_vs_HCMSA",
    output_dir = output_subdir_sentinel_definition
)

# --- 为MSA模型执行Top-N验证 ---
msa_top_n_results <- validate_top_n_features(
    n_features_list = c(10, 15, 20),
    sentinel_df = msa_sentinel_candidates_df,
    full_X_train = X_train_msa_eval,
    full_X_test = X_test_msa_eval,
    Y_train = Y_train_msa_eval,
    Y_test = Y_test_msa_eval,
    positive_class = "Disease_MSA",
    levels_ordered = c("Other_HCPD", "Disease_MSA"),
    model_name_prefix = "MSA_vs_HCPD",
    output_dir = output_subdir_sentinel_definition
)


# --- 整合所有二分类评估模型的性能，包括全特征模型和Top-N模型 ---
# 从 all_binary_eval_performance 中提取全特征模型的结果
full_model_performance <- all_binary_eval_performance

# 合并所有结果
final_performance_comparison <- bind_rows(
    full_model_performance,
    pd_top_n_results,
    msa_top_n_results
)

# 打印最终的性能对比表
cat("\n\n=== 所有二分类模型性能对比 (测试集) ===\n")
print(final_performance_comparison, row.names = FALSE)

# 保存最终的性能对比表
write.csv(final_performance_comparison, file.path(output_subdir_sentinel_definition, "performance_summary_ALL_BINARY_MODELS_COMPARISON.csv"), row.names = FALSE)
try({
  # 生成扩展版：补充F1、BA、AUC CI（如上已在TopN内生成；对eval两行单独计算）
  # 读取评估表以扩展（确保已有）
  eval_df <- read.csv(file.path(output_subdir_sentinel_definition, "performance_summary_ALL_BINARY_MODELS_EVAL.csv"))
  # 对eval_df无法直接有F1/CI，这里保持原样附加NA列，保证列齐全
  eval_df$AUC_CI95_lower <- NA_real_
  eval_df$AUC_CI95_upper <- NA_real_
  eval_df$F1 <- NA_real_
  eval_df$BalancedAccuracy <- (eval_df$Sensitivity + eval_df$Specificity)/2
  # 合并TopN（已含F1/BA/CI），与eval扩展
  comp_ext <- bind_rows(eval_df, pd_top_n_results, msa_top_n_results)
  write.csv(comp_ext, file.path(output_subdir_sentinel_definition, "performance_summary_ALL_BINARY_MODELS_COMPARISON_extended.csv"), row.names = FALSE)
}, silent = TRUE)
cat("\n所有二分类模型的最终性能对比表已保存。\n")

# =================================================================================================
# --- Part 2.3 (新增部分) 结束 ---
# =================================================================================================

# --- 清理并行计算环境 ---

if(exists("cl") && inherits(cl, "cluster")) { # 检查cl是否存在且为集群对象

stopCluster(cl)

cat("已停止并行计算集群。\n")

} else {

cat("并行计算集群未运行或已停止。\n")

}


# ==========================================================================

# 第3部分 (新): 基于疾病特异性哨兵对RBD进行亚型分析和可视化

# ==========================================================================

cat("\n第3部分: 基于疾病特征谱对RBD进行亚型分析和高级可视化...\n")

output_subdir_RBD_subtyping <- file.path(output_basedir, "Part3_RBD_Subtyping_AdvancedViz")

dir.create(output_subdir_RBD_subtyping, showWarnings = FALSE, recursive = TRUE)


# --- 3.1: 选取核心哨兵特征 ---

cat("根据绝对值系数选取核心哨兵特征用于RBD分析...\n")


# 从已按绝对值排序的哨兵候选列表中选取Top N (例如Top 10或15)

# 确保pd_sentinel_candidates_df 和 msa_sentinel_candidates_df 已在PART 2中正确生成和排序

num_core_sentinels_per_group <- 10 # 每组选10个，总数可能少于20（如果重叠）

pd_core_sentinels_abs <- head(pd_sentinel_candidates_df, num_core_sentinels_per_group)

msa_core_sentinels_abs <- head(msa_sentinel_candidates_df, num_core_sentinels_per_group)


cat(paste0("Top ", num_core_sentinels_per_group, " PD核心哨兵 (基于abs(系数)):\n")); print(pd_core_sentinels_abs)

cat(paste0("Top ", num_core_sentinels_per_group, " MSA核心哨兵 (基于abs(系数)):\n")); print(msa_core_sentinels_abs)


# 合并所有核心哨兵特征名，并确保唯一

# unique_core_sentinel_names 现在将包含那些系数为正和为负的最重要特征

unique_core_sentinel_names <- unique(c(pd_core_sentinels_abs$Feature, msa_core_sentinels_abs$Feature))
# TOP20：不截断，直接使用并集
top20_features <- unique_core_sentinel_names[seq_len(min(20, length(unique_core_sentinel_names)))]
# 将TOP20作为统一主特征集落盘
out_def_dir <- file.path(output_basedir, "Part2_Disease_Sentinel_Definition")
dir.create(out_def_dir, showWarnings = FALSE, recursive = TRUE)
write.csv(data.frame(Feature = top20_features), file.path(out_def_dir, "top_features_master_TOP20.csv"), row.names = FALSE)
# 为兼容后续变量名，令top15_features等同于TOP20
top15_features <- top20_features

cat("用于RBD分析的独立核心哨兵特征数量:", length(unique_core_sentinel_names), "\n")

if(length(unique_core_sentinel_names) == 0) {

stop("未能选取任何核心哨兵特征。后续RBD亚型分析无法进行。请检查PART 2的哨兵定义步骤。")

}

# print(unique_core_sentinel_names)


# --- 3.2: 为所有受试者 (包括RBD) 计算"PD疾病签名得分"和"MSA疾病签名得分" ---

cat("为所有受试者计算PD签名得分和MSA签名得分...\n")


# 提取用于计算PD签名的核心哨兵及其【原始】系数 (来自PD vs HC+MSA模型)

pd_sentinel_coefs_for_scoring <- pd_sentinel_candidates_df %>%

filter(Feature %in% unique_core_sentinel_names) %>%

select(Feature, Coefficient_PD = Coefficient)


# 提取用于计算MSA签名的核心哨兵及其【原始】系数 (来自MSA vs HC+PD模型)

msa_sentinel_coefs_for_scoring <- msa_sentinel_candidates_df %>%

filter(Feature %in% unique_core_sentinel_names) %>%

select(Feature, Coefficient_MSA = Coefficient)


# 准备所有受试者的完整特征数据 (combined_measures_all)

# 并对其进行与各自模型训练时相同的标准化处理

all_subjects_features_scaled_for_pd_model <- predict(model_pd_vs_hcmsa_sentinel$preProcess, combined_measures_all)

all_subjects_features_scaled_for_msa_model <- predict(model_msa_vs_hcpd_sentinel$preProcess, combined_measures_all)


pd_signature_scores_all_subjects <- rep(0, nrow(combined_measures_all))

msa_signature_scores_all_subjects <- rep(0, nrow(combined_measures_all))


# 计算PD签名得分 (标准化特征值 * 原始系数)

for(i in 1:nrow(pd_sentinel_coefs_for_scoring)){

feature_name <- pd_sentinel_coefs_for_scoring$Feature[i]

coef_val <- pd_sentinel_coefs_for_scoring$Coefficient_PD[i]

if(feature_name %in% colnames(all_subjects_features_scaled_for_pd_model)){

pd_signature_scores_all_subjects <- pd_signature_scores_all_subjects + (all_subjects_features_scaled_for_pd_model[, feature_name] * coef_val)

}

}

# 计算MSA签名得分

for(i in 1:nrow(msa_sentinel_coefs_for_scoring)){

feature_name <- msa_sentinel_coefs_for_scoring$Feature[i]

coef_val <- msa_sentinel_coefs_for_scoring$Coefficient_MSA[i]

if(feature_name %in% colnames(all_subjects_features_scaled_for_msa_model)){

msa_signature_scores_all_subjects <- msa_signature_scores_all_subjects + (all_subjects_features_scaled_for_msa_model[, feature_name] * coef_val)

}

}


# 将签名得分添加到demo_all数据框中

demo_all$PD_Signature_Score <- pd_signature_scores_all_subjects

demo_all$MSA_Signature_Score <- msa_signature_scores_all_subjects


# --- 3.3: 定义RBD亚型 (基于PD和MSA签名得分) & 可视化RBD患者得分 ---

cat("基于签名得分为RBD患者定义亚型...\n")

rbd_scores_df <- demo_all %>%

filter(group == "RBD") %>%

select(ID, PD_Signature_Score, MSA_Signature_Score)


# 示例分层：基于双维度得分的象限或聚类

# 这里我们用一个更灵活的方法：基于PD得分和MSA得分相对于RBD队列自身的分布来定义亚型

# 例如，可以使用分位数（如中位数、四分位数）或K-均值聚类

# 为了简化和演示，我们继续使用基于中位数的四象限法，但建议您探索更优的分型方法

pd_score_median_rbd <- median(rbd_scores_df$PD_Signature_Score, na.rm=TRUE)

msa_score_median_rbd <- median(rbd_scores_df$MSA_Signature_Score, na.rm=TRUE)


rbd_scores_df$Subtype <- case_when(

rbd_scores_df$PD_Signature_Score > pd_score_median_rbd & rbd_scores_df$MSA_Signature_Score <= msa_score_median_rbd ~ "RBD-PDsig_High", # PD签名高，MSA签名不高

rbd_scores_df$MSA_Signature_Score > msa_score_median_rbd & rbd_scores_df$PD_Signature_Score <= pd_score_median_rbd ~ "RBD-MSAsig_High",# MSA签名高，PD签名不高

rbd_scores_df$PD_Signature_Score > pd_score_median_rbd & rbd_scores_df$MSA_Signature_Score > msa_score_median_rbd ~ "RBD-MixedSig_High", # 两者都高

TRUE ~ "RBD-LowSig_Profile" # 两者都不高（或低）

)

rbd_subtype_levels <- c("RBD-PDsig_High", "RBD-MSAsig_High", "RBD-MixedSig_High", "RBD-LowSig_Profile")

rbd_scores_df$Subtype <- factor(rbd_scores_df$Subtype, levels = rbd_subtype_levels)


cat("RBD亚型分布 (基于签名得分中位数划分):\n"); print(table(rbd_scores_df$Subtype))

write.csv(rbd_scores_df, file.path(output_subdir_RBD_subtyping, "RBD_Subtypes_from_Signature_Scores.csv"), row.names = FALSE)

    # --- Prepare profile_data for heatmap (Top-15, Z relative to HC) ---
    hc_mask <- demo_all$group == "HC"
    top_feats <- top20_features
    hc_mean <- apply(combined_measures_all[hc_mask, top_feats, drop = FALSE], 2, mean, na.rm = TRUE)
    hc_sd   <- apply(combined_measures_all[hc_mask, top_feats, drop = FALSE], 2, sd,   na.rm = TRUE)
    hc_sd[hc_sd == 0 | is.na(hc_sd)] <- 1e-6
    z_mat <- sweep(combined_measures_all[, top_feats, drop = FALSE], 2, hc_mean, FUN = "-")
    z_mat <- sweep(z_mat, 2, hc_sd,   FUN = "/")

    display_group <- as.character(demo_all$group)
    display_group[display_group == "HC"]  <- "HC (Ref)"
    display_group[display_group == "PD"]  <- "PD (Ref)"
    display_group[display_group == "MSA"] <- "MSA (Ref)"
    # map RBD to subtypes
    rbd_map <- rbd_scores_df[, c("ID", "Subtype")] ; colnames(rbd_map) <- c("ID", "Subtype")
    demo_tmp <- demo_all[, c("ID", "group")]
    demo_tmp <- merge(demo_tmp, rbd_map, by = "ID", all.x = TRUE)
    idx_rbd <- which(demo_all$group == "RBD")
    display_group[idx_rbd] <- as.character(demo_tmp$Subtype[idx_rbd])

    z_df <- as.data.frame(z_mat)
    z_df$Group <- display_group
    z_long <- z_df %>%
        tidyr::pivot_longer(cols = all_of(top_feats), names_to = "Feature", values_to = "Z_Score")
    profile_data <- z_long %>%
        dplyr::group_by(Feature, Group) %>%
        dplyr::summarise(Z_Score = mean(Z_Score, na.rm = TRUE), .groups = "drop")
    max_val_radar <- max(abs(profile_data$Z_Score), na.rm = TRUE)

   # --- Heatmap ---
    cat("Generating heatmap...\n")
    heatmap_matrix <- profile_data %>%
        dplyr::filter(Feature %in% top20_features) %>%
        pivot_wider(names_from = Group, values_from = Z_Score) %>%
        column_to_rownames(var = "Feature") %>% as.matrix()

    if (nrow(heatmap_matrix) > 0 && ncol(heatmap_matrix) > 0) {
        desired_col_order <- c("HC (Ref)", "PD (Ref)", "MSA (Ref)",
                               "RBD-LowSig_Profile", "RBD-PDsig_High", "RBD-MSAsig_High", "RBD-MixedSig_High")
        actual_col_order <- desired_col_order[desired_col_order %in% colnames(heatmap_matrix)]
        heatmap_matrix_ordered <- heatmap_matrix[, actual_col_order, drop=FALSE]

        heatmap_breaks <- seq(-max_val_radar, max_val_radar, length.out=101)
        color_palette_heatmap <- colorRampPalette(rev(brewer.pal(n = 7, name = "RdBu")))(100)

        pheatmap(heatmap_matrix_ordered,
                 color = color_palette_heatmap, breaks = heatmap_breaks,
                 cluster_rows = TRUE, cluster_cols = FALSE,
                 fontsize_row = 8, fontsize_col = 10,
                 main = "Heatmap of imaging profiles across core sentinel features\n(Z-scores relative to HC)",
                 border_color = "grey80", angle_col = 45)

        # Save to file using pheatmap's filename argument
        out_heatmap <- file.path(output_subdir_RBD_subtyping, "Heatmap_RBD_subtype_profiles.png")
        pheatmap(heatmap_matrix_ordered,
                 color = color_palette_heatmap, breaks = heatmap_breaks,
                 cluster_rows = TRUE, cluster_cols = FALSE,
                 fontsize_row = 8, fontsize_col = 10,
                 main = "Heatmap of imaging profiles across core sentinel features\n(Z-scores relative to HC)",
                 border_color = "grey80", angle_col = 45,
                 filename = out_heatmap,
                 width = 10, height = max(8, nrow(heatmap_matrix_ordered) * 0.25))
        cat("Heatmap saved to: ", out_heatmap, "\n", sep = "")
    } else {cat("Heatmap data preparation failed.\n")}


# ==========================================================================
# 第4部分: 基于MSA亚型的扩展分析
# ==========================================================================

cat("\n第4部分: 基于MSA亚型的扩展分析...\n")

output_subdir_MSA_subtype <- file.path(output_basedir, "Part4_MSA_Subtype_Analysis")
dir.create(output_subdir_MSA_subtype, showWarnings = FALSE, recursive = TRUE)

# 新增：为第4部分重新初始化并行后端（避免之前stopCluster影响）
try({
    cores_to_use_part4 <- parallel::detectCores() - 2
    if (cores_to_use_part4 < 1) cores_to_use_part4 <- 1
    cl_part4 <- makePSOCKcluster(cores_to_use_part4)
    registerDoParallel(cl_part4)
    cat("(Part4) 已重新注册并行后端，使用核心数:", cores_to_use_part4, "\n")
}, silent = TRUE)

# --- 4.1: 重新读取包含MSA_group信息的demo数据 ---
cat("正在重新读取包含MSA亚型信息的demo数据...\n")

demo_all_with_msa_subtype <- read_excel('demo_all.xlsx')
demo_all_with_msa_subtype$group <- as.character(demo_all_with_msa_subtype$group)
demo_all_with_msa_subtype$group <- factor(demo_all_with_msa_subtype$group, levels = c('HC', 'MSA', 'PD', 'RBD'))

# 检查数据一致性
if(nrow(demo_all_with_msa_subtype) != nrow(demo_all)) {
    stop("新读取的数据行数与原始数据不一致！")
}

cat("完整数据集中各组分布:\n"); print(table(demo_all_with_msa_subtype$group))
cat("MSA亚组分布:\n"); print(table(demo_all_with_msa_subtype$MSA_group, useNA='ifany'))

# --- 4.2: 使用PD和MSA的TOP20特征进行HC-PD-MSA三分类模型分析 ---
cat("\n--- 4.2: 使用TOP20哨兵特征进行HC-PD-MSA三分类模型分析 ---\n")

# 确保我们有TOP20特征
if(!exists("top20_features") || length(top20_features) == 0) {
    stop("TOP20特征未定义，请确保第3部分已正确执行")
}

cat("使用的TOP20特征:\n")
print(top20_features)

# 准备HC, PD, MSA数据（使用TOP20特征）
hc_pd_msa_indices_top15 <- which(demo_all_with_msa_subtype$group %in% c('HC', 'PD', 'MSA'))
demo_hc_pd_msa_top15 <- demo_all_with_msa_subtype[hc_pd_msa_indices_top15, ]
demo_hc_pd_msa_top15$group <- droplevels(demo_hc_pd_msa_top15$group)

X_hc_pd_msa_top15 <- combined_measures_all[hc_pd_msa_indices_top15, top20_features]
Y_hc_pd_msa_top15 <- demo_hc_pd_msa_top15$group

cat("TOP20特征三分类数据集各组分布:\n"); print(table(Y_hc_pd_msa_top15))

# 训练/测试集划分
set.seed(401)
split_indices_hc_pd_msa_top15 <- createDataPartition(Y_hc_pd_msa_top15, p = 0.8, list = FALSE)
X_train_hc_pd_msa_top15 <- X_hc_pd_msa_top15[split_indices_hc_pd_msa_top15, ]
Y_train_hc_pd_msa_top15 <- Y_hc_pd_msa_top15[split_indices_hc_pd_msa_top15]
X_test_hc_pd_msa_top15 <- X_hc_pd_msa_top15[-split_indices_hc_pd_msa_top15, ]
Y_test_hc_pd_msa_top15 <- Y_hc_pd_msa_top15[-split_indices_hc_pd_msa_top15]

# 训练TOP20特征三分类模型
cat("正在训练基于TOP20特征的HC-PD-MSA三分类模型...\n")
model_hc_pd_msa_top15 <- train(
    x = X_train_hc_pd_msa_top15, y = Y_train_hc_pd_msa_top15, 
    metric = "Accuracy", method = "glmnet",
    trControl = train_control_diag, tuneLength = 50, 
    preProc = c("center", "scale")
)

cat("\n=== HC, PD, MSA (TOP20特征) 模型性能 ===\n")
print(model_hc_pd_msa_top15$bestTune)

predictions_hc_pd_msa_top15_test <- predict(model_hc_pd_msa_top15, newdata = X_test_hc_pd_msa_top15)
cm_hc_pd_msa_top15 <- confusionMatrix(predictions_hc_pd_msa_top15_test, Y_test_hc_pd_msa_top15)
print(cm_hc_pd_msa_top15)

# 混淆矩阵图
cm_table_hc_pd_msa_top15 <- as.data.frame(cm_hc_pd_msa_top15$table)
plot_cm_hc_pd_msa_top15 <- create_confusion_matrix_plot(cm_table_hc_pd_msa_top15, "Confusion Matrix - HC/PD/MSA (TOP20 Features)")

ggsave(filename = file.path(output_subdir_MSA_subtype, "Confusion_Matrix_HPDMSA_TOP20.png"), 
       plot = plot_cm_hc_pd_msa_top15, width = 7, height = 6, dpi = 300)

cat("HC-PD-MSA (TOP20特征) 模型混淆矩阵图已保存。\n")

# 保存性能总结
performance_summary_hc_pd_msa_top15 <- data.frame(
    Model = "HC_PD_MSA_TOP20",
    Accuracy = cm_hc_pd_msa_top15$overall["Accuracy"],
    Kappa = cm_hc_pd_msa_top15$overall["Kappa"],
    stringsAsFactors = FALSE
)
rownames(performance_summary_hc_pd_msa_top15) <- NULL

# --- 4.3: HC-PD-MSA-P三分类模型分析 ---
cat("\n--- 4.3: HC-PD-MSA-P三分类模型分析 ---\n")

# 4.3.1: 准备HC-PD-MSA-P数据
# 筛选HC, PD, 和MSA-P患者
demo_hc_pd_msap <- demo_all_with_msa_subtype %>%
    filter((group == "HC") | (group == "PD") | (group == "MSA" & MSA_group == "MSA-P")) %>%
    mutate(group_new = case_when(
        group == "HC" ~ "HC",
        group == "PD" ~ "PD", 
        group == "MSA" & MSA_group == "MSA-P" ~ "MSA_P"
    )) %>%
    mutate(group_new = factor(group_new, levels = c("HC", "PD", "MSA_P")))

cat("HC-PD-MSA-P数据集各组分布:\n"); print(table(demo_hc_pd_msap$group_new))

# 获取对应的特征数据索引
hc_pd_msap_indices <- which(
    (demo_all_with_msa_subtype$group == "HC") | 
    (demo_all_with_msa_subtype$group == "PD") | 
    (demo_all_with_msa_subtype$group == "MSA" & demo_all_with_msa_subtype$MSA_group == "MSA-P")
)

# 4.3.2: 使用全部特征的HC-PD-MSA-P模型
cat("\n使用全部特征进行HC-PD-MSA-P三分类分析...\n")

X_hc_pd_msap_full <- combined_measures_all[hc_pd_msap_indices, ]
Y_hc_pd_msap <- demo_hc_pd_msap$group_new

# 训练/测试集划分
set.seed(402)
split_indices_hc_pd_msap_full <- createDataPartition(Y_hc_pd_msap, p = 0.8, list = FALSE)
X_train_hc_pd_msap_full <- X_hc_pd_msap_full[split_indices_hc_pd_msap_full, ]
Y_train_hc_pd_msap_full <- Y_hc_pd_msap[split_indices_hc_pd_msap_full]
X_test_hc_pd_msap_full <- X_hc_pd_msap_full[-split_indices_hc_pd_msap_full, ]
Y_test_hc_pd_msap_full <- Y_hc_pd_msap[-split_indices_hc_pd_msap_full]

# 训练全特征模型
model_hc_pd_msap_full <- train(
    x = X_train_hc_pd_msap_full, y = Y_train_hc_pd_msap_full, 
    metric = "Accuracy", method = "glmnet",
    trControl = train_control_diag, tuneLength = 50, 
    preProc = c("center", "scale")
)

cat("\n=== HC, PD, MSA-P (全特征) 模型性能 ===\n")
print(model_hc_pd_msap_full$bestTune)

predictions_hc_pd_msap_full_test <- predict(model_hc_pd_msap_full, newdata = X_test_hc_pd_msap_full)
cm_hc_pd_msap_full <- confusionMatrix(predictions_hc_pd_msap_full_test, Y_test_hc_pd_msap_full)
print(cm_hc_pd_msap_full)

# 混淆矩阵图 - 全特征
cm_table_hc_pd_msap_full <- as.data.frame(cm_hc_pd_msap_full$table)
colnames(cm_table_hc_pd_msap_full) <- c("Prediction", "Reference", "Freq")
# 重新映射显示标签
cm_table_hc_pd_msap_full$Prediction <- gsub("MSA_P", "MSA-P", cm_table_hc_pd_msap_full$Prediction)
cm_table_hc_pd_msap_full$Reference  <- gsub("MSA_P", "MSA-P", cm_table_hc_pd_msap_full$Reference)
plot_cm_hc_pd_msap_full <- create_confusion_matrix_plot(cm_table_hc_pd_msap_full, "Confusion Matrix - HC/PD/MSA-P (All Features)")

ggsave(filename = file.path(output_subdir_MSA_subtype, "Confusion_Matrix_HC_PD_MSAP_Full.png"), 
       plot = plot_cm_hc_pd_msap_full, width = 7, height = 6, dpi = 300)

# 4.3.3: 使用TOP20特征的HC-PD-MSA-P模型  
cat("\n使用TOP20特征进行HC-PD-MSA-P三分类分析...\n")

X_hc_pd_msap_top15 <- combined_measures_all[hc_pd_msap_indices, top20_features]

# 训练/测试集划分（使用相同的随机种子确保一致性）
set.seed(403)
split_indices_hc_pd_msap_top15 <- createDataPartition(Y_hc_pd_msap, p = 0.8, list = FALSE)
X_train_hc_pd_msap_top15 <- X_hc_pd_msap_top15[split_indices_hc_pd_msap_top15, ]
Y_train_hc_pd_msap_top15 <- Y_hc_pd_msap[split_indices_hc_pd_msap_top15]
X_test_hc_pd_msap_top15 <- X_hc_pd_msap_top15[-split_indices_hc_pd_msap_top15, ]
Y_test_hc_pd_msap_top15 <- Y_hc_pd_msap[-split_indices_hc_pd_msap_top15]

# 训练TOP20特征模型
model_hc_pd_msap_top15 <- train(
    x = X_train_hc_pd_msap_top15, y = Y_train_hc_pd_msap_top15, 
    metric = "Accuracy", method = "glmnet",
    trControl = train_control_diag, tuneLength = 50, 
    preProc = c("center", "scale")
)

cat("\n=== HC, PD, MSA-P (TOP20特征) 模型性能 ===\n")
print(model_hc_pd_msap_top15$bestTune)

predictions_hc_pd_msap_top15_test <- predict(model_hc_pd_msap_top15, newdata = X_test_hc_pd_msap_top15)
cm_hc_pd_msap_top15 <- confusionMatrix(predictions_hc_pd_msap_top15_test, Y_test_hc_pd_msap_top15)
print(cm_hc_pd_msap_top15)

# 混淆矩阵图 - TOP15特征
cm_table_hc_pd_msap_top15 <- as.data.frame(cm_hc_pd_msap_top15$table)
colnames(cm_table_hc_pd_msap_top15) <- c("Prediction", "Reference", "Freq")
# 重新映射显示标签
cm_table_hc_pd_msap_top15$Prediction <- gsub("MSA_P", "MSA-P", cm_table_hc_pd_msap_top15$Prediction)
cm_table_hc_pd_msap_top15$Reference  <- gsub("MSA_P", "MSA-P", cm_table_hc_pd_msap_top15$Reference)
plot_cm_hc_pd_msap_top15 <- create_confusion_matrix_plot(cm_table_hc_pd_msap_top15, "Confusion Matrix - HC/PD/MSA-P (TOP20 Features)")

ggsave(filename = file.path(output_subdir_MSA_subtype, "Confusion_Matrix_HC_PD_MSAP_TOP20.png"), 
       plot = plot_cm_hc_pd_msap_top15, width = 7, height = 6, dpi = 300)

# --- 4.4: 多分类ROC曲线分析 ---
cat("\n--- 4.4: 多分类ROC曲线分析 ---\n")

# 由于pROC包主要用于二分类，对于多分类ROC，我们使用multiclass.roc或手动计算OvR ROC
library(pROC)

# 4.4.1: HC-PD-MSA (TOP20特征) 多分类ROC
cat("计算HC-PD-MSA (TOP20特征) 多分类ROC曲线...\n")
predictions_hc_pd_msa_top15_prob <- predict(model_hc_pd_msa_top15, newdata = X_test_hc_pd_msa_top15, type = "prob")

# 使用multiclass.roc
multiclass_roc_hc_pd_msa_top15 <- multiclass.roc(Y_test_hc_pd_msa_top15, predictions_hc_pd_msa_top15_prob)
auc_hc_pd_msa_top15 <- auc(multiclass_roc_hc_pd_msa_top15)

cat("HC-PD-MSA (TOP20特征) 多分类AUC:", as.numeric(auc_hc_pd_msa_top15), "\n")

# 4.4.2: HC-PD-MSA-P (全特征) 多分类ROC
cat("计算HC-PD-MSA-P (全特征) 多分类ROC曲线...\n")
predictions_hc_pd_msap_full_prob <- predict(model_hc_pd_msap_full, newdata = X_test_hc_pd_msap_full, type = "prob")
multiclass_roc_hc_pd_msap_full <- multiclass.roc(Y_test_hc_pd_msap_full, predictions_hc_pd_msap_full_prob)
auc_hc_pd_msap_full <- auc(multiclass_roc_hc_pd_msap_full)

cat("HC-PD-MSA-P (全特征) 多分类AUC:", as.numeric(auc_hc_pd_msap_full), "\n")

# 4.4.3: HC-PD-MSA-P (TOP20特征) 多分类ROC
cat("计算HC-PD-MSA-P (TOP20特征) 多分类ROC曲线...\n")
predictions_hc_pd_msap_top15_prob <- predict(model_hc_pd_msap_top15, newdata = X_test_hc_pd_msap_top15, type = "prob")
multiclass_roc_hc_pd_msap_top15 <- multiclass.roc(Y_test_hc_pd_msap_top15, predictions_hc_pd_msap_top15_prob)
auc_hc_pd_msap_top15 <- auc(multiclass_roc_hc_pd_msap_top15)

cat("HC-PD-MSA-P (TOP20特征) 多分类AUC:", as.numeric(auc_hc_pd_msap_top15), "\n")

# --- 4.5: 性能总结和数据保存 ---
cat("\n--- 4.5: 性能总结和数据保存 ---\n")

# 整合所有模型性能
performance_summary_hc_pd_msap_full <- data.frame(
    Model = "HC_PD_MSAP_Full",
    Accuracy = cm_hc_pd_msap_full$overall["Accuracy"],
    Kappa = cm_hc_pd_msap_full$overall["Kappa"],
    MultiClass_AUC = as.numeric(auc_hc_pd_msap_full),
    stringsAsFactors = FALSE
)

performance_summary_hc_pd_msap_top15 <- data.frame(
    Model = "HC_PD_MSAP_TOP20", 
    Accuracy = cm_hc_pd_msap_top15$overall["Accuracy"],
    Kappa = cm_hc_pd_msap_top15$overall["Kappa"],
    MultiClass_AUC = as.numeric(auc_hc_pd_msap_top15),
    stringsAsFactors = FALSE
)

# 添加多分类AUC到TOP15模型
performance_summary_hc_pd_msa_top15$MultiClass_AUC <- as.numeric(auc_hc_pd_msa_top15)

# 合并所有新模型性能
all_new_models_performance <- bind_rows(
    performance_summary_hc_pd_msa_top15,
    performance_summary_hc_pd_msap_full,
    performance_summary_hc_pd_msap_top15
)

rownames(all_new_models_performance) <- NULL

cat("\n=== 第4部分新增模型性能总结 ===\n")
print(all_new_models_performance)

# 保存性能总结
write.csv(all_new_models_performance, 
          file.path(output_subdir_MSA_subtype, "performance_summary_MSA_subtype_models.csv"), 
          row.names = FALSE)

# 保存混淆矩阵数据
write.csv(cm_table_hc_pd_msa_top15, 
          file.path(output_subdir_MSA_subtype, "confusion_matrix_HC_PD_MSA_TOP20.csv"), 
          row.names = FALSE)

write.csv(cm_table_hc_pd_msap_full, 
          file.path(output_subdir_MSA_subtype, "confusion_matrix_HC_PD_MSAP_Full.csv"), 
          row.names = FALSE)

write.csv(cm_table_hc_pd_msap_top15, 
          file.path(output_subdir_MSA_subtype, "confusion_matrix_HC_PD_MSAP_TOP20.csv"), 
          row.names = FALSE)

# 保存预测概率数据
predictions_df_hc_pd_msa_top15 <- data.frame(
    True_Label = Y_test_hc_pd_msa_top15,
    Predicted_Label = predictions_hc_pd_msa_top15_test,
    predictions_hc_pd_msa_top15_prob
)

predictions_df_hc_pd_msap_full <- data.frame(
    True_Label = Y_test_hc_pd_msap_full,
    Predicted_Label = predictions_hc_pd_msap_full_test,
    predictions_hc_pd_msap_full_prob
)

predictions_df_hc_pd_msap_top15 <- data.frame(
    True_Label = Y_test_hc_pd_msap_top15,
    Predicted_Label = predictions_hc_pd_msap_top15_test,
    predictions_hc_pd_msap_top15_prob
)

write.csv(predictions_df_hc_pd_msa_top15, 
          file.path(output_subdir_MSA_subtype, "predictions_HC_PD_MSA_TOP20.csv"), 
          row.names = FALSE)

write.csv(predictions_df_hc_pd_msap_full, 
          file.path(output_subdir_MSA_subtype, "predictions_HC_PD_MSAP_Full.csv"), 
          row.names = FALSE)

write.csv(predictions_df_hc_pd_msap_top15, 
          file.path(output_subdir_MSA_subtype, "predictions_HC_PD_MSAP_TOP20.csv"), 
          row.names = FALSE)

cat("\n第4部分：基于MSA亚型的扩展分析已完成！\n")
cat("- HC-PD-MSA (TOP20特征) 三分类模型已训练\n")
cat("- HC-PD-MSA-P (全特征) 三分类模型已训练\n")
cat("- HC-PD-MSA-P (TOP20特征) 三分类模型已训练\n")
cat("- 所有混淆矩阵图已保存到:", output_subdir_MSA_subtype, "\n")
cat("- 所有性能数据和预测结果已保存\n")

cat("\n=== 脚本执行完成 ===\n")

# === 新增[保存sessionInfo，不改变结果] ===
try({
  writeLines(capture.output(sessionInfo()),
             file.path(output_basedir, "Part2_Disease_Sentinel_Definition", "sessionInfo.txt"))
}, silent = TRUE)

# 落盘统一主特征清单，供后续脚本读取
dir.create(file.path(output_basedir, "Part2_Disease_Sentinel_Definition"), showWarnings = FALSE, recursive = TRUE)
write.csv(data.frame(Feature = top20_features),
          file.path(output_basedir, "Part2_Disease_Sentinel_Definition", "top_features_master_TOP20.csv"),
          row.names = FALSE)


