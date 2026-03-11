#%%
#形成包含所有结构和功能连接度量的大型数据集combined_measures数据框，用于进一步分析或机器学习
rm(list = ls())
setwd('/media/neurox/T7_Shield/PD_analyse')  # 保持原始工作目录设置
run_tag <- format(Sys.time(), "%Y%m%d_%H%M%S")
output_subdir <- file.path("features_glmnet", paste0("all_optimized_", run_tag))  # 优化版本输出目录
dir.create(output_subdir, showWarnings = FALSE, recursive = TRUE)

# 加载必要包（移除重复安装语句）
library(reticulate)
use_python("/usr/local/fsl/bin/python", required = TRUE)  # 系统路径保持不变
np <- import('numpy')

# 加载其他必要的R包
library(glmnet) # 弹性网络模型
library(readxl) # 用于读取Excel文件
library(caret) # 数据划分、特征工程等工具
library(Matrix) # 矩阵运算支持
library(tidyverse) # 数据处理与可视化工具集
# 注：移除未使用且可能缺失的依赖，避免执行时中断
# ggplot2由tidyverse附带
library(doParallel) # 并行计算支持
registerDoParallel(cores = parallel::detectCores()) # 根据可用核心数注册并行后端
library(pROC) # 计算AUC值等指标
# 移除未使用库，保持最小依赖

# --- 与 9.1test_RBD_glmnet.R 对齐的绘图工具函数 ---
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

# --- 扩展指标工具函数（与 TOP20 一致口径） ---
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

#%%结合了FC SC图论测量指标以及脑区的形态学度量T1（皮层厚度、体积等），为后续的统计分析或机器学习模型训练准备数据
# 读取完整demo（与9.1一致，包含RBD；本脚本仅分析HC/PD/MSA）
demo <- read_excel('demo_all.xlsx')
# 对年龄和体积进行标准化
# demo[,5] <- scale(demo$age,center = T,scale = T)
# demo[,6] <- scale(demo$volume,center = T,scale = T)

# 将组别列转换为字符类型，并修正组别名称
demo$group <- as.character(demo$group)
demo$group[demo$group == 'HC'] <- 'HC'
demo$group[demo$group == 'MSA'] <- 'MSA'
demo$group[demo$group == 'PD'] <- 'PD'

keep_idx <- which(demo$group %in% c("HC", "MSA", "PD"))
if (length(keep_idx) == 0) stop("未在 demo_all.xlsx 中找到 HC/MSA/PD 组别，无法继续。")

# 从CSV文件加载局部图论指标
clustering_fc <- read.csv('FC_SC_MatPlots/clustering_fc.csv', header = FALSE)
clustering_sc <- read.csv('FC_SC_MatPlots/clustering_sc.csv', header = FALSE)
local_efficiency_fc <- read.csv('FC_SC_MatPlots/local_efficiency_fc.csv', header = FALSE)
local_efficiency_sc <- read.csv('FC_SC_MatPlots/local_efficiency_sc.csv', header = FALSE)
degree_fc <- read.csv('FC_SC_MatPlots/degree_fc.csv', header = FALSE)
degree_sc <- read.csv('FC_SC_MatPlots/degree_sc.csv', header = FALSE)
betweenness_fc <- read.csv('FC_SC_MatPlots/betweenness_fc.csv', header = FALSE)
betweenness_sc <- read.csv('FC_SC_MatPlots/betweenness_sc.csv', header = FALSE)
# 全局指标
characteristic_path_length_fc <- read.csv('FC_SC_MatPlots/characteristic_path_length_fc.csv', header = FALSE)
characteristic_path_length_sc <- read.csv('FC_SC_MatPlots/characteristic_path_length_sc.csv', header = FALSE)
global_efficiency_fc <- read.csv('FC_SC_MatPlots/global_efficiency_fc.csv', header = FALSE)
global_efficiency_sc <- read.csv('FC_SC_MatPlots/global_efficiency_sc.csv', header = FALSE)
modularity_fc <- read.csv('FC_SC_MatPlots/modularity_fc.csv', header = FALSE)
modularity_sc <- read.csv('FC_SC_MatPlots/modularity_sc.csv', header = FALSE)
# small_worldness_fc <- read.csv('E:/PD_analyse/FC_SC_MatPlots/small_worldness_fc.csv', header = FALSE)
# small_worldness_sc <- read.csv('E:/PD_analyse/FC_SC_MatPlots/small_worldness_sc.csv', header = FALSE)

# 将多个图论测量指标组合成一个多维数据结构
graph_measures <- cbind(clustering_fc, clustering_sc, local_efficiency_fc, local_efficiency_sc, degree_fc, degree_sc,
                        betweenness_fc, betweenness_sc, characteristic_path_length_fc, characteristic_path_length_sc,
                        global_efficiency_fc, global_efficiency_sc, modularity_fc, modularity_sc)
graph_measures <- graph_measures[keep_idx, , drop = FALSE]
rm(list = c('clustering_fc', 'clustering_sc', 'local_efficiency_fc', 'local_efficiency_sc', 'degree_fc', 'degree_sc',
            'betweenness_fc', 'betweenness_sc', 'characteristic_path_length_fc', 'characteristic_path_length_sc',
            'global_efficiency_fc', 'global_efficiency_sc', 'modularity_fc', 'modularity_sc')) # 删除中间变量节省内存

# 从npy文件加载表面形态学指标（与9.1一致）
data <- np$load('surface_ALL.npy')
data_dim <- tryCatch(dim(data), error = function(e) NULL)
if (!is.null(data_dim) && length(data_dim) == 2) {
  stopifnot(data_dim[1] >= max(keep_idx))
  stopifnot(data_dim[2] >= 800)
}
struc_measures <- array(data, dim = c(nrow(demo), 800)) # 将NumPy数组转换为R数组
struc_measures <- as.data.frame(struc_measures) # 将数组转换为DataFrame方便后续处理
struc_measures <- struc_measures[keep_idx, , drop = FALSE]
rm(data) # 删除中间变量

# 加载ROI标签信息
annot <- read.csv("/media/neurox/T7_Shield/R_done/analysis/surfaces/parcellations/lut/lut_schaefer-400_mics.csv")
annot <- annot[-c(1,202), ] # 移除不需要的行
annot$label <- gsub("7Networks_", "", annot$label) # 去除标签前缀
roi_labels <- annot$label # 提取ROI标签

# 定义结构指标标签
struc_labels <- c('area', 'thickness')

# 重复结构指标标签400次
struc_labels <- rep(struc_labels, each = 400)

# 合并ROI标签和结构指标标签形成新的列名
combined_labels <- paste(roi_labels, struc_labels, sep = "_")

# 定义图论指标标签
graph_labels <- c('clustering_fc', 'clustering_sc', 'local_efficiency_fc', 'local_efficiency_sc', 'degree_fc', 'degree_sc',
                  'betweenness_fc', 'betweenness_sc')
graph_labels2 <- c('characteristic_path_length_fc', 'characteristic_path_length_sc',
                   'global_efficiency_fc', 'global_efficiency_sc', 'modularity_fc', 'modularity_sc')

# 重复图论指标标签400次
graph_labels <- rep(graph_labels, each = 400)

# 合并ROI标签和图论指标标签形成新的列名
combined_labels2 <- paste(roi_labels, graph_labels, sep = "_")

# 添加第二个部分的图论指标标签到列名列表中
combined_labels2 <- c(combined_labels2, graph_labels2)

# 为结构和图论指标数据框设置列名
colnames(struc_measures) <- combined_labels
colnames(graph_measures) <- combined_labels2

# 加载皮层下指标（与9.1一致）
asegstats_path <- "/media/neurox/T7_Shield/PD_analyse/asegstats_all.xlsx"
asegstats_measures <- read_excel(asegstats_path, col_names = TRUE)
# 验证并按 keep_idx 同步截断
stopifnot(nrow(asegstats_measures) >= max(keep_idx))
asegstats_measures <- asegstats_measures[keep_idx, , drop = FALSE]

# --------------------------
# 按9.1风格：特征缩放由caret::preProc在建模阶段完成
# --------------------------

# 将结构和图论指标合并形成最终的数据集
combined_measures <- cbind(struc_measures, graph_measures, asegstats_measures)

# 将人口统计信息与结构和功能连接测量结合在一起（不加入age，保持与9.1一致）
demo <- demo[keep_idx, , drop = FALSE]
features <- cbind(demo$group, combined_measures)
rm(struc_measures, graph_measures, asegstats_measures) # 删除中间变量
# 设置新数据集的列名
colnames(features)[1:1] <- c('group')
# 仅保留三组：HC/MSA/PD
features <- features[features$group %in% c('HC','MSA','PD'), ]
colnames(features) <- make.unique(colnames(features)) # 确保列名唯一
# 提前保存标签，避免后续rm(features)导致引用错误
group <- as.factor(features$group)

# 将数据转换为模型矩阵形式
model_mat <- model.matrix(group ~ ., data = features)
rm(features) # 删除原始数据集
model_mat <- model_mat[, -1] # 移除截距项列

# 将合并后的数据集保存为Excel文件
write_excel_csv(combined_measures, file = file.path(output_subdir, "combined_measures_all.csv"))

# --------------------- 建模模型训练 ----------------------------
set.seed(314159)  # 改变随机种子以获得不同的训练/测试划分
colnames(combined_measures) <- make.unique(colnames(combined_measures))

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
  alpha = seq(0.22, 0.40, by = 0.02),
  lambda = exp(seq(-0.8, 1.4, length.out = 50))
)

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

# --------最小化标准输出（与9.1风格一致，仅保留所需图表与概要） --------------------------------------

#---------最佳模型超参数组合-------------
print(model$bestTune)

## 移除3D调参可视化以保持最简输出（与TOP20脚本一致）

# Make predictions on the test set
predictions <- predict(model, X_test)

# Calculate the accuracy of the predictions
accuracy <- mean(predictions == Y_test)
print(accuracy)

# 多分类AUC（与9.1风格一致，使用multiclass.roc数值汇总）
prediction_probabilities <- predict(model, X_test, type = "prob")
mc_roc <- pROC::multiclass.roc(Y_test, prediction_probabilities)
mc_auc <- as.numeric(pROC::auc(mc_roc))

# Extract the coefficients of the best model 提取训练好的glmnet模型的最佳系数，并绘制这些系数随正则化参数lambda变化的图。
best_model_coefficients <- coef(model$finalModel, model$bestTune$lambda)

## 移除系数可视化与词云等与论文本段无关的输出，仅保留混淆矩阵与ROC

# confusion matrix（与参考文件格式一致）
cm <- confusionMatrix(predictions, Y_test)
output_subdir_HC_PD_MSA <- output_subdir  # 直接输出到主目录
cat("特征数量:", ncol(model_mat), "\n")
n_features <- ncol(model_mat)

cm_table_df <- as.data.frame(cm$table)
plot_cm <- create_confusion_matrix_plot(cm_table_df, "Confusion Matrix - HC/PD/MSA (Test)")
ggsave(filename = file.path(output_subdir_HC_PD_MSA, "Confusion_Matrix_HPDMSA_model_test.png"), plot = plot_cm, width = 7, height = 6, dpi = 300)

# 训练集混淆矩阵
pred_train <- predict(model, X_train)
cm_train <- confusionMatrix(pred_train, Y_train)
cm_table_df_tr <- as.data.frame(cm_train$table)
plot_cm_tr <- create_confusion_matrix_plot(cm_table_df_tr, "Confusion Matrix - HC/PD/MSA (Train)")
ggsave(filename = file.path(output_subdir_HC_PD_MSA, "Confusion_Matrix_HPDMSA_model_train.png"), plot = plot_cm_tr, width = 7, height = 6, dpi = 300)

# 保存测试集混淆矩阵CSV
write.csv(cm_table_df, file.path(output_subdir_HC_PD_MSA, "confusion_matrix_part1_test.csv"), row.names = FALSE)

# 训练、测试 ROC（使用ggplot2专业风格）
suppressPackageStartupMessages(require(pROC))
suppressPackageStartupMessages(require(ggplot2))
prob_train <- predict(model, X_train, type = "prob")
prob_test  <- prediction_probabilities
classes <- colnames(prob_test)

# 计算总AUC
mc_auc_test <- as.numeric(pROC::auc(pROC::multiclass.roc(Y_test, prob_test)))
mc_auc_train <- as.numeric(pROC::auc(pROC::multiclass.roc(Y_train, prob_train)))

# 定义颜色：HC蓝色，MSA红色，PD绿色（与参考图一致）
roc_colors <- c("HC" = "#1F78B4", "MSA" = "#E31A1C", "PD" = "#33A02C")

# 创建测试集ROC数据
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

# 绘制测试集ROC（ggplot2专业风格，标题不重叠）
roc_plot_test <- ggplot(roc_data_test, aes(x = 1 - specificity, y = sensitivity, color = class)) +
  geom_line(linewidth = 1.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60", linewidth = 0.8) +
  scale_color_manual(values = roc_colors, 
                     labels = sapply(classes, function(c) sprintf("%s (AUC=%.3f)", c, unique(roc_data_test$auc[roc_data_test$class == c])))) +
  labs(title = sprintf("One-vs-Rest ROC (Test)\nOverall AUC = %.3f", mc_auc_test),
       x = "1 - Specificity",
       y = "Sensitivity") +
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

# 创建训练集ROC数据
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

# 绘制训练集ROC（ggplot2专业风格，标题不重叠）
roc_plot_train <- ggplot(roc_data_train, aes(x = 1 - specificity, y = sensitivity, color = class)) +
  geom_line(linewidth = 1.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60", linewidth = 0.8) +
  scale_color_manual(values = roc_colors,
                     labels = sapply(classes, function(c) sprintf("%s (AUC=%.3f)", c, unique(roc_data_train$auc[roc_data_train$class == c])))) +
  labs(title = sprintf("One-vs-Rest ROC (Train)\nOverall AUC = %.3f", mc_auc_train),
       x = "1 - Specificity",
       y = "Sensitivity") +
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


# 扩展性能指标（训练/测试），与参考文件格式一致
try({
  # Test-set
  acc_ci_te <- acc_ci_from_cm(cm)
  mauc_te <- as.numeric(pROC::auc(pROC::multiclass.roc(Y_test, prob_test)))
  ext_test <- data.frame(
    Model = "HC_PD_MSA_T1_AllFeatures",
    Dataset = "Test",
    Accuracy = unname(cm$overall["Accuracy"]),
    Accuracy_CI95_lower = acc_ci_te["lower"],
    Accuracy_CI95_upper = acc_ci_te["upper"],
    Kappa = unname(cm$overall["Kappa"]),
    BalancedAccuracy = balanced_acc(cm),
    MacroF1 = macro_f1(cm),
    MultiClass_AUC = mauc_te,
    stringsAsFactors = FALSE
  )
  # Train-set
  acc_ci_tr <- acc_ci_from_cm(cm_train)
  mauc_tr <- as.numeric(pROC::auc(pROC::multiclass.roc(Y_train, prob_train)))
  ext_train <- data.frame(
    Model = "HC_PD_MSA_T1_AllFeatures",
    Dataset = "Train",
    Accuracy = unname(cm_train$overall["Accuracy"]),
    Accuracy_CI95_lower = acc_ci_tr["lower"],
    Accuracy_CI95_upper = acc_ci_tr["upper"],
    Kappa = unname(cm_train$overall["Kappa"]),
    BalancedAccuracy = balanced_acc(cm_train),
    MacroF1 = macro_f1(cm_train),
    MultiClass_AUC = mauc_tr,
    stringsAsFactors = FALSE
  )
  ext_out <- rbind(ext_train, ext_test)
  write.csv(ext_out, file.path(output_subdir_HC_PD_MSA, "performance_summary_HC_PD_MSA_T1_AllFeatures_extended.csv"), row.names = FALSE)
  
  # Overall summary (multiclass AUC CI via bootstrap percentile)
  n_train <- length(Y_train)
  n_test <- length(Y_test)
  auc_ci_tr <- multiclass_auc_ci_bootstrap(Y_train, prob_train, n_boot = 1000, seed = 2026L)
  auc_ci_te <- multiclass_auc_ci_bootstrap(Y_test, prob_test, n_boot = 1000, seed = 2027L)
  auc_ci_tr_lower <- auc_ci_tr[1]
  auc_ci_tr_upper <- auc_ci_tr[2]
  auc_ci_te_lower <- auc_ci_te[1]
  auc_ci_te_upper <- auc_ci_te[2]
  
  overall_summary <- data.frame(
    Set = c("Train", "Test"),
    Class = "Overall",
    FeatureCount = n_features,
    AUC = c(mauc_tr, mauc_te),
    AUC_Low = c(auc_ci_tr_lower, auc_ci_te_lower),
    AUC_High = c(auc_ci_tr_upper, auc_ci_te_upper),
    Accuracy = c(unname(cm_train$overall["Accuracy"]), unname(cm$overall["Accuracy"])),
    Acc_Low = c(acc_ci_tr["lower"], acc_ci_te["lower"]),
    Acc_High = c(acc_ci_tr["upper"], acc_ci_te["upper"]),
    Sensitivity = NA, Sens_Low = NA, Sens_High = NA,
    Specificity = NA, Spec_Low = NA, Spec_High = NA,
    PPV = NA, PPV_Low = NA, PPV_High = NA,
    NPV = NA, NPV_Low = NA, NPV_High = NA
  )
  write.csv(overall_summary, file.path(output_subdir_HC_PD_MSA, "performance_part1_overall_summary.csv"), row.names = FALSE)
  
  # Byclass data
  byclass_data <- rbind(
    data.frame(Set = "Train", Class = rownames(cm_train$byClass), cm_train$byClass),
    data.frame(Set = "Test", Class = rownames(cm$byClass), cm$byClass)
  )
  write.csv(byclass_data, file.path(output_subdir_HC_PD_MSA, "byclass_HC_PD_MSA_T1_AllFeatures.csv"), row.names = FALSE)
  
  # All metrics with CI
  all_metrics_list <- list()
  idx <- 1
  
  # Train Overall
  all_metrics_list[[idx]] <- data.frame(
    Set = "Train", Class = "Overall", FeatureCount = n_features,
    AUC = mauc_tr, AUC_Low = auc_ci_tr_lower, AUC_High = auc_ci_tr_upper,
    Accuracy = unname(cm_train$overall["Accuracy"]),
    Acc_Low = acc_ci_tr["lower"], Acc_High = acc_ci_tr["upper"],
    Sensitivity = NA, Sens_Low = NA, Sens_High = NA,
    Specificity = NA, Spec_Low = NA, Spec_High = NA,
    PPV = NA, PPV_Low = NA, PPV_High = NA,
    NPV = NA, NPV_Low = NA, NPV_High = NA
  )
  idx <- idx + 1
  
  # Train classes - 使用Clopper-Pearson exact CI
  for (i in seq_along(classes)) {
    bin_true <- factor(ifelse(Y_train == classes[i], classes[i], "other"), levels = c(classes[i], "other"))
    bin_pred <- factor(ifelse(pred_train == classes[i], classes[i], "other"), levels = c(classes[i], "other"))
    roc_obj <- pROC::roc(bin_true, prob_train[, i], levels = c(classes[i], "other"), quiet = TRUE)
    auc_ci <- auc_ci95_delong(roc_obj)
    
    # 构建二分类混淆矩阵以提取TP/FN/TN/FP
    cm_binary <- confusionMatrix(bin_pred, bin_true, positive = classes[i])
    tab_bin <- as.matrix(cm_binary$table)
    tp <- tab_bin[classes[i], classes[i]]
    fn <- tab_bin["other", classes[i]]
    tn <- tab_bin["other", "other"]
    fp <- tab_bin[classes[i], "other"]
    
    # 使用Clopper-Pearson exact计算置信区间
    sens_ci <- suppressWarnings(binom.test(tp, tp + fn, conf.level = 0.95)$conf.int)
    spec_ci <- suppressWarnings(binom.test(tn, tn + fp, conf.level = 0.95)$conf.int)
    ppv_ci  <- suppressWarnings(binom.test(tp, tp + fp, conf.level = 0.95)$conf.int)
    npv_ci  <- suppressWarnings(binom.test(tn, tn + fn, conf.level = 0.95)$conf.int)
    
    all_metrics_list[[idx]] <- data.frame(
      Set = "Train", Class = classes[i], FeatureCount = n_features,
      AUC = as.numeric(pROC::auc(roc_obj)), AUC_Low = auc_ci[1], AUC_High = auc_ci[3],
      Accuracy = NA, Acc_Low = NA, Acc_High = NA,
      Sensitivity = cm_binary$byClass["Sensitivity"],
      Sens_Low = sens_ci[1],
      Sens_High = sens_ci[2],
      Specificity = cm_binary$byClass["Specificity"],
      Spec_Low = spec_ci[1],
      Spec_High = spec_ci[2],
      PPV = cm_binary$byClass["Pos Pred Value"],
      PPV_Low = ppv_ci[1],
      PPV_High = ppv_ci[2],
      NPV = cm_binary$byClass["Neg Pred Value"],
      NPV_Low = npv_ci[1],
      NPV_High = npv_ci[2]
    )
    idx <- idx + 1
  }
  
  # Test Overall
  all_metrics_list[[idx]] <- data.frame(
    Set = "Test", Class = "Overall", FeatureCount = n_features,
    AUC = mauc_te, AUC_Low = auc_ci_te_lower, AUC_High = auc_ci_te_upper,
    Accuracy = unname(cm$overall["Accuracy"]),
    Acc_Low = acc_ci_te["lower"], Acc_High = acc_ci_te["upper"],
    Sensitivity = NA, Sens_Low = NA, Sens_High = NA,
    Specificity = NA, Spec_Low = NA, Spec_High = NA,
    PPV = NA, PPV_Low = NA, PPV_High = NA,
    NPV = NA, NPV_Low = NA, NPV_High = NA
  )
  idx <- idx + 1
  
  # Test classes - 使用Clopper-Pearson exact CI
  for (i in seq_along(classes)) {
    bin_true <- factor(ifelse(Y_test == classes[i], classes[i], "other"), levels = c(classes[i], "other"))
    bin_pred <- factor(ifelse(predictions == classes[i], classes[i], "other"), levels = c(classes[i], "other"))
    roc_obj <- pROC::roc(bin_true, prob_test[, i], levels = c(classes[i], "other"), quiet = TRUE)
    auc_ci <- auc_ci95_delong(roc_obj)
    
    # 构建二分类混淆矩阵以提取TP/FN/TN/FP
    cm_binary <- confusionMatrix(bin_pred, bin_true, positive = classes[i])
    tab_bin <- as.matrix(cm_binary$table)
    tp <- tab_bin[classes[i], classes[i]]
    fn <- tab_bin["other", classes[i]]
    tn <- tab_bin["other", "other"]
    fp <- tab_bin[classes[i], "other"]
    
    # 使用Clopper-Pearson exact计算置信区间
    sens_ci <- suppressWarnings(binom.test(tp, tp + fn, conf.level = 0.95)$conf.int)
    spec_ci <- suppressWarnings(binom.test(tn, tn + fp, conf.level = 0.95)$conf.int)
    ppv_ci  <- suppressWarnings(binom.test(tp, tp + fp, conf.level = 0.95)$conf.int)
    npv_ci  <- suppressWarnings(binom.test(tn, tn + fn, conf.level = 0.95)$conf.int)
    
    all_metrics_list[[idx]] <- data.frame(
      Set = "Test", Class = classes[i], FeatureCount = n_features,
      AUC = as.numeric(pROC::auc(roc_obj)), AUC_Low = auc_ci[1], AUC_High = auc_ci[3],
      Accuracy = NA, Acc_Low = NA, Acc_High = NA,
      Sensitivity = cm_binary$byClass["Sensitivity"],
      Sens_Low = sens_ci[1],
      Sens_High = sens_ci[2],
      Specificity = cm_binary$byClass["Specificity"],
      Spec_Low = spec_ci[1],
      Spec_High = spec_ci[2],
      PPV = cm_binary$byClass["Pos Pred Value"],
      PPV_Low = ppv_ci[1],
      PPV_High = ppv_ci[2],
      NPV = cm_binary$byClass["Neg Pred Value"],
      NPV_Low = npv_ci[1],
      NPV_High = npv_ci[2]
    )
    idx <- idx + 1
  }
  
  write.csv(do.call(rbind, all_metrics_list),
            file.path(output_subdir_HC_PD_MSA, "performance_part1_all_metrics_with_ci.csv"), row.names = FALSE)
  
  cat("\n========== 最终结果 ==========\n")
  cat("训练集总AUC:", round(mauc_tr, 4), "\n")
  cat("测试集总AUC:", round(mauc_te, 4), "\n")
  cat("输出文件夹:", output_subdir_HC_PD_MSA, "\n")
  cat("==============================\n")
}, silent = TRUE)

# 保存划分索引与逐例分类明细（与9.1保持）
try({
  write.csv(data.frame(train_idx = as.integer(split_indices)), file.path(output_subdir_HC_PD_MSA, "split_HC_PD_MSA_train_idx.csv"), row.names = FALSE)
  write.csv(data.frame(test_idx  = setdiff(seq_len(nrow(model_mat)), as.integer(split_indices))), file.path(output_subdir_HC_PD_MSA, "split_HC_PD_MSA_test_idx.csv"), row.names = FALSE)
  pred_detail <- data.frame(
    Dataset = c(rep("Train", length(Y_train)), rep("Test", length(Y_test))),
    True = c(as.character(Y_train), as.character(Y_test)),
    Pred = c(as.character(pred_train), as.character(predictions))
  )
  write.csv(pred_detail, file.path(output_subdir_HC_PD_MSA, "classification_details_HC_PD_MSA.csv"), row.names = FALSE)
}, silent = TRUE)
