message(
  paste(
    "Legacy notice: this script is preserved for full reproduction from original workflow.",
    "For peer-review submission mainline, prefer `submission_release/04_validate_external_clinical_mri.R`."
  )
)

rm(list = ls())
setwd('/media/neurox/T7_Shield/PD_analyse')

# 统一外部验证集配置
validation_configs <- list(
  list(
    name = "ruijin",
    demo_file = "demo_clinical_ruijin.xlsx",
    surface_file = "surface_yanzheng_ruijin.npy",
    aseg_file = "asegstats_yanzheng_clinical_ruijin.csv",
    output_dir = "features_glmnet/PD_vs_MSA_Publication_clinical_ruijin"
  ),
  list(
    name = "suzhou",
    demo_file = "demo_clinical_suzhou.xlsx",
    surface_file = "surface_yanzheng_suzhou.npy",
    aseg_file = "asegstats_yanzheng_clinical_suzhou.csv",
    output_dir = "features_glmnet/PD_vs_MSA_Publication_clinical_suzhou"
  ),
  list(
    name = "xuzhou",
    demo_file = "demo_clinical_xuzhou.xlsx",
    surface_file = "surface_yanzheng_xuzhou.npy",
    aseg_file = "asegstats_yanzheng_clinical_xuzhou.csv",
    output_dir = "features_glmnet/PD_vs_MSA_Publication_clinical_xuzhou"
  ),
  list(
    name = "jiangsu",
    demo_file = "demo_clinical_jiangsu.xlsx",
    surface_file = "surface_yanzheng_jiangsu.npy",
    aseg_file = "asegstats_yanzheng_clinical_jiangsu.csv",
    output_dir = "features_glmnet/PD_vs_MSA_Publication_clinical_jiangsu"
  ),
  list(
    name = "xuanwu",
    demo_file = "demo_xuanwuyanzheng.xls",
    surface_file = "surface_xuanwuyanzheng.npy",
    aseg_file = "asegstats_xuanwuyanzheng.xlsx",
    output_dir = "features_glmnet/PD_vs_MSA_Publication_xuanwu_gongchenghua"
  )
)

# 加载依赖
library(reticulate); library(readxl); library(caret); library(Matrix)
library(tidyverse); library(ggplot2); library(pROC); library(neuroCombat)
utils::globalVariables(c('Prediction','Reference','Freq','%>%','mutate','across','everything'))

# 绘图与指标工具
create_confusion_matrix_plot <- function(cm_table_df, title_text) {
  colnames(cm_table_df) <- c('Prediction','Reference','Freq')
  ggplot(data = cm_table_df, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile(color = 'black') +
    geom_text(aes(label = Freq), color = 'magenta', size = 5) +
    scale_fill_gradient(low = 'white', high = 'navyblue') +
    labs(title = title_text, x = 'Predicted', y = 'True') +
    theme_classic(base_size = 16) +
    theme(plot.title = element_text(hjust = 0.5, face = 'bold'))
}

create_single_roc_plot_legacy <- function(roc_obj, title_text) {
  base_family <- 'Arial'
  g <- pROC::ggroc(roc_obj, colour = '#33A02C', size = 1.6) +
    xlab('1 - Specificity') +
    ylab('Sensitivity') +
    geom_abline(slope = -1, intercept = 1, linetype = 'dashed', color = 'grey60', linewidth = 0.7) +
    ggtitle(title_text) +
    theme_classic(base_size = 16, base_family = base_family) +
    theme(
      plot.title = element_text(hjust = 0.5, face = 'bold', family = base_family),
      axis.text = element_text(family = base_family),
      axis.title = element_text(family = base_family),
      plot.margin = margin(t = 10, r = 40, b = 24, l = 14),
      aspect.ratio = 1
    ) +
    scale_x_reverse(limits = c(1, 0), breaks = seq(1, 0, -0.25), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25), expand = c(0, 0)) +
    coord_fixed(ratio = 1, xlim = c(1, 0), ylim = c(0, 1), clip = 'on')
  g
}

f1_from_cm_binary <- function(cm) {
  bc <- cm$byClass
  ppv <- suppressWarnings(as.numeric(bc['Pos Pred Value']))
  sens <- suppressWarnings(as.numeric(bc['Sensitivity']))
  if (is.na(ppv) || is.na(sens) || (ppv + sens) == 0) return(NA_real_)
  2 * ppv * sens / (ppv + sens)
}
exact_ci <- function(x, n, conf.level = 0.95) {
  if (is.na(x) || is.na(n) || n <= 0) return(c(NA_real_, NA_real_))
  as.numeric(binom.test(x, n, conf.level = conf.level)$conf.int)
}
binary_exact_ci_from_cm <- function(cm) {
  tab <- as.matrix(cm$table)
  # confusionMatrix table: rows=Prediction, cols=Reference
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

auc_ci95_delong <- function(roc_obj) {
  ci <- suppressWarnings(pROC::ci.auc(roc_obj, conf.level = 0.95, method = 'delong'))
  as.numeric(ci)
}

# 全域阈值穷举 + 平局裁决
find_best_threshold_exhaustive <- function(prob_pd, y_true) {
  u <- sort(unique(as.numeric(prob_pd)))
  mids <- if (length(u) >= 2) (u[-1] + u[-length(u)]) / 2 else numeric(0)
  thr <- sort(unique(pmin(pmax(c(0, 1, u, mids), 0), 1)))
  best <- list(th = NA_real_, balanced_acc = -Inf, accuracy = -Inf, f1 = -Inf, rule = 'ge')
  for (t in thr) {
    for (rule in c('ge','gt')) {
      if (rule == 'ge') {
        cls_tmp <- factor(ifelse(prob_pd >= t, 'PD', 'MSA'), levels = c('MSA','PD'))
      } else {
        cls_tmp <- factor(ifelse(prob_pd >  t, 'PD', 'MSA'), levels = c('MSA','PD'))
      }
      cm_tmp <- caret::confusionMatrix(cls_tmp, y_true, positive = 'PD')
      ba  <- (as.numeric(cm_tmp$byClass['Sensitivity']) + as.numeric(cm_tmp$byClass['Specificity']))/2
      acc <- as.numeric(cm_tmp$overall['Accuracy'])
      f1  <- suppressWarnings({
        bc <- cm_tmp$byClass
        ppv <- as.numeric(bc['Pos Pred Value']); sens <- as.numeric(bc['Sensitivity'])
        if (is.na(ppv) || is.na(sens) || (ppv + sens) == 0) NA_real_ else 2 * ppv * sens / (ppv + sens)
      })
      better <- FALSE
      if (ba > best$balanced_acc + 1e-12) better <- TRUE else if (abs(ba - best$balanced_acc) <= 1e-12) {
        if (acc > best$accuracy + 1e-12) better <- TRUE else if (abs(acc - best$accuracy) <= 1e-12) {
          if (f1 > best$f1 + 1e-12) better <- TRUE else if (abs(f1 - best$f1) <= 1e-12) {
            if (!is.na(best$th)) {
              if (abs(t - 0.5) < abs(best$th - 0.5)) better <- TRUE
            } else better <- TRUE
          }
        }
      }
      if (better) best <- list(th = t, balanced_acc = ba, accuracy = acc, f1 = f1, rule = rule)
    }
  }
  best
}

# Python环境 & numpy
use_python('/usr/local/fsl/bin/python', required = TRUE)
np <- import('numpy')

cat('=== Unified External Validation with Youden Optimal Thresholds (All Sites) ===\n')
USE_LOCKED_THRESHOLD <- TRUE

# 1) 读取固定模型与TOP20特征清单
old_out <- 'features_glmnet/Final_PD_vs_MSA_Publication'
model_path <- file.path(old_out, 'model_TOP20_FrozenComBat_glmnet.rds')
feat_path  <- file.path(old_out, 'TOP20_features_PD_vs_MSA_specific.csv')
if (!file.exists(model_path) || !file.exists(feat_path)) {
  stop('缺少固定模型或TOP20清单，请先运行训练脚本以生成：model_TOP20_FrozenComBat_glmnet.rds 与 TOP20_features_PD_vs_MSA_specific.csv')
}
mdl <- readRDS(model_path)
top20_features <- read.csv(feat_path, stringsAsFactors = FALSE)$Feature
combat_assets_path <- file.path(old_out, "TOP20_frozen_combat_assets.rds")
combat_assets <- if (file.exists(combat_assets_path)) readRDS(combat_assets_path) else NULL
thres_path <- file.path(old_out, 'TOP20_winsorize_thresholds.csv')
thres_df <- if (file.exists(thres_path)) read.csv(thres_path, stringsAsFactors = FALSE) else NULL
fixed_th_path <- file.path(old_out, 'TOP20_fixed_threshold.csv')
locked_threshold <- NA_real_
if (file.exists(fixed_th_path)) {
  tmp_th <- read.csv(fixed_th_path, stringsAsFactors = FALSE)
  if ("Threshold" %in% colnames(tmp_th)) locked_threshold <- as.numeric(tmp_th$Threshold[1])
}
if (isTRUE(USE_LOCKED_THRESHOLD) && !is.finite(locked_threshold)) {
  stop('USE_LOCKED_THRESHOLD=TRUE 但缺少有效固定阈值（TOP20_fixed_threshold.csv），已停止以避免外部集重估阈值。')
}

# 2) 数据装载与预处理（可复用）
load_validation_data <- function(config) {
  demo_val_full <- read_excel(config$demo_file)
  val_indices <- which(demo_val_full$group %in% c('PD','MSA'))
  demo_val <- demo_val_full[val_indices, ] %>%
    mutate(site = as.factor(site), group = factor(group, levels = c('MSA','PD')))

  data_val <- np$load(config$surface_file)
  arr <- as.array(data_val)
  if (length(dim(arr)) == 2) {
    if (ncol(arr) == 800 && nrow(arr) == nrow(demo_val_full)) {
      mat2d <- arr
    } else if (nrow(arr) == 800 && ncol(arr) == nrow(demo_val_full)) {
      mat2d <- t(arr)
    } else {
      stop(sprintf('%s 为二维但维度不匹配: %s vs 期望 N×800',
                   config$surface_file, paste(dim(arr), collapse = 'x')))
    }
  } else if (length(dim(arr)) == 3) {
    if (dim(arr)[1] != 400) stop(sprintf('surface 第一维应为400个parcel，实际=%d', dim(arr)[1]))
    ch <- dim(arr)[3]
    if (ch >= 3) {
      area_mat <- t(arr[, , 2, drop = TRUE])
      thick_mat <- t(arr[, , 3, drop = TRUE])
    } else if (ch == 2) {
      area_mat <- t(arr[, , 1, drop = TRUE])
      thick_mat <- t(arr[, , 2, drop = TRUE])
    } else {
      stop(sprintf('surface 第三维通道不足，实际=%d', ch))
    }
    if (nrow(area_mat) != nrow(demo_val_full)) {
      stop(sprintf('surface 被试数与 %s 行数不一致: %d vs %d',
                   config$demo_file, nrow(area_mat), nrow(demo_val_full)))
    }
    mat2d <- cbind(area_mat, thick_mat)
  } else {
    stop(sprintf('%s 维度异常: %s', config$surface_file, paste(dim(arr), collapse = 'x')))
  }

  struc_val_all <- as.data.frame(mat2d)
  rm(arr, mat2d)
  struc_val <- struc_val_all[val_indices, , drop = FALSE]
  # 支持 xlsx/xls/CSV 三种格式，避免把 Excel 当 CSV 读出“embedded nulls”
  if (grepl("\\.xlsx?$", config$aseg_file, ignore.case = TRUE)) {
    aseg_val_all <- readxl::read_excel(config$aseg_file)
  } else {
    aseg_val_all <- read.csv(config$aseg_file, check.names = FALSE)
  }
  aseg_val <- aseg_val_all[val_indices, , drop = FALSE]
  if ('sub' %in% colnames(aseg_val)) aseg_val$sub <- NULL

  annot <- read.csv('/media/neurox/T7_Shield/R_done/analysis/surfaces/parcellations/lut/lut_schaefer-400_mics.csv')
  annot <- annot[-c(1,202), ]
  annot$label <- gsub('7Networks_', '', annot$label)
  roi_labels <- annot$label
  struc_labels <- rep(c('area','thickness'), each = 400)
  combined_labels <- paste(roi_labels, struc_labels, sep = '_')
  colnames(struc_val) <- make.unique(combined_labels)
  colnames(aseg_val) <- paste0('subcort_', colnames(aseg_val))
  X_val_raw <- cbind(struc_val, aseg_val)

  shared_features <- intersect(top20_features, colnames(X_val_raw))
  missing <- setdiff(top20_features, shared_features)
  if (length(missing) > 0) {
    stop(sprintf('%s 中缺少模型需要的特征列: %s', config$name, paste(missing, collapse = ', ')))
  }
  X_val <- X_val_raw[, top20_features, drop = FALSE]
  Y_val <- droplevels(factor(demo_val$group))

  # 冻结ComBat外推：仅使用训练阶段保存的estimates，不在外部集重拟合
  if (!is.null(combat_assets) && isTRUE(combat_assets$applied) && !is.null(combat_assets$estimates)) {
    val_sites <- as.character(demo_val$site)
    known_idx <- which(val_sites %in% combat_assets$training_sites)
    unknown_idx <- setdiff(seq_len(nrow(X_val)), known_idx)
    if (length(known_idx) > 0) {
      mod_val <- model.matrix(~ age + sex, data = demo_val[known_idx, , drop = FALSE])
      val_combat <- neuroCombat::neuroCombatFromTraining(
        dat = t(as.matrix(X_val[known_idx, , drop = FALSE])),
        batch = val_sites[known_idx],
        mod = mod_val,
        estimates = combat_assets$estimates
      )
      X_tmp <- X_val
      X_tmp[known_idx, ] <- as.data.frame(t(val_combat$dat.combat))
      X_val <- X_tmp
    }
    if (length(unknown_idx) > 0) {
      warning(sprintf(
        "%s: 外部样本中存在训练未见site（%s），这些样本保持原始值，不执行ComBat重拟合。",
        config$name,
        paste(unique(val_sites[unknown_idx]), collapse = ", ")
      ))
    }
  }

  if (!is.null(thres_df)) {
    for (f in top20_features) {
      if (f %in% thres_df$Feature) {
        row_idx <- which(thres_df$Feature == f)[1]
        lo <- thres_df$Lower[row_idx]
        hi <- thres_df$Upper[row_idx]
        x <- X_val[[f]]
        x[x < lo] <- lo; x[x > hi] <- hi
        X_val[[f]] <- x
      }
    }
  }

  list(
    X_val = X_val,
    Y_val = Y_val,
    source_site = rep(config$name, length(Y_val))
  )
}

# 3) 单站点处理函数
process_validation_set <- function(config) {
  cat(sprintf('\n=== Processing %s ===\n', config$name))
  if (dir.exists(config$output_dir)) {
    try(unlink(config$output_dir, recursive = TRUE, force = TRUE), silent = TRUE)
  }
  dir.create(config$output_dir, recursive = TRUE, showWarnings = FALSE)

  data_list <- load_validation_data(config)
  X_val <- data_list$X_val
  Y_val <- data_list$Y_val

  prob <- predict(mdl, newdata = X_val, type = 'prob')
  cls  <- predict(mdl, newdata = X_val)

  roc_pd <- roc(Y_val, prob$PD, levels = c('MSA','PD'))
  roc_msa <- roc(Y_val, prob$MSA, levels = c('PD','MSA'))

  if (isTRUE(USE_LOCKED_THRESHOLD) && is.finite(locked_threshold)) {
    th_optimal <- locked_threshold
    cls_optimal <- factor(ifelse(prob$PD >= th_optimal, 'PD', 'MSA'), levels = c('MSA','PD'))
    cm_optimal <- confusionMatrix(cls_optimal, Y_val, positive = 'PD')
    best_balanced_acc <- (as.numeric(cm_optimal$byClass['Sensitivity']) + as.numeric(cm_optimal$byClass['Specificity']))/2
  } else {
    coords_youden <- try(pROC::coords(roc_pd, x = 'best', best.method = 'youden',
                                      ret = c('threshold','sensitivity','specificity'), transpose = FALSE), silent = TRUE)
    if (inherits(coords_youden, 'try-error')) {
      cat(sprintf('Warning: Could not optimize threshold for %s\n', config$name))
      return(NULL)
    }
    th_youden <- as.numeric(coords_youden['threshold'])
    best_exh <- find_best_threshold_exhaustive(prob$PD, Y_val)
    cls_youden <- factor(ifelse(prob$PD >= th_youden, 'PD', 'MSA'), levels = c('MSA','PD'))
    cm_youden <- confusionMatrix(cls_youden, Y_val, positive = 'PD')
    ba_youden <- (as.numeric(cm_youden$byClass['Sensitivity']) + as.numeric(cm_youden$byClass['Specificity']))/2
    acc_youden <- as.numeric(cm_youden$overall['Accuracy'])

    use_exh <- FALSE
    if (best_exh$balanced_acc > ba_youden + 1e-12) use_exh <- TRUE else if (abs(best_exh$balanced_acc - ba_youden) <= 1e-12) {
      if (best_exh$accuracy > acc_youden + 1e-12) use_exh <- TRUE
    }
    if (use_exh) {
      th_optimal <- best_exh$th
      if (identical(best_exh$rule, 'ge')) {
        cls_optimal <- factor(ifelse(prob$PD >= th_optimal, 'PD', 'MSA'), levels = c('MSA','PD'))
      } else {
        cls_optimal <- factor(ifelse(prob$PD >  th_optimal, 'PD', 'MSA'), levels = c('MSA','PD'))
      }
      cm_optimal <- confusionMatrix(cls_optimal, Y_val, positive = 'PD')
      best_balanced_acc <- best_exh$balanced_acc
    } else {
      th_optimal <- th_youden
      cls_optimal <- cls_youden
      cm_optimal <- cm_youden
      best_balanced_acc <- ba_youden
    }
  }

  auc_val <- auc(roc_pd)
  ci_val <- auc_ci95_delong(roc_pd)
  ci_bin <- binary_exact_ci_from_cm(cm_optimal)

  cat(sprintf('%s - Locked-threshold external validation:\n', config$name))
    cat(sprintf('  AUC: %.3f [%.3f, %.3f]\n', auc_val, ci_val[1], ci_val[3]))
    cat(sprintf('  Accuracy: %.3f\n', cm_optimal$overall['Accuracy']))
    cat(sprintf('  Sensitivity: %.3f\n', cm_optimal$byClass['Sensitivity']))
    cat(sprintf('  Specificity: %.3f\n', cm_optimal$byClass['Specificity']))
    cat(sprintf('  F1: %.3f\n', f1_from_cm_binary(cm_optimal)))
    cat(sprintf('  Balanced Accuracy: %.3f\n', best_balanced_acc))
  cat(sprintf('  Locked Threshold: %.3f\n', th_optimal))

    write.csv(data.frame(
      Subject_ID = seq_len(nrow(X_val)),
      True_Label = Y_val,
      Predicted_Label = cls_optimal,
      Prob_MSA = prob$MSA,
      Prob_PD = prob$PD,
      stringsAsFactors = FALSE
    ), file.path(config$output_dir, sprintf('classification_details_%s.csv', config$name)), row.names = FALSE)

    perf <- data.frame(
      Model = sprintf('PD_vs_MSA_TOP20_%s', config$name),
      Accuracy = cm_optimal$overall['Accuracy'],
      Accuracy_CI95_lower = ci_bin$acc[1],
      Accuracy_CI95_upper = ci_bin$acc[2],
      Kappa = cm_optimal$overall['Kappa'],
      Sensitivity = cm_optimal$byClass['Sensitivity'],
      Sensitivity_CI95_lower = ci_bin$sens[1],
      Sensitivity_CI95_upper = ci_bin$sens[2],
      Specificity = cm_optimal$byClass['Specificity'],
      Specificity_CI95_lower = ci_bin$spec[1],
      Specificity_CI95_upper = ci_bin$spec[2],
      PPV = cm_optimal$byClass['Pos Pred Value'],
      PPV_CI95_lower = ci_bin$ppv[1],
      PPV_CI95_upper = ci_bin$ppv[2],
      NPV = cm_optimal$byClass['Neg Pred Value'],
      NPV_CI95_lower = ci_bin$npv[1],
      NPV_CI95_upper = ci_bin$npv[2],
      F1 = f1_from_cm_binary(cm_optimal),
      BalancedAccuracy = (as.numeric(cm_optimal$byClass['Sensitivity']) + as.numeric(cm_optimal$byClass['Specificity']))/2,
      AUC = as.numeric(auc_val),
      AUC_CI95_lower = ci_val[1],
      AUC_CI95_upper = ci_val[3],
      Optimal_Threshold = th_optimal,
      stringsAsFactors = FALSE
    )
    write.csv(perf, file.path(config$output_dir, sprintf('performance_%s.csv', config$name)), row.names = FALSE)

    write.csv(as.data.frame(cm_optimal$table), file.path(config$output_dir, sprintf('confusion_matrix_%s.csv', config$name)), row.names = FALSE)

    roc_plot <- create_single_roc_plot_legacy(roc_pd, title_text = 'ROC Curves for Model Predictions')
    auc3 <- as.numeric(sprintf('%.3f', as.numeric(auc_val)))
    auc2 <- floor(auc3 * 100 + 0.5) / 100
    auc_txt <- sprintf('AUC = %.2f\n95%% CI = [%.2f, %.2f]', auc2, ci_val[1], ci_val[3])
    roc_plot <- roc_plot + annotate('text', x = 0.02, y = 0.10, label = auc_txt,
                                    color = 'black', hjust = 1, vjust = 0,
                                    family = 'Arial', size = 4.6)
    ggsave(filename = file.path(config$output_dir, sprintf('ROC_%s.png', config$name)),
           plot = roc_plot, width = 8, height = 6, dpi = 300)

    cm_plot <- create_confusion_matrix_plot(as.data.frame(cm_optimal$table),
                                          sprintf('Confusion Matrix - %s', config$name))
    ggsave(filename = file.path(config$output_dir, sprintf('Confusion_Matrix_%s.png', config$name)),
           plot = cm_plot, width = 7, height = 6, dpi = 300)

    return(list(
      name = config$name,
      auc = auc_val,
      accuracy = cm_optimal$overall['Accuracy'],
      sensitivity = cm_optimal$byClass['Sensitivity'],
      specificity = cm_optimal$byClass['Specificity'],
      f1 = f1_from_cm_binary(cm_optimal),
      threshold = th_optimal
    ))
}

# 4) 合并四站点数据进行联合外部验证
process_combined_validation <- function(configs, combined_name = 'all_sites_combined',
                                        output_dir = 'features_glmnet/PD_vs_MSA_Publication_clinical_all_sites') {
  cat('\n=== Processing Combined External Validation (All Sites) ===\n')
  if (dir.exists(output_dir)) {
    try(unlink(output_dir, recursive = TRUE, force = TRUE), silent = TRUE)
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  X_list <- list(); Y_list <- list(); site_list <- list()
  for (cfg in configs) {
    dl <- load_validation_data(cfg)
    X_list[[cfg$name]] <- dl$X_val
    Y_list[[cfg$name]] <- dl$Y_val
    site_list[[cfg$name]] <- dl$source_site
  }
  if (length(X_list) == 0) {
    cat('No data loaded for combined validation.\n')
    return(NULL)
  }

  X_all <- do.call(rbind, X_list)
  Y_all <- factor(unlist(Y_list), levels = c('MSA','PD'))
  site_all <- unlist(site_list)

  prob <- predict(mdl, newdata = X_all, type = 'prob')
  cls  <- predict(mdl, newdata = X_all)

  roc_pd <- roc(Y_all, prob$PD, levels = c('MSA','PD'))
  if (isTRUE(USE_LOCKED_THRESHOLD) && is.finite(locked_threshold)) {
    th_optimal <- locked_threshold
    cls_optimal <- factor(ifelse(prob$PD >= th_optimal, 'PD', 'MSA'), levels = c('MSA','PD'))
    cm_optimal <- confusionMatrix(cls_optimal, Y_all, positive = 'PD')
    best_balanced_acc <- (as.numeric(cm_optimal$byClass['Sensitivity']) + as.numeric(cm_optimal$byClass['Specificity']))/2
  } else {
    coords_youden <- try(pROC::coords(roc_pd, x = 'best', best.method = 'youden',
                                      ret = c('threshold','sensitivity','specificity'), transpose = FALSE), silent = TRUE)
    if (inherits(coords_youden, 'try-error')) {
      cat('Warning: Could not optimize threshold for combined set\n')
      return(NULL)
    }
    th_youden <- as.numeric(coords_youden['threshold'])
    best_exh <- find_best_threshold_exhaustive(prob$PD, Y_all)
    cls_youden <- factor(ifelse(prob$PD >= th_youden, 'PD', 'MSA'), levels = c('MSA','PD'))
    cm_youden <- confusionMatrix(cls_youden, Y_all, positive = 'PD')
    ba_youden <- (as.numeric(cm_youden$byClass['Sensitivity']) + as.numeric(cm_youden$byClass['Specificity']))/2
    acc_youden <- as.numeric(cm_youden$overall['Accuracy'])

    use_exh <- FALSE
    if (best_exh$balanced_acc > ba_youden + 1e-12) use_exh <- TRUE else if (abs(best_exh$balanced_acc - ba_youden) <= 1e-12) {
      if (best_exh$accuracy > acc_youden + 1e-12) use_exh <- TRUE
    }
    if (use_exh) {
      th_optimal <- best_exh$th
      if (identical(best_exh$rule, 'ge')) {
        cls_optimal <- factor(ifelse(prob$PD >= th_optimal, 'PD', 'MSA'), levels = c('MSA','PD'))
      } else {
        cls_optimal <- factor(ifelse(prob$PD >  th_optimal, 'PD', 'MSA'), levels = c('MSA','PD'))
      }
      cm_optimal <- confusionMatrix(cls_optimal, Y_all, positive = 'PD')
      best_balanced_acc <- best_exh$balanced_acc
    } else {
      th_optimal <- th_youden
      cls_optimal <- cls_youden
      cm_optimal <- cm_youden
      best_balanced_acc <- ba_youden
    }
  }

  auc_val <- auc(roc_pd)
  ci_val <- auc_ci95_delong(roc_pd)
  ci_bin <- binary_exact_ci_from_cm(cm_optimal)

  cat(sprintf('%s - Combined locked-threshold results:\n', combined_name))
  cat(sprintf('  AUC: %.3f [%.3f, %.3f]\n', auc_val, ci_val[1], ci_val[3]))
  cat(sprintf('  Accuracy: %.3f\n', cm_optimal$overall['Accuracy']))
  cat(sprintf('  Sensitivity: %.3f\n', cm_optimal$byClass['Sensitivity']))
  cat(sprintf('  Specificity: %.3f\n', cm_optimal$byClass['Specificity']))
  cat(sprintf('  F1: %.3f\n', f1_from_cm_binary(cm_optimal)))
  cat(sprintf('  Balanced Accuracy: %.3f\n', best_balanced_acc))
  cat(sprintf('  Locked Threshold: %.3f\n', th_optimal))

  write.csv(data.frame(
    Subject_ID = seq_len(nrow(X_all)),
    Source_Site = site_all,
    True_Label = Y_all,
    Predicted_Label = cls_optimal,
    Prob_MSA = prob$MSA,
    Prob_PD = prob$PD,
    stringsAsFactors = FALSE
  ), file.path(output_dir, sprintf('classification_details_%s.csv', combined_name)), row.names = FALSE)

  perf <- data.frame(
    Model = sprintf('PD_vs_MSA_TOP20_%s', combined_name),
    Accuracy = cm_optimal$overall['Accuracy'],
    Accuracy_CI95_lower = ci_bin$acc[1],
    Accuracy_CI95_upper = ci_bin$acc[2],
    Kappa = cm_optimal$overall['Kappa'],
    Sensitivity = cm_optimal$byClass['Sensitivity'],
    Sensitivity_CI95_lower = ci_bin$sens[1],
    Sensitivity_CI95_upper = ci_bin$sens[2],
    Specificity = cm_optimal$byClass['Specificity'],
    Specificity_CI95_lower = ci_bin$spec[1],
    Specificity_CI95_upper = ci_bin$spec[2],
    PPV = cm_optimal$byClass['Pos Pred Value'],
    PPV_CI95_lower = ci_bin$ppv[1],
    PPV_CI95_upper = ci_bin$ppv[2],
    NPV = cm_optimal$byClass['Neg Pred Value'],
    NPV_CI95_lower = ci_bin$npv[1],
    NPV_CI95_upper = ci_bin$npv[2],
    F1 = f1_from_cm_binary(cm_optimal),
    BalancedAccuracy = (as.numeric(cm_optimal$byClass['Sensitivity']) + as.numeric(cm_optimal$byClass['Specificity']))/2,
    AUC = as.numeric(auc_val),
    AUC_CI95_lower = ci_val[1],
    AUC_CI95_upper = ci_val[3],
    Optimal_Threshold = th_optimal,
    stringsAsFactors = FALSE
  )
  write.csv(perf, file.path(output_dir, sprintf('performance_%s.csv', combined_name)), row.names = FALSE)

  write.csv(as.data.frame(cm_optimal$table), file.path(output_dir, sprintf('confusion_matrix_%s.csv', combined_name)), row.names = FALSE)

  roc_plot <- create_single_roc_plot_legacy(roc_pd, title_text = 'ROC Curves for Combined Model Predictions')
  auc3 <- as.numeric(sprintf('%.3f', as.numeric(auc_val)))
  auc2 <- floor(auc3 * 100 + 0.5) / 100
  auc_txt <- sprintf('AUC = %.2f\n95%% CI = [%.2f, %.2f]', auc2, ci_val[1], ci_val[3])
  roc_plot <- roc_plot + annotate('text', x = 0.02, y = 0.10, label = auc_txt,
                                  color = 'black', hjust = 1, vjust = 0,
                                  family = 'Arial', size = 4.6)
  ggsave(filename = file.path(output_dir, sprintf('ROC_%s.png', combined_name)),
         plot = roc_plot, width = 8, height = 6, dpi = 300)

  cm_plot <- create_confusion_matrix_plot(as.data.frame(cm_optimal$table),
                                          sprintf('Confusion Matrix - %s', combined_name))
  ggsave(filename = file.path(output_dir, sprintf('Confusion_Matrix_%s.png', combined_name)),
         plot = cm_plot, width = 7, height = 6, dpi = 300)

  list(
    name = combined_name,
    auc = auc_val,
    accuracy = cm_optimal$overall['Accuracy'],
    sensitivity = cm_optimal$byClass['Sensitivity'],
    specificity = cm_optimal$byClass['Specificity'],
    f1 = f1_from_cm_binary(cm_optimal),
    threshold = th_optimal
  )
}

# 5) 处理所有验证集
results <- list()
for (config in validation_configs) {
  result <- process_validation_set(config)
  if (!is.null(result)) {
    results[[config$name]] <- result
  }
}

# 6) 合并四中心联合外部验证
combined_result <- process_combined_validation(validation_configs)
if (!is.null(combined_result)) {
  results[[combined_result$name]] <- combined_result
}

# 7) 生成汇总报告
if (length(results) > 0) {
  cat('\n=== Summary of External Validation Results ===\n')
  summary_df <- do.call(rbind, lapply(results, function(x) {
    data.frame(
      Validation_Set = x$name,
      AUC = x$auc,
      Accuracy = x$accuracy,
      Sensitivity = x$sensitivity,
      Specificity = x$specificity,
      F1 = x$f1,
      Optimal_Threshold = x$threshold,
      stringsAsFactors = FALSE
    )
  }))

  print(summary_df)
  write.csv(summary_df, 'features_glmnet/external_validation_summary_with_all_sites.csv', row.names = FALSE)

  cat('\nExternal validation completed successfully!\n')
  cat('All results saved with locked discovery threshold and exact CI for non-AUC metrics.\n')
} else {
  cat('No valid results generated.\n')
}
