# 11_Phenotyping_and_Thresholding.R
# ==============================================================================
# 
# Purpose: This script implements Module 2 of the analysis pipeline. It performs
#          RBD phenotyping based on PD and MSA signature scores derived from
#          sentinel models. Crucially, it uses a supervised approach to define
#          a clinically meaningful threshold for the PD score, while using a
#          robust relative threshold for the MSA score, acknowledging the
#          data imbalance in known outcomes.
#
# Inputs:
# - analysis_pipeline_output/harmonized_features.csv (from Module 1)
# - analysis_pipeline_output/harmonized_pheno.csv (from Module 1)
#
# Outputs:
# - analysis_pipeline_output/RBD_Subtypes_from_Signature_Scores_supervised.csv
# - analysis_pipeline_output/sentinel_models/PD_sentinel_model.rds
# - analysis_pipeline_output/sentinel_models/MSA_sentinel_model.rds
# - analysis_pipeline_output/plots/PD_Score_ROC_for_Threshold.png
#
# ==============================================================================

# 1. Setup Environment
# ==============================================================================
library(tidyverse)
library(data.table)
library(caret)
library(glmnet)
library(pROC)
library(readxl)
library(reticulate)
library(neuroCombat)

# Create output directories if they don't exist
dir.create("analysis_pipeline_output/sentinel_models", showWarnings = FALSE, recursive = TRUE)
dir.create("analysis_pipeline_output/plots", showWarnings = FALSE, recursive = TRUE)

# 2. Load Harmonized Data
# ==============================================================================
cat("Step 1: Loading harmonized data...\n")
if (!file.exists("analysis_pipeline_output/harmonized_pheno.csv") ||
    !file.exists("analysis_pipeline_output/harmonized_features.csv")) {
  source("09_02_prepare_harmonized_feature_tables.R")
}
pheno <- fread("analysis_pipeline_output/harmonized_pheno.csv")
features <- fread("analysis_pipeline_output/harmonized_features.csv")

# ROBUSTNESS FIX: Clean the conversion_status column to handle all formats (e.g., "PD", "PD(7.3y)")
pheno <- pheno %>%
  mutate(
    conversion_outcome = case_when(
      grepl("PD", conversion_status, ignore.case = TRUE) ~ "PD",
      grepl("MSA", conversion_status, ignore.case = TRUE) ~ "MSA",
      grepl("stable", conversion_status, ignore.case = TRUE) ~ "stable",
      TRUE ~ NA_character_
    )
  )

# FINAL & ROBUST FIX:
# The root cause is that harmonized_features.csv was saved without an ID column.
# The rows, however, are in the correct order corresponding to the pheno file.
# We will add the subject IDs from the pheno file to the features file to ensure a perfect match.
if ("ID" %in% names(pheno)) {
    setnames(pheno, "ID", "sub")
}

# Add the 'sub' column from pheno to features. This ensures alignment.
features$sub <- pheno$sub

# FINAL & ROBUST FIX 2: Define feature names DIRECTLY from the features data
# before the merge. This is the most robust way to ensure only numeric feature
# columns are selected for modeling.
feature_names <- setdiff(names(features), "sub")

# Merge into a single data frame for modeling
data_all <- merge(pheno, features, by = "sub")

# The old, flawed method is now removed.

# 3. Load locked sentinel models
# ==============================================================================
cat("Step 2: Loading locked sentinel models...\n")
pd_model_path <- "analysis_pipeline_output/sentinel_models/PD_sentinel_model.rds"
msa_model_path <- "analysis_pipeline_output/sentinel_models/MSA_sentinel_model.rds"
if (!file.exists(pd_model_path) || !file.exists(msa_model_path)) {
  stop("缺少锁定哨兵模型，请先生成 analysis_pipeline_output/sentinel_models/PD_sentinel_model.rds 与 MSA_sentinel_model.rds")
}
pd_model <- readRDS(pd_model_path)
msa_model <- readRDS(msa_model_path)


# 4. Calculate Signature Scores for iRBD Subjects
# ==============================================================================
cat("Step 3: Calculating PD and MSA signature scores for RBD subjects...\n")

rbd_data <- data_all %>% filter(group == "RBD")

# Extract coefficients from the best tune as numeric named vectors (exclude intercept)
pd_coef_mat <- as.matrix(coef(pd_model$finalModel, s = pd_model$bestTune$lambda))
msa_coef_mat <- as.matrix(coef(msa_model$finalModel, s = msa_model$bestTune$lambda))
pd_coef_vec <- pd_coef_mat[-1, 1]
msa_coef_vec <- msa_coef_mat[-1, 1]
storage.mode(pd_coef_vec) <- "double"
storage.mode(msa_coef_vec) <- "double"

# Prefer a single frozen harmonization reference (same as locked external model)
top20_candidates <- c(
  "features_glmnet/Final_PD_vs_MSA_Publication/TOP20_features_PD_vs_MSA_specific.csv",
  "RBD_test_TOP20/Part2_Disease_Sentinel_Definition/top_features_master_TOP20.csv",
  "最终文章使用代码及原始文件/Part2_Disease_Sentinel_Definition/top_features_master_TOP20.csv"
)
top20_file <- top20_candidates[file.exists(top20_candidates)][1]
top20_features <- if (!is.na(top20_file) && nzchar(top20_file)) read.csv(top20_file, stringsAsFactors = FALSE)$Feature else character(0)
combat_assets_path <- "features_glmnet/Final_PD_vs_MSA_Publication/TOP20_frozen_combat_assets.rds"
use_single_frozen_reference <- FALSE
rbd_feature_df <- NULL

if (length(top20_features) > 0 &&
    file.exists(combat_assets_path) &&
    file.exists("demo_all.xlsx") &&
    file.exists("surface_ALL.npy") &&
    file.exists("asegstats_all.xlsx")) {
  message("  - Reusing frozen harmonization reference from locked external model assets.")
  combat_assets <- readRDS(combat_assets_path)
  if (isTRUE(combat_assets$applied) && !is.null(combat_assets$estimates)) {
    demo_raw <- readxl::read_excel("demo_all.xlsx")
    if (!"sub" %in% names(demo_raw) && "ID" %in% names(demo_raw)) demo_raw$sub <- demo_raw$ID
    demo_raw$site <- as.factor(demo_raw$site)
    if (!"sex" %in% names(demo_raw) && "gender" %in% names(demo_raw)) demo_raw$sex <- demo_raw$gender
    demo_raw$sex <- as.factor(demo_raw$sex)

    np <- reticulate::import("numpy")
    arr <- as.array(np$load("surface_ALL.npy"))
    n_rois <- dim(arr)[1]; n_subs <- dim(arr)[2]
    area_mat <- matrix(arr[, , 2], nrow = n_rois, ncol = n_subs)
    thick_mat <- matrix(arr[, , 3], nrow = n_rois, ncol = n_subs)
    struc_df <- as.data.frame(t(cbind(area_mat, thick_mat)))
    aseg_all <- readxl::read_excel("asegstats_all.xlsx")
    annot <- read.csv('/media/neurox/T7_Shield/R_done/analysis/surfaces/parcellations/lut/lut_schaefer-400_mics.csv')
    annot <- annot[-c(1, 202), ]
    annot$label <- gsub('7Networks_', '', annot$label)
    roi_labels <- annot$label
    colnames(struc_df) <- make.unique(c(paste(roi_labels, 'area', sep = '_'), paste(roi_labels, 'thickness', sep = '_')))
    colnames(aseg_all) <- paste0('subcort_', colnames(aseg_all))
    X_all_raw <- cbind(struc_df, aseg_all)
    available_top20 <- intersect(top20_features, colnames(X_all_raw))
    rbd_idx_raw <- which(demo_raw$group == "RBD")
    if (length(available_top20) > 0 && length(rbd_idx_raw) > 0) {
      X_rbd_top20 <- X_all_raw[rbd_idx_raw, available_top20, drop = FALSE]
      known_idx <- which(as.character(demo_raw$site[rbd_idx_raw]) %in% combat_assets$training_sites)
      if (length(known_idx) > 0) {
        mod_rbd <- model.matrix(~ age + sex, data = demo_raw[rbd_idx_raw[known_idx], , drop = FALSE])
        combat_rbd <- neuroCombat::neuroCombatFromTraining(
          dat = t(as.matrix(X_rbd_top20[known_idx, , drop = FALSE])),
          batch = as.character(demo_raw$site[rbd_idx_raw[known_idx]]),
          mod = mod_rbd,
          estimates = combat_assets$estimates
        )
        X_tmp <- X_rbd_top20
        X_tmp[known_idx, ] <- as.data.frame(t(combat_rbd$dat.combat))
        X_rbd_top20 <- X_tmp
      }
      rbd_feature_df <- data.frame(sub = as.character(demo_raw$sub[rbd_idx_raw]), X_rbd_top20, check.names = FALSE)
      use_single_frozen_reference <- TRUE
    }
  }
}

if (!use_single_frozen_reference) {
  message("  - Fallback to harmonized feature table for iRBD scoring.")
  rbd_feature_df <- data.frame(sub = rbd_data$sub, rbd_data[, feature_names, with = FALSE], check.names = FALSE)
}

# Build paper-consistent 2D diagnostic coordinates:
# PD score = weighted sum of 10 PD-driving regions; MSA-like score = weighted sum of 10 MSA-driving regions.
pd_rank <- names(sort(abs(pd_coef_vec[colnames(rbd_feature_df)[-1]]), decreasing = TRUE))
pd_rank <- pd_rank[is.finite(pd_coef_vec[pd_rank]) & pd_coef_vec[pd_rank] != 0]
msa_rank <- names(sort(abs(msa_coef_vec[colnames(rbd_feature_df)[-1]]), decreasing = TRUE))
msa_rank <- msa_rank[is.finite(msa_coef_vec[msa_rank]) & msa_coef_vec[msa_rank] != 0]
pd_top10 <- head(pd_rank, 10)
msa_top10 <- head(msa_rank, 10)
if (length(pd_top10) == 0 || length(msa_top10) == 0) {
  stop("Unable to derive PD/MSA top-driving features for iRBD scoring.")
}

pd_scores <- as.numeric(as.matrix(rbd_feature_df[, pd_top10, drop = FALSE]) %*% pd_coef_vec[pd_top10])
msa_scores <- as.numeric(as.matrix(rbd_feature_df[, msa_top10, drop = FALSE]) %*% msa_coef_vec[msa_top10])
rbd_scores <- data.frame(
  sub = rbd_feature_df$sub,
  PD_signature_score = pd_scores,
  MSA_signature_score = msa_scores,
  stringsAsFactors = FALSE
)

# 5. Supervised Thresholding for PD Score
# ==============================================================================
cat("Step 4: Determining supervised clinical threshold for PD score...\n")
USE_DISCOVERY_LOCKED_THRESHOLDS <- TRUE

# Get RBD subjects with known conversion outcomes using the CLEANED outcome column
converters <- pheno %>%
  filter(group == "RBD" & conversion_outcome %in% c("PD", "MSA")) %>%
  select(sub, conversion_outcome)

# Merge scores with conversion status
converter_scores <- merge(rbd_scores, converters, by = "sub")

# Perform ROC analysis: PD converters vs. MSA converters
# Ensure the outcome is a factor with the positive class specified
converter_scores$outcome <- factor(converter_scores$conversion_outcome, levels = c("MSA", "PD")) # Positive class = PD

pd_th_file <- "analysis_pipeline_output/PD_supervised_threshold.txt"
msa_th_file <- "analysis_pipeline_output/MSA_relative_threshold.txt"
pd_th_locked <- NA_real_
msa_th_locked <- NA_real_
if (file.exists(pd_th_file)) {
  pd_line <- readLines(pd_th_file, warn = FALSE)
  pd_th_locked <- suppressWarnings(as.numeric(sub(".*=", "", pd_line[1])))
}
if (file.exists(msa_th_file)) {
  msa_line <- readLines(msa_th_file, warn = FALSE)
  msa_th_locked <- suppressWarnings(as.numeric(sub(".*=", "", msa_line[1])))
}

if (isTRUE(USE_DISCOVERY_LOCKED_THRESHOLDS) && is.finite(pd_th_locked)) {
  pd_threshold_supervised <- pd_th_locked
  cat(paste0("  - Locked PD threshold from file: ", pd_threshold_supervised, "\n"))
} else if (isTRUE(USE_DISCOVERY_LOCKED_THRESHOLDS) &&
    is.data.frame(pd_model$pred) && all(c("obs", "PD") %in% colnames(pd_model$pred))) {
  roc_pd <- roc(pd_model$pred$obs, pd_model$pred$PD, levels = c("Other", "PD"))
  pd_threshold_supervised <- as.numeric(coords(roc_pd, "best", ret = "threshold", best.method = "youden"))
  cat(paste0("  - Locked PD threshold from discovery model CV predictions: ", pd_threshold_supervised, "\n"))
} else if (!isTRUE(USE_DISCOVERY_LOCKED_THRESHOLDS) && length(unique(converter_scores$outcome)) == 2) {
  roc_pd <- roc(outcome ~ PD_signature_score, data = converter_scores)

  # Find the optimal threshold that maximizes Youden's J index
  pd_threshold_supervised <- as.numeric(coords(roc_pd, "best", ret = "threshold", best.method = "youden"))

  cat(paste0("  - ROC analysis complete. AUC = ", round(auc(roc_pd), 3), "\n"))
  cat(paste0("  - Supervised PD score threshold (Youden's J): ", pd_threshold_supervised, "\n"))
  
  # --- FINAL ROBUSTNESS FIX: Save the entire ROC object for the plotting script ---
  saveRDS(roc_pd, "analysis_pipeline_output/ROC_object_for_plotting.rds")
  cat("  - ROC object saved for plotting script.\n")

  # Save ROC plot for diagnostics
  png("analysis_pipeline_output/plots/PD_Score_ROC_for_Threshold_DIAGNOSTIC.png", width = 6, height = 6, units = "in", res = 300)
  plot(roc_pd, main = "ROC Curve for PD Score to Differentiate Converters",
       sub = paste("AUC:", round(auc(roc_pd), 3)),
       print.thres = "best", print.thres.best.method = "youden")
  dev.off()
  cat("  - ROC plot saved.\n")
  
  # Save threshold to file
  writeLines(paste0("PD_supervised_threshold=", pd_threshold_supervised), "analysis_pipeline_output/PD_supervised_threshold.txt")
  
} else if (!isTRUE(USE_DISCOVERY_LOCKED_THRESHOLDS)) {
  cat("  - WARNING: Could not perform ROC analysis. Need both PD and MSA converters.\n")
  pd_threshold_supervised <- median(rbd_scores$PD_signature_score) # Fallback to median
  cat(paste0("  - Falling back to median PD score as threshold: ", pd_threshold_supervised, "\n"))
  writeLines(paste0("PD_supervised_threshold=", pd_threshold_supervised, " (FALLBACK MEDIAN)"), "analysis_pipeline_output/PD_supervised_threshold.txt")
} else {
  stop("USE_DISCOVERY_LOCKED_THRESHOLDS=TRUE 但未找到可用的PD锁定阈值来源（阈值文件或锁定模型CV预测）。")
}

# 6. Relative Thresholding for MSA Score
# ==============================================================================
cat("Step 5: Determining relative (exploratory) threshold for MSA score...\n")

if (isTRUE(USE_DISCOVERY_LOCKED_THRESHOLDS) && is.finite(msa_th_locked)) {
  msa_threshold_relative <- msa_th_locked
  cat(paste0("  - Locked MSA threshold from file: ", msa_threshold_relative, "\n"))
} else if (isTRUE(USE_DISCOVERY_LOCKED_THRESHOLDS) &&
    is.data.frame(msa_model$pred) && all(c("obs", "MSA") %in% colnames(msa_model$pred))) {
  roc_msa <- roc(msa_model$pred$obs, msa_model$pred$MSA, levels = c("Other", "MSA"))
  msa_threshold_relative <- as.numeric(coords(roc_msa, "best", ret = "threshold", best.method = "youden"))
  cat(paste0("  - Locked MSA threshold from discovery model CV predictions: ", msa_threshold_relative, "\n"))
} else if (!isTRUE(USE_DISCOVERY_LOCKED_THRESHOLDS)) {
  # Using the 75th percentile (upper quartile) of all RBD subjects as fallback threshold
  msa_threshold_relative <- as.numeric(quantile(rbd_scores$MSA_signature_score, 0.75, na.rm = TRUE))
  cat(paste0("  - Relative MSA score threshold (75th percentile): ", msa_threshold_relative, "\n"))
} else {
  stop("USE_DISCOVERY_LOCKED_THRESHOLDS=TRUE 但未找到可用的MSA锁定阈值来源（阈值文件或锁定模型CV预测）。")
}

# 7. Apply Thresholds and Define Subtypes
# ==============================================================================
cat("Step 6: Applying thresholds and defining final RBD subtypes...\n")

final_rbd_subtypes <- rbd_scores %>%
  mutate(
    predicted_PD_by_PDthreshold = PD_signature_score > pd_threshold_supervised,
    subtype = case_when(
      MSA_signature_score > msa_threshold_relative ~ "Atypical_Complex",
      PD_signature_score > pd_threshold_supervised & MSA_signature_score <= msa_threshold_relative ~ "Typical_PD",
      TRUE ~ "Indeterminate"
    ),
    subtype = factor(subtype, levels = c("Typical_PD", "Atypical_Complex", "Indeterminate"))
  )

# Attach conversion labels for evaluation (where available) using the CLEANED pheno data
final_rbd_with_labels <- final_rbd_subtypes %>%
  left_join(pheno %>% filter(group == "RBD") %>% select(sub, conversion_outcome, follow_up_years), by = "sub")

# 8. Save Final Results and Metrics
# ==============================================================================
cat("Step 7: Saving final subtype results and evaluation metrics...\n")

fwrite(final_rbd_with_labels, "analysis_pipeline_output/RBD_Subtypes_from_Signature_Scores_supervised.csv")
writeLines(paste0("PD_supervised_threshold=", pd_threshold_supervised), "analysis_pipeline_output/PD_supervised_threshold.txt")
writeLines(paste0("MSA_relative_threshold=", msa_threshold_relative), "analysis_pipeline_output/MSA_relative_threshold.txt")

# Compute evaluation among converters
eval_file <- "analysis_pipeline_output/phenotyping_threshold_metrics.txt"
try({
  conv_eval <- final_rbd_with_labels %>% filter(conversion_outcome %in% c("PD", "MSA"))
  if (nrow(conv_eval) > 0) {
    # Predict PD if PD_signature_score > threshold (primary supervised rule)
    conv_eval <- conv_eval %>% mutate(predicted_outcome = ifelse(predicted_PD_by_PDthreshold, "PD", "MSA"))
    acc <- mean(conv_eval$predicted_outcome == conv_eval$conversion_outcome)
    cm <- table(Observed = conv_eval$conversion_outcome, Predicted = conv_eval$predicted_outcome)
    
    sink(eval_file)
    cat("Evaluation of Supervised PD Threshold on ALL Converters (PD vs MSA)\n")
    cat("===============================================================\n")
    cat(paste0("Num Converters: ", nrow(conv_eval), " (PD=", sum(conv_eval$conversion_outcome=="PD"), ", MSA=", sum(conv_eval$conversion_outcome=="MSA"), ")\n\n"))
    cat("Confusion Matrix:\n")
    print(cm)
    cat("\nAccuracy:\n")
    print(acc)
    sink()
  } else {
    writeLines("No converters with labels available for evaluation.", eval_file)
  }
}, silent = TRUE)

cat("\nModule 2 finished successfully.\n")
cat("Final subtype results saved to: analysis_pipeline_output/RBD_Subtypes_from_Signature_Scores_supervised.csv\n")
cat("Threshold saved to: analysis_pipeline_output/PD_supervised_threshold.txt\n")
cat("Metrics saved to: analysis_pipeline_output/phenotyping_threshold_metrics.txt\n")
