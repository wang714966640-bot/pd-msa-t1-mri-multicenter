rm(list=ls())

suppressPackageStartupMessages({
  library(readxl); library(dplyr); library(tidyr); library(ggplot2)
  library(pheatmap); library(RColorBrewer); library(reticulate)
})

base_dir <- "/media/neurox/T7_Shield/PD_analyse"
setwd(base_dir)

# ---------- 目录与文件 ----------
orig_part3_dir <- file.path(base_dir, "RBD_test_TOP20", "Part3_RBD_Subtyping_AdvancedViz")
stopifnot(dir.exists(orig_part3_dir))
existing_csv <- file.path(orig_part3_dir, "RBD_Subtypes_from_Signature_Scores.csv")
stopifnot(file.exists(existing_csv))

# 新的运行输出目录（不覆盖旧图）
run_tag <- format(Sys.time(), "%Y%m%d_%H%M%S")
out_basedir <- file.path(base_dir, paste0("RBD_test_TOP20_update_", run_tag))
out_part3_dir <- file.path(out_basedir, "Part3_RBD_Subtyping_AdvancedViz")
dir.create(out_part3_dir, recursive = TRUE, showWarnings = FALSE)

# ---------- 读取全量样本信息 ----------
demo_all <- read_excel(file.path(base_dir, "demo_all.xlsx"))
demo_all$group <- as.character(demo_all$group)
old_res <- read.csv(existing_csv, stringsAsFactors = FALSE)
# 统一ID策略：若旧结果ID大多为 Subject_数字，则按行序生成以保持兼容；否则优先使用 sub 列
use_subject_index_id <- mean(grepl("^Subject_[0-9]+$", old_res$ID)) > 0.5
if (!"ID" %in% colnames(demo_all)) {
  if (use_subject_index_id) {
    demo_all$ID <- paste0("Subject_", seq_len(nrow(demo_all)))
    message("已按行序生成 ID = Subject_#，以与既有结果兼容")
  } else if ("sub" %in% colnames(demo_all)) {
    demo_all$ID <- as.character(demo_all$sub)
    message("已使用 demo_all$sub 作为 ID 列")
  } else {
    warning("demo_all 中无 ID 与 sub，按顺序生成: Subject_1..N")
    demo_all$ID <- paste0("Subject_", seq_len(nrow(demo_all)))
  }
}

# 找到新增RBD（仅追加这些）
all_rbd_ids <- demo_all %>% filter(group == "RBD") %>% pull(ID)
new_ids <- setdiff(all_rbd_ids, old_res$ID)
if (length(new_ids) == 0L) {
  message("没有需要追加的新RBD样本。")
  quit(save="no")
}
cat("检测到需要追加的新RBD样本数:", length(new_ids), "\n")

# ---------- 组合成像特征（与主脚本一致） ----------
# 1) 皮层：仅使用 surface_ALL.npy（已包含所有被试）
npy_path <- file.path(base_dir, "surface_ALL.npy")
stopifnot(file.exists(npy_path))
try({ use_python("/usr/local/fsl/bin/python", required = FALSE) }, silent = TRUE)
np <- import("numpy")
data_all_npy <- np$load(npy_path)
arr <- as.array(data_all_npy)
dims <- dim(arr)
# 兼容旧逻辑：若为二维且800列，直接使用；若为三维(400, N, 5)，按8.features_glmnet_all.R与9.1的口径取area+thickness两通道并展开为N×800
if (length(dims) == 2) {
  if (dims[1] == nrow(demo_all) && dims[2] == 800) {
    mat2d <- arr
  } else if (dims[2] == nrow(demo_all) && dims[1] == 800) {
    mat2d <- t(arr)
  } else {
    stop("surface_ALL.npy 为二维但维度不匹配: ", paste(dims, collapse='x'), " vs n=", nrow(demo_all))
  }
} else if (length(dims) == 3) {
  if (dims[1] != 400) stop("surface_ALL.npy 第一维应为400个parcel，实际=", dims[1])
  if (dims[3] < 3) stop("surface_ALL.npy 第三维应至少包含 area 与 thickness 两个通道，实际=", dims[3])
  # R索引：1=volume, 2=area, 3=thickness
  area_mat <- t(arr[, , 2, drop = TRUE])       # N×400
  thick_mat <- t(arr[, , 3, drop = TRUE])      # N×400
  if (nrow(area_mat) != nrow(demo_all)) {
    stop("surface_ALL.npy 的被试数与 demo_all 不一致: ", nrow(area_mat), " vs ", nrow(demo_all))
  }
  mat2d <- cbind(area_mat, thick_mat)          # N×800（按列拼接）
} else {
  stop("surface_ALL.npy 维度异常: ", paste(dims, collapse='x'))
}
struc_measures_cortical_all <- as.data.frame(mat2d)
rm(data_all_npy, arr, mat2d, dims)

annot <- read.csv("/media/neurox/T7_Shield/R_done/analysis/surfaces/parcellations/lut/lut_schaefer-400_mics.csv")
annot <- annot[-c(1,202), ]
annot$label <- gsub("7Networks_", "", annot$label)
roi_labels <- annot$label
struc_labels <- rep(c("area","thickness"), each=400)
if (ncol(struc_measures_cortical_all) != 800) stop("皮层特征数不是800，请检查 surface_ALL.npy")
colnames(struc_measures_cortical_all) <- make.unique(paste(roi_labels, struc_labels, sep="_"))

# 2) 皮层下
aseg_all <- read_excel(file.path(base_dir, "asegstats_all.xlsx"), col_names = TRUE)
stopifnot(nrow(aseg_all) == nrow(demo_all))
colnames(aseg_all) <- paste0("subcort_", colnames(aseg_all))

# 3) 合并 + 中位数填补
combined_measures_all <- cbind(struc_measures_cortical_all, aseg_all)
rm(struc_measures_cortical_all, aseg_all)
if (any(duplicated(colnames(combined_measures_all))))
  colnames(combined_measures_all) <- make.unique(colnames(combined_measures_all))
if (sum(is.na(combined_measures_all)) > 0) {
  for (i in seq_len(ncol(combined_measures_all))) {
    if (is.numeric(combined_measures_all[[i]])) {
      med <- median(combined_measures_all[[i]], na.rm=TRUE)
      combined_measures_all[is.na(combined_measures_all[[i]]), i] <- med
    }
  }
}
stopifnot(sum(is.na(combined_measures_all)) == 0)

# ---------- 恢复训练(仅HC/PD/MSA)的标准化 ----------
train_mask <- demo_all$group %in% c("HC","PD","MSA")
X_train_all <- combined_measures_all[train_mask, , drop=FALSE]
mu <- suppressWarnings(colMeans(X_train_all, na.rm=TRUE))
sdv <- apply(X_train_all, 2, sd, na.rm=TRUE); sdv[is.na(sdv) | sdv==0] <- 1e-6

# ---------- 载入已保存的哨兵系数（不重训） ----------
pd_sent_path  <- file.path(base_dir, "RBD_test_TOP20/Part2_Disease_Sentinel_Definition/sentinels_PD_vs_HCMSA_abs_sorted.csv")
msa_sent_path <- file.path(base_dir, "RBD_test_TOP20/Part2_Disease_Sentinel_Definition/sentinels_MSA_vs_HCPD_abs_sorted.csv")
stopifnot(file.exists(pd_sent_path), file.exists(msa_sent_path))
pd_sent  <- read.csv(pd_sent_path, stringsAsFactors = FALSE)
msa_sent <- read.csv(msa_sent_path, stringsAsFactors = FALSE)

pd_core  <- head(pd_sent, 10)
msa_core <- head(msa_sent, 10)
top20_features <- unique(c(pd_core$Feature, msa_core$Feature))
top20_features <- top20_features[top20_features %in% colnames(combined_measures_all)]
stopifnot(length(top20_features) >= 1)

# ---------- 计算新RBD的签名分数与分型 ----------
new_mask <- demo_all$ID %in% new_ids
X_new <- combined_measures_all[new_mask, top20_features, drop=FALSE]
ids_new <- demo_all$ID[new_mask]

Z_new <- sweep(X_new, 2, mu[top20_features], "-")
Z_new <- sweep(Z_new, 2, sdv[top20_features], "/")

pd_coef_all  <- setNames(pd_sent$Coefficient,  pd_sent$Feature)
msa_coef_all <- setNames(msa_sent$Coefficient, msa_sent$Feature)
pd_coef_vec  <- pd_coef_all[top20_features];  pd_coef_vec[is.na(pd_coef_vec)]   <- 0
msa_coef_vec <- msa_coef_all[top20_features]; msa_coef_vec[is.na(msa_coef_vec)] <- 0

pd_score_new  <- as.numeric(as.matrix(Z_new) %*% pd_coef_vec)
msa_score_new <- as.numeric(as.matrix(Z_new) %*% msa_coef_vec)

# 阈值优先使用锁定阈值文件；缺失时回退到旧结果中位数
pd_lock_path <- file.path(base_dir, "features_glmnet", "Final_PD_vs_MSA_Publication", "TOP20_fixed_threshold.csv")
if (file.exists(pd_lock_path)) {
  pd_th_df <- read.csv(pd_lock_path, stringsAsFactors = FALSE)
  pd_threshold <- suppressWarnings(as.numeric(pd_th_df$Threshold[1]))
} else {
  pd_threshold <- NA_real_
}
if (!is.finite(pd_threshold)) pd_threshold <- median(old_res$PD_Signature_Score, na.rm = TRUE)

msa_lock_path <- file.path(base_dir, "analysis_pipeline_output", "MSA_relative_threshold.txt")
if (file.exists(msa_lock_path)) {
  msa_line <- readLines(msa_lock_path, warn = FALSE)
  msa_threshold <- suppressWarnings(as.numeric(sub(".*=", "", msa_line[1])))
} else {
  msa_threshold <- NA_real_
}
if (!is.finite(msa_threshold)) msa_threshold <- median(old_res$MSA_Signature_Score, na.rm = TRUE)

Subtype_new <- ifelse(pd_score_new > pd_threshold & msa_score_new <= msa_threshold, "RBD-PDsig_High",
                ifelse(msa_score_new > msa_threshold & pd_score_new <= pd_threshold, "RBD-MSAsig_High",
                ifelse(pd_score_new > pd_threshold & msa_score_new > msa_threshold,  "RBD-MixedSig_High",
                       "RBD-LowSig_Profile")))

new_rows <- data.frame(
  ID = ids_new,
  PD_Signature_Score = pd_score_new,
  MSA_Signature_Score = msa_score_new,
  Subtype = Subtype_new,
  stringsAsFactors = FALSE
)

# ---------- 备份旧CSV并写入新结果（仅追加新样本） ----------
backup_dir <- file.path(orig_part3_dir, paste0("backup_before_append_", run_tag))
dir.create(backup_dir, showWarnings = FALSE)
file.copy(existing_csv, file.path(backup_dir, basename(existing_csv)), overwrite = TRUE)

out_final <- old_res %>% bind_rows(anti_join(new_rows, old_res, by = "ID"))
write.csv(out_final, existing_csv, row.names = FALSE)
cat("已追加", nrow(new_rows), "条记录到原文件：", existing_csv, "\n", sep="")

# 也把新的总表另存到本次输出目录
write.csv(out_final, file.path(out_part3_dir, "RBD_Subtypes_from_Signature_Scores_updated.csv"), row.names = FALSE)

# ---------- 生成热图（与主文风格一致；输出到本次目录） ----------
hc_mask <- demo_all$group == "HC"
hc_mean <- apply(combined_measures_all[hc_mask, top20_features, drop = FALSE], 2, mean, na.rm = TRUE)
hc_sd   <- apply(combined_measures_all[hc_mask, top20_features, drop = FALSE], 2, sd,   na.rm = TRUE)
hc_sd[hc_sd == 0 | is.na(hc_sd)] <- 1e-6
z_mat <- sweep(combined_measures_all[, top20_features, drop = FALSE], 2, hc_mean, "-")
z_mat <- sweep(z_mat, 2, hc_sd, "/")

# 显示分组：HC/PD/MSA保留为Ref；RBD映射为最新分型
display_group <- as.character(demo_all$group)
display_group[display_group == "HC"]  <- "HC (Ref)"
display_group[display_group == "PD"]  <- "PD (Ref)"
display_group[display_group == "MSA"] <- "MSA (Ref)"
rbd_map <- out_final[, c("ID","Subtype")]
demo_tmp <- demo_all[, c("ID","group")]
demo_tmp <- merge(demo_tmp, rbd_map, by = "ID", all.x = TRUE)
idx_rbd <- which(demo_all$group == "RBD")
display_group[idx_rbd] <- as.character(demo_tmp$Subtype[idx_rbd])

z_df <- as.data.frame(z_mat); z_df$Group <- display_group
z_long <- z_df %>% tidyr::pivot_longer(cols = all_of(top20_features), names_to = "Feature", values_to = "Z_Score")
profile_data <- z_long %>% dplyr::group_by(Feature, Group) %>% dplyr::summarise(Z_Score = mean(Z_Score, na.rm = TRUE), .groups = "drop")

max_val <- max(abs(profile_data$Z_Score), na.rm = TRUE)
heatmap_matrix <- profile_data %>%
  dplyr::filter(Feature %in% top20_features) %>%
  tidyr::pivot_wider(names_from = Group, values_from = Z_Score) %>%
  tibble::column_to_rownames("Feature") %>% as.matrix()

if (nrow(heatmap_matrix) > 0 && ncol(heatmap_matrix) > 0) {
  desired_col_order <- c("HC (Ref)", "PD (Ref)", "MSA (Ref)",
                         "RBD-LowSig_Profile", "RBD-PDsig_High", "RBD-MSAsig_High", "RBD-MixedSig_High")
  actual_col_order <- desired_col_order[desired_col_order %in% colnames(heatmap_matrix)]
  heatmap_matrix <- heatmap_matrix[, actual_col_order, drop=FALSE]
  brks <- seq(-max_val, max_val, length.out=101)
  pal  <- colorRampPalette(rev(brewer.pal(7,"RdBu")))(100)
  pheatmap(heatmap_matrix, color = pal, breaks = brks,
           cluster_rows = TRUE, cluster_cols = FALSE,
           fontsize_row = 8, fontsize_col = 10,
           main = "Heatmap of imaging profiles across core sentinel features\n(Z-scores relative to HC)",
           border_color = "grey80", angle_col = 45,
           filename = file.path(out_part3_dir, "Heatmap_RBD_subtype_profiles_updated.png"),
           width = 10, height = max(8, nrow(heatmap_matrix) * 0.25))
  cat("热图已保存：", file.path(out_part3_dir, "Heatmap_RBD_subtype_profiles_updated.png"), "\n", sep="")
}

# ---------- 生成散点图（RBD分型分布；虚线为旧中位数阈值） ----------
scatter_df <- out_final
scatter_df$IsNew <- ifelse(scatter_df$ID %in% new_ids, "New 20", "Existing")

p_scatter <- ggplot(scatter_df, aes(x = PD_Signature_Score, y = MSA_Signature_Score,
                                    color = Subtype, shape = IsNew)) +
  geom_hline(yintercept = msa_threshold, linetype = "dashed", color = "grey40") +
  geom_vline(xintercept = pd_threshold,  linetype = "dashed", color = "grey40") +
  geom_point(size = 3, alpha = 0.85) +
  scale_shape_manual(values = c("Existing" = 16, "New 20" = 21)) +
  scale_color_manual(values = c("RBD-PDsig_High"="#1f77b4","RBD-MSAsig_High"="#ff7f0e",
                                "RBD-MixedSig_High"="#2ca02c","RBD-LowSig_Profile"="#d62728")) +
  labs(title = "Distribution of PD and MSA disease-signature scores in RBD (updated)",
       x = "PD disease-signature score (higher = more PD-like)",
       y = "MSA disease-signature score (higher = more MSA-like)",
       color = "RBD imaging phenotype", shape = NULL) +
  theme_classic(base_size = 12)

ggsave(file.path(out_part3_dir, "Scatter_PD_MSA_signature_RBD_updated.png"),
       p_scatter, width = 9, height = 7, dpi = 300)
cat("散点图已保存：", file.path(out_part3_dir, "Scatter_PD_MSA_signature_RBD_updated.png"), "\n", sep="")

cat("\n完成。本次输出目录：", out_part3_dir, "\n", sep="")

 