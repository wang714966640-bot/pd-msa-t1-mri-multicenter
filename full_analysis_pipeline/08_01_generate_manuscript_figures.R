# 13_Manuscript_Figure_Generation.R
# ==============================================================================
# 
# Purpose: FINAL Visualization Module for Manuscript
#          - Generates Figure 1 (Thresholds, ROC, Density, Confusion Matrix)
#          - Generates Figure 2 (Stratification Scatter + Clinical Outcomes)
# 
# Features:
#          - Top-tier aesthetics (Nature/JAMA style)
#          - Corrected data sources and conversion times
#          - Prism-style statistics (Fine mean lines + Error bars)
#          - Strict factor ordering (Quiescent -> MSA -> PD)
#
# Inputs:
# - analysis_pipeline_output/harmonized_pheno.csv
# - analysis_pipeline_output/RBD_Subtypes_from_Signature_Scores_supervised.csv
# - analysis_pipeline_output/plots/Figure2_Data.csv (Critical for conversion times)
# - analysis_pipeline_output/PD_supervised_threshold.txt
#
# Outputs:
# - analysis_pipeline_output/plots/Figure1a...Figure1e.png
# - analysis_pipeline_output/plots/Figure1f_Heatmap_iRBD_Subtype_Profiles.png
# - analysis_pipeline_output/plots/Figure2_Composite_A4_FINAL_CorrectTime.png
#
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(data.table)
  library(pROC)
  library(ggrepel)
  library(patchwork)
  library(grid)
  library(stringr)
  library(pheatmap)
  library(RColorBrewer)
})

# ==============================================================================
# 1. Global Setup & Themes
# ==============================================================================
cat("Step 1: Setting up environment...\n")

# Define Nature-like palette
nat_blue <- "#4C72B0"   # Nature blue
nat_green <- "#55A868"  # Nature green
pd_col <- nat_green
msa_col <- nat_blue
base_sz <- 16

# Figure 2 Specific Colors (Strict Consistency)
col_pd_fig2  <- "#2E8B57"  # Deep Green
col_msa_fig2 <- "#B22222"  # Firebrick Red
col_low_fig2 <- "#778899"  # Light Slate Gray

# Theme for Figure 1
theme_manuscript <- function(base_size = base_sz) {
  theme_classic(base_size = base_size) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = base_size + 2, margin = margin(b = 8)),
    axis.text = element_text(color = "black", size = base_size - 3),
    axis.title = element_text(color = "black", size = base_size - 1, face = "bold"),
    legend.title = element_text(face = "bold", size = base_size - 3),
    legend.text = element_text(size = base_size - 4),
    panel.grid.major = element_line(color = "grey90", linewidth = 0.4),
    panel.grid.minor = element_blank(),
    plot.margin = margin(10, 12, 10, 12)
  )
}

# Directory Setup
out_dir <- "analysis_pipeline_output/plots"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ==============================================================================
# 2. Data Loading (For Figure 1)
# ==============================================================================
cat("Step 2: Loading data for Figure 1...\n")
pheno_all <- fread("analysis_pipeline_output/harmonized_pheno.csv")
rbd_subtypes <- fread("analysis_pipeline_output/RBD_Subtypes_from_Signature_Scores_supervised.csv")

if ("ID" %in% names(pheno_all) && !("sub" %in% names(pheno_all))) setnames(pheno_all, "ID", "sub")
pheno_tbl <- as_tibble(pheno_all)
if ("ID" %in% names(pheno_tbl) && !("sub" %in% names(pheno_tbl))) pheno_tbl <- pheno_tbl %>% rename(sub = ID)
rbd_tbl   <- as_tibble(rbd_subtypes)

# Thresholds
pd_threshold_supervised <- as.numeric(gsub("PD_supervised_threshold=", "", readLines("analysis_pipeline_output/PD_supervised_threshold.txt")))

# Converter Data for Threshold Analysis
converters_data <- pheno_tbl %>%
  mutate(
    conversion_outcome = case_when(
      grepl("PD", conversion_status, ignore.case = TRUE) ~ "PD",
      grepl("MSA", conversion_status, ignore.case = TRUE) ~ "MSA",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(group == "RBD" & conversion_outcome %in% c("PD", "MSA")) %>%
  transmute(sub, conv_out = conversion_outcome) %>%
  left_join(rbd_tbl, by = "sub") %>%
  mutate(outcome_factor = factor(conv_out, levels = c("MSA", "PD")))

# ==============================================================================
# 3. Generate Figure 1 (a-e)
# ==============================================================================
cat("Step 3: Generating Figure 1 components...\n")

# Fig 1a: Density
fig1a <- ggplot(converters_data, aes(x = PD_signature_score, fill = outcome_factor)) +
  geom_density(alpha = 0.55, linewidth = 0) +
  geom_vline(xintercept = pd_threshold_supervised, linetype = "dashed", color = "black", linewidth = 1) +
  scale_fill_manual(values = c("PD" = pd_col, "MSA" = msa_col), name = "True outcome") +
  annotate("label", x = pd_threshold_supervised, y = Inf, label = paste0("Optimal threshold\n", round(pd_threshold_supervised, 2)), vjust = 1.6, hjust = -0.02, size = base_sz/3.4, label.size = 0) +
  labs(title = "PD signature score distribution", x = "PD signature score", y = "Density") +
  theme_manuscript()
ggsave(file.path(out_dir, "Figure1a_Threshold_Density.png"), plot = fig1a, width = 8, height = 6, dpi = 300)

# Fig 1b / 1c: Youden Index and ROC (robust handling if no MSA converters)
# If we have both PD and MSA converters use converters_data; otherwise fall back to model-derived ROC
if (length(unique(converters_data$conv_out)) == 2) {
  roc_obj <- pROC::roc(converters_data$conv_out, converters_data$PD_signature_score, levels = c("MSA", "PD"), quiet = TRUE)
  source_label <- "Converters (PD vs MSA)"
} else {
  # Fallback: derive ROC from PD sentinel model predictions (training PD vs MSA subset)
  pd_model_path <- "analysis_pipeline_output/sentinel_models/PD_sentinel_model.rds"
  if (file.exists(pd_model_path)) {
    pd_model <- readRDS(pd_model_path)
    # Reconstruct training labels from harmonized pheno (robust merging)
    ph <- readr::read_csv("analysis_pipeline_output/harmonized_pheno.csv", show_col_types = FALSE)
    feat <- readr::read_csv("analysis_pipeline_output/harmonized_features.csv", show_col_types = FALSE)
    if ("ID" %in% names(ph)) names(ph)[names(ph) == "ID"] <- "sub"
    if (!("sub" %in% names(feat))) feat$sub <- ph$sub
    data_all <- dplyr::left_join(ph, feat, by = "sub")
    # use pheno group from the pheno side (avoid duplicate columns)
    if ("group.x" %in% names(data_all)) data_all$group <- data_all$group.x else if ("group" %in% names(data_all)) data_all$group <- data_all$group
    train_idx <- which(data_all$group %in% c("PD", "MSA"))
    if (length(train_idx) >= 4) {
      feature_names <- setdiff(names(feat), "sub")
      X_train <- data_all[train_idx, feature_names, drop = FALSE]
      # ensure types
      X_train[] <- lapply(X_train, function(x) if(is.character(x)) as.numeric(as.character(x)) else x)
      probs <- predict(pd_model, newdata = X_train, type = "prob")
      eval_df <- data.frame(outcome = factor(data_all$group[train_idx], levels = c("MSA", "PD")),
                            PD_prob = probs$PD)
      roc_obj <- pROC::roc(eval_df$outcome, eval_df$PD_prob, levels = c("MSA", "PD"), quiet = TRUE)
      source_label <- "PD sentinel model (train PD vs MSA)"
    } else {
      stop("Insufficient PD/MSA cases in training/phenotype files to derive ROC. Cannot plot ROC/Youden reliably.")
    }
  } else {
    stop("No PD sentinel model found and converters do not include both PD and MSA — cannot compute ROC.")
  }
}

# Compute AUC and CI
auc_val <- as.numeric(pROC::auc(roc_obj))
auc_ci <- as.numeric(pROC::ci.auc(roc_obj, conf.level = 0.95))

# Compute Youden curve and best threshold; bootstrap threshold CI for robustness
roc_coords <- pROC::coords(roc_obj, "all", ret = c("threshold", "youden"), transpose = FALSE)
roc_coords_df <- data.frame(threshold = as.numeric(roc_coords["threshold", ]),
                            youden = as.numeric(roc_coords["youden", ]))

best <- try(pROC::coords(roc_obj, "best", ret = c("threshold", "sensitivity", "specificity"), best.method = "youden", transpose = FALSE), silent = TRUE)
if (inherits(best, "try-error")) {
  pd_threshold_plot <- NA
  sens_best <- NA
  spec_best <- NA
} else {
  pd_threshold_plot <- as.numeric(best["threshold"])
  sens_best <- as.numeric(best["sensitivity"])
  spec_best <- as.numeric(best["specificity"])
}

# Bootstrap Youden threshold CI (faster default B=1000)
set.seed(2026)
B <- 1000
th_boot <- numeric(B)
for (i in seq_len(B)) {
  ii <- sample(seq_len(length(roc_obj$cases)), replace = TRUE)
  # sample by indices using original predictors and responses via pROC bootstrap approach
  # pROC has a bootstrap for AUC but threshold bootstrapping is implemented here manually using case sampling
  resp <- roc_obj$case[ii]
  pred <- roc_obj$predictor[ii]
  r <- try(pROC::roc(resp, pred, quiet = TRUE), silent = TRUE)
  if (inherits(r, "try-error")) { th_boot[i] <- NA; next }
  bt <- try(pROC::coords(r, "best", ret = "threshold", best.method = "youden"), silent = TRUE)
  if (inherits(bt, "try-error")) { th_boot[i] <- NA; next }
  th_boot[i] <- as.numeric(bt[1])
}
th_boot <- th_boot[is.finite(th_boot)]
if (length(th_boot) > 0) {
  th_ci <- quantile(th_boot, c(0.025, 0.5, 0.975))
} else {
  th_ci <- c(NA, pd_threshold_supervised, NA)
}

# Fig 1b: improved Youden plot
fig1b <- ggplot(roc_coords_df, aes(x = threshold, y = youden)) +
  geom_line(linewidth = 1.2, color = msa_col) +
  geom_vline(xintercept = pd_threshold_plot, linetype = "dashed", color = "black", linewidth = 0.8) +
  annotate("rect", xmin = th_ci[1], xmax = th_ci[3], ymin = -Inf, ymax = Inf, alpha = 0.18, fill = "grey60") +
  annotate("text", x = pd_threshold_plot, y = max(roc_coords_df$youden, na.rm = TRUE) * 0.95,
           label = paste0("Threshold = ", sprintf('%.3f', pd_threshold_plot), "\nSens=", sprintf('%.2f', sens_best),
                          " Spec=", sprintf('%.2f', spec_best)), hjust = -0.02, size = 4) +
  labs(title = "Youden's J vs Threshold", subtitle = source_label, x = "Threshold", y = "Youden's J") +
  theme_manuscript()
ggsave(file.path(out_dir, "Figure1b_Threshold_Youden_Index.png"), plot = fig1b, width = 8, height = 6, dpi = 300)

# Fig 1c: improved ROC with AUC and CI
fig1c <- pROC::ggroc(roc_obj, colour = msa_col, size = 1.6) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey70") +
  labs(title = "ROC Curve", subtitle = source_label, x = "1 - Specificity", y = "Sensitivity") +
  annotate("text", x = 0.6, y = 0.2, label = paste0("AUC = ", sprintf('%.3f', auc_val), " (95% CI: ", sprintf('%.3f', auc_ci[1]), "-", sprintf('%.3f', auc_ci[3]), ")"), size = 5) +
  theme_manuscript()
ggsave(file.path(out_dir, "Figure1c_PD_Score_ROC_for_Threshold.png"), plot = fig1c, width = 7, height = 7, dpi = 300)

# Fig 1d: PR Curve
if (requireNamespace("precrec", quietly = TRUE)) {
  labels_bin <- as.integer(converters_data$outcome_factor == "PD")
  pr_obj <- precrec::evalmod(scores = converters_data$PD_signature_score, labels = labels_bin)
  fig1d <- autoplot(pr_obj) + theme_manuscript()
  ggsave(file.path(out_dir, "Figure1d_Precision_Recall.png"), plot = fig1d, width = 7, height = 7, dpi = 300)
}

# Fig 1e: Confusion Matrix
conv_pred <- ifelse(converters_data$PD_signature_score > pd_threshold_supervised, "PD", "MSA")
cm_df <- as.data.frame(table(Reference = converters_data$conv_out, Prediction = conv_pred))
fig1e <- ggplot(cm_df, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "black") + geom_text(aes(label = Freq), color = "magenta", size = 6) +
  scale_fill_gradient(low = "white", high = "navyblue") +
  theme_classic(base_size = 16)
ggsave(file.path(out_dir, "Figure1e_CM_Converters.png"), plot = fig1e, width = 7, height = 6, dpi = 300)

# Fig 1f: Heatmap of iRBD Subtype Profiles (TOP20 Features)
cat("Step 3f: Generating Figure 1f (Heatmap of iRBD Subtype Profiles)...\n")

# Load TOP20 features
top20_path <- "features_glmnet/Final_PD_vs_MSA_Publication/TOP20_features_PD_vs_MSA_specific.csv"
if (!file.exists(top20_path)) {
  top20_path <- "RBD_test_TOP20/Part2_Disease_Sentinel_Definition/top_features_master_TOP20.csv"
}
if (!file.exists(top20_path)) stop("TOP20 features file not found!")
top20_features <- read.csv(top20_path, stringsAsFactors = FALSE)$Feature

# Load harmonized features
harmonized_features_path <- "analysis_pipeline_output/harmonized_features.csv"
if (!file.exists(harmonized_features_path)) stop("harmonized_features.csv not found!")
features_raw <- fread(harmonized_features_path)

# Ensure pheno and features are aligned
pheno_ordered <- pheno_tbl
if (!("sub" %in% names(pheno_ordered)) && "ID" %in% names(pheno_ordered)) {
  pheno_ordered <- pheno_ordered %>% rename(sub = ID)
}
if (nrow(features_raw) != nrow(pheno_ordered)) {
  stop("Mismatch: harmonized_features.csv rows (", nrow(features_raw), ") != pheno rows (", nrow(pheno_ordered), ")")
}

# Filter to TOP20 features that exist in data
top20_available <- intersect(top20_features, names(features_raw))
if (length(top20_available) < 15) {
  warning("Only ", length(top20_available), " TOP20 features found in data. Proceeding with available features.")
}
top20_available <- top20_available[seq_len(min(20, length(top20_available)))]

# Calculate Z-scores relative to HC
hc_mask <- pheno_ordered$group == "HC"
if (sum(hc_mask) == 0) stop("No HC subjects found!")

# Convert to data.frame for easier column selection
features_df <- as.data.frame(features_raw)
features_top20 <- features_df[, top20_available, drop = FALSE]

hc_mean <- apply(features_top20[hc_mask, , drop = FALSE], 2, mean, na.rm = TRUE)
hc_sd   <- apply(features_top20[hc_mask, , drop = FALSE], 2, sd, na.rm = TRUE)
hc_sd[hc_sd == 0 | is.na(hc_sd)] <- 1e-6

z_mat <- sweep(features_top20, 2, hc_mean, FUN = "-")
z_mat <- sweep(as.matrix(z_mat), 2, hc_sd, FUN = "/")
z_df <- as.data.frame(z_mat)

# Create group labels with iRBD subtypes
fig2_csv <- file.path(out_dir, "Figure2_Data.csv")
if (!file.exists(fig2_csv)) stop("Figure2_Data.csv not found for subtype mapping!")
fig2_data <- fread(fig2_csv)

# Map RBD to subtypes (ensure PD-like vs MSA-like mapping corresponds to imaging signatures)
rbd_subtype_map <- fig2_data %>%
  mutate(
    group3_display = case_when(
      subtype %in% c("Typical_PD", "RBD-PDSig_High", "PD-Dominant") ~ "PD-like Pattern",
      subtype %in% c("Atypical_Complex", "RBD-MSASig_High", "MSA-like") ~ "MSA-like Pattern",
      TRUE ~ "Indeterminate Pattern"
    )
  ) %>%
  select(sub, group3_display)

pheno_with_subtype <- pheno_ordered %>%
  left_join(rbd_subtype_map, by = "sub") %>%
  mutate(
    display_group = case_when(
      group == "HC" ~ "HC (Ref)",
      group == "PD" ~ "PD (Ref)",
      group == "MSA" ~ "MSA (Ref)",
      group == "RBD" & !is.na(group3_display) ~ group3_display,
      TRUE ~ as.character(group)
    )
  )

z_df$Group <- pheno_with_subtype$display_group

# Aggregate by group (mean Z-score per group per feature)
profile_data <- z_df %>%
  pivot_longer(cols = all_of(top20_available), names_to = "Feature", values_to = "Z_Score") %>%
  group_by(Feature, Group) %>%
  summarise(Z_Score = mean(Z_Score, na.rm = TRUE), .groups = "drop")

# Create heatmap matrix
heatmap_matrix <- profile_data %>%
  pivot_wider(names_from = Group, values_from = Z_Score) %>%
  column_to_rownames(var = "Feature") %>%
  as.matrix()

# Column order: HC (Ref) → PD (Ref) → MSA (Ref) → Indeterminate → MSA-like → PD-like
desired_col_order <- c("HC (Ref)", "PD (Ref)", "MSA (Ref)", 
                       "Indeterminate Pattern", "MSA-like Pattern", "PD-like Pattern")
actual_col_order <- desired_col_order[desired_col_order %in% colnames(heatmap_matrix)]
heatmap_matrix_ordered <- heatmap_matrix[, actual_col_order, drop = FALSE]

# Standard color palette (uniform breaks, professional RdBu scheme)
z_range <- max(abs(heatmap_matrix_ordered), na.rm = TRUE)
z_max <- max(ceiling(z_range), 5)

# Uniform breaks (standard scientific visualization)
z_breaks <- seq(-z_max, z_max, length.out = 101)

# 11-color RdBu palette (balanced red-blue visual weight)
color_palette_heatmap <- colorRampPalette(rev(brewer.pal(n = 11, name = "RdBu")))(100)

# Standard legend breaks (uniform intervals)
legend_breaks_select <- seq(-z_max, z_max, by = 1)
# Filter to reasonable range
legend_breaks_select <- legend_breaks_select[abs(legend_breaks_select) <= z_max]

# Generate heatmap with large fonts (Arial, matching Figure 2 X-axis: 16pt bold)
out_heatmap <- file.path(out_dir, "Figure1f_Heatmap_iRBD_Subtype_Profiles.png")

# Set Arial font family for the plot
png(out_heatmap, width = 12, height = 14, units = "in", res = 600, family = "sans")

pheatmap(heatmap_matrix_ordered,
         color = color_palette_heatmap,
         breaks = z_breaks,
         cluster_rows = TRUE,  # Hierarchical clustering for features
         cluster_cols = FALSE, # Fixed column order (HC → PD → MSA → Indeterminate → MSA-like → PD-like)
         fontsize_row = 19,    # Increased from 16 to 19 (base_size + 3 as requested)
         fontsize_col = 19,    # Increased from 16 to 19 (base_size + 3 as requested)
         fontsize = 19,        # Increased general font size (affects legend)
         main = "",            # Remove title (user will add manually)
         border_color = "grey60",
         angle_col = 45,
         cellwidth = NA,
         cellheight = NA,
         legend_breaks = legend_breaks_select,  # Standard uniform breaks
         legend_labels = legend_breaks_select,  # Standard labels
         display_numbers = FALSE,  # Clean look without numbers
         gaps_col = NULL,
         treeheight_row = 50,
         treeheight_col = 0
)

dev.off()

cat("Figure 1f heatmap saved to:", out_heatmap, "\n")

# ==============================================================================
# 4. Generate Figure 2: The Masterpiece (Integrated from 13c)
# ==============================================================================
cat("Step 4: Generating Final Figure 2 (Scatter + Clinical Outcomes)...\n")

# 4.1 Load Correct Data (with converter_label for time extraction)
fig2_csv <- file.path(out_dir, "Figure2_Data.csv")
if (!file.exists(fig2_csv)) stop("Figure2_Data.csv not found! Ensure pipeline upstream generates it.")
df_fig2 <- fread(fig2_csv) %>% as_tibble()

# 4.2 Thresholds for Visualization
msa_threshold_vis <- df_fig2 %>%
  filter(subtype %in% c("Atypical_Complex", "RBD-MSASig_High", "MSA-like")) %>%
  summarise(v = min(MSA_signature_score, na.rm = TRUE)) %>%
  pull(v)
if(is.infinite(msa_threshold_vis)) msa_threshold_vis <- quantile(df_fig2$MSA_signature_score, 0.75, na.rm=TRUE)

# 4.3 Data Cleaning & Logic Application
df_fig2_clean <- df_fig2 %>%
  mutate(
    # Mapping subtypes to 3 groups
    group3 = case_when(
      subtype %in% c("Typical_PD", "RBD-PDSig_High", "PD-Dominant", "Mixed") ~ "PD-Predominant",
      subtype %in% c("Atypical_Complex", "RBD-MSASig_High", "MSA-like") ~ "MSA-Predominant",
      TRUE ~ "Quiescent Profile"
    ),
    # Strict Ordering: Quiescent -> MSA -> PD
    group3 = factor(group3, levels = c("Quiescent Profile", "MSA-Predominant", "PD-Predominant")),
    
    # Converter Status
    is_converter = !is.na(conversion_outcome) & conversion_outcome %in% c("PD", "MSA"),
    
    # Time Extraction Logic (Crucial Fix)
    time_from_label = suppressWarnings({
      label_str <- as.character(converter_label)
      extracted <- stringr::str_extract(label_str, "[0-9.]+(?=y)")
      as.numeric(extracted)
    }),
    # Fallback to follow_up_years_pheno only if label missing (shouldn't happen for converters)
    time = ifelse(!is.na(time_from_label), time_from_label, suppressWarnings(as.numeric(follow_up_years_pheno)))
  ) %>%
  filter(!is.na(time) & time >= 0)

# 4.4 Aesthetics Configuration
shapes_fig2 <- c("PD-Predominant" = 16, "MSA-Predominant" = 17, "Quiescent Profile" = 18)
cols_fig2   <- c("PD-Predominant" = col_pd_fig2, "MSA-Predominant" = col_msa_fig2, "Quiescent Profile" = col_low_fig2)
font_fam <- "sans"
base_font_size <- 20

# 4.5 Panel A: Stratification Scatter
x_rng <- range(df_fig2_clean$PD_signature_score, na.rm = TRUE)
y_rng <- range(df_fig2_clean$MSA_signature_score, na.rm = TRUE)
x_pad <- 0.05 * diff(x_rng)
y_pad <- 0.05 * diff(y_rng)
x_lim <- c(x_rng[1] - x_pad, x_rng[2] + x_pad)
y_lim <- c(y_rng[1] - y_pad, y_rng[2] + y_pad)

conv_subset <- df_fig2_clean %>% filter(is_converter == TRUE)

pA <- ggplot(df_fig2_clean, aes(x = PD_signature_score, y = MSA_signature_score)) +
  # Background Shading
  annotate("rect", xmin = pd_threshold_supervised, xmax = Inf, ymin = -Inf, ymax = msa_threshold_vis, fill = col_pd_fig2, alpha = 0.06) + 
  annotate("rect", xmin = -Inf, xmax = pd_threshold_supervised, ymin = msa_threshold_vis, ymax = Inf, fill = col_msa_fig2, alpha = 0.06) + 
  annotate("rect", xmin = -Inf, xmax = pd_threshold_supervised, ymin = -Inf, ymax = msa_threshold_vis, fill = col_low_fig2, alpha = 0.06) + 
  # Thresholds
  geom_vline(xintercept = pd_threshold_supervised, linetype = "dashed", color = "grey40", linewidth = 0.8) +
  geom_hline(yintercept = msa_threshold_vis, linetype = "dashed", color = "grey40", linewidth = 0.8) +
  # Points
  geom_point(aes(color = group3, shape = group3), size = 3.2, alpha = 0.85) +
  # Converter Rings
  geom_point(data = conv_subset, shape = 1, size = 4.2, color = "black", stroke = 1.0) +
  # Scales & Theme
  scale_color_manual(values = cols_fig2) +
  scale_shape_manual(values = shapes_fig2) +
  coord_cartesian(xlim = x_lim, ylim = y_lim, expand = FALSE) +
  labs(title = "A. Neuroanatomical Stratification", x = "PD-Specificity Score", y = "MSA-Specificity Score") +
  theme_classic(base_size = base_font_size, base_family = font_fam) +
  theme(
    plot.title = element_text(face = "bold", size = 24),
    axis.title = element_text(face = "bold", size = 20),
    axis.text = element_text(size = 18, color = "black"),
    legend.position = "none",
    plot.margin = margin(10, 15, 10, 10),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1.5)
  )

# 4.6 Panel B: Clinical Outcomes (Prism Style)
plot_df_B <- df_fig2_clean %>% filter(is_converter == TRUE)
y_max_B <- max(plot_df_B$time, na.rm = TRUE) * 1.35

# Stats Calculation
rate_stats <- df_fig2_clean %>%
  group_by(group3) %>%
  summarise(n_tot = n(), n_cnv = sum(is_converter), pct = round(100*n_cnv/n_tot, 0),
            label = paste0("Conv. Rate:\n", pct, "% (", n_cnv, "/", n_tot, ")")) %>% ungroup()

stats_B <- plot_df_B %>%
  group_by(group3) %>%
  summarise(
    n = n(),
    mean_time = mean(time, na.rm = TRUE),
    se_time = ifelse(n >= 2, sd(time, na.rm = TRUE)/sqrt(n), NA_real_)
  ) %>% filter(n >= 1) # Keep all for plotting points, but logic for bars handles n=1

# Filter for bars (n>=2)
stats_B_bars <- stats_B %>% filter(n >= 2)

pB <- ggplot(plot_df_B, aes(x = group3, y = time, color = group3)) +
  # 1. Error Bars (SE)
  geom_errorbar(data = stats_B_bars, aes(x = group3, ymin = mean_time - se_time, ymax = mean_time + se_time),
                inherit.aes = FALSE, width = 0.12, color = "black", linewidth = 0.6) +
  # 2. Mean Line (Slim Horizontal)
  geom_errorbar(data = stats_B_bars, aes(x = group3, ymin = mean_time, ymax = mean_time),
                inherit.aes = FALSE, width = 0.18, color = "black", linewidth = 0.50) +
  # 3. Points
  # NOTE: For professional alignment, do NOT jitter the MSA group when n=1.
  geom_jitter(
    data = dplyr::filter(plot_df_B, group3 != "MSA-Predominant"),
    aes(shape = group3),
    width = 0.2,
    size = 3.5,
    alpha = 0.9
  ) +
  geom_point(
    data = dplyr::filter(plot_df_B, group3 == "MSA-Predominant"),
    aes(shape = group3),
    size = 3.5,
    alpha = 0.9,
    position = position_nudge(x = 0)
  ) +
  # 4. Annotations
  geom_text(data = rate_stats, aes(x = group3, y = y_max_B * 0.95, label = label),
            color = "grey30", size = 5, fontface = "italic", lineheight = 0.9, inherit.aes = FALSE) +
  # Scales
  scale_color_manual(values = cols_fig2) +
  scale_shape_manual(values = shapes_fig2) +
  scale_y_continuous(limits = c(0, y_max_B), expand = c(0, 0), labels = function(x) paste0(x, "y")) +
  scale_x_discrete(limits = c("Quiescent Profile", "MSA-Predominant", "PD-Predominant"),
                  labels = c("Quiescent Profile" = "Indeterminate\nPattern",
                             "MSA-Predominant" = "MSA-like\nPattern",
                             "PD-Predominant" = "PD-like\nPattern")) +
  labs(title = "B. Clinical Progression & Outcomes", x = NULL, y = "Years to Conversion") +
  theme_classic(base_size = base_font_size, base_family = font_fam) +
  theme(
    plot.title = element_text(face = "bold", size = 24),
    axis.title.y = element_text(face = "bold", size = 20),
    axis.text.x = element_text(size = 16, color = "black", face = "bold", lineheight = 0.95),
    axis.text.y = element_text(size = 18, color = "black"),
    legend.position = "none",
    plot.margin = margin(10, 15, 10, 10),
    panel.border = element_blank(),
    axis.line = element_line(color = "black", linewidth = 1.2)
  )

# 4.7 Combine & Save
final_fig2 <- pA / pB + plot_layout(heights = c(1, 0.8))
out_file_2 <- file.path(out_dir, "Figure2_Composite_A4_FINAL_CorrectTime.png")
ggsave(out_file_2, plot = final_fig2, width = 8, height = 11, units = "in", dpi = 600)

cat("Success! Generated Figure 1 series and Final Figure 2 (Correct Time/Order) at:", out_file_2, "\n")
