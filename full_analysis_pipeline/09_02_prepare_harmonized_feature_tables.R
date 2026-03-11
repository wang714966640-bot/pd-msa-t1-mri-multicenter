# Script 10: Data Preparation and Harmonization
# Purpose: Load all raw data, preprocess, apply ComBat for batch effect
# correction, and save clean, harmonized datasets for downstream analysis.

# --- 0. Setup Environment ---
message("--- Script 10: Data Preparation and Harmonization ---")
message("Step 0: Setting up the environment...")

# Silence package startup messages for cleaner output
suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(reticulate)
  library(sva)
})

# Pin Python for reticulate to avoid automatic Miniconda downloads
if (file.exists("/usr/local/fsl/bin/python")) {
  reticulate::use_python("/usr/local/fsl/bin/python", required = TRUE)
} else if (nzchar(Sys.which("python3"))) {
  reticulate::use_python(Sys.which("python3"), required = TRUE)
}

# --- 1. Define Paths and Create Output Directory ---
base_dir <- "/media/neurox/T7_Shield/PD_analyse"
output_dir <- file.path(base_dir, "analysis_pipeline_output")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Input file paths
pheno_path <- file.path(base_dir, "demo_all.xlsx")
npy_path <- file.path(base_dir, "surface_ALL.npy")
aseg_path <- file.path(base_dir, "asegstats_all.xlsx")

# Output file paths
harmonized_features_path <- file.path(output_dir, "harmonized_features.csv")
harmonized_pheno_path <- file.path(output_dir, "harmonized_pheno.csv")

# --- 2. Load and Preprocess Phenotype Data ---
message("Step 1: Loading and preprocessing phenotype data...")
stopifnot(file.exists(pheno_path))
pheno <- read_excel(pheno_path)

# Ensure required columns exist
required_cols <- c("sub", "group", "site", "age", "gender")
missing_cols <- setdiff(required_cols, colnames(pheno))
if (length(missing_cols) > 0) {
  stop("Phenotype file is missing required columns: ", paste(missing_cols, collapse=", "))
}

# Data cleaning and type conversion
pheno <- pheno %>%
  rename(ID = sub, sex = gender) %>%
  mutate(
    group = factor(group, levels = c("HC", "PD", "MSA", "RBD")),
    site = factor(site),
    sex = factor(sex)
  )

# Handle potential NAs in covariates (critical for ComBat's model matrix)
if (any(is.na(pheno$age))) {
  median_age <- median(pheno$age, na.rm = TRUE)
  pheno$age[is.na(pheno$age)] <- median_age
  message("Imputed ", sum(is.na(pheno$age)), " missing age values with median (", median_age, ").")
}
if (any(is.na(pheno$sex))) {
  stop("Missing values found in 'sex' column. Please remove these subjects or impute.")
}

# Check site counts (ComBat requires >1 subject per site)
site_counts <- table(pheno$site)
message("Site counts:")
print(site_counts)
if (any(site_counts < 2)) {
  warning("Some sites have only one subject. ComBat may be unstable. Consider merging small sites.")
}

# Save the cleaned phenotype data
write.csv(pheno, harmonized_pheno_path, row.names = FALSE)
message("Cleaned phenotype data saved to: ", harmonized_pheno_path)


# --- 3. Load and Preprocess Imaging Data ---
message("Step 2: Loading and preprocessing imaging data...")

# Load cortical features from 3D NumPy array
stopifnot(file.exists(npy_path))
np <- import("numpy")
arr_3d <- np$load(npy_path)

# Extract area (channel index 2 in R -> Python 1) and thickness (3 -> Python 2)
message("Extracting area (channel 2) and thickness (channel 3) from 3D array...")
area_mat  <- t(arr_3d[ , , 2, drop = TRUE])
thick_mat <- t(arr_3d[ , , 3, drop = TRUE])
cortical_features <- as.data.frame(cbind(area_mat, thick_mat))

# Generate cortical feature names with fallback if LUT file is unavailable
annot_path <- "/media/neurox/T7_Shield/R_done/analysis/surfaces/parcellations/lut/lut_schaefer-400_mics.csv"
if (file.exists(annot_path)) {
  annot <- read.csv(annot_path)
  annot <- annot[-c(1, 202), ]
  annot$label <- gsub("7Networks_", "", annot$label)
  roi_labels <- annot$label
} else {
  roi_labels <- paste0("Schaefer400_", seq_len(400))
}
colnames(cortical_features) <- make.unique(paste(roi_labels, rep(c('area', 'thickness'), each = 400), sep = "_"))

# Load subcortical features (numeric columns only)
stopifnot(file.exists(aseg_path))
subcortical_features <- read_excel(aseg_path) %>% select(where(is.numeric))
colnames(subcortical_features) <- paste0("subcort_", colnames(subcortical_features))

# Combine into one matrix and ensure dimensions match
stopifnot(nrow(cortical_features) == nrow(pheno))
stopifnot(nrow(subcortical_features) == nrow(pheno))
features_raw <- cbind(cortical_features, subcortical_features)

# Impute any remaining missing values (e.g., from aseg) using column medians
impute_median <- function(x) {
  if (is.numeric(x)) {
    x[is.na(x)] <- median(x, na.rm = TRUE)
  }
  return(x)
}
features_imputed <- as.data.frame(lapply(features_raw, impute_median))
if (!all(sapply(features_imputed, is.numeric))) {
  stop("Non-numeric columns present in feature matrix; please check inputs.")
}
if (sum(sapply(features_imputed, function(x) sum(is.na(x)))) > 0) {
  stop("Missing values remain after imputation. Please check the data.")
}

message(sprintf("Feature matrix dims (samples x features): %d x %d", nrow(features_imputed), ncol(features_imputed)))

# --- 4. Apply ComBat Harmonization ---
message("Step 3: Applying frozen ComBat harmonization (fit on discovery-train only)...")

stratified_train_idx <- function(labels, p = 0.8, seed = 2026L) {
  set.seed(seed)
  idx <- integer(0)
  lv <- unique(as.character(labels))
  for (g in lv) {
    g_idx <- which(as.character(labels) == g)
    n_g <- length(g_idx)
    if (n_g <= 1) {
      idx <- c(idx, g_idx)
    } else {
      n_take <- max(1L, floor(p * n_g))
      idx <- c(idx, sample(g_idx, size = n_take, replace = FALSE))
    }
  }
  sort(unique(idx))
}

# Freeze harmonizer parameters on discovery train split only.
# If no explicit cohort field exists, fallback to HC/PD/MSA split in current table.
primary_idx <- which(pheno$group %in% c("HC", "PD", "MSA"))
if (length(primary_idx) < 10) {
  stop("Not enough HC/PD/MSA samples to fit frozen ComBat harmonizer.")
}
primary_train_rel <- stratified_train_idx(pheno$group[primary_idx], p = 0.8, seed = 2026L)
train_idx <- primary_idx[primary_train_rel]
apply_idx <- seq_len(nrow(pheno))

features_matrix <- as.matrix(features_imputed)
features_transposed <- t(features_matrix) # features x samples
mod_combat <- model.matrix(~ age + sex, data = pheno)

# Start from raw values and overwrite harmonized columns where valid
harmonized_features_transposed <- features_transposed
train_sites <- unique(as.character(pheno$site[train_idx]))

if (requireNamespace("neuroCombat", quietly = TRUE)) {
  train_fit <- neuroCombat::neuroCombat(
    dat = features_transposed[, train_idx, drop = FALSE],
    batch = as.character(pheno$site[train_idx]),
    mod = mod_combat[train_idx, , drop = FALSE]
  )
  harmonized_features_transposed[, train_idx] <- train_fit$dat.combat

  remaining_idx <- setdiff(apply_idx, train_idx)
  known_site_idx <- remaining_idx[as.character(pheno$site[remaining_idx]) %in% train_sites]
  unknown_site_idx <- setdiff(remaining_idx, known_site_idx)

  if (length(known_site_idx) > 0) {
    applied <- neuroCombat::neuroCombatFromTraining(
      dat = features_transposed[, known_site_idx, drop = FALSE],
      batch = as.character(pheno$site[known_site_idx]),
      mod = mod_combat[known_site_idx, , drop = FALSE],
      estimates = train_fit$estimates
    )
    harmonized_features_transposed[, known_site_idx] <- applied$dat.combat
  }

  # For unseen sites, keep raw values and log explicitly (no refit to avoid leakage)
  if (length(unknown_site_idx) > 0) {
    warning(
      "Frozen ComBat skipped for unseen sites: ",
      paste(unique(as.character(pheno$site[unknown_site_idx])), collapse = ", "),
      ". Raw values retained for these samples."
    )
  }
} else {
  warning("Package neuroCombat not available; frozen ComBat skipped and raw features retained.")
}

# Transpose back: samples x features
harmonized_features <- as.data.frame(t(harmonized_features_transposed))

# --- 5. Save Harmonized Data ---
message("Step 4: Saving harmonized feature matrix...")
write.csv(harmonized_features, harmonized_features_path, row.names = FALSE)
message("Harmonized features saved to: ", harmonized_features_path)
writeLines(
  c(
    "Frozen ComBat policy in this legacy full script copy:",
    "1) Frozen harmonizer is fit on discovery-train split only (stratified HC/PD/MSA, seed=2026).",
    "2) Batch variable: site; protected covariates: age + sex.",
    "3) Remaining samples are transformed via neuroCombatFromTraining without refit.",
    "4) Unseen sites are logged and retained as raw values to avoid leakage/refit."
  ),
  file.path(output_dir, "combat_policy_legacy_full.txt")
)

message("--- Script 10 completed successfully! ---")
