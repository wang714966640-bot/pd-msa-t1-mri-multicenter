# 以 ggseg 生成 Schaefer-400 四视图脑图（出版风格）
# 依赖：dplyr, ggseg, ggplot2, ggsegSchaefer
# 可选输入：命令行第一个参数为 CSV，需包含列 region, hemi, value；否则将使用随机值演示
# 输出：PD_analyse/MSN Result (Figure)/new_style/ggseg_schaefer400_publication.png

suppressPackageStartupMessages({
  library(dplyr)
  library(ggseg)
  library(ggplot2)
  library(ggsegSchaefer)
  library(readr)
  library(tibble)
  library(patchwork)
  library(scales)
})

# 通过脚本自身路径推断项目根目录（兼容任意工作目录运行）
this_file <- tryCatch({ rstudioapi::getSourceEditorContext()$path }, error = function(e) NA)
if (is.na(this_file) || this_file == "") {
  # 非 RStudio 环境时，fallback 到命令参数或当前工作目录
  args_all <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args_all[grepl("^--file=", args_all)])
  this_file <- if (length(file_arg) > 0) file_arg[1] else getwd()
}
script_dir <- normalizePath(if (dir.exists(this_file)) this_file else dirname(this_file))
# PD_analyse/最终文章使用代码及原始文件/1.code_newstyle/ → 项目根目录为上两级的上一级（共三级）
root_dir <- normalizePath(file.path(script_dir, "..", "..", ".."))
out_dir  <- file.path(root_dir, "MSN Result (Figure)", "new_style")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# 加载 atlas 数据
atlas_tbl <- tibble::as_tibble(ggsegSchaefer::schaefer17_400)

# 小工具：对每个度量使用稳健分位数设色，并使用更“自然”的饱和度范围
plot_one_metric <- function(df_metric, title = NULL, edge_col = "#4a4a4a") {
  # df_metric 需包含列：region, hemi, value
  stopifnot(all(c("region", "hemi", "value") %in% colnames(df_metric)))
  vals <- df_metric$value[is.finite(df_metric$value)]
  cr <- quantile(vals, probs = c(0.05, 0.95), na.rm = TRUE)
  mid <- stats::median(vals, na.rm = TRUE)
  # 避免上下限相等
  if (!is.finite(cr[[1]]) || !is.finite(cr[[2]]) || abs(cr[[2]] - cr[[1]]) < .Machine$double.eps) {
    cr <- range(df_metric$value[is.finite(df_metric$value)], na.rm = TRUE)
    if (!all(is.finite(cr)) || abs(cr[[2]] - cr[[1]]) < .Machine$double.eps) cr <- c(0, 1)
  }
  plot_df <- left_join(atlas_tbl, df_metric, by = c("region", "hemi"))
  ggplot() +
    ggseg::geom_brain(
      atlas       = plot_df,
      mapping     = aes(fill = value),
      position    = ggseg::position_brain(hemi ~ side),
      hemi        = NULL,
      color       = edge_col,
      size        = 0.35,
      show.legend = FALSE
    ) +
    theme_void(base_family = "Arial") +
    # 使用温暖的发文级发散调色：柔和蓝 → 乳白 → 暖橙
    scale_fill_gradient2(
      low = "#8FB0FF",
      mid = "#FAF7F2",
      high = "#E67E5F",
      midpoint = mid,
      limits = cr,
      oob = scales::squish
    ) +
    theme(
      plot.margin = margin(4, 4, 4, 4),
      panel.spacing.x = unit(14, "pt"),
      panel.spacing.y = unit(14, "pt"),
      plot.title = element_text(hjust = 0.5, face = "bold", size = 12)
    ) +
    labs(title = title)
}

# 读取外部数据或使用演示数据
args <- commandArgs(trailingOnly = TRUE)
in_csv  <- if (length(args) >= 1) args[1] else NA
out_png <- if (length(args) >= 2) args[2] else file.path(out_dir, "ggseg_schaefer400_publication.png")
ncol    <- if (length(args) >= 3) suppressWarnings(as.integer(args[3])) else NA
width_i <- if (length(args) >= 4) suppressWarnings(as.numeric(args[4])) else NA
height_i<- if (length(args) >= 5) suppressWarnings(as.numeric(args[5])) else NA
edge_col<- if (length(args) >= 6) args[6] else "#b5b5b5"

if (!is.na(in_csv) && file.exists(in_csv)) {
  message("[ggseg] 读取外部数据: ", in_csv)
  df <- readr::read_csv(in_csv, show_col_types = FALSE)
  # 兼容仅提供 value（和可选 metric）而无 region/hemi 的输入
  has_region <- all(c("region", "hemi") %in% colnames(df))
  has_metric <- "metric" %in% colnames(df)
  if (!has_region) {
    base <- distinct(na.omit(data.frame(region = atlas_tbl$region, hemi = atlas_tbl$hemi)))
    if (has_metric) {
      # 期望每个 metric 对应恰好 400 个值，按 atlas 顺序赋值
      df <- df %>% group_by(metric) %>% mutate(idx = row_number()) %>% ungroup()
      base <- base %>% mutate(idx = row_number())
      df <- left_join(base, df, by = "idx") %>% select(region, hemi, metric, value)
    } else {
      base$value <- df$value
      df <- base
    }
  }
  if (has_metric || ("metric" %in% colnames(df))) {
    metrics <- unique(df$metric)
    plots <- lapply(metrics, function(m) {
      plot_one_metric(df %>% filter(metric == m) %>% select(region, hemi, value), title = as.character(m), edge_col = edge_col)
    })
    ncol_final <- if (is.na(ncol)) ceiling(sqrt(length(plots))) else ncol
    p <- wrap_plots(plots, ncol = ncol_final)
  } else {
    p <- plot_one_metric(df %>% select(region, hemi, value), edge_col = edge_col)
  }
} else {
  message("[ggseg] 未提供外部数据，使用随机值演示（请替换为真实数据）")
  region <- atlas_tbl$region
  hemi   <- atlas_tbl$hemi
  value_df <- distinct(na.omit(data.frame(region = region, hemi = hemi)))
  set.seed(123)
  value_df$value <- scales::rescale(runif(nrow(value_df)), to = c(0, 100))
  p <- plot_one_metric(value_df)
}

width_final  <- ifelse(is.na(width_i), 9.5, width_i)
height_final <- ifelse(is.na(height_i), 6.5, height_i)
ggsave(filename = out_png, plot = p, width = width_final, height = height_final, units = "in", dpi = 600, bg = "white")
message("[ggseg] 已保存: ", out_png)

