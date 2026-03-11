# 复现脚本（ggseg 路线）：基于 Schaefer-400 生成出版风脑图（带色条）
#
# 设计目标：
# - 不覆盖旧结果：输出到全新文件夹
# - 两类图：
#   1) local measures（8项：Clustering/Efficiency/Degree/Betweenness × FC/SC）
#   2) Area/Thickness（来自 ggseg 输入 CSV；推荐先跑 Python 复现脚本生成）
#
# 输入来源（两种任选其一）：
# A. 直接读 PD_analyse/FC_SC_MatPlots/*.csv（计算列均值） -> local measures
# B. 读 Python 导出的 ggseg_input_*.csv（更稳定）
#
# 用法示例：
# Rscript 7.ggseg_schaefer400_publication_reproduce.R
# Rscript 7.ggseg_schaefer400_publication_reproduce.R \
#   --local_csv "/path/to/ggseg_input_local_measures.csv" \
#   --at_csv "/path/to/ggseg_input_area_thickness.csv" \
#   --out_dir "/path/to/output"

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

infer_pd_analyse_dir <- function() {
  this_file <- tryCatch({ rstudioapi::getSourceEditorContext()$path }, error = function(e) NA)
  if (is.na(this_file) || this_file == "") {
    args_all <- commandArgs(trailingOnly = FALSE)
    file_arg <- sub("^--file=", "", args_all[grepl("^--file=", args_all)])
    this_file <- if (length(file_arg) > 0) file_arg[1] else getwd()
  }
  script_dir <- normalizePath(if (dir.exists(this_file)) this_file else dirname(this_file))
  # .../PD_analyse/最终文章使用代码及原始文件/1.code/
  normalizePath(file.path(script_dir, "..", ".."))
}

args <- commandArgs(trailingOnly = TRUE)
# 解析简单参数：--key value
parse_args <- function(args) {
  out <- list()
  i <- 1
  while (i <= length(args)) {
    k <- args[[i]]
    if (startsWith(k, "--") && i < length(args)) {
      out[[sub("^--", "", k)]] <- args[[i + 1]]
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  out
}
opt <- parse_args(args)

pd_analyse_dir <- if (!is.null(opt$pd_analyse_dir)) normalizePath(opt$pd_analyse_dir) else infer_pd_analyse_dir()

# 输出目录：默认新建到 MSN Result (Figure)/reproduce_<timestamp>/ggseg
if (!is.null(opt$out_dir)) {
  # out_dir 可能尚未存在，normalizePath 会警告；这里直接使用原始路径
  out_dir <- opt$out_dir
} else {
  ts <- format(Sys.time(), "%Y%m%d_%H%M%S")
  out_dir <- file.path(pd_analyse_dir, "demo_1126_orig", "MSN Result (Figure)", paste0("reproduce_", ts), "ggseg")
}
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

message("[ggseg reproduce] pd_analyse_dir: ", pd_analyse_dir)
message("[ggseg reproduce] out_dir: ", out_dir)

atlas_tbl <- tibble::as_tibble(ggsegSchaefer::schaefer17_400)
base_regions <- atlas_tbl %>% distinct(region, hemi) %>% mutate(idx = row_number())

# 统一出版风配色（与你现有 7.ggseg 脚本保持一致）
make_scale <- function(vals) {
  vals <- vals[is.finite(vals)]
  cr <- quantile(vals, probs = c(0.05, 0.95), na.rm = TRUE)
  mid <- stats::median(vals, na.rm = TRUE)
  if (!is.finite(cr[[1]]) || !is.finite(cr[[2]]) || abs(cr[[2]] - cr[[1]]) < .Machine$double.eps) {
    cr <- range(vals, na.rm = TRUE)
    if (!all(is.finite(cr)) || abs(cr[[2]] - cr[[1]]) < .Machine$double.eps) cr <- c(0, 1)
  }
  list(cr = cr, mid = mid)
}

plot_one_metric <- function(df_metric, title = NULL, edge_col = "#b5b5b5") {
  stopifnot(all(c("region", "hemi", "value") %in% colnames(df_metric)))
  sc <- make_scale(df_metric$value)
  plot_df <- left_join(atlas_tbl, df_metric, by = c("region", "hemi"))

  ggplot() +
    ggseg::geom_brain(
      atlas       = plot_df,
      mapping     = aes(fill = value),
      position    = ggseg::position_brain(hemi ~ side),
      hemi        = NULL,
      color       = edge_col,
      size        = 0.35,
      show.legend = TRUE
    ) +
    theme_void(base_family = "Arial") +
    scale_fill_gradient2(
      low = "#8FB0FF",
      mid = "#FAF7F2",
      high = "#E67E5F",
      midpoint = sc$mid,
      limits = sc$cr,
      oob = scales::squish,
      # 色条保留刻度数字，但不显示“指标名”(legend title)
      name = NULL
    ) +
    guides(fill = guide_colourbar(barheight = unit(44, "mm"), barwidth = unit(3.3, "mm"))) +
    theme(
      # 增大边距与面板间距，避免多图拼版时互相压住/重叠
      plot.margin = margin(6, 10, 6, 6),
      plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
      legend.title = element_blank(),
      legend.text = element_text(size = 9, color = "#333333"),
      legend.margin = margin(0, 0, 0, 0)
    ) +
    labs(title = title)
}

# 读取 local measures（优先 local_csv，否则从 FC_SC_MatPlots 计算）
read_local_measures_long <- function() {
  if (!is.null(opt$local_csv) && file.exists(opt$local_csv)) {
    message("[ggseg reproduce] 读取 local_csv: ", opt$local_csv)
    df <- readr::read_csv(opt$local_csv, show_col_types = FALSE)
    stopifnot(all(c("metric", "idx", "value") %in% colnames(df)))
    return(df)
  }
  fc_sc_dir <- file.path(pd_analyse_dir, "FC_SC_MatPlots")
  files <- c(
    clustering_fc = "clustering_fc.csv",
    clustering_sc = "clustering_sc.csv",
    efficiency_fc = "local_efficiency_fc.csv",
    efficiency_sc = "local_efficiency_sc.csv",
    degree_fc = "degree_fc.csv",
    degree_sc = "degree_sc.csv",
    betweenness_fc = "betweenness_fc.csv",
    betweenness_sc = "betweenness_sc.csv"
  )
  titles <- c(
    clustering_fc = "Clustering FC",
    clustering_sc = "Clustering SC",
    efficiency_fc = "Efficiency FC",
    efficiency_sc = "Efficiency SC",
    degree_fc = "Degree FC",
    degree_sc = "Degree SC",
    betweenness_fc = "Betweenness FC",
    betweenness_sc = "Betweenness SC"
  )
  rows <- list()
  for (k in names(files)) {
    fp <- file.path(fc_sc_dir, files[[k]])
    if (!file.exists(fp)) stop("缺少输入文件: ", fp)
    m <- as.matrix(readr::read_csv(fp, col_names = FALSE, show_col_types = FALSE))
    v <- colMeans(m)
    rows[[k]] <- tibble(metric = titles[[k]], idx = seq_along(v), value = as.numeric(v))
  }
  bind_rows(rows)
}

# 将 idx 映射到 ggseg atlas 的 region/hemi
attach_region_hemi <- function(df_long) {
  df_long %>%
    left_join(base_regions, by = "idx") %>%
    select(metric, region, hemi, value)
}

# 1) local measures 生成：网格图（每个 panel 自带色条会很拥挤，所以：网格图不收集 legend；另外额外输出单图版）
local_long <- read_local_measures_long()
local_mapped <- attach_region_hemi(local_long)
local_metrics <- unique(local_mapped$metric)

plots_local_single <- lapply(local_metrics, function(m) {
  plot_one_metric(local_mapped %>% filter(metric == m) %>% select(region, hemi, value), title = m)
})

# 网格版（不强制合并 legend，避免不同指标范围导致误导）
local_grid <- wrap_plots(plots_local_single, ncol = 4) +
  plot_layout(guides = "keep") &
  theme(panel.spacing = unit(10, "pt"))

ggsave(
  filename = file.path(out_dir, "local_measures_ggseg_grid.png"),
  plot = local_grid,
  # 保持 2 行布局，但给色条和标题留足空间，避免重叠
  width = 13.6,
  height = 7.6,
  units = "in",
  dpi = 600,
  bg = "white"
)

# 单图版（每个指标 1 张，带色条）
local_single_dir <- file.path(out_dir, "local_measures_single")
if (!dir.exists(local_single_dir)) dir.create(local_single_dir, recursive = TRUE, showWarnings = FALSE)
for (i in seq_along(local_metrics)) {
  m <- local_metrics[[i]]
  ggsave(
    filename = file.path(local_single_dir, paste0(gsub("[^A-Za-z0-9]+", "_", m), ".png")),
    plot = plots_local_single[[i]],
    width = 4.2,
    height = 3.2,
    units = "in",
    dpi = 600,
    bg = "white"
  )
}

message("[ggseg reproduce] 已输出 local measures: ", out_dir)

# 2) Area/Thickness（优先 at_csv；否则跳过，因为 R 端不直接读 npy）
if (!is.null(opt$at_csv) && file.exists(opt$at_csv)) {
  message("[ggseg reproduce] 读取 at_csv: ", opt$at_csv)
  at_long <- readr::read_csv(opt$at_csv, show_col_types = FALSE)
  stopifnot(all(c("metric", "idx", "value") %in% colnames(at_long)))

  at_mapped <- attach_region_hemi(at_long)
  at_metrics <- unique(at_mapped$metric)
  plots_at <- lapply(at_metrics, function(m) {
    plot_one_metric(at_mapped %>% filter(metric == m) %>% select(region, hemi, value), title = m)
  })

  at_grid <- wrap_plots(plots_at, ncol = 2)
  ggsave(
    filename = file.path(out_dir, "area_thickness_ggseg_grid.png"),
    plot = at_grid,
    width = 8.4,
    height = 3.8,
    units = "in",
    dpi = 600,
    bg = "white"
  )
  message("[ggseg reproduce] 已输出 Area/Thickness")
} else {
  message("[ggseg reproduce] 未提供 --at_csv（或文件不存在），已跳过 Area/Thickness ggseg 输出。")
  message("[ggseg reproduce] 你可以先运行 Python 复现脚本生成 ggseg_input_area_thickness.csv")
}

message("[ggseg reproduce] 全部完成：请查看输出目录：", out_dir)







