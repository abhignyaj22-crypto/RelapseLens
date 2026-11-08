# IEEE conference figure/table toolkit (R) — shared theme, column-width helpers, honest-data guards.
#
# Real data only: every figure/table script sources this and reads the tracked result artifacts under
# outputs/ (produced by the autoresearch SCORECARD + make-all pipeline). Nothing here fabricates numbers.
# Run all scripts with the repo root as the working directory.

suppressMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr)
  library(jsonlite); library(scales); library(stringr); library(forcats)
})

# --- IEEE geometry (IEEEtran): single column 3.5in, full width 7.16in; captions ~8pt ---
IEEE_COL_1  <- 3.5
IEEE_COL_2  <- 7.16
IEEE_BASE   <- 8

# Okabe–Ito colorblind-safe palette
IEEE_PALETTE <- c("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#999999")

theme_ieee <- function(base_size = IEEE_BASE, base_family = "serif") {
  theme_bw(base_size = base_size, base_family = base_family) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.25, colour = "grey88"),
      panel.border     = element_rect(linewidth = 0.4, colour = "grey30", fill = NA),
      axis.ticks       = element_line(linewidth = 0.3, colour = "grey30"),
      axis.title       = element_text(size = base_size),
      axis.text        = element_text(size = base_size - 1, colour = "grey15"),
      legend.key.size  = unit(0.85, "lines"),
      legend.title     = element_text(size = base_size - 1),
      legend.text      = element_text(size = base_size - 1),
      legend.background = element_blank(),
      legend.margin    = margin(1, 1, 1, 1),
      plot.title       = element_text(size = base_size, face = "bold"),
      plot.subtitle    = element_text(size = base_size - 1, colour = "grey35"),
      plot.caption     = element_text(size = base_size - 2, colour = "grey40", hjust = 0),
      strip.background = element_rect(fill = "grey92", colour = "grey30", linewidth = 0.4),
      strip.text       = element_text(size = base_size, margin = margin(2, 2, 2, 2))
    )
}

# Save at IEEE column width, 600 dpi — vector PDF (for LaTeX \includegraphics) + PNG preview.
save_ieee <- function(plot, path_noext, width = IEEE_COL_1, height = 2.6) {
  dir.create(dirname(path_noext), recursive = TRUE, showWarnings = FALSE)
  ok_pdf <- tryCatch({ ggsave(paste0(path_noext, ".pdf"), plot, width = width, height = height,
                              units = "in", device = cairo_pdf); TRUE },
                     error = function(e) { ggsave(paste0(path_noext, ".pdf"), plot, width = width,
                                                  height = height, units = "in"); FALSE })
  ggsave(paste0(path_noext, ".png"), plot, width = width, height = height, units = "in", dpi = 600)
  cat(sprintf("  wrote %s.{pdf,png}  (%.2f x %.2f in, cairo_pdf=%s)\n",
              path_noext, width, height, ok_pdf))
  invisible(path_noext)
}

# fail-loud artifact reader — refuses to proceed on a missing real-data file (never silently fabricates).
# JSON path: Python's json.dump emits bare NaN/Infinity (Cox NLL is NaN), which is invalid JSON that jsonlite
# rejects — sanitise those tokens to null before parsing so real numbers are read faithfully.
read_artifact <- function(path, reader = read_csv) {
  if (!file.exists(path)) stop(sprintf("missing real-data artifact: %s (run the pipeline first)", path))
  if (identical(reader, fromJSON)) {
    txt <- paste(readLines(path, warn = FALSE), collapse = "\n")
    txt <- gsub("\\bNaN\\b", "null", txt)
    txt <- gsub("-?\\bInfinity\\b", "null", txt)
    return(fromJSON(txt))
  }
  suppressMessages(reader(path))
}

FIG_DIR <- "paper/ieee/figures"
TAB_DIR <- "paper/ieee/tables"
