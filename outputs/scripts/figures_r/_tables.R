# Shared LaTeX-table helper for the IEEE results: kableExtra booktabs -> (1) a \input-able .tex fragment for the
# paper and (2) a compiled PNG preview (tectonic -> pdftoppm) so the table can be visually verified. Real data
# only; every table script reads a tracked artifact. Run from the repo root.

source("scripts/figures_r/ieee_theme.R")
suppressMessages(library(kableExtra))

TAB_DIR  <- "paper/ieee/tables"
RENV_BIN <- file.path(Sys.getenv("HOME"), "micromamba/envs/renv/bin")

# tabular: a BARE LaTeX tabular string (kbl(format="latex", booktabs=TRUE, ...) with NO kable_styling table
# wrapper — pass only column_spec/pack_rows, so this helper owns the single table environment). size: a LaTeX
# size macro applied around the tabular.
emit_table <- function(tabular, name, caption, label, full_width = FALSE, note = NULL, size = "\\footnotesize") {
  dir.create(TAB_DIR, recursive = TRUE, showWarnings = FALSE)
  env  <- if (full_width) "table*" else "table"
  body <- c(sprintf("{%s", size), tabular, "}")
  frag <- c(sprintf("\\begin{%s}[t]", env), "\\centering",
            sprintf("\\caption{%s}", caption), sprintf("\\label{%s}", label),
            body,
            if (!is.null(note)) sprintf("\\vspace{1pt}\\par{\\footnotesize %s}", note),
            sprintf("\\end{%s}", env))
  writeLines(frag, file.path(TAB_DIR, paste0(name, ".tex")))

  # standalone preview: wrap the sized tabular in a minimal doc, compile, rasterise
  standalone <- c("\\documentclass[border=8pt,varwidth=\\maxdimen]{standalone}",
                  "\\usepackage{booktabs,array,amssymb,textcomp}",
                  "\\begin{document}", body, "\\end{document}")
  bd <- file.path(TAB_DIR, "_build"); dir.create(bd, showWarnings = FALSE)
  sfile <- file.path(bd, paste0(name, ".tex")); writeLines(standalone, sfile)
  ok <- tryCatch(system2(file.path(RENV_BIN, "tectonic"),
                         c("-X", "compile", shQuote(sfile), "--outdir", shQuote(bd), "--keep-logs"),
                         stdout = FALSE, stderr = FALSE), error = function(e) 1L)
  pdf <- file.path(bd, paste0(name, ".pdf"))
  if (file.exists(pdf)) {
    system2(file.path(RENV_BIN, "pdftoppm"),
            c("-png", "-r", "260", shQuote(pdf), shQuote(file.path(TAB_DIR, paste0(name, "_preview")))))
    cat(sprintf("  wrote %s/%s.tex  + %s_preview*.png\n", TAB_DIR, name, name))
  } else {
    cat(sprintf("  wrote %s/%s.tex  (PREVIEW COMPILE FAILED — check %s)\n", TAB_DIR, name, bd))
  }
  invisible(file.path(TAB_DIR, paste0(name, ".tex")))
}
