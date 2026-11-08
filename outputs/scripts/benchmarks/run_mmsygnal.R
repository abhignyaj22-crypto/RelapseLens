#!/usr/bin/env Rscript
# Score patients with the upstream mmSYGNAL risk models (research benchmark only).
#
# Models are caret `train` (glmnet) objects expecting 141 program-activity columns
# labelled 0..140. Risk = predict(model, newdata, type="prob")$high. Subtype model
# selection follows the upstream grade order: t(4;14)=A, FGFR3=A, amp(1q)/del(13)/
# del(1p)=B, agnostic=C (highest grade the patient exhibits; agnostic otherwise).
#
# Usage: Rscript run_mmsygnal.R <program_activity.csv> <cytogenetics.csv> <out.csv>

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(caret)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript run_mmsygnal.R <program_activity.csv> <cytogenetics.csv> <out.csv>")
}
program_path <- args[[1]]; cyto_path <- args[[2]]; out_path <- args[[3]]
repo <- Sys.getenv("MMSYGNAL_REPO", "external/mmSYGNAL-risk-prediction-models")
model_dir <- file.path(repo, "data")

program <- read_csv(program_path, show_col_types = FALSE)
has_cyto <- file.exists(cyto_path)
cyto <- if (has_cyto) read_csv(cyto_path, show_col_types = FALSE) else NULL

stopifnot("patient_id" %in% names(program))
# normalise program labels: program_0 -> 0, X0 -> 0
names(program) <- gsub("^(program_|X)", "", names(program))
program_cols <- as.character(0:140)
missing_programs <- setdiff(program_cols, names(program))
if (length(missing_programs) > 0) {
  stop(paste("Missing program activity columns (need 0..140):",
             paste(head(missing_programs, 10), collapse = ", "), "..."))
}

models <- list(
  agnostic = readRDS(file.path(model_dir, "agnostic_risk_model.Rds")),
  amp1q    = readRDS(file.path(model_dir, "amp(1q)_risk_model.Rds")),
  del13    = readRDS(file.path(model_dir, "del(13)_risk_model.Rds")),
  del1p    = readRDS(file.path(model_dir, "del(1p)_risk_model.Rds")),
  t_4_14   = readRDS(file.path(model_dir, "t(4;14)_risk_model.Rds")),
  FGFR3    = readRDS(file.path(model_dir, "FGFR3_risk_model.Rds"))
)

newdata <- as.data.frame(program[, program_cols])
score_high <- function(model, x) {
  p <- predict(model, x, type = "prob")
  as.numeric(p$high)
}

scores <- data.frame(patient_id = program$patient_id, stringsAsFactors = FALSE)
for (m in names(models)) {
  scores[[paste0("mmsygnal_", m, "_score")]] <- score_high(models[[m]], newdata)
}

# grade-based subtype selection (A > B > C); agnostic if no relevant subtype.
# Vectorised case_when (NOT apply, which coerces the row to character).
if (!is.null(cyto) && "patient_id" %in% names(cyto)) {
  scores <- scores %>% left_join(cyto, by = "patient_id")
  pos <- function(col) if (col %in% names(scores)) !is.na(scores[[col]]) & scores[[col]] == 1 else rep(FALSE, nrow(scores))
  scores <- scores %>%
    mutate(mmsygnal_selected_score = dplyr::case_when(
      pos("t_4_14") ~ mmsygnal_t_4_14_score,   # grade A
      pos("amp1q")  ~ mmsygnal_amp1q_score,    # grade B
      pos("del13q") ~ mmsygnal_del13_score,    # grade B
      pos("del1p")  ~ mmsygnal_del1p_score,    # grade B
      TRUE          ~ mmsygnal_agnostic_score  # grade C
    ))
} else {
  scores$mmsygnal_selected_score <- scores$mmsygnal_agnostic_score
}
scores$mmsygnal_selected_score <- as.numeric(scores$mmsygnal_selected_score)
scores$mmsygnal_risk_class <- cut(scores$mmsygnal_selected_score,
  breaks = c(-Inf, 0.5, 0.6, Inf), labels = c("low", "high", "extreme"), right = FALSE)

write_csv(scores, out_path)
cat("Wrote", out_path, "with", nrow(scores), "patients\n")
