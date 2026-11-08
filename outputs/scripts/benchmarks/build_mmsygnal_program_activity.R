#!/usr/bin/env Rscript

# Build mmSYGNAL-compatible 141-program activity (labels 0..140) from expression.
#
# STATUS: SCAFFOLD — FAILS BY DESIGN. The exact upstream program-activity scoring
# is NOT reproducible from the cloned risk-models repo:
#   * `generateProgramActivity` is referenced in code/utilities.R comments but is
#     NOT defined anywhere in the repo;
#   * utilities.R READS pre-computed program/regulon activity from an external
#     `MINER/` path (the miner3 Python pipeline output);
#   * README step 2 directs to "apply mmSYGNAL ... See Wall et al, Precision
#     Oncology 2021" to GENERATE program activity.
#
# mmSYGNAL/MINER program activity uses sample-level over/under-expression
# membership of each regulon (miner3), with the cohort-specific "minernorm"
# normalisation the risk models were trained against. Re-deriving it heuristically
# would produce OUT-OF-DISTRIBUTION inputs for the glmnet models and an invalid,
# misleading comparison. Therefore this script FAILS rather than approximate.
#
# To produce a valid matrix, run the upstream miner3/mmSYGNAL pipeline on the RNA
# (regulons.json + transcriptional_programs.json are shipped as the canonical
# definitions) and emit data/real/mmsygnal_program_activity_0_140.csv.

suppressPackageStartupMessages({
  library(readr); library(jsonlite); library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Usage: Rscript build_mmsygnal_program_activity.R <expr_log2_tpm.csv> <regulons.json> <transcriptional_programs.json> <out.csv>")
}
expr_path <- args[[1]]; regulons_path <- args[[2]]
programs_path <- args[[3]]; out_path <- args[[4]]

expr <- read_csv(expr_path, show_col_types = FALSE)
regulons <- fromJSON(regulons_path, simplifyVector = FALSE)   # regulon_id -> [ENSG...]
programs <- fromJSON(programs_path, simplifyVector = FALSE)    # program 0..140 -> [regulon_id...]

if (!"gene" %in% names(expr)) {
  stop("Expression must have a 'gene' column (ENSEMBL ids) and patient/sample columns.")
}

cat(sprintf("Loaded %d regulons, %d programs, expression %d genes x %d samples.\n",
            length(regulons), length(programs), nrow(expr), ncol(expr) - 1))

# DO NOT approximate silently. Exact upstream MINER/mmSYGNAL scoring (miner3
# over/under-expression membership + minernorm) is required and is not wired here.
stop(paste(
  "BLOCKED: exact mmSYGNAL/MINER program-activity scoring is not reproduced in",
  "this repo (generateProgramActivity is undefined here). Generate program",
  "activity with the upstream miner3/mmSYGNAL pipeline (Wall et al. 2021) and",
  "save it as the out.csv path. Refusing to emit approximate activity."))
