#!/usr/bin/env Rscript
# Verify the mmSYGNAL .Rds risk models are present and loadable WITHOUT
# modifying or redistributing them.

repo <- Sys.getenv("MMSYGNAL_REPO", "external/mmSYGNAL-risk-prediction-models")
model_dir <- file.path(repo, "data")

required <- c(
  "agnostic_risk_model.Rds",
  "amp(1q)_risk_model.Rds",
  "del(13)_risk_model.Rds",
  "del(1p)_risk_model.Rds",
  "t(4;14)_risk_model.Rds",
  "FGFR3_risk_model.Rds"
)

missing <- required[!file.exists(file.path(model_dir, required))]
if (length(missing) > 0) {
  stop(paste("Missing mmSYGNAL model artifacts:", paste(missing, collapse = ", ")))
}

cat("mmSYGNAL model artifacts found:\n")
for (f in required) {
  path <- file.path(model_dir, f)
  obj <- readRDS(path)
  cat("\n---", f, "---\n")
  cat("R class:", paste(class(obj), collapse = ", "), "\n")
  cat("Size bytes:", file.info(path)$size, "\n")
  if (inherits(obj, "train")) {
    cat("caret method:", obj$method, "\n")
    cat("n input features:", length(obj$coefnames), "\n")
    cat("predict API: predict(model, newdata, type='prob')$high\n")
  }
}
cat("\nAll mmSYGNAL artifacts loadable.\n")
