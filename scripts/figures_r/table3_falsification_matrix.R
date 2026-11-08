# Table III — the falsification matrix: paired ΔC-index of each candidate arm (and negative control) vs its
# baseline, with 95% bootstrap CI, BH-adjusted permutation p, and verdict. Real data: outputs/
# falsification_open_gdc_os/falsification_paired_deltas.csv. Honest: NO arm survives BH correction — every CI
# crosses 0 (the OS rank ceiling, Fig 1).

source("scripts/figures_r/_tables.R")

d <- read_artifact("outputs/falsification_open_gdc_os/falsification_paired_deltas.csv")

labmap <- c(
  "clinical+omics_vs_clinical"               = "Clinical + Omics",
  "residual_risk_vs_clinical"                = "Residual-risk",
  "clinical+cyto+omics_vs_clinical"          = "Clinical + Cyto + Omics",
  "pooled_neural_vs_clinical+omics"          = "Pooled-neural",
  "scrambled_subtype_vs_clinical+cyto+omics" = "Scrambled subtype (ctrl)",
  "permuted_omics_vs_clinical+omics"         = "Permuted omics (ctrl)")

df <- d |>
  mutate(Arm    = ifelse(comparison %in% names(labmap), labmap[comparison], comparison),
         UnoC   = sprintf("%.3f", cindex_model_a),
         Delta  = sprintf("$%+.3f$", delta_cindex),
         CI     = sprintf("$[%+.3f, %+.3f]$", delta_ci_low, delta_ci_high),
         BHp    = sprintf("%.2f", p_fdr_bh),
         Verdict = ifelse(fdr_significant, "sig.", "n.s.")) |>
  arrange(desc(delta_cindex)) |>
  select(Arm, UnoC, Delta, CI, BHp, Verdict)

tab <- kbl(df, format = "latex", booktabs = TRUE, escape = FALSE,
           align = c("l", "c", "c", "c", "c", "c"),
           col.names = c("Arm vs baseline", "Uno-C", "$\\Delta$C", "95\\% CI", "BH $p$", "Verdict"),
           linesep = "") |>
  column_spec(1, width = "3.1cm")

emit_table(tab, "table3_falsification_matrix",
           caption = "Falsification matrix: paired $\\Delta$ Uno C-index of each arm vs its baseline (5-fold, patient-disjoint), with 95\\% bootstrap CI and BH-adjusted permutation $p$. No arm survives BH correction; every CI includes 0 (Fig.~\\ref{fig:forest}). Negative controls (scrambled/permuted) collapse to 0.",
           label = "tab:falsification",
           note = "n.s. = not significant after BH correction across the family.")

cat(sprintf("Table III: %d comparisons; BH-significant=%d; max Δ=%.3f\n",
            nrow(df), sum(d$fdr_significant), max(d$delta_cindex)))
