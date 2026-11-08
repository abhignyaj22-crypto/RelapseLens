# Table II — leak-proof survival leaderboard (representative PCA rank k=8). Uno C-index (5-fold mean, SD) and
# IPCW-IBS per feature set, distinguishing leak-free in-fold PCA from leaky pre-computed PCA. Real data:
# outputs/rebaseline_open_gdc_os/leakproof_leaderboard.csv. Honest: omics adds a modest, n.s. gain over clinical
# (Fig 1 / Table III), and the pre-computed variant inflates it.

source("scripts/figures_r/_tables.R")

K <- 8
d <- read_artifact("outputs/rebaseline_open_gdc_os/leakproof_leaderboard.csv") |> filter(k == K)

meta <- tribble(
  ~feature_set,                     ~fset,             ~pca,                ~ord,
  "clinical",                       "Clinical",        "---",               1,
  "clinical+cyto",                  "+ Cyto",          "---",               2,
  "clinical+omics_infold",          "+ Omics",         "in-fold",           3,
  "clinical+omics_precomp",         "+ Omics",         "pre-comp (leaky)",  4,
  "clinical+cyto+omics_infold",     "+ Cyto + Omics",  "in-fold",           5,
  "clinical+cyto+omics_precomp",    "+ Cyto + Omics",  "pre-comp (leaky)",  6)

df <- d |>
  inner_join(meta, by = "feature_set") |>
  arrange(ord) |>
  mutate(cidx = sprintf("%.3f\\,(%.3f)", mean_cindex, sd_cindex),
         ibs  = sprintf("%.3f", mean_ibs)) |>
  select(fset, pca, cidx, ibs)

tab <- kbl(df, format = "latex", booktabs = TRUE, escape = FALSE, align = c("l", "l", "c", "c"),
           col.names = c("Feature set", "PCA", "Uno C-index (SD)", "IPCW-IBS"), linesep = "") |>
  column_spec(3, width = "2.3cm") |>
  row_spec(1, italic = TRUE)   # clinical baseline row

emit_table(tab, "table2_leakproof_leaderboard",
           caption = sprintf("Leak-proof survival leaderboard (PCA rank $k{=}%d$, 5-fold). In-fold PCA is leak-free; pre-computed PCA leaks test variance and inflates the C-index. Adding omics gains only ${\\sim}{+}0.07$ over the clinical baseline (not significant, Table~\\ref{tab:falsification}).", K),
           label = "tab:leaderboard",
           note = "Clinical baseline in italics; higher C-index / lower IBS is better.")

cat(sprintf("Table II: %d rows at k=%d; clinical=%.3f, best in-fold=%.3f\n",
            nrow(df), K, d$mean_cindex[d$feature_set == "clinical"],
            max(d$mean_cindex[grepl("infold", d$feature_set)])))
