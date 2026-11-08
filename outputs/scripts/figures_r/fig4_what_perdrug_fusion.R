# Fig 4 — WHAT (which drugs fail): the honest per-drug picture. (A) Per-drug bortezomib response carries real
# above-chance signal (AUROC 0.589, CI excludes the permuted control / chance). (B) Leave-drug-out generalization:
# structure-conditioned fusion beats the floor (real) but only TIES DeepCDR (matches SOTA, does not beat) — and
# the advertised multimodal 0.82→0.89 lift over the best single modality is NULL (GDSC2, caption). Real data:
# outputs/autoresearch/{drug_e2,drug_holdout,fusion_gdsc2}.json.

source("scripts/figures_r/ieee_theme.R")
suppressMessages(library(patchwork))

e2 <- read_artifact("outputs/autoresearch/drug_e2.json", fromJSON)
hd <- read_artifact("outputs/autoresearch/drug_holdout.json", fromJSON)
fu <- read_artifact("outputs/autoresearch/fusion_gdsc2.json", fromJSON)

# --- Panel A: per-drug AUROC vs permuted control (chance = 0.5) ---
dA <- tibble(
  model = factor(c("Bortezomib model", "Permuted labels"), levels = c("Permuted labels", "Bortezomib model")),
  est   = c(e2$auroc, e2$permuted_auroc),
  lo    = c(e2$auroc_ci[1], NA), hi = c(e2$auroc_ci[2], NA),
  kind  = c("Real signal", "Control"))

pA <- ggplot(dA, aes(est, model, colour = kind)) +
  geom_vline(xintercept = 0.5, linetype = "dashed", linewidth = 0.4, colour = "grey45") +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.2, linewidth = 0.45, na.rm = TRUE) +
  geom_point(size = 1.7) +
  scale_colour_manual(values = c("Real signal" = "#0072B2", "Control" = "#D55E00"), guide = "none") +
  scale_x_continuous(limits = c(0.44, 0.68), breaks = c(0.45, 0.5, 0.55, 0.6, 0.65)) +
  labs(x = "AUROC (95% CI; dashed = chance)", y = NULL,
       subtitle = "A  Per-drug: bortezomib (GSE9782, n=239)") +
  theme_ieee() + theme(plot.subtitle = element_text(face = "bold"))

# --- Panel B: leave-drug-out R² contrasts (fusion vs floor = real; fusion vs DeepCDR = tie) ---
dB <- tibble(
  contrast = factor(c("Fusion − floor", "Fusion − DeepCDR"),
                    levels = c("Fusion − DeepCDR", "Fusion − floor")),
  est = c(hd$fusion_minus_global_floor, hd$fusion_minus_deepcdr),
  lo  = c(hd$floor_ci_drug_bootstrap[1], hd$deepcdr_ci_drug_bootstrap[1]),
  hi  = c(hd$floor_ci_drug_bootstrap[2], hd$deepcdr_ci_drug_bootstrap[2]),
  sig = c("Real (CI ≠ 0)", "Tie (CI ∋ 0)"))

pB <- ggplot(dB, aes(est, contrast, colour = sig)) +
  geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.4, colour = "grey45") +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.2, linewidth = 0.45) +
  geom_point(size = 1.7) +
  scale_colour_manual(values = c("Real (CI ≠ 0)" = "#009E73", "Tie (CI ∋ 0)" = "#7570b3"), name = NULL) +
  labs(x = expression(Delta*" leave-drug-out "*R^2*" (95% CI)"), y = NULL,
       subtitle = "B  Leave-drug-out fusion vs floor & SOTA") +
  theme_ieee() +
  theme(plot.subtitle = element_text(face = "bold"),
        legend.position = "bottom", legend.key.size = unit(0.6, "lines"))

fusion_null <- sprintf("GDSC2: fusion−best-single ΔR²=%+.3f [%.3f, %+.3f] — no multimodal lift.",
                       fu$fusion_minus_best_single, fu$fusion_minus_best_single_ci[1], fu$fusion_minus_best_single_ci[2])

p <- (pA / pB) + plot_annotation(
  title = "WHAT: real per-drug signal, but no multimodal lift",
  caption = fusion_null,
  theme = theme_ieee() + theme(plot.title = element_text(face = "bold", size = IEEE_BASE),
                               plot.caption = element_text(size = IEEE_BASE - 2.5, colour = "grey40", hjust = 0)))

save_ieee(p, file.path(FIG_DIR, "fig4_what_perdrug_fusion"), width = IEEE_COL_1, height = 3.2)

cat(sprintf("Fig4: bortezomib AUROC=%.3f [%.3f,%.3f] vs permuted %.3f; LDO fusion R²=%.3f (floor %.3f, DeepCDR %.3f); fusion-DeepCDR=%.3f [%.3f,%.3f]\n",
            e2$auroc, e2$auroc_ci[1], e2$auroc_ci[2], e2$permuted_auroc,
            hd$r2$fusion, hd$r2$global_floor, hd$r2$deepcdr,
            hd$fusion_minus_deepcdr, hd$deepcdr_ci_drug_bootstrap[1], hd$deepcdr_ci_drug_bootstrap[2]))
