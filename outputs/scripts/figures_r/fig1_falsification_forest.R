# Fig 1 — the honest-negative headline: no molecular/neural arm beats a flat penalised clinical Cox on OS rank.
# Forest of paired ΔC-index (arm − its baseline) with 95% bootstrap CI + BH-adjusted permutation p; scrambled/
# permuted negative controls collapse to 0. Real data: outputs/falsification_open_gdc_os/
# falsification_paired_deltas.csv (produced by `make benchmark`).

source("scripts/figures_r/ieee_theme.R")

d <- read_artifact("outputs/falsification_open_gdc_os/falsification_paired_deltas.csv")

# concise, publication-ready labels + control flag (keyed on the exact comparison strings in the artifact)
labmap <- c(
  "clinical+omics_vs_clinical"                     = "+ Omics",
  "residual_risk_vs_clinical"                      = "Residual-risk",
  "clinical+cyto+omics_vs_clinical"                = "+ Cyto + Omics",
  "pooled_neural_vs_clinical+omics"                = "Pooled-neural",
  "scrambled_subtype_vs_clinical+cyto+omics"       = "Scrambled subtype",
  "permuted_omics_vs_clinical+omics"               = "Permuted omics")
ctrl <- c("scrambled_subtype_vs_clinical+cyto+omics", "permuted_omics_vs_clinical+omics")

df <- d |>
  mutate(arm  = ifelse(comparison %in% names(labmap), labmap[comparison], comparison),
         kind = ifelse(comparison %in% ctrl, "Negative control", "Candidate arm"),
         arm  = fct_reorder(arm, delta_cindex),
         plab = sprintf("p==%.2f", p_fdr_bh))

xr <- range(c(df$delta_ci_low, df$delta_ci_high))
xpad <- diff(xr) * 0.02
p_x <- xr[2] + diff(xr) * 0.10               # fixed x for the right-hand p-value column

p <- ggplot(df, aes(x = delta_cindex, y = arm, colour = kind)) +
  geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.4, colour = "grey45") +
  geom_errorbar(aes(xmin = delta_ci_low, xmax = delta_ci_high),
                width = 0.26, linewidth = 0.45, orientation = "y") +
  geom_point(aes(shape = kind), size = 1.5, fill = "white") +
  geom_text(aes(x = p_x, label = plab), parse = TRUE, hjust = 0,
            size = (IEEE_BASE - 2.5) / .pt, colour = "grey30") +
  annotate("text", x = p_x, y = nlevels(df$arm) + 0.55, label = "BH~italic(p)", parse = TRUE,
           hjust = 0, size = (IEEE_BASE - 2.5) / .pt, colour = "grey30") +
  scale_colour_manual(values = c("Candidate arm" = "#0072B2", "Negative control" = "#D55E00"), name = NULL) +
  scale_shape_manual(values = c("Candidate arm" = 21, "Negative control" = 24), name = NULL) +
  scale_x_continuous(limits = c(xr[1] - xpad, p_x + diff(xr) * 0.16),
                     breaks = scales::pretty_breaks(5)) +
  coord_cartesian(clip = "off") +
  labs(x = expression(Delta*" Uno C-index vs baseline (95% CI)"), y = NULL,
       title = "OS rank ceiling: nothing beats clinical",
       caption = "Δ>0 favours richer model; none survives BH (5-fold).") +
  theme_ieee() +
  theme(plot.title.position = "plot",
        plot.margin = margin(6, 6, 3, 3),
        legend.position = "bottom", legend.margin = margin(0, 0, 0, 0),
        legend.box.spacing = unit(2, "pt"))

save_ieee(p, file.path(FIG_DIR, "fig1_falsification_forest"), width = IEEE_COL_1, height = 2.9)

cat(sprintf("Fig1: %d comparisons; max Δ=%.3f (%s); BH-significant=%d/%d\n",
            nrow(df), max(df$delta_cindex),
            as.character(df$arm[which.max(df$delta_cindex)]),
            sum(df$fdr_significant), nrow(df)))
