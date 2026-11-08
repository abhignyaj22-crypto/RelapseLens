# Fig 3 — leak-proof feature-set ladder: the honest in-fold gain of omics over a clinical baseline is modest,
# and the leaky (pre-computed PCA) variant inflates it. Uno C-index by feature set at a representative PCA
# rank (k=8), leak-free in-fold vs leaky pre-computed, clinical baseline as reference. Real data:
# outputs/rebaseline_open_gdc_os/leakproof_leaderboard.csv (make benchmark / rebaseline).

source("scripts/figures_r/ieee_theme.R")

K <- 8  # representative PCA rank (full k-sensitivity lives in Table II)
d <- read_artifact("outputs/rebaseline_open_gdc_os/leakproof_leaderboard.csv") |> filter(k == K)

labmap <- c("clinical" = "Clinical", "clinical+cyto" = "+ Cyto",
            "clinical+omics_infold" = "+ Omics", "clinical+omics_precomp" = "+ Omics",
            "clinical+cyto+omics_infold" = "+ Cyto + Omics", "clinical+cyto+omics_precomp" = "+ Cyto + Omics")

df <- d |>
  mutate(fset  = labmap[feature_set],
         leak  = case_when(str_detect(feature_set, "precomp") ~ "Pre-computed PCA (leaky)",
                           str_detect(feature_set, "infold")  ~ "In-fold PCA (leak-free)",
                           TRUE ~ "In-fold PCA (leak-free)"),
         fset  = factor(fset, levels = c("Clinical", "+ Cyto", "+ Omics", "+ Cyto + Omics")))

clin <- df$mean_cindex[df$feature_set == "clinical"][1]

p <- ggplot(df, aes(x = fset, y = mean_cindex, colour = leak, group = leak)) +
  geom_hline(yintercept = clin, linetype = "dashed", linewidth = 0.4, colour = "grey45") +
  geom_errorbar(aes(ymin = mean_cindex - sd_cindex, ymax = mean_cindex + sd_cindex),
                width = 0.16, linewidth = 0.4, position = position_dodge(width = 0.4)) +
  geom_point(aes(shape = leak), size = 1.7, position = position_dodge(width = 0.4)) +
  scale_colour_manual(values = c("In-fold PCA (leak-free)" = "#0072B2",
                                 "Pre-computed PCA (leaky)" = "#D55E00"), name = NULL) +
  scale_shape_manual(values = c("In-fold PCA (leak-free)" = 16, "Pre-computed PCA (leaky)" = 17), name = NULL) +
  labs(x = NULL, y = "Uno C-index (5-fold mean ± SD)",
       title = "Leak-proof ladder: modest, leakage-sensitive",
       caption = sprintf("Dashed = clinical (%.3f), k=%d. Omics ~+0.07 (n.s.); leaky PCA inflates.", clin, K)) +
  theme_ieee() +
  theme(plot.title.position = "plot", legend.position = "bottom",
        legend.key.size = unit(0.7, "lines"))

save_ieee(p, file.path(FIG_DIR, "fig3_leakproof_leaderboard"), width = IEEE_COL_1, height = 2.9)

cat(sprintf("Fig3: clinical=%.3f; best in-fold=%.3f; best precomp(leaky)=%.3f\n",
            clin, max(df$mean_cindex[str_detect(df$feature_set, "infold")]),
            max(df$mean_cindex[str_detect(df$feature_set, "precomp")])))
