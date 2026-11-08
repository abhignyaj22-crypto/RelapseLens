# Fig 5 — the dynamic-when axis is a method-exhaustive negative. Near-term ΔAUC (dynamic − static) at landmarks
# 3/6/12 mo for the two recency-metric methods: C3 recency features and D1 the literature-grounded joint model
# (MixedLM BLUP). A real but underpowered near-term signal that never clears 0 (every CI includes 0), even for
# the joint model. Real data: outputs/autoresearch/{recency_auc,joint_model}.json. (B2 LOCF, C1 trajectory, C2
# time-varying Cox are also null on OS td-C — caption.)

source("scripts/figures_r/ieee_theme.R")

rec <- read_artifact("outputs/autoresearch/recency_auc.json", fromJSON)
jm  <- read_artifact("outputs/autoresearch/joint_model.json", fromJSON)

grab <- function(pl, method) {
  tibble(landmark = names(pl),
         est = sapply(pl, function(x) x$delta_auc_near),
         lo  = sapply(pl, function(x) x$ci_near[1]),
         hi  = sapply(pl, function(x) x$ci_near[2]),
         method = method)
}

df <- bind_rows(grab(rec$per_landmark, "Recency features (C3)"),
                grab(jm$per_landmark,  "Joint model / BLUP (D1)")) |>
  mutate(landmark = factor(str_remove_all(landmark, "t_L=|mo"), levels = c("3", "6", "12")),
         landmark = fct_recode(landmark, "3 mo" = "3", "6 mo" = "6", "12 mo" = "12")) |>
  mutate(landmark = fct_rev(landmark))

p <- ggplot(df, aes(x = est, y = landmark, colour = method)) +
  geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.4, colour = "grey45") +
  geom_errorbar(aes(xmin = lo, xmax = hi), width = 0.28, linewidth = 0.45,
                position = position_dodge(width = 0.5), orientation = "y") +
  geom_point(aes(shape = method), size = 1.7, position = position_dodge(width = 0.5)) +
  scale_colour_manual(values = c("Recency features (C3)" = "#0072B2",
                                 "Joint model / BLUP (D1)" = "#009E73"), name = NULL) +
  scale_shape_manual(values = c("Recency features (C3)" = 16, "Joint model / BLUP (D1)" = 17), name = NULL) +
  labs(x = expression(Delta*" near-term td-AUC (dynamic − static, 95% CI)"), y = "Landmark",
       title = "Dynamic-when: near signal, never significant",
       caption = "Every CI crosses 0; BH n.s. B2/C1/C2 also null on OS td-C.") +
  theme_ieee() +
  theme(plot.title.position = "plot", legend.position = "bottom",
        legend.key.size = unit(0.7, "lines"))

save_ieee(p, file.path(FIG_DIR, "fig5_dynamic_sweep"), width = IEEE_COL_1, height = 2.7)

cat(sprintf("Fig5: C3 max near ΔAUC=%.3f; D1 max near ΔAUC=%.3f; all CIs include 0\n",
            max(sapply(rec$per_landmark, function(x) x$delta_auc_near)),
            max(sapply(jm$per_landmark, function(x) x$delta_auc_near))))
