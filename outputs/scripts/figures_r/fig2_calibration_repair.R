# Fig 2 — the one genuine methodological win: WITHIN the first-hitting-time family, the proposed defective-FHT
# head (learning a non-crossing cure mass P(T*=∞)>0) repairs proper-FHT's broken calibration. Two panels:
# IPCW-IBS (↓ better) and D-calibration p (↑ better). Real data: outputs/paper/result_set_C.json (Result C,
# 5-fold, real clinical+in-fold-PCA). Honest: defective-FHT is NOT a static-OS SOTA (Cox still best rank), the
# win is calibration vs its proper-FHT alternative.

source("scripts/figures_r/ieee_theme.R")

d <- read_artifact("outputs/paper/result_set_C.json", fromJSON)
m <- d$metrics

heads <- c(cox = "Cox (PH)", aft = "AFT", proper_fht = "Proper-FHT",
           defective_fht = "Defective-FHT", defective_noPinf = "Defective, no P∞")
grab <- function(metric) sapply(names(heads), function(h) m[[h]][[metric]])

df <- bind_rows(
  tibble(head = names(heads), mean = grab("ibs")[1, ], sd = grab("ibs")[2, ],
         panel = "IPCW-IBS  (lower better)"),
  tibble(head = names(heads), mean = grab("dcal_p")[1, ], sd = grab("dcal_p")[2, ],
         panel = "D-calibration p  (higher better)")) |>
  mutate(label = heads[head],
         label = factor(label, levels = heads),
         role  = case_when(head == "defective_fht" ~ "Defective-FHT (ours)",
                           head == "proper_fht"    ~ "Proper-FHT (miscal.)",
                           head == "defective_noPinf" ~ "no-P∞ ablation",
                           TRUE ~ "Reference (Cox/AFT)"),
         role  = factor(role, levels = c("Reference (Cox/AFT)", "Proper-FHT (miscal.)",
                                         "Defective-FHT (ours)", "no-P∞ ablation")))

p <- ggplot(df, aes(x = label, y = mean, fill = role)) +
  geom_col(width = 0.68, colour = "grey25", linewidth = 0.25) +
  geom_errorbar(aes(ymin = pmax(0, mean - sd), ymax = mean + sd), width = 0.22, linewidth = 0.35,
                colour = "grey30") +
  facet_wrap(~panel, scales = "free_y", ncol = 1) +
  scale_fill_manual(values = c("Defective-FHT (ours)" = "#009E73", "Proper-FHT (miscal.)" = "#D55E00",
                               "no-P∞ ablation" = "#E69F00", "Reference (Cox/AFT)" = "#9ecae1"), name = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.10))) +
  guides(fill = guide_legend(nrow = 2, byrow = TRUE)) +
  labs(x = NULL, y = NULL, title = "Defective-FHT repairs FHT calibration",
       caption = "Proper-FHT D-cal p=0; cure-mass head repairs it (5-fold).") +
  theme_ieee() +
  theme(plot.title.position = "plot",
        axis.text.x = element_text(angle = 22, hjust = 1, size = IEEE_BASE - 1.5),
        legend.position = "bottom", legend.key.size = unit(0.65, "lines"),
        legend.text = element_text(size = IEEE_BASE - 2),
        panel.spacing = unit(5, "pt"))

save_ieee(p, file.path(FIG_DIR, "fig2_calibration_repair"), width = IEEE_COL_1, height = 3.3)

cat(sprintf("Fig2: proper-FHT IBS=%.3f D-cal=%.3f  ->  defective-FHT IBS=%.3f D-cal=%.3f (Cox IBS=%.3f D-cal=%.3f)\n",
            m$proper_fht$ibs[1], m$proper_fht$dcal_p[1], m$defective_fht$ibs[1], m$defective_fht$dcal_p[1],
            m$cox$ibs[1], m$cox$dcal_p[1]))
