# Table IV — Related-work positioning. Transposed (axes as rows, the 3 papers + this work as columns) so the
# many comparison dimensions read cleanly at full column width. Source: paper/ieee/tables/related_work.csv —
# the ONE hand-entered table in this paper, holding PUBLISHED-LITERATURE facts about the three cited papers
# (P1 PMC12954042, P2 PMC13094027, P3 PMC12081869); it contains no repo results. This work's cells trace to the
# same real artifacts as Figs 1-5 / Tables I-III.

source("scripts/figures_r/_tables.R")
suppressMessages(library(tidyr))

d <- read_artifact("paper/ieee/tables/related_work.csv",
                   function(p) readr::read_csv(p, comment = "#", show_col_types = FALSE))

axes <- tribble(
  ~col,          ~label,
  "endpoint",    "Endpoint",
  "cohort",      "Cohort / $n$",
  "modalities",  "Modalities",
  "dynamic",     "Dynamic?",
  "leakage",     "Leakage control",
  "controls",    "Mult. / neg. ctrl",
  "calibration", "Calibration",
  "external",    "External val.",
  "verdict",     "Headline verdict")

works <- d$work   # P1, P2, P3, This work (row order preserved)

wide <- d |>
  pivot_longer(-work, names_to = "col", values_to = "val") |>
  pivot_wider(names_from = work, values_from = val) |>
  arrange(match(col, axes$col)) |>
  mutate(Axis = axes$label[match(col, axes$col)]) |>
  select(Axis, all_of(works))

tab <- kbl(wide, format = "latex", booktabs = TRUE, escape = FALSE,
           align = c("l", "l", "l", "l", "l"),
           col.names = c("", "P1: ensemble", "P2: dynamic", "P3: mmSYGNAL", "This work"),
           linesep = "") |>
  column_spec(1, bold = TRUE, width = "2.2cm") |>
  column_spec(2:4, width = "2.7cm") |>
  column_spec(5, bold = TRUE, width = "2.7cm")

emit_table(tab, "table4_related_work",
           caption = "Positioning vs.\\ recent MM ML models. P1~\\cite{p1ensemble}, P2~\\cite{p2dynamic}, and P3/mmSYGNAL~\\cite{p3mmsygnal} are positive-result predictors; \\emph{this work} is a leak-proof falsification benchmark + calibration mechanism (Figs.~\\ref{fig:forest}--\\ref{fig:dynamic}, Tables~\\ref{tab:scoreboard}--\\ref{tab:falsification}). Literature cells are published facts, not reproduced here.",
           label = "tab:related",
           full_width = TRUE, size = "\\scriptsize",
           note = "P2 shows dynamic prediction works \\emph{with serial data} (this work's dynamic axis is data-blocked, not refuted); P3 is external-validated and used here as a substrate.")

cat(sprintf("Table IV: %d axes x %d works\n", nrow(wide), length(works)))
