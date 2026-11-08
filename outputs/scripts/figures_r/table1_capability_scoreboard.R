# Table I — the honest WHAT/WHEN/WHY capability scoreboard, straight from the goal registry (resistancemap.goal
# -> outputs/paper/capability_scoreboard.csv, dumped by the pipeline; never hand-typed). Status: ✓ answerable,
# ~ partial, ✗ blocked. This is the honest goal-validation summary: mostly partial/blocked, the durable
# contribution is the methodology + the negatives.

source("scripts/figures_r/_tables.R")

d <- read_artifact("outputs/paper/capability_scoreboard.csv") |>
  mutate(Status = recode(status, yes = "$\\checkmark$", partial = "$\\sim$", blocked = "$\\times$"),
         Question = factor(question, levels = c("WHEN", "WHAT", "WHY")),
         Capability = capability) |>
  arrange(Question) |>
  select(Question, Capability, Layer = layer, Status, Tier = tier)

tab <- kbl(select(d, -Question), format = "latex", booktabs = TRUE, escape = FALSE,
           align = c("l", "c", "c", "c"),
           col.names = c("Capability", "Layer", "Status", "Tier"),
           linesep = "") |>
  column_spec(1, width = "5.4cm") |>
  pack_rows(index = table(d$Question), bold = TRUE, latex_gap_space = "0.4em")

emit_table(tab, "table1_capability_scoreboard",
           caption = "Honest capability scoreboard for the dashboard goal (WHAT/WHEN/WHY). $\\checkmark$ answerable on real open data, $\\sim$ partial, $\\times$ blocked pending controlled data. Source: \\texttt{resistancemap.goal} registry.",
           label = "tab:scoreboard",
           note = "Tiers: T1 open GDC baseline+OS (on disk); T2 serial clinical (controlled).")

cat(sprintf("Table I: %d capabilities (%d answerable, %d partial, %d blocked)\n",
            nrow(d), sum(d$Status == "$\\checkmark$"), sum(d$Status == "$\\sim$"), sum(d$Status == "$\\times$")))
