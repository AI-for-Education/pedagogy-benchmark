# Handover: multilingual translation scoping and mapping

Last updated: 2026-07-28. Author: Paul Atherton (with Claude Code).

This note hands over the work done to scope the target language universe for the
pedagogy benchmark translations, tier those languages by priority, cost the
programme, and write up the Clear Global partnership proposal.

## What this covers

Four connected pieces of work:

1. A de jure mapping of the official mother-tongue / medium-of-instruction (MoI)
   languages across every African education system.
2. A prioritised, tiered translation cost model built on top of that mapping.
3. Reconciliation of the current machine-translated language set against the
   mapping (what we have, what is verified, what the gaps are).
4. The partnership proposal write-up, brought into Fab house style.

## Deliverable files

All live in the Dropbox folder:
`Fab Inc BMGF AI/13. Benchmarks/Multilingual/Translations_project/`

| File | What it is |
| --- | --- |
| `African MoT languages by education system (de jure) - 2026-07-03.xlsx` | The mapping deliverable. Sheets: Priority summary, Languages by country (401 rows), Translation backlog, Target languages (dedup), Country summary. |
| `translation_cost_model.xlsx` | Prioritised cost model, 256 languages, tagged with Priority tier / Status / MT cost. Backup: `translation_cost_model_backup_2026-07-03.xlsx` in the repo `data/` folder on the Dropbox side. |
| `language_speaker_regions.xlsx` | Current translation status source. `Complete` tab = human-verified (Kiswahili, Luganda only). `Gemini_translated` tab = 75 machine-translated (unverified) languages. |
| `Multilingual AI benchmarks for education.docx` | The Clear Global partnership proposal. |

## 1. The de jure mapping

Scope, agreed with Paul: **de jure** (a language must be named in national
policy, curriculum, or an education act), **Africa-wide** including colonial
languages as reference, **comprehensive** (all officially recognised languages),
delivered as a **separate Excel sorted by country**.

Coverage: all **55** African education systems (54 sovereign states plus Western
Sahara).

Key numbers:
- **401** country x language rows.
- **~253 unique African languages** named as de jure MoI or early-grade-literacy
  languages across the continent.
- The current machine-translated set covers roughly **68** of the ~253 target
  languages, so there is a long tail of gaps (mostly pilot or marginal).

Method: research was run via roughly 20 parallel background sub-agents (region
then country clusters), each returning sourced JSON, then compiled into the
mapping. A methods summary of this de jure searching and mapping is written into
the proposal.

De jure nuances worth remembering:
- Rwanda has been English-MoI from P1 since December 2019; Kinyarwanda is now a
  subject only.
- Tanzania is Kiswahili-only, with no ethnic-language MoI.
- Ethiopia teaches in roughly 51 mother-tongue-education languages but no single
  official list enumerates them all (about 33 are sourced).
- Nigeria, South Sudan and Kenya use open-ended "language of the immediate
  environment / catchment" wording rather than a closed statutory list.

## 2. Priority tiers

Tiering logic:
- **Tier 1**: national MoI in at least one country, or official in at least two
  countries.
- **Tier 2**: single-country regional or bilingual MoI.
- **Tier 3**: subject-only or pilot.

Result: **72 Tier-1 languages**, of which 42 are already machine-translated and
30 remain to translate. This is the "~70 priority-one languages, about half of
which are machine-translated" framing used in the proposal.

## 3. Translation status (reconciled)

- **Human-verified: 2** languages only, Kiswahili and Luganda (`Complete` tab).
- **Machine-translated, not verified: 75** languages (`Gemini_translated` tab):
  the 69-language African batch plus Arabic, Hausa, Yoruba, Runyankore
  (Nyankore), Dari and Pashto.
- Hausa and Yoruba **are** translated (unverified), not missing as an earlier
  draft implied. They should be prioritised for human verification.
- Dari and Pashto are non-African (Afghanistan) and belong to the wider
  benchmark, not the African mapping.
- Flagged: 7 languages in the current set (edo, ibibio, tiv, meru, sukuma,
  tarifit, plus afrikaans as a classification artefact) are not named as de jure
  MoI anywhere in the mapping.

The proposal's Languages section was rewritten to reflect this: only Kiswahili
and Luganda are described as verified, and everything else as machine-translated
but not yet verified.

## 4. Cost model

The model covers 256 languages: 75 already machine-translated (review only) and
181 backlog (machine-translation plus review). Each row is tagged with Priority
tier, Status and MT cost. The Summary sheet has a "BY PRIORITY TIER" block plus a
cumulative roadmap.

Approximate totals at current toggles ($20 reviewer / $25 senior, 6.5 min per Q,
920 Q per language, 15% overhead):
- Tier 1 (73 langs): ~$271k
- Tier 1 + 2: ~$890k
- Tier 1 + 2 + 3: ~$952k
- Everything: ~$981k

Review cost dominates. Machine translation (~$38 per language) is negligible by
comparison. Cost model maths was validated in Python against the model's own
per-language figures.

## 5. Proposal write-up

The proposal `Multilingual AI benchmarks for education.docx` was filled in and
brought into Fab house style. Sections added or rewritten: Languages intro,
Human-verified vs Machine-translated breakdown, Prioritising coverage (Tier-1
table), external Alignment paragraph (UIS BLGF / IRT plus the Fab AI readability
rubric), Method summary, and the priority-coverage reframe.

Consistency and style pass:
- Language totals standardised to **75** across the Overview, Volume/scope and
  Languages sections. The Costing section's separate "$38 for 70 African
  languages" statement was left as is, because that figure was averaged over the
  original ~70 and changing the count would misstate it.
- All em-dashes removed (Fab house style: never use em-dashes; use commas,
  parentheses or colons). UK English throughout.
- Placeholder text filled, one grammar slip fixed.

Open point for Paul to decide: the Overview says "75 languages", but 2 of those
(Dari, Pashto) are non-African. If preferred, it could read "75 languages (73
African plus Dari and Pashto)".

## Fab house style (applies to all prose)

- **Never use em-dashes.** Use commas, parentheses or colons.
- **UK English** throughout (prioritise, programme, colour, licence, analyse).

## Where the working files are

Source chunks, the mapping compiler (`compile.py`), backlog and cost-model
scripts were kept in the Claude Code session scratchpad, not in this repo. The
translation pipeline scripts themselves live in `scripts/` (see
`run_all_translations.py`, `translate_benchmark.py`,
`prepare_cdpk_dataset_multilingual.py`).
