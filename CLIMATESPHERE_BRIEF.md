# FloodLens Bangladesh | ClimateSphere judging brief

**Team Candy Crush · Idea to Unicorn · CloudCamp Bangladesh**

## Problem and first user

A rising river reading tells a union disaster volunteer that a flood is developing. The volunteer still has to work out which roads could become unusable, which households to contact first, and how to explain the next action in Bangla. FloodLens brings those decisions into one review screen.

## What to try

Open the [live dashboard](https://nahidzaman1996271-sketch.github.io/floodlens/), select Sunamganj, and search **B0002**. From hour 0, advance to hour 1 to see the sample home and access roads change state. Open the household plan and play the sample audio in Bangla. The map uses illustrative geography and sample households.

For the corresponding **seed-42 Python case**, use the [reproducible result](results/judging_evaluation.json). B0002 is categorized as critical, its sample cell first floods at hour 1, and the nearest sample shelter is 2.07 km away in a straight line. The backend case contains 102 sample buildings, with 91 modeled as affected within the forecast and 89 flagged critical by hour 10. The dashboard’s summary counts its smaller illustrative map cohort. The [API docs](https://floodlens-api-vq31.onrender.com/docs) show the backend payload.

## How the system works

The Python pipeline estimates river stage from recent gauge levels and rainfall inputs, propagates connected inundation across an elevation grid, and derives building and road impacts. The alert layer can draft Bangla and English from those structured facts, with an offline template available. A pilot would put a volunteer in the approval path to verify real roads and shelter availability before any alert is shared.

## Evidence and limits

- **Historical context.** UNOSAT mapped 106 km² of possible flooding within the 138 km² it analyzed in Sunamganj using imagery from 18 June 2022, 17:54 local time. The analyzed area covered part of the district. The source describes the mapping as preliminary. [UNOSAT assessment map](https://unosat.org/static/unosat_filesystem/3303/UNOSAT_A3_Natural_Portrait_FL20220525BGD_Sylhet_Bangladesh_18Jun2022.pdf)
- **Measured prototype behavior.** In 12 seeded simulations, first-inundation times matched the sampled grids for 1,170 buildings and 36 roads. Across 1,023 offline bilingual alert pairs, both versions included the expected hour, shelter name and distance. These checks cover consistency on synthetic cases. Real-event forecast accuracy, shelter availability, and field reach remain open. [Evaluation script](scripts/evaluate_judging_case.py) · [Machine-readable results](results/judging_evaluation.json)

## First pilot

Start with one union disaster-management team and one workflow: review a draft family warning before sharing it. Measure the time to identify when to act, the first action, and the shelter. Treat alert comprehension and verified shelter or road availability as separate measures. Repeat in a second union only after reviewing the first pilot’s errors and data needs.
