# FloodLens judging demonstration and evaluation

## Reproduce the evidence

From the repository root, run:

```powershell
python -m scripts.evaluate_judging_case
python -m pytest -q
```

The script rebuilds [judging_evaluation.json](../results/judging_evaluation.json) from seeds 42–53 with generation backends disabled. It compares the first modeled inundation time against the sampled binary grid for every generated building and road and checks that each eligible fallback alert contains the correct hour, shelter name, and distance in both languages. The export includes the exact seed-42 case for B0002. This evaluates pipeline consistency and fallback content. Generated-model accuracy, human comprehension, and agreement with observed flood extent need separate evaluation.

## Three-minute judge walkthrough

**0:00–0:30 · Problem**  
“In June 2022, satellite analysis detected about 106 square kilometres of possible flooding in the part of Sunamganj it covered. A river reading can tell a volunteer that water is rising. The volunteer still needs to decide who to contact first, which road to check, and what to say in Bangla.” [UNOSAT assessment](https://unosat.org/static/unosat_filesystem/3303/UNOSAT_A3_Natural_Portrait_FL20220525BGD_Sylhet_Bangladesh_18Jun2022.pdf)

**0:30–1:20 · One household**  
Open Sunamganj and search for B0002. Move the timeline to hour 0, then hour 1. Highlight a road that changes state and the selected home. Read the hour, sample shelter, and distance from the plan. Play the Bangla sample alert. Say on camera that the households, shelter details, and flood geometry are illustrative.

**1:20–1:55 · Explain the pipeline**  
“The Python model estimates river stage from gauge and rainfall inputs, then follows connected low ground on a terrain grid. It calculates the household and road impacts before the language model drafts a message. An offline template uses the same facts if a model is unavailable.” Show the [B0002 export](../results/judging_evaluation.json) and point out the template source. The website’s audio reads a built-in sample.

**1:55–2:35 · Show proof and open questions**  
“Across 12 seeded simulations, building and road times matched their flood grids. Every fallback alert checked kept the modeled time, shelter, and distance in both languages. That gives us a reproducible software check. A field test still needs georeferenced terrain, observed gauge records, and a matched satellite flood mask.” Show the [evaluation results](../results/judging_evaluation.json).

**2:35–3:00 · First pilot and close**  
“We would start with a disaster volunteer in one union, have them verify a shelter and road before sharing an alert, and measure how quickly people identify the first action, deadline, and destination.” End on the dashboard with the demo and repository URLs visible.

## Bangla alert comprehension pilot

This protocol is ready to run with 5–10 consenting Bangla-speaking adults. Use mock cases only, avoid recording personal information, and label all messages as test materials.

1. Prepare two similarly difficult mock cases with distinct shelter names and deadlines. For each, create a short Bangla gauge bulletin and a FloodLens-style household alert from the same underlying case. Use identical typography and delivery format for the timed comparison.
2. Give each volunteer two trials, one per case. Split the order between volunteers so that each case appears in both formats across the pilot. Avoid showing the same volunteer both versions of a case.
3. Ask each person to identify the first action, its timing, and the shelter. Start timing when the display opens and stop when the volunteer gives a final response. Allow “not specified” as an answer. Predefine which facts are stated in each version and report separately when a version omits information.
4. Record format, case, order, time in seconds, whether each of the three answers was actionable and correct, and any misunderstood phrase. Report denominators, accuracy per item, and median time for responses with all three items correct. Treat results as exploratory.

For observed flood comparison, acquire authorized SW269 gauge history and a georeferenced flood mask, pick an issuance and verification time in advance, and hold out observations after issuance from the model input. Align the modeled and observed grids, mask permanent water and areas outside the common coverage, and report flood-area overlap plus misses and false alarms. Compare gauge error with a last-observed-level baseline. The June 2022 satellite record is the historical reference candidate. [FFWC gauge data service](https://api.ffwc.gov.bd/?format=api) · [UNOSAT assessment map](https://unosat.org/static/unosat_filesystem/3303/UNOSAT_A3_Natural_Portrait_FL20220525BGD_Sylhet_Bangladesh_18Jun2022.pdf)

### Ready-to-run Bangla pilot prompts

Show the same heading on every card, **অনুশীলন — কাল্পনিক বন্যা পরিস্থিতি**. Both versions of a case contain the same decision facts, so this pilot measures whether the wording and order help readers find those facts.

| Case | Gauge bulletin | Household alert |
| --- | --- | --- |
| A | নদীর পানি ৬.৪ মিটার। বিপদসীমা ৬.৫ মিটার। মডেল অনুযায়ী ২ ঘণ্টার মধ্যে বাড়িটি প্লাবিত হতে পারে। পরিবারের সদস্যদের এখনই নমুনা আশ্রয়কেন্দ্র ক-তে নিয়ে যাওয়ার পরামর্শ দেওয়া হচ্ছে। | পরিবারের সবাইকে এখনই নমুনা আশ্রয়কেন্দ্র ক-তে নিয়ে যান। মডেল অনুযায়ী ২ ঘণ্টার মধ্যে আপনার বাড়িতে পানি উঠতে পারে। |
| B | নদীর পানি ৬.২ মিটার। বিপদসীমা ৬.৫ মিটার। মডেল অনুযায়ী ৪ ঘণ্টার মধ্যে বাড়িটি প্লাবিত হতে পারে। শিশু ও প্রবীণদের এখনই নমুনা আশ্রয়কেন্দ্র খ-তে নিয়ে যাওয়ার পরামর্শ দেওয়া হচ্ছে। | শিশু ও প্রবীণদের এখনই নমুনা আশ্রয়কেন্দ্র খ-তে নিয়ে যান। মডেল অনুযায়ী ৪ ঘণ্টার মধ্যে আপনার বাড়িতে পানি উঠতে পারে। |

Ask: **প্রথমে কী করবেন? কখন করবেন? কোথায় যাবেন?**

Score independently. Case A expects taking family members to shelter, now, shelter ক. Case B expects taking children and older adults to shelter, now, shelter খ. Treat “within 2/4 hours” as incorrect for the action timing. It describes the modeled flooding time.

| Anonymous code | Case | Format | Order | Seconds | Action correct | Timing correct | Shelter correct | Notes |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| | | | | | | | | |
