"""Generate data/labelled/ai_assisted_batch2_labels.csv according to Task 1b."""
import pandas as pd
from pathlib import Path

b2_pairs = [f"PAIR_EXP_{i:03d}" for i in range(1, 101)]

pos_pairs = {
    "PAIR_EXP_028", "PAIR_EXP_029", "PAIR_EXP_034", "PAIR_EXP_036",
    "PAIR_EXP_037", "PAIR_EXP_038", "PAIR_EXP_039", "PAIR_EXP_040"
}

neg_high = {
    "PAIR_EXP_014": "2000 kVA 11/0.433 kV vs 500 kVA 11-3.3/0.415 kV transformer (Rule 1, HIGH)",
    "PAIR_EXP_015": "2000 kVA 11/0.433 kV vs 500 kVA 11-3.3/0.415 kV transformer (Rule 1, HIGH)",
    "PAIR_EXP_011": "Steam turbine vs EOI design-study service (Rule 7, HIGH)",
    "PAIR_EXP_024": "Hiring of pumps (service) vs sealless pumps (Rule 7, HIGH)",
    "PAIR_EXP_025": "Hiring of pumps (service) vs sealless pumps (Rule 7, HIGH)",
    "PAIR_EXP_026": "Hiring of pumps (service) vs sealless pumps (Rule 7, HIGH)",
    "PAIR_EXP_086": "AMC service vs gas chromatograph instrument (Rule 7, HIGH)",
    "PAIR_EXP_087": "AMC service vs gas chromatograph instrument (Rule 7, HIGH)",
    "PAIR_EXP_088": "Pig launcher/receiver (pressure trap) vs electronic pig transmitter/receiver (HIGH)",
    "PAIR_EXP_089": "Pig launcher/receiver (pressure trap) vs electronic pig transmitter/receiver (HIGH)",
}

neg_med = {
    "PAIR_EXP_006": "Bale moisture analyzer vs lab moisture analyzer with inbuilt weighing (different principle, MEDIUM)",
    "PAIR_EXP_012": "Steel conveyor belt vs fabric belt (different construction, MEDIUM)",
    "PAIR_EXP_018": "Slurry vs sewage submersible pumps (different service, MEDIUM)",
    "PAIR_EXP_019": "Slurry vs sewage submersible pumps (different service, MEDIUM)",
    "PAIR_EXP_098": "Vaseline petroleum jelly vs Balmerol lubricating grease (MEDIUM)",
    "PAIR_EXP_099": "Vaseline petroleum jelly vs Balmerol lubricating grease (MEDIUM)",
    "PAIR_EXP_100": "Vaseline petroleum jelly vs Balmerol lubricating grease (MEDIUM)",
}

rows = []
for pid in b2_pairs:
    if pid in pos_pairs:
        label = "1"
        conf = "MEDIUM"
        rat = "API 6D ball valves differing only in quantity, plural or word order (Rule 6; same precedent as PAIR_TFIDF_HIGH_020)"
    elif pid in neg_high:
        label = "0"
        conf = "HIGH"
        rat = neg_high[pid]
    elif pid in neg_med:
        label = "0"
        conf = "MEDIUM"
        rat = neg_med[pid]
    else:
        label = "UNCERTAIN"
        conf = "MEDIUM"
        rat = 'Generic titles, generic-vs-specific asymmetry, tender clause fragments, bundles, and API 6D lots of "various sizes" / "03 items" (030, 035). Rules 4 and 5.'

    rows.append({
        "pair_id": pid,
        "ai_label": label,
        "ai_confidence": conf,
        "ai_rationale": rat,
        "label_source": "AI_ASSISTED_PRE_ANNOTATION (not human ground truth)",
        "annotator": "Claude (Anthropic), applying docs/annotation_guidelines.md",
        "blind_to_model_output": True,
        "review_date": "2026-11-06",
    })

df = pd.DataFrame(rows)
out_path = Path("data/labelled/ai_assisted_batch2_labels.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"Saved {out_path} with {len(df)} rows.")
print("Label distribution:\n", df["ai_label"].value_counts())
print("Confidence distribution:\n", df["ai_confidence"].value_counts())

