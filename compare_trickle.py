import csv

for label, path in [
    ("Trickle ON",  "results/trickle_on/summary.csv"),
    ("Trickle OFF", "results/trickle_off/summary.csv"),
]:
    with open(path) as f:
        row = list(csv.DictReader(f))[0]
    print(f"{label}:  DIO messages = {row['dio_count']},  DIO bytes = {row['dio_bytes']},  convergence = {row['convergence_time_s']}s")
