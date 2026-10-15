from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import wfdb


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
PTBXL_ROOT = PROJECT_ROOT / "PTB_XL"
META_CSV = PTBXL_ROOT / "ptbxl_database.csv"


# ============================================================
# Load metadata
# ============================================================

df = pd.read_csv(META_CSV)

print("PTB-XL root:", PTBXL_ROOT)
print("Metadata file:", META_CSV)
print("Total metadata records:", len(df))

required_columns = [
    "ecg_id",
    "patient_id",
    "strat_fold",
    "filename_lr"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# Storage
# ============================================================

hash_to_records = {}

failed_records = []


# ============================================================
# Process records
# ============================================================

for i, row in enumerate(
    df.itertuples(index=False),
    start=1
):

    record_path = PTBXL_ROOT / row.filename_lr

    # WFDB record paths normally have no extension.
    # We check the .hea file to make sure the record exists.
    header_path = Path(str(record_path) + ".hea")

    if not header_path.exists():
        failed_records.append({
            "ecg_id": row.ecg_id,
            "reason": "Header file not found",
            "path": str(header_path)
        })
        continue

    try:
        # Read DIGITAL samples instead of converted floating-point signal.
        record = wfdb.rdrecord(
            str(record_path),
            physical=False
        )

        signal = record.d_signal

        if signal is None:
            raise ValueError(
                "WFDB returned no digital signal"
            )

        # Ensure consistent contiguous array representation
        signal = np.ascontiguousarray(signal)

        # Include shape + dtype in the hash input as an extra safeguard
        hasher = hashlib.sha256()

        hasher.update(
            str(signal.shape).encode("utf-8")
        )

        hasher.update(
            str(signal.dtype).encode("utf-8")
        )

        hasher.update(
            signal.tobytes()
        )

        waveform_hash = hasher.hexdigest()

        record_info = {
            "ecg_id": int(row.ecg_id),
            "patient_id": int(row.patient_id),
            "strat_fold": int(row.strat_fold),
            "filename_lr": row.filename_lr
        }

        hash_to_records.setdefault(
            waveform_hash,
            []
        ).append(record_info)

    except Exception as e:

        failed_records.append({
            "ecg_id": row.ecg_id,
            "reason": str(e),
            "path": str(record_path)
        })


    if i % 1000 == 0 or i == len(df):
        print(
            f"Processed {i}/{len(df)} | "
            f"Successful: {sum(len(v) for v in hash_to_records.values())} | "
            f"Failed: {len(failed_records)}"
        )


# ============================================================
# Find duplicate waveform groups
# ============================================================

duplicate_groups = {
    waveform_hash: records
    for waveform_hash, records
    in hash_to_records.items()
    if len(records) > 1
}

duplicate_record_count = sum(
    len(records) - 1
    for records in duplicate_groups.values()
)


# ============================================================
# Analyze whether duplicates cross folds / patients
# ============================================================

cross_fold_groups = []
cross_patient_groups = []

for waveform_hash, records in duplicate_groups.items():

    folds = {
        record["strat_fold"]
        for record in records
    }

    patients = {
        record["patient_id"]
        for record in records
    }

    if len(folds) > 1:
        cross_fold_groups.append({
            "hash": waveform_hash,
            "records": records
        })

    if len(patients) > 1:
        cross_patient_groups.append({
            "hash": waveform_hash,
            "records": records
        })


# ============================================================
# Summary
# ============================================================

successful_records = sum(
    len(records)
    for records in hash_to_records.values()
)

print("\n" + "=" * 60)
print("WAVEFORM DUPLICATE AUDIT SUMMARY")
print("=" * 60)

print("Metadata records:          ", len(df))
print("Successfully processed:    ", successful_records)
print("Failed records:            ", len(failed_records))
print("Unique waveform hashes:    ", len(hash_to_records))
print("Duplicate waveform groups: ", len(duplicate_groups))
print("Duplicate waveform records:", duplicate_record_count)

print(
    "Duplicates crossing folds: ",
    len(cross_fold_groups)
)

print(
    "Duplicates across patients:",
    len(cross_patient_groups)
)


# ============================================================
# Show examples
# ============================================================

if duplicate_groups:

    print("\n===== EXAMPLE DUPLICATE GROUPS =====")

    for group_num, records in enumerate(
        duplicate_groups.values(),
        start=1
    ):

        print(
            f"\nDuplicate group {group_num}:"
        )

        for record in records:
            print(record)

        if group_num >= 10:
            break


# ============================================================
# Cross-fold duplicates
# ============================================================

if cross_fold_groups:

    print(
        "\nWARNING: DUPLICATE SIGNALS FOUND "
        "ACROSS STRATIFICATION FOLDS"
    )

    for group in cross_fold_groups[:10]:

        print("\nHash:", group["hash"])

        for record in group["records"]:
            print(record)


# ============================================================
# Failed records
# ============================================================

if failed_records:

    print("\n===== FAILED RECORDS =====")

    for failure in failed_records[:20]:
        print(failure)


# ============================================================
# Save audit results
# ============================================================

results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(
    parents=True,
    exist_ok=True
)

if duplicate_groups:

    duplicate_rows = []

    for waveform_hash, records in duplicate_groups.items():

        for record in records:

            duplicate_rows.append({
                "waveform_hash": waveform_hash,
                **record
            })

    duplicate_df = pd.DataFrame(
        duplicate_rows
    )

    duplicate_df.to_csv(
        results_dir /
        "ptbxl_waveform_duplicates.csv",
        index=False
    )


if failed_records:

    pd.DataFrame(
        failed_records
    ).to_csv(
        results_dir /
        "ptbxl_waveform_audit_failures.csv",
        index=False
    )


print("\nAudit complete.")