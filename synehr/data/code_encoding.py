from __future__ import annotations
from synehr.data.time_bins import FINE_BIN_LABELS, BIN_MIDPOINTS

VISIT_TYPE_MAP = {'EMERGENCY': 0, 'URGENT': 1, 'ELECTIVE': 2, 'NEWBORN': 3}
FINE_LABEL_TO_DAYS = {
    label: float(mid)
    for label, mid in zip(FINE_BIN_LABELS, BIN_MIDPOINTS)
}
PAD = 0
UNK = 1


def _enc(codes_raw: list | None, prefix: str, vocab: dict[str,
                                                          int]) -> list[int]:
    out = [vocab.get(f'{prefix}{c}', UNK) for c in codes_raw or []]

    return out or [PAD]


def _enc_direct(names: list | None, vocab: dict[str, int]) -> list[int]:
    out = [vocab.get(n, UNK) for n in names or []]

    return out or [PAD]


def gap_value_to_days(delta) -> tuple[float, int]:
    if delta is None:
        return (0.0, 1)

    if isinstance(delta, str):
        return (FINE_LABEL_TO_DAYS[delta], 0)

    return (float(delta), 0)


def encode_patient_demographics(patient: dict) -> list[float]:
    sex_raw = str(patient.get('sex', '')).strip().upper()
    race_raw = str(patient.get('race', '')).strip().upper()
    age = float(patient.get('age', 0.0))
    sex_m = 1.0 if sex_raw.startswith('M') else 0.0
    sex_f = 1.0 if sex_raw.startswith('F') else 0.0
    sex_o = 1.0 if not (sex_m or sex_f) else 0.0
    race_white = 1.0 if 'WHITE' in race_raw else 0.0
    race_black = 1.0 if 'BLACK' in race_raw else 0.0
    race_asian = 1.0 if 'ASIAN' in race_raw else 0.0
    race_hispanic = 1.0 if 'HISPANIC' in race_raw or 'LATINO' in race_raw else 0.0
    race_other = 1.0 if not (race_white or race_black or race_asian
                             or race_hispanic) else 0.0

    age_scaled = age / 100.0
    age_sq = age_scaled * age_scaled

    return [
        age_scaled, age_sq, sex_m, sex_f, sex_o, race_white, race_black,
        race_asian, race_hispanic, race_other
    ]


def _encode_visit_fields(
    visit: dict, code_vocabs: dict
) -> tuple[list[int], list[int], list[int], list[int], int]:
    dx = _enc(visit.get('diagnosis_ccs'), 'DIAG_', code_vocabs['dx'])
    proc = _enc(visit.get('procedure_ccs'), 'PROC_', code_vocabs['proc'])
    lab = _enc(visit.get('lab_categories'), 'LABFLUID_', code_vocabs['lab'])
    med = _enc_direct(visit.get('medication_ingredients'), code_vocabs['med'])
    vtype = VISIT_TYPE_MAP.get(visit.get('type', ''), 0)

    return (dx, proc, med, lab, vtype)


def encode_visit_for_static_branch(visit: dict, code_vocabs: dict) -> dict:
    dx, proc, med, lab, vtype = _encode_visit_fields(visit, code_vocabs)

    return {'dx': dx, 'proc': proc, 'med': med, 'lab': lab, 'vtype': vtype}


def encode_visit_for_inference(visit: dict, code_vocabs: dict) -> dict:
    dx, proc, med, lab, vtype = _encode_visit_fields(visit, code_vocabs)
    gap_prev_days, gap_missing = gap_value_to_days(visit.get('delta_days'))

    return {
        'dx': dx,
        'proc': proc,
        'med': med,
        'lab': lab,
        'vtype': vtype,
        'gap_prev_days': gap_prev_days,
        'gap_missing': gap_missing
    }
