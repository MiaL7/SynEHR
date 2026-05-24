from __future__ import annotations
import json
import re

VISIT_EOS_TOKEN = '<VISIT_EOS>'
SYSTEM_PROMPT = "You are a medical data synthesis assistant for EHR. You predict the next clinical visit in JSON format given a patient's demographic information and visit history."
USER_PREAMBLE = 'You are a medical data synthesis assistant. Your task is to generate realistic synthetic electronic health records (EHR) that follow clinical patterns and temporal logic.\n\nGuidelines:\n- Generate clinically plausible visit sequences based on patient history\n- Maintain temporal consistency (delta_days should be realistic for the clinical scenario)\n- Diagnosis and procedure codes should align with visit type\n- Medications should be appropriate for diagnoses\n- Lab tests should be relevant to the clinical context\n- Follow the exact JSON format provided.\n- Output only valid JSON without any additional text or explanation.\n- If the patient has no further visits, output the special token <VISIT_EOS> instead of a JSON object.\n\nGiven the following patient\'s medical history, generate the next visit in the SAME JSON schema as the existing visits.\n\nPatient history:\n\n{history_json}\n\nOutput ONLY valid JSON representing the next visit, with the same fields as one element of the "visits" list.\n'
_HISTORY_BLOCK_RE = re.compile(
    '(Patient history:\\s*\\n\\s*\\n)(\\{.*?\\})(\\s*\\n\\s*\\nOutput ONLY valid JSON)',
    re.DOTALL)


def make_user_message(patient: dict, visits_so_far: list[dict]) -> str:
    history_json = json.dumps({
        'patient': patient,
        'visits': visits_so_far
    },
        indent=2)

    return USER_PREAMBLE.format(history_json=history_json)


def extract_history_payload(user_content: str) -> dict | None:
    match = _HISTORY_BLOCK_RE.search(user_content)
    if match is None:
        return None

    try:
        return json.loads(match.group(2))

    except json.JSONDecodeError:
        return None


def replace_history_payload(user_content: str, patient: dict,
                            visits: list[dict]) -> str | None:
    history_json = json.dumps({'patient': patient, 'visits': visits}, indent=2)

    match = _HISTORY_BLOCK_RE.search(user_content)
    if match is None:
        return None

    return _HISTORY_BLOCK_RE.sub(
        lambda m: f'{m.group(1)}{history_json}{m.group(3)}',
        user_content,
        count=1)


def count_visits_in_user_message(user_content: str) -> int:
    payload = extract_history_payload(user_content)
    visits = payload.get('visits') if isinstance(payload, dict) else None

    return len(visits) if isinstance(visits, list) else 0


def append_visit_to_history(user_content: str,
                            new_visit: dict) -> tuple[str | None, str | None]:
    payload = extract_history_payload(user_content)

    if payload is None:
        return (None, 'JSON block not found')

    patient = payload.get('patient')
    visits = payload.get('visits')

    if not isinstance(patient, dict) or not isinstance(visits, list):
        return (None, 'history payload missing patient/visits')

    next_visit = dict(new_visit)
    next_visit['visit_index'] = (visits[-1]['visit_index']
                                 if visits else 0) + 1

    updated = replace_history_payload(user_content, patient,
                                      visits + [next_visit])

    if updated is None:
        return (None, 'failed to rebuild history payload')

    return (updated, None)


def rebuild_prompt_with_visit_suffix(messages: list[dict], asst_pos: int,
                                     keep_from: int) -> list[dict] | None:
    prompt_messages = messages[:asst_pos]

    if len(prompt_messages) < 2 or prompt_messages[0]['role'] != 'system':
        return None

    latest_user = prompt_messages[-1]

    if latest_user['role'] != 'user':
        return None

    payload = extract_history_payload(latest_user['content'])

    if payload is None:
        return None

    patient = payload.get('patient')
    visits = payload.get('visits')

    if not isinstance(patient, dict) or not isinstance(visits, list):
        return None

    kept_visits = visits[keep_from:]

    if not kept_visits:
        return None

    rebuilt = [prompt_messages[0]]

    for idx in range(len(kept_visits)):
        user_content = replace_history_payload(latest_user['content'], patient,
                                               kept_visits[:idx + 1])
        if user_content is None:
            return None
        rebuilt.append({'role': 'user', 'content': user_content})
        if idx < len(kept_visits) - 1:
            rebuilt.append({
                'role':
                'assistant',
                'content':
                json.dumps(kept_visits[idx + 1], indent=2)
            })

    return rebuilt


def fit_step_messages_by_visit(
        messages: list[dict], asst_pos: int, tokenizer,
        max_len: int) -> tuple[list[dict], list[dict], dict, dict] | None:
    target_message = messages[asst_pos]
    prompt_messages = messages[:asst_pos]
    latest_user = prompt_messages[-1] if prompt_messages else None
    payload = extract_history_payload(
        latest_user['content']

    ) if latest_user is not None and latest_user['role'] == 'user' else None
    visits_so_far = payload.get('visits') if isinstance(payload,
                                                        dict) else None

    max_drop = max(0,
                   len(visits_so_far) -
                   1) if isinstance(visits_so_far, list) else 0

    for keep_from in range(max_drop + 1):
        if keep_from == 0:
            current_prompt = prompt_messages
        else:
            current_prompt = rebuild_prompt_with_visit_suffix(
                messages, asst_pos, keep_from)
            if current_prompt is None:
                return None
        full_messages = current_prompt + [target_message]
        try:
            prompt_text = tokenizer.apply_chat_template(
                current_prompt, tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False)
            prompt_enc = tokenizer(prompt_text,
                                   truncation=False,
                                   padding=False,
                                   return_tensors=None,
                                   add_special_tokens=False,
                                   verbose=False)
            full_enc = tokenizer(full_text,
                                 truncation=False,
                                 padding=False,
                                 return_tensors=None,
                                 add_special_tokens=False,
                                 verbose=False)
        except Exception:
            return None
        prompt_len = len(prompt_enc['input_ids'])
        full_len = len(full_enc['input_ids'])
        if full_len <= prompt_len:
            return None
        if full_len > max_len:
            continue
        return (current_prompt, full_messages, prompt_enc, full_enc)

    return None
