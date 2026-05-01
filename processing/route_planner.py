import json
import os
import re
from typing import List

from google import genai


def _gemini_config():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("กรุณาตั้งค่า GEMINI_API_KEY หรือ GOOGLE_API_KEY ก่อนใช้งาน planner")
    model_name = os.getenv("GEMINI_MODEL")
    if not model_name:
        raise ValueError("กรุณาตั้งค่า GEMINI_MODEL ใน .env ก่อนใช้งาน planner")
    return api_key, model_name


def _build_evidence_section(evidences: List[dict]) -> str:
    lines = []
    for idx, item in enumerate(evidences, start=1):
        title = item.get("title") or "Untitled"
        source_url = item.get("source_url") or "N/A"
        snippet = (item.get("text") or "").strip().replace("\n", " ")
        snippet = snippet[:600]
        lines.append(
            f"[E{idx}] title={title} | source_url={source_url}\n"
            f"snippet={snippet}"
        )
    return "\n\n".join(lines)


def _extract_json(text: str):
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if "\n" in raw:
            raw = raw.split("\n", 1)[1]
    return json.loads(raw)


def _detect_user_language(user_query: str):
    thai_chars = len(re.findall(r"[\u0E00-\u0E7F]", user_query))
    latin_chars = len(re.findall(r"[A-Za-z]", user_query))
    cjk_chars = len(re.findall(r"[\u4E00-\u9FFF]", user_query))
    if cjk_chars >= max(thai_chars, latin_chars):
        return "zh"
    return "th" if thai_chars >= latin_chars else "en"


def _norm_place_name(name: str):
    return re.sub(r"[^a-z0-9\u0E00-\u0E7F]+", "", (name or "").lower())


def _match_place_name(candidate_name: str, place_names: List[str]):
    cand = _norm_place_name(candidate_name)
    if not cand:
        return None
    for place_name in place_names:
        target = _norm_place_name(place_name)
        if not target:
            continue
        if cand == target or cand in target or target in cand:
            return place_name
    return None


def _auto_split_day_plan(place_names: List[str]):
    midpoint = (len(place_names) + 1) // 2
    return {
        "Day 1": place_names[:midpoint],
        "Day 2": place_names[midpoint:],
    }


def _detect_requested_days(user_query: str):
    patterns = [
        r"(\d+)\s*วัน",
        r"(\d+)\s*days?",
        r"(\d+)\s*天",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_query.lower())
        if match:
            try:
                value = int(match.group(1))
                return max(1, min(value, 7))
            except ValueError:
                return 2
    return 2


def _build_day_labels(day_count: int):
    return [f"Day {idx}" for idx in range(1, day_count + 1)]


def _split_places_by_days(place_names: List[str], day_labels: List[str]):
    plan = {label: [] for label in day_labels}
    if not place_names:
        return plan
    for idx, place_name in enumerate(place_names):
        plan[day_labels[idx % len(day_labels)]].append(place_name)
    return plan


def _derive_name_from_evidence(item: dict):
    title = (item.get("title") or "").strip()
    if not title:
        return "Unknown place"
    # Keep the first concise segment from long SEO titles.
    for sep in [" - ", " | ", " — ", " – "]:
        if sep in title:
            title = title.split(sep)[0].strip()
            break
    return title[:90] if title else "Unknown place"


def _fallback_plan_from_evidence(evidences: List[dict], day_labels: List[str]):
    places = []
    used_urls = set()
    for item in evidences:
        source_url = item.get("source_url")
        if not source_url or source_url in used_urls:
            continue
        used_urls.add(source_url)
        places.append(
            {
                "name": _derive_name_from_evidence(item),
                "province": "Unknown",
                "zone": "Unknown",
                "source_url": source_url,
                "source_title": item.get("title", ""),
            }
        )
        if len(places) >= 5:
            break

    day_plan = _split_places_by_days([p["name"] for p in places], day_labels)
    return {
        "places": places,
        "day_plan": day_plan,
        "travel_tips": [],
        "travel_tips_th": [],
        "travel_tips_en": [],
        "travel_tips_zh": [],
        "guide_story_th": "",
        "guide_story_en": "",
        "guide_story_zh": "",
        "place_notes": [],
    }


def _guardrail_filter(plan: dict, evidences: List[dict], day_labels: List[str]):
    evidence_urls = {item.get("source_url") for item in evidences if item.get("source_url")}
    evidence_lookup = {
        item.get("source_url"): item.get("title", "")
        for item in evidences
        if item.get("source_url")
    }

    places = []
    for place in plan.get("places", []):
        source_url = place.get("source_url")
        if not source_url or source_url not in evidence_urls:
            continue
        places.append(
            {
                "name": place.get("name", "").strip(),
                "province": place.get("province", "ไม่ระบุจังหวัด"),
                "zone": place.get("zone", "ไม่ระบุโซน"),
                "source_url": source_url,
                "source_title": evidence_lookup.get(source_url, ""),
            }
        )

    day_plan = plan.get("day_plan", {})
    normalized_day_plan = {}
    place_names = [place["name"] for place in places]
    for day in day_labels:
        day_places = day_plan.get(day, [])
        valid_day_places = []
        for day_place in day_places:
            matched_name = _match_place_name(day_place, place_names)
            if matched_name and matched_name not in valid_day_places:
                valid_day_places.append(matched_name)
        normalized_day_plan[day] = valid_day_places

    # If the model gives unmatchable day names, still provide a usable itinerary.
    if places and all(not normalized_day_plan[day] for day in day_labels):
        normalized_day_plan = _split_places_by_days(place_names, day_labels)

    raw_place_notes = plan.get("place_notes", [])
    normalized_place_notes = []
    for note in raw_place_notes:
        source_url = note.get("source_url")
        if not source_url or source_url not in evidence_urls:
            continue
        matched_name = _match_place_name(note.get("name", ""), place_names)
        if not matched_name:
            continue
        normalized_place_notes.append(
            {
                "name": matched_name,
                "source_url": source_url,
                "history_note_th": note.get("history_note_th", ""),
                "history_note_en": note.get("history_note_en", ""),
                "history_note_zh": note.get("history_note_zh", ""),
                "extra_spot_th": note.get("extra_spot_th", ""),
                "extra_spot_en": note.get("extra_spot_en", ""),
                "extra_spot_zh": note.get("extra_spot_zh", ""),
            }
        )

    return {
        "places": places,
        "day_plan": normalized_day_plan,
        "travel_tips": plan.get("travel_tips", []),
        "travel_tips_th": plan.get("travel_tips_th", []),
        "travel_tips_en": plan.get("travel_tips_en", []),
        "travel_tips_zh": plan.get("travel_tips_zh", []),
        "guide_story_th": plan.get("guide_story_th", ""),
        "guide_story_en": plan.get("guide_story_en", ""),
        "guide_story_zh": plan.get("guide_story_zh", ""),
        "place_notes": normalized_place_notes,
    }


def _render_fallback_insufficient(language: str):
    return (
        "ข้อมูลยังไม่พอ / Not enough evidence yet / 目前证据不足\n"
        "- ยังไม่พบหลักฐานเพียงพอสำหรับจัดแผนเที่ยวที่เชื่อถือได้\n"
        "- There is not enough reliable evidence to produce a trustworthy trip plan.\n"
        "- 目前可用的可靠证据不足，暂时无法生成可信的旅行规划。\n"
        "- ลองระบุชื่อซีรีส์/พื้นที่/ช่วงเวลาให้ชัดขึ้น แล้วถามใหม่อีกครั้ง"
        "\n- Please provide a more specific series name, location, or time period and try again."
        "\n- 请提供更具体的剧名、地点或时间范围后再试一次。"
    )


def _pick_by_language(language: str, th_value, en_value, zh_value):
    if language == "th":
        return th_value or en_value or zh_value
    if language == "zh":
        return zh_value or en_value or th_value
    return en_value or th_value or zh_value


def build_route_plan_with_gemini(
    user_query: str,
    evidences: List[dict],
    model_name: str | None = None,
):
    language = _detect_user_language(user_query)
    day_count = _detect_requested_days(user_query)
    day_labels = _build_day_labels(day_count)

    if len(evidences) < 3:
        return {
            "status": "insufficient_evidence",
            "answer_text": _render_fallback_insufficient(language),
            "plan": {"places": [], "day_plan": {label: [] for label in day_labels}},
        }

    api_key, env_model_name = _gemini_config()
    selected_model = model_name or env_model_name

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a cautious travel planner assistant.
Use ONLY the evidence below. Do not invent places.

User question:
{user_query}

Evidence:
{_build_evidence_section(evidences)}

Rules:
1) Return JSON only.
2) Every place must include source_url from evidence exactly.
3) If evidence is weak, return empty places.
4) Write in a friendly travel-guide tone.
5) Always provide Thai, English, and Chinese story/tips fields.
6) Include short historical/cultural context and one nearby recommendation for each place when possible.
7) Build exactly {day_count} trip day(s), using day_plan keys: {", ".join(day_labels)}.

JSON schema:
{{
  "places": [
    {{
      "name": "string",
      "province": "string",
      "zone": "string",
      "source_url": "string"
    }}
  ],
  "day_plan": {{
    "Day 1": ["place name"],
    "Day 2": ["place name"]
  }},
  "guide_story_th": "2-5 sentences",
  "guide_story_en": "2-5 sentences",
  "guide_story_zh": "2-5 sentences",
  "travel_tips_th": ["string", "string"],
  "travel_tips_en": ["string", "string"],
  "travel_tips_zh": ["string", "string"],
  "place_notes": [
    {{
      "name": "string",
      "source_url": "string",
      "history_note_th": "string",
      "history_note_en": "string",
      "history_note_zh": "string",
      "extra_spot_th": "string",
      "extra_spot_en": "string",
      "extra_spot_zh": "string"
    }}
  ]
}}
""".strip()

    model_error = None
    try:
        response = client.models.generate_content(model=selected_model, contents=prompt)
        parsed = _extract_json((response.text or "{}"))
        filtered_plan = _guardrail_filter(parsed, evidences, day_labels=day_labels)
    except Exception as exc:
        model_error = str(exc)
        filtered_plan = {"places": [], "day_plan": {label: [] for label in day_labels}, "travel_tips": []}

    if not filtered_plan["places"]:
        fallback_plan = _fallback_plan_from_evidence(evidences, day_labels=day_labels)
        if fallback_plan["places"]:
            filtered_plan = fallback_plan
        else:
            return {
                "status": "insufficient_evidence",
                "answer_text": _render_fallback_insufficient(language),
                "plan": filtered_plan,
            }

    if not filtered_plan["places"]:
        return {
            "status": "insufficient_evidence",
            "answer_text": _render_fallback_insufficient(language),
            "plan": filtered_plan,
        }

    if language == "th":
        lines = [f"คำถาม: {user_query}", "", "แผนทริปแนะนำ (evidence-backed):"]
    elif language == "zh":
        lines = [f"问题：{user_query}", "", "推荐行程（基于证据）："]
    else:
        lines = [f"Question: {user_query}", "", "Recommended trip plan (evidence-backed):"]

    for idx, place in enumerate(filtered_plan["places"], start=1):
        lines.append(
            f"{idx}. {place['name']} - {place['province']} / {place['zone']} ({place['source_url']})"
        )

    if model_error:
        lines.append("")
        if language == "th":
            lines.append("หมายเหตุ: บริการโมเดลหนาแน่นชั่วคราว ระบบใช้แผนสำรองจากหลักฐานที่ดึงได้")
        elif language == "zh":
            lines.append("备注：模型服务暂时繁忙，当前结果使用检索证据回退方案生成。")
        else:
            lines.append("Note: The model service is temporarily busy. This answer uses retrieval-based fallback.")

    lines.append("")
    if language == "th":
        lines.append("แผนเที่ยว:")
    elif language == "zh":
        lines.append("行程安排:")
    else:
        lines.append("Trip Plan:")
    for day in day_labels:
        entries = filtered_plan["day_plan"].get(day, [])
        if entries:
            lines.append(f"- {day}: " + " -> ".join(entries))
        else:
            if language == "th":
                lines.append(f"- {day}: (ข้อมูลไม่เพียงพอ)")
            elif language == "zh":
                lines.append(f"- {day}: （证据不足）")
            else:
                lines.append(f"- {day}: (insufficient evidence)")

    guide_story_th = filtered_plan.get("guide_story_th", "").strip()
    guide_story_en = filtered_plan.get("guide_story_en", "").strip()
    guide_story_zh = filtered_plan.get("guide_story_zh", "").strip()
    guide_story = _pick_by_language(language, guide_story_th, guide_story_en, guide_story_zh)
    if guide_story:
        lines.append("")
        if language == "th":
            lines.append("ไกด์เล่าเรื่อง:")
        elif language == "zh":
            lines.append("导览故事:")
        else:
            lines.append("Guide Story:")
        lines.append(guide_story)

    place_notes = filtered_plan.get("place_notes", [])
    if place_notes:
        lines.append("")
        if language == "th":
            lines.append("เกร็ดประวัติและจุดแนะนำเพิ่มเติม:")
        elif language == "zh":
            lines.append("历史与周边推荐:")
        else:
            lines.append("History & Nearby Recommendations:")
        for note in place_notes:
            lines.append(f"- {note['name']}")
            history_note = _pick_by_language(
                language,
                note.get("history_note_th", ""),
                note.get("history_note_en", ""),
                note.get("history_note_zh", ""),
            )
            nearby_note = _pick_by_language(
                language,
                note.get("extra_spot_th", ""),
                note.get("extra_spot_en", ""),
                note.get("extra_spot_zh", ""),
            )
            if history_note:
                if language == "th":
                    lines.append(f"  ประวัติ: {history_note}")
                elif language == "zh":
                    lines.append(f"  历史: {history_note}")
                else:
                    lines.append(f"  History: {history_note}")
            if nearby_note:
                if language == "th":
                    lines.append(f"  แนะนำเพิ่ม: {nearby_note}")
                elif language == "zh":
                    lines.append(f"  周边推荐: {nearby_note}")
                else:
                    lines.append(f"  Nearby: {nearby_note}")

    tips_th = filtered_plan.get("travel_tips_th", [])
    tips_en = filtered_plan.get("travel_tips_en", [])
    tips_zh = filtered_plan.get("travel_tips_zh", [])
    fallback_tips = filtered_plan.get("travel_tips", [])
    selected_tips = _pick_by_language(language, tips_th, tips_en, tips_zh) or fallback_tips
    if selected_tips:
        lines.append("")
        if language == "th":
            lines.append("คำแนะนำ:")
        elif language == "zh":
            lines.append("旅行建议:")
        else:
            lines.append("Tips:")
        for tip in selected_tips:
            lines.append(f"- {tip}")

    return {
        "status": "ok",
        "answer_text": "\n".join(lines),
        "plan": filtered_plan,
    }
