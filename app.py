import gc
import io
import os
import re
import time
import json
import asyncio
import threading
from typing import List, Literal, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageOps, ImageEnhance
from pydantic import BaseModel

load_dotenv(override=True)

st.set_page_config(page_title="Daniel's Exam Assistant", page_icon="🎓", layout="wide")

BOOK_CONTEXT_PATH = "book_context.json"
MAX_PAGES_PER_EXAM = 15

# Fast model first for instant response (2-4 seconds)
MODELS_TO_TRY = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
]

EXAM_TYPES = {
    "Short Quiz (10 Marks)": 10,
    "Monthly Exam (100 Marks)": 100,
}


class QuestionEvaluation(BaseModel):
    question_number: str
    visual_mark_found: str
    student_answer: str
    correct_answer: str
    status: Literal["Correct", "Partially Correct", "Incorrect"]
    score_awarded: float
    max_score: float
    reason: str


class PaperEvaluation(BaseModel):
    questions: List[QuestionEvaluation]
    total_score: float
    total_max_score: float


def get_api_key() -> str:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "")


def apply_theme(dark_mode: bool) -> None:
    if dark_mode:
        bg_main = "#09090B"
        bg_sidebar = "#111113"
        card_bg = "#141417"
        card_border = "#27272A"
        input_bg = "#18181B"
        text_color = "#FAFAFA"
        subtext_color = "#A1A1AA"
        banner_bg = "linear-gradient(135deg, #18181B 0%, #27272A 100%)"
        banner_border = "#3F3F46"
        banner_title = "#FFFFFF"
        banner_sub = "#D4D4D8"
        primary_btn_bg = "#FFFFFF"
        primary_btn_text = "#09090B"
        primary_btn_hover = "#E4E4E7"
        secondary_btn_bg = "#18181B"
        secondary_btn_hover = "#27272A"
        accent_color = "#FFFFFF"
        shadow = "0 8px 24px rgba(0, 0, 0, 0.5)"
    else:
        bg_main = "#FAFAFA"
        bg_sidebar = "#F4F4F5"
        card_bg = "#FFFFFF"
        card_border = "#E4E4E7"
        input_bg = "#FFFFFF"
        text_color = "#09090B"
        subtext_color = "#52525B"
        banner_bg = "linear-gradient(135deg, #09090B 0%, #27272A 100%)"
        banner_border = "#18181B"
        banner_title = "#FFFFFF"
        banner_sub = "#E4E4E7"
        primary_btn_bg = "#09090B"
        primary_btn_text = "#FFFFFF"
        primary_btn_hover = "#27272A"
        secondary_btn_bg = "#F4F4F5"
        secondary_btn_hover = "#E4E4E7"
        accent_color = "#09090B"
        shadow = "0 4px 16px rgba(0, 0, 0, 0.05)"

    st.markdown(
        f"""
        <style>
        header[data-testid="stHeader"] {{
            background: transparent !important;
        }}
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        [data-testid="stAppDeployButton"],
        [data-testid="manage-app-button"],
        .stAppDeployButton,
        .viewerBadge_container__r5tak,
        .viewerBadge_link__qRIco,
        div[class*="viewerBadge"],
        div[class*="StatusWidget"],
        div[class*="profileContainer"],
        a[href*="streamlit.io"],
        #MainMenu,
        footer {{
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            pointer-events: none !important;
            height: 0 !important;
            width: 0 !important;
        }}
        .block-container {{
            padding-top: 2.2rem !important;
            padding-bottom: 1.5rem !important;
        }}
        .stApp {{
            background-color: {bg_main};
            color: {text_color};
        }}
        [data-testid="stSidebar"] {{
            background-color: {bg_sidebar};
            border-right: 1px solid {card_border};
        }}
        .stMarkdown, .stText, p, label, span, div {{
            color: {text_color};
        }}
        .hero-banner {{
            background: {banner_bg};
            padding: 1.35rem 1.75rem;
            border-radius: 14px;
            box-shadow: {shadow};
            margin-bottom: 1rem;
            border: 1px solid {banner_border};
        }}
        .hero-banner h1 {{
            color: {banner_title} !important;
            margin: 0;
            font-size: 1.85rem;
            font-weight: 800;
            letter-spacing: -0.4px;
        }}
        .hero-banner p {{
            color: {banner_sub} !important;
            margin: 0.35rem 0 0 0;
            font-size: 0.95rem;
        }}
        .score-banner {{
            background: {card_bg};
            border: 1.5px solid {accent_color};
            border-radius: 14px;
            padding: 1.05rem 1.45rem;
            margin: 0.8rem 0 1.1rem 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: {shadow};
        }}
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: {card_bg};
            border: 1px solid {card_border} !important;
            border-radius: 14px !important;
            box-shadow: {shadow};
            padding: 0.35rem;
        }}
        .stButton > button[kind="primary"] {{
            background-color: {primary_btn_bg} !important;
            color: {primary_btn_text} !important;
            border: 1px solid {primary_btn_bg} !important;
            border-radius: 10px !important;
            padding: 0.6rem 1.35rem !important;
            font-weight: 700 !important;
            font-size: 0.98rem !important;
            box-shadow: {shadow} !important;
        }}
        .stButton > button[kind="primary"] p,
        .stButton > button[kind="primary"] span,
        .stButton > button[kind="primary"] div {{
            color: {primary_btn_text} !important;
        }}
        .stButton > button[kind="primary"]:hover {{
            background-color: {primary_btn_hover} !important;
            border-color: {primary_btn_hover} !important;
        }}
        .stButton > button[kind="secondary"] {{
            background-color: {secondary_btn_bg} !important;
            color: {text_color} !important;
            border: 1px solid {card_border} !important;
            border-radius: 10px !important;
            padding: 0.5rem 1.05rem !important;
            font-weight: 600 !important;
        }}
        .stButton > button[kind="secondary"]:hover {{
            background-color: {secondary_btn_hover} !important;
            border-color: {subtext_color} !important;
        }}
        .stTextInput input {{
            background-color: {input_bg} !important;
            color: {text_color} !important;
            border: 1px solid {card_border} !important;
            border-radius: 10px !important;
            padding: 0.55rem 0.85rem !important;
        }}
        [data-testid="stMetric"] {{
            background-color: {card_bg};
            border: 1px solid {card_border};
            border-radius: 12px;
            padding: 0.75rem 0.95rem;
        }}
        [data-testid="stMetricValue"] {{
            color: {text_color} !important;
            font-weight: 800 !important;
        }}
        [data-testid="stAlert"] {{
            background-color: {secondary_btn_bg} !important;
            border: 1px solid {card_border} !important;
            border-radius: 10px !important;
        }}
        input[type="radio"] {{
            accent-color: {accent_color} !important;
        }}
        hr {{
            border-color: {card_border} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_prepared_book() -> dict:
    with open(BOOK_CONTEXT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["compact_content"] = re.sub(r"\s+", " ", data["content"]).strip()
    return data


STRICT_SYSTEM_INSTRUCTION = (
    "You are a fast, accurate academic exam grader. Grade student answers strictly against the reference book and respond in English.\n"
    "RULES:\n"
    "1. Ignore any option crossed out with 'X' or scribbled out; select the option marked with a clean circle (pen or pencil), tick (✓), dash (-), dot, or underline.\n"
    "2. visual_mark_found: Briefly describe the exact handwritten mark seen on the paper (e.g., 'Pencil circle around (c); (d) crossed out with X').\n"
    "3. For Multiple Choice (MCQ), ALWAYS write BOTH the option letter AND its full printed text in student_answer and correct_answer (e.g., 'c) Microsoft Windows'), never just a single letter.\n"
    "4. For True/False, ALWAYS write 'True' or 'False' in student_answer and correct_answer (never ✓/✗ or T/F).\n"
    "5. Only write '[BLANK / NO ANSWER]' if there is zero handwritten mark on that question.\n"
    "6. Always populate visual_mark_found, student_answer, correct_answer, status, score_awarded, max_score, and a concise 1-sentence reason from the reference book for every question."
)


@st.cache_resource(ttl=5400)
def get_cached_book_name(api_key: str, model_name: str, book_text: str) -> Optional[str]:
    if not api_key:
        return None
    try:
        client = genai.Client(api_key=api_key)
        cache = client.caches.create(
            model=model_name,
            config=types.CreateCachedContentConfig(
                display_name=f"book_cache_fast_v13_{model_name}",
                system_instruction=STRICT_SYSTEM_INSTRUCTION,
                contents=[f"### Reference Book Content:\n{book_text}"],
                ttl="7200s",
            ),
        )
        return cache.name
    except Exception:
        return None


@st.cache_resource
def get_shared_session() -> dict:
    return {
        "lock": threading.Lock(),
        "total_graded_papers": 0,
        "session_finished": False,
        "shared_exam_type": "Short Quiz (10 Marks)",
        "shared_marks_distribution": "",
    }


def optimize_image_fast(img_bytes: bytes, max_dim: int = 1200) -> bytes:
    """Lightweight fast resize + mild contrast so upload & inference take ~2-4 seconds."""
    img = Image.open(io.BytesIO(img_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.BILINEAR)
    img = ImageEnhance.Contrast(img).enhance(1.18)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82, optimize=True)
    return buf.getvalue()


async def grade_paper_async(
    api_key: str,
    book_text: str,
    uploaded_files: list,
    total_marks: int,
    marks_distribution: str,
) -> Tuple[PaperEvaluation, str, float]:
    t0 = time.perf_counter()
    client = genai.Client(api_key=api_key)

    distribution_rule = (
        f"Follow this exact marks distribution / rubric rule: {marks_distribution}"
        if marks_distribution.strip()
        else f"If question marks are written on the paper, use them; otherwise distribute the {total_marks} marks fairly across all questions so they sum up to {total_marks}."
    )

    grading_instructions = (
        "All attached images belong to ONE SINGLE student's exam submission.\n"
        "- Ignore crossed-out ('X') options; choose the option with a clean circle (pen or pencil), tick, dash, or underline.\n"
        "- In visual_mark_found, briefly state what handwritten mark/cross-out you saw on the paper.\n"
        "- For MCQ, ALWAYS include BOTH letter and full option text (e.g., 'c) Microsoft Windows') in student_answer and correct_answer.\n"
        "- For True/False, ALWAYS write 'True' or 'False' (never ✓/✗ or T/F).\n"
        "- Always provide correct_answer and a short 1-sentence reason from the reference book.\n"
        f"Total Marks: {total_marks} | Pages: {len(uploaded_files)} | Rubric: {distribution_rule}"
    )

    image_parts = []
    for f in uploaded_files:
        opt_bytes = optimize_image_fast(f.getvalue())
        image_parts.append(types.Part.from_bytes(data=opt_bytes, mime_type="image/jpeg"))

    last_error = None
    for model_name in MODELS_TO_TRY:
        cache_name = get_cached_book_name(api_key, model_name, book_text)
        attempts = [True, False] if cache_name else [False]
        for attempt_cached in attempts:
            try:
                config_kwargs = {
                    "response_mime_type": "application/json",
                    "response_schema": PaperEvaluation,
                    "temperature": 0.0,
                }
                if attempt_cached and cache_name:
                    contents = [grading_instructions] + image_parts
                    config_kwargs["cached_content"] = cache_name
                else:
                    contents = [f"### Reference Book:\n{book_text}\n\n{grading_instructions}"] + image_parts
                    config_kwargs["system_instruction"] = STRICT_SYSTEM_INSTRUCTION

                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(**config_kwargs),
                    ),
                    timeout=30.0,
                )
                elapsed = round(time.perf_counter() - t0, 1)
                status_tag = f"{model_name} ({'Cached' if (attempt_cached and cache_name) else 'Direct'})"
                return PaperEvaluation.model_validate_json(response.text), status_tag, elapsed
            except Exception as e:
                last_error = e
                if attempt_cached:
                    get_cached_book_name.clear()
                    continue
                break

    raise last_error


# Initialize resources
api_key = get_api_key()
book_data = load_prepared_book()
shared = get_shared_session()

if "my_last_result" not in st.session_state:
    st.session_state.my_last_result = None
if "last_meta" not in st.session_state:
    st.session_state.last_meta = None
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 1
if "is_grading" not in st.session_state:
    st.session_state.is_grading = False

apply_theme(st.session_state.dark_mode)

# Top Header Banner + Theme Toggle
header_col1, header_col2 = st.columns([5.2, 1])
with header_col1:
    st.markdown(
        """
        <div class="hero-banner">
            <h1>🎓 Daniel's Exam Assistant</h1>
            <p>Fast, accurate multi-page handwritten & printed exam grading powered by your reference book</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_col2:
    st.write("")
    theme_label = "☀️ Light" if st.session_state.dark_mode else "🌙 Dark"
    if st.button(theme_label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# Always-Visible Session & Counter Bar
with st.container(border=True):
    bar_col1, bar_col2, bar_col3, bar_col4 = st.columns([2.2, 1.5, 1.3, 1.6])
    with bar_col1:
        st.markdown(f"📖 **Reference Book:** `{book_data['source_file']}`")
    with bar_col2:
        st.markdown(f"👨‍🎓 **Students Graded:** **{shared['total_graded_papers']}**")
    with bar_col3:
        if not shared["session_finished"] and shared["total_graded_papers"] > 0:
            if st.button("↩️ Undo (-1)", key="top_undo", use_container_width=True):
                with shared["lock"]:
                    if shared["total_graded_papers"] > 0:
                        shared["total_graded_papers"] -= 1
                st.rerun()
    with bar_col4:
        if not shared["session_finished"]:
            if st.button("🏁 End & Submit Session", key="top_end", type="primary", use_container_width=True):
                with shared["lock"]:
                    shared["session_finished"] = True
                st.rerun()
        else:
            if st.button("🔄 Start New Session", key="top_new", type="primary", use_container_width=True):
                with shared["lock"]:
                    shared["total_graded_papers"] = 0
                    shared["session_finished"] = False
                    shared["shared_marks_distribution"] = ""
                st.session_state.my_last_result = None
                st.session_state.last_meta = None
                st.session_state.uploader_key += 1
                gc.collect()
                st.rerun()

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Session Dashboard")
    st.info(f"📖 **Reference Book:**\n\n{book_data['source_file']}")
    st.metric(label="👨‍🎓 Total Students Graded", value=shared["total_graded_papers"])

    if not shared["session_finished"]:
        if shared["total_graded_papers"] > 0:
            if st.button("↩️ Undo Last Student (-1)", key="side_undo", use_container_width=True):
                with shared["lock"]:
                    if shared["total_graded_papers"] > 0:
                        shared["total_graded_papers"] -= 1
                st.rerun()
        st.divider()
        if st.button("🏁 End Session & Submit", key="side_end", type="primary", use_container_width=True):
            with shared["lock"]:
                shared["session_finished"] = True
            st.rerun()
    else:
        if st.button("🔄 Start New Session", key="side_new", type="primary", use_container_width=True):
            with shared["lock"]:
                shared["total_graded_papers"] = 0
                shared["session_finished"] = False
                shared["shared_marks_distribution"] = ""
            st.session_state.my_last_result = None
            st.session_state.last_meta = None
            st.session_state.uploader_key += 1
            gc.collect()
            st.rerun()

# Main Workspace
if shared["session_finished"]:
    st.balloons()
    st.success(
        f"🎉 Session completed! Total number of students graded in this session: "
        f"**{shared['total_graded_papers']}** students."
    )
    if st.button("🔄 Start New Session", key="main_new", type="primary"):
        with shared["lock"]:
            shared["total_graded_papers"] = 0
            shared["session_finished"] = False
            shared["shared_marks_distribution"] = ""
        st.session_state.my_last_result = None
        st.session_state.last_meta = None
        st.session_state.uploader_key += 1
        gc.collect()
        st.rerun()
else:
    with st.container(border=True):
        col_type, col_dist = st.columns([1, 2])
        with col_type:
            exam_options = list(EXAM_TYPES.keys())
            default_idx = exam_options.index(shared["shared_exam_type"])
            exam_type = st.radio("1️⃣ Select Assessment Type:", exam_options, index=default_idx)
            if exam_type != shared["shared_exam_type"]:
                with shared["lock"]:
                    shared["shared_exam_type"] = exam_type

        with col_dist:
            marks_distribution = st.text_input(
                "2️⃣ Question Marks Rule (Saved for the whole session):",
                value=shared["shared_marks_distribution"],
                placeholder="e.g., Essay questions = 2.5 marks each | or: Q1=2.5, Q2=2.5, Q3=5",
            )
            if marks_distribution != shared["shared_marks_distribution"]:
                with shared["lock"]:
                    shared["shared_marks_distribution"] = marks_distribution

    with st.container(border=True):
        st.markdown(f"**3️⃣ Snap or Attach Pages for ONE Student (Max {MAX_PAGES_PER_EXAM} pages):**")
        uploaded_files = st.file_uploader(
            "Tap 'Browse files' to Take Photo or Choose from Photo Library (select multiple pages at once):",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key=f"fast_uploader_{st.session_state.uploader_key}",
        )

        if uploaded_files:
            if len(uploaded_files) > MAX_PAGES_PER_EXAM:
                st.warning(f"Only the first {MAX_PAGES_PER_EXAM} pages will be graded.")
                uploaded_files = uploaded_files[:MAX_PAGES_PER_EXAM]

            st.caption(f"✅ {len(uploaded_files)} page(s) attached and ready.")
            if st.button(
                f"✨ Grade This Student ({len(uploaded_files)} page(s))",
                type="primary",
                use_container_width=True,
                disabled=(not api_key or st.session_state.is_grading),
            ):
                st.session_state.is_grading = True
                with st.spinner(f"Grading {len(uploaded_files)} page(s)..."):
                    try:
                        result, status_tag, elapsed = asyncio.run(
                            grade_paper_async(
                                api_key=api_key,
                                book_text=book_data["compact_content"],
                                uploaded_files=uploaded_files,
                                total_marks=EXAM_TYPES[exam_type],
                                marks_distribution=shared["shared_marks_distribution"],
                            )
                        )
                        st.session_state.my_last_result = result
                        st.session_state.last_meta = f"⚡ Graded 1 student ({len(uploaded_files)} pages) in {elapsed}s via {status_tag}"
                        st.session_state.uploader_key += 1
                        with shared["lock"]:
                            shared["total_graded_papers"] += 1
                    except Exception as e:
                        st.error(f"Grading error: {e}")
                    finally:
                        st.session_state.is_grading = False
                        gc.collect()
                st.rerun()

    # Results Section
    if st.session_state.my_last_result:
        res = st.session_state.my_last_result
        res_top1, res_top2 = st.columns([4, 1.4])
        with res_top1:
            if st.session_state.last_meta:
                st.caption(st.session_state.last_meta)
        with res_top2:
            if st.button("➡️ Next Student (Clear)", use_container_width=True):
                st.session_state.my_last_result = None
                st.session_state.last_meta = None
                st.session_state.uploader_key += 1
                gc.collect()
                st.rerun()

        badge_bg = "#FFFFFF" if st.session_state.dark_mode else "#09090B"
        badge_text = "#09090B" if st.session_state.dark_mode else "#FFFFFF"

        st.markdown(
            f"""
            <div class="score-banner">
                <div>
                    <span style="font-size: 0.85rem; opacity: 0.75; font-weight: 600;">FINAL STUDENT SCORE</span>
                    <h2 style="margin: 0; font-size: 1.85rem; font-weight: 800;">🏆 {res.total_score} / {res.total_max_score}</h2>
                </div>
                <div style="text-align: right;">
                    <span style="background: {badge_bg}; color: {badge_text}; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 0.88rem;">
                        {len(res.questions)} Question(s) Evaluated
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for q in res.questions:
            icon = "🟢" if q.status == "Correct" else ("🟡" if q.status == "Partially Correct" else "🔴")
            with st.container(border=True):
                col1, col2 = st.columns([4.2, 1])
                with col1:
                    st.markdown(f"**{icon} Question {q.question_number} — {q.status}**")
                    st.caption(f"👁️ Detected Mark: {q.visual_mark_found}")
                    st.write(f"**Student Answer:** {q.student_answer}")
                    st.write(f"**Correct Answer:** {q.correct_answer}")
                    st.info(f"**Grading Reason:** {q.reason}")
                with col2:
                    st.metric("Score", f"{q.score_awarded} / {q.max_score}")
