import gc
import io
import os
import re
import time
import json
import base64
import asyncio
import threading
from typing import List, Literal, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageOps
from pydantic import BaseModel

load_dotenv(override=True)

st.set_page_config(page_title="Daniel's Exam Assistant", page_icon="🎓", layout="wide")

BOOK_CONTEXT_PATH = "book_context.json"
MAX_PAGES_PER_EXAM = 15

# Image resolution sent to the model (higher = faint pencil marks survive)
IMAGE_MAX_DIM = 2000
IMAGE_JPEG_QUALITY = 90

# Stronger model first (reading handwriting is the hard part). The lite model is only the fallback.
# NOTE: check these names against the models available on your API key.
MODELS_TO_TRY = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
]

EXAM_TYPES = {
    "Short Quiz (10 Marks)": 10,
    "Monthly Exam (100 Marks)": 100,
}


# ----------------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------------
class ExtractedAnswer(BaseModel):
    """Stage 1: what is PHYSICALLY on the paper. The model does not know the correct answers here."""
    page: int
    question_number: str
    question_snippet: str
    question_type: Literal["mcq", "true_false", "essay"]
    max_score_on_paper: float  # marks printed next to the question, 0 if not shown
    # Field order matters: the model first DESCRIBES what it sees, then decides.
    visual_mark_found: str
    has_handwriting: bool
    student_answer: str
    confidence: Literal["high", "medium", "low"]


class ExtractionResult(BaseModel):
    answers: List[ExtractedAnswer]


class GradedQuestion(BaseModel):
    """Stage 2: grading from text only (no images)."""
    question_number: str
    correct_answer: str
    status: Literal["Correct", "Partially Correct", "Incorrect"]
    score_awarded: float
    max_score: float
    reason: str


class GradingResult(BaseModel):
    questions: List[GradedQuestion]


class QuestionEvaluation(BaseModel):
    question_number: str
    question_snippet: str
    is_blank: bool
    visual_mark_found: str
    student_answer: str
    correct_answer: str
    status: Literal["Correct", "Partially Correct", "Incorrect"]
    score_awarded: float
    max_score: float
    reason: str
    confidence: str = "high"
    needs_review: bool = False


class PaperEvaluation(BaseModel):
    questions: List[QuestionEvaluation]
    total_score: float
    total_max_score: float
    warnings: List[str] = []


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
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


def render_mobile_multi_capture(dark_mode: bool, reset_token: int, max_pages: int = 15) -> List[Tuple[bytes, str]]:
    card = "#18181B" if dark_mode else "#F4F4F5"
    border = "#27272A" if dark_mode else "#E4E4E7"
    text = "#FAFAFA" if dark_mode else "#09090B"
    subtext = "#A1A1AA" if dark_mode else "#52525B"
    btn_bg = "#FFFFFF" if dark_mode else "#09090B"
    btn_fg = "#09090B" if dark_mode else "#FFFFFF"

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{ box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        body {{ margin: 0; padding: 4px; background: transparent; color: {text}; }}
        .btn-row {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }}
        .cap-btn {{
            flex: 1; min-width: 145px; padding: 12px 14px; border-radius: 10px;
            font-weight: 700; font-size: 14px; cursor: pointer; text-align: center;
            border: 1.5px solid {btn_bg}; transition: opacity 0.15s; user-select: none;
        }}
        .cap-btn.primary {{ background: {btn_bg}; color: {btn_fg}; }}
        .cap-btn.secondary {{ background: {card}; color: {text}; border-color: {border}; }}
        .cap-btn:active {{ opacity: 0.75; }}
        .status {{ font-size: 13px; color: {subtext}; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }}
        .clear-link {{ color: #EF4444; cursor: pointer; font-weight: 700; text-decoration: underline; padding: 2px 6px; }}
        .grid {{ display: flex; gap: 10px; overflow-x: auto; padding-bottom: 6px; }}
        .thumb {{
            position: relative; width: 78px; height: 98px; flex-shrink: 0;
            border-radius: 10px; overflow: hidden; border: 1.5px solid {border}; background: {card};
        }}
        .thumb img {{ width: 100%; height: 100%; object-fit: cover; }}
        .thumb .badge {{
            position: absolute; bottom: 4px; left: 4px; background: rgba(0,0,0,0.75);
            color: #fff; font-size: 11px; padding: 2px 6px; border-radius: 6px; font-weight: 700;
        }}
        .thumb .del {{
            position: absolute; top: 4px; right: 4px; width: 22px; height: 22px;
            border-radius: 50%; background: rgba(220,38,38,0.9); color: #fff;
            border: none; font-size: 13px; font-weight: bold; cursor: pointer;
            display: flex; align-items: center; justify-content: center;
        }}
    </style>
    </head>
    <body data-session="{reset_token}">
        <input type="file" id="camInput" accept="image/*" capture="environment" style="display:none" />
        <input type="file" id="galInput" accept="image/*" multiple style="display:none" />

        <div class="btn-row">
            <div class="cap-btn primary" onclick="document.getElementById('camInput').click()">
                📸 Snap Page (Camera)
            </div>
            <div class="cap-btn secondary" onclick="document.getElementById('galInput').click()">
                🖼️ Choose from Gallery
            </div>
        </div>

        <div class="status">
            <span id="statusText">Tap "Snap Page" to take photos one by one, or "Choose from Gallery".</span>
            <span id="clearBtn" class="clear-link" style="display:none" onclick="clearAll()">🗑️ Clear</span>
        </div>

        <div class="grid" id="thumbGrid"></div>

        <script>
            const MAX_PAGES = {max_pages};
            const SESSION_TOKEN = {reset_token};
            let images = [];

            function sendToStreamlit() {{
                window.parent.postMessage({{
                    isStreamlitMessage: true,
                    type: "streamlit:setComponentValue",
                    value: {{ token: SESSION_TOKEN, images: images }}
                }}, "*");
                updateHeight();
            }}

            function updateHeight() {{
                const h = images.length > 0 ? 195 : 95;
                window.parent.postMessage({{
                    isStreamlitMessage: true,
                    type: "streamlit:setFrameHeight",
                    height: h
                }}, "*");
            }}

            function compressImage(file) {{
                return new Promise((resolve) => {{
                    const reader = new FileReader();
                    reader.onload = (e) => {{
                        const img = new Image();
                        img.onload = () => {{
                            const maxDim = {IMAGE_MAX_DIM};
                            let w = img.width;
                            let h = img.height;
                            if (Math.max(w, h) > maxDim) {{
                                if (w > h) {{ h = Math.round(h * maxDim / w); w = maxDim; }}
                                else {{ w = Math.round(w * maxDim / h); h = maxDim; }}
                            }}
                            const canvas = document.createElement("canvas");
                            canvas.width = w;
                            canvas.height = h;
                            const ctx = canvas.getContext("2d");
                            ctx.drawImage(img, 0, 0, w, h);
                            resolve(canvas.toDataURL("image/jpeg", 0.92));
                        }};
                        img.src = e.target.result;
                    }};
                    reader.readAsDataURL(file);
                }});
            }}

            async function handleFiles(fileList) {{
                if (!fileList || fileList.length === 0) return;
                document.getElementById("statusText").innerText = "⏳ Adding photo(s)...";
                for (let i = 0; i < fileList.length; i++) {{
                    if (images.length >= MAX_PAGES) break;
                    const dataUrl = await compressImage(fileList[i]);
                    images.push(dataUrl);
                }}
                render();
                sendToStreamlit();
            }}

            function removeAt(idx) {{
                images.splice(idx, 1);
                render();
                sendToStreamlit();
            }}

            function clearAll() {{
                images = [];
                render();
                sendToStreamlit();
            }}

            function render() {{
                const grid = document.getElementById("thumbGrid");
                const status = document.getElementById("statusText");
                const clearBtn = document.getElementById("clearBtn");
                grid.innerHTML = "";
                if (images.length === 0) {{
                    status.innerText = 'Tap "Snap Page" to take photos one by one, or "Choose from Gallery".';
                    clearBtn.style.display = "none";
                }} else {{
                    status.innerText = `✅ ${{images.length}} page(s) ready — tap "Snap Page" again for more.`;
                    clearBtn.style.display = "inline";
                }}
                images.forEach((src, idx) => {{
                    const div = document.createElement("div");
                    div.className = "thumb";
                    div.innerHTML = `
                        <img src="${{src}}" />
                        <span class="badge">P${{idx + 1}}</span>
                        <button class="del" onclick="removeAt(${{idx}})">×</button>
                    `;
                    grid.appendChild(div);
                }});
            }}

            document.getElementById("camInput").addEventListener("change", (e) => {{
                handleFiles(e.target.files);
                e.target.value = "";
            }});
            document.getElementById("galInput").addEventListener("change", (e) => {{
                handleFiles(e.target.files);
                e.target.value = "";
            }});

            window.parent.postMessage({{
                isStreamlitMessage: true,
                type: "streamlit:componentReady",
                apiVersion: 1
            }}, "*");
            sendToStreamlit();
        </script>
    </body>
    </html>
    """

    comp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mobile_cam_comp")
    os.makedirs(comp_dir, exist_ok=True)
    index_path = os.path.join(comp_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html_code)

    comp_func = components.declare_component("mobile_multi_capture", path=comp_dir)
    payload = comp_func(key=f"multi_cap_v8_{reset_token}", default={"token": reset_token, "images": []})

    pages: List[Tuple[bytes, str]] = []
    if isinstance(payload, dict) and payload.get("token") == reset_token:
        raw_list = payload.get("images", [])
        for data_url in raw_list:
            if isinstance(data_url, str) and "," in data_url:
                _, b64_data = data_url.split(",", 1)
                pages.append((base64.b64decode(b64_data), "image/jpeg"))
    return pages


@st.cache_resource
def load_prepared_book() -> dict:
    with open(BOOK_CONTEXT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    text = data["content"]
    # Keep line breaks / paragraphs (they help the model locate the right passage).
    # Only collapse runs of spaces/tabs and excessive blank lines.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    data["compact_content"] = text.strip()
    return data


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------
EXTRACT_SYSTEM_INSTRUCTION = (
    "You are a TRANSCRIBER of student exam papers, NOT a grader. You do NOT know the correct answers and must never "
    "try to work them out. Your only job is to report what is physically visible on the pages.\n"
    "RULES:\n"
    "- List ONLY questions that are actually visible on the pages. Never invent questions or answers.\n"
    "- Printed parentheses around option letters like '(a)', '(b)', '(c)', '(d)' are PRINTED TEXT, not student marks.\n"
    "- A student mark is anything added by hand on top of the printed paper: a pen/pencil circle, tick, check mark, underline, dash, "
    "highlight, cross, filled bubble, or handwritten text. Marks can be faint, thin, pencil-light, or in blue/black/red ink. "
    "Look carefully at EVERY question before deciding it is blank.\n"
    "- Options crossed out with 'X' are rejected by the student; the chosen option is the one with a clean circle/tick/underline.\n"
    "- First describe in visual_mark_found exactly what handwritten marks you see next to this question. Then set has_handwriting.\n"
    "- has_handwriting=true if you see ANY hand-made mark for that question, even a faint one (then use confidence 'medium' or 'low' if unsure). "
    "Only set has_handwriting=false when you looked and the question is clearly untouched. Then visual_mark_found='None (blank)' "
    "and student_answer='[BLANK / NO ANSWER]'.\n"
    "- If you are unsure whether a mark exists, report your best reading with confidence='low' instead of marking it blank.\n"
    "- For MCQ, student_answer = option letter + full printed option text (e.g. 'c) Microsoft Windows').\n"
    "- For True/False, student_answer = 'True' or 'False' (never a symbol or T/F).\n"
    "- For essay questions, student_answer = a faithful transcription of the handwriting. Write [unclear] for words you cannot read. Never complete or correct the student's text.\n"
    "- visual_mark_found: describe exactly what you see and where (e.g. 'pen circle around option b on page 2').\n"
    "- confidence: 'high' only if the mark/text is clearly readable; 'medium' if somewhat unclear; 'low' if you are guessing.\n"
    "- question_snippet: the first 5-8 words of the printed question so each answer stays tied to its own question.\n"
    "- max_score_on_paper: the marks printed beside the question, or 0 if none is shown.\n"
    "- Write all fields in English except quoted student/printed text, which stays in its original language."
)

GRADER_SYSTEM_INSTRUCTION = (
    "You are an honest, accurate academic exam grader. You are given (1) a reference book and (2) a list of student "
    "answers that were already transcribed from the paper. Grade ONLY from the reference book.\n"
    "RULES:\n"
    "- Do NOT change or reinterpret the student's answer; it is already extracted.\n"
    "- correct_answer: the exact answer according to the reference book (letter + full text for MCQ, 'True'/'False' for true/false, "
    "a short model answer for essay questions).\n"
    "- If the reference book does not contain the answer, set correct_answer='NOT FOUND IN BOOK', status='Incorrect', "
    "score_awarded=0 and say so in the reason. Never answer from your own outside knowledge.\n"
    "- If a student's answer is '[BLANK / NO ANSWER]', score_awarded must be 0 and status 'Incorrect'.\n"
    "- Essay answers: award partial credit by how many required points from the book are present; status 'Partially Correct' when partially right.\n"
    "- reason: one short sentence that cites the book content used.\n"
    "- Return exactly one result per input question, using the same question_number."
)


@st.cache_resource(ttl=5400)
def get_cached_book_name(api_key: str, model_name: str, book_text: str) -> Optional[str]:
    """Context cache of the book for the grading (text-only) stage."""
    if not api_key:
        return None
    try:
        client = genai.Client(api_key=api_key)
        cache = client.caches.create(
            model=model_name,
            config=types.CreateCachedContentConfig(
                display_name=f"book_cache_v19_{model_name}",
                system_instruction=GRADER_SYSTEM_INSTRUCTION,
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


def prepare_clean_image(img_bytes: bytes) -> bytes:
    """Fix orientation, cap size, encode once at high quality (no double low-quality compression)."""
    img = Image.open(io.BytesIO(img_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > IMAGE_MAX_DIM:
        img.thumbnail((IMAGE_MAX_DIM, IMAGE_MAX_DIM), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=IMAGE_JPEG_QUALITY)
    return buf.getvalue()


def _high_res_kwargs() -> dict:
    level = getattr(getattr(types, "MediaResolution", None), "MEDIA_RESOLUTION_HIGH", None)
    return {"media_resolution": level} if level is not None else {}


# ----------------------------------------------------------------------------
# Stage 1: extraction (images only, NO book)
# ----------------------------------------------------------------------------
async def extract_answers_async(
    client: "genai.Client",
    pages: List[Tuple[bytes, str]],
    total_marks: int,
    errors: List[str],
) -> Tuple[ExtractionResult, str]:
    contents: list = [
        f"These {len(pages)} page(s) belong to ONE student's exam (total marks: {total_marks}). "
        "Transcribe every visible question and the student's answer to it."
    ]
    for i, (img_bytes, _) in enumerate(pages, start=1):
        contents.append(f"Page {i}:")
        contents.append(types.Part.from_bytes(data=prepare_clean_image(img_bytes), mime_type="image/jpeg"))

    all_blank_candidate: Optional[Tuple[ExtractionResult, str]] = None
    for model_name in MODELS_TO_TRY:
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=EXTRACT_SYSTEM_INSTRUCTION,
                        response_mime_type="application/json",
                        response_schema=ExtractionResult,
                        **_high_res_kwargs(),
                    ),
                ),
                timeout=75.0,
            )
            result = ExtractionResult.model_validate_json(response.text)
            if not result.answers:
                errors.append(f"[extract / {model_name}]: no questions detected")
                continue
            # If the model says EVERY question is blank, it probably failed to see the marks:
            # try the next model before accepting that.
            if not any(a.has_handwriting for a in result.answers):
                errors.append(f"[extract / {model_name}]: all questions read as blank")
                if all_blank_candidate is None:
                    all_blank_candidate = (result, model_name)
                continue
            return result, model_name
        except Exception as e:
            errors.append(f"[extract / {model_name}]: {type(e).__name__}: {e}")
    if all_blank_candidate is not None:
        return all_blank_candidate
    raise RuntimeError(" | ".join(errors))


# ----------------------------------------------------------------------------
# Stage 2: grading (text only: extracted answers + book)
# ----------------------------------------------------------------------------
async def grade_extracted_async(
    client: "genai.Client",
    api_key: str,
    book_text: str,
    extraction: ExtractionResult,
    total_marks: int,
    marks_distribution: str,
    errors: List[str],
) -> Tuple[GradingResult, str]:
    distribution_rule = (
        f"Follow this exact marks distribution / rubric rule: {marks_distribution}"
        if marks_distribution.strip()
        else (
            "If marks_on_paper is greater than 0 for a question, use it as max_score; otherwise distribute the remaining "
            f"marks fairly so that all max_score values sum to exactly {total_marks}."
        )
    )

    student_payload = [
        {
            "question_number": a.question_number,
            "question_snippet": a.question_snippet,
            "question_type": a.question_type,
            "marks_on_paper": a.max_score_on_paper,
            "student_answer": a.student_answer,
        }
        for a in extraction.answers
    ]
    grading_instructions = (
        f"Total Marks: {total_marks}\nRubric: {distribution_rule}\n\n"
        "Student answers (already transcribed, do not change them):\n"
        f"{json.dumps(student_payload, ensure_ascii=False, indent=2)}"
    )

    for model_name in MODELS_TO_TRY:
        cache_name = get_cached_book_name(api_key, model_name, book_text)
        attempts = [True, False] if cache_name else [False]
        for attempt_cached in attempts:
            try:
                config_kwargs = {
                    "response_mime_type": "application/json",
                    "response_schema": GradingResult,
                }
                if attempt_cached and cache_name:
                    contents = grading_instructions
                    config_kwargs["cached_content"] = cache_name
                else:
                    contents = f"### Reference Book:\n{book_text}\n\n{grading_instructions}"
                    config_kwargs["system_instruction"] = GRADER_SYSTEM_INSTRUCTION

                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(**config_kwargs),
                    ),
                    timeout=60.0,
                )
                result = GradingResult.model_validate_json(response.text)
                tag = f"{model_name} ({'Cached' if (attempt_cached and cache_name) else 'Direct'})"
                return result, tag
            except Exception as e:
                mode_str = "Cached" if (attempt_cached and cache_name) else "Direct"
                errors.append(f"[grade / {model_name} / {mode_str}]: {type(e).__name__}: {e}")
                if attempt_cached:
                    get_cached_book_name.clear()
                    continue
                break
    raise RuntimeError(" | ".join(errors))


# ----------------------------------------------------------------------------
# Merge + validate in code (do not blindly trust model numbers)
# ----------------------------------------------------------------------------
def merge_and_validate(extraction: ExtractionResult, grading: GradingResult, total_marks: int) -> PaperEvaluation:
    graded_by_q = {}
    for g in grading.questions:
        graded_by_q.setdefault(g.question_number.strip(), g)

    warnings: List[str] = []
    questions: List[QuestionEvaluation] = []

    for a in extraction.answers:
        g = graded_by_q.get(a.question_number.strip())
        is_blank = (not a.has_handwriting) or a.student_answer.strip().startswith("[BLANK")
        needs_review = a.confidence != "high"

        if g is None:
            warnings.append(f"Question {a.question_number} was not graded by the model — review manually.")
            questions.append(
                QuestionEvaluation(
                    question_number=a.question_number,
                    question_snippet=a.question_snippet,
                    is_blank=is_blank,
                    visual_mark_found=a.visual_mark_found,
                    student_answer=a.student_answer,
                    correct_answer="—",
                    status="Incorrect",
                    score_awarded=0.0,
                    max_score=a.max_score_on_paper,
                    reason="Not graded by the model; manual review required.",
                    confidence=a.confidence,
                    needs_review=True,
                )
            )
            continue

        max_score = max(0.0, g.max_score)
        score = max(0.0, min(g.score_awarded, max_score))
        status = g.status
        reason = g.reason

        if is_blank:
            score = 0.0
            status = "Incorrect"
        elif status == "Correct" and score < max_score:
            status = "Partially Correct"
        elif status == "Incorrect" and score > 0:
            status = "Partially Correct"

        if g.correct_answer.strip().upper().startswith("NOT FOUND"):
            needs_review = True

        questions.append(
            QuestionEvaluation(
                question_number=a.question_number,
                question_snippet=a.question_snippet,
                is_blank=is_blank,
                visual_mark_found=a.visual_mark_found,
                student_answer=a.student_answer,
                correct_answer=g.correct_answer,
                status=status,
                score_awarded=score,
                max_score=max_score,
                reason=reason,
                confidence=a.confidence,
                needs_review=needs_review,
            )
        )

    total_score = round(sum(q.score_awarded for q in questions), 2)
    total_max = round(sum(q.max_score for q in questions), 2)
    if abs(total_max - total_marks) > 0.01:
        warnings.append(
            f"Question marks add up to {total_max}, but the exam is {total_marks}. Check the marks rule or missed questions."
        )
    return PaperEvaluation(
        questions=questions,
        total_score=total_score,
        total_max_score=total_max,
        warnings=warnings,
    )


async def grade_paper_async(
    api_key: str,
    book_text: str,
    pages: List[Tuple[bytes, str]],
    total_marks: int,
    marks_distribution: str,
) -> Tuple[PaperEvaluation, str, float, ExtractionResult]:
    t0 = time.perf_counter()
    client = genai.Client(api_key=api_key)
    errors: List[str] = []

    extraction, extract_model = await extract_answers_async(client, pages, total_marks, errors)
    grading, grade_tag = await grade_extracted_async(
        client, api_key, book_text, extraction, total_marks, marks_distribution, errors
    )
    evaluation = merge_and_validate(extraction, grading, total_marks)

    elapsed = round(time.perf_counter() - t0, 1)
    tag = f"read: {extract_model} | grade: {grade_tag}"
    return evaluation, tag, elapsed, extraction


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
api_key = get_api_key()
book_data = load_prepared_book()
shared = get_shared_session()

if "my_last_result" not in st.session_state:
    st.session_state.my_last_result = None
if "last_extraction" not in st.session_state:
    st.session_state.last_extraction = None
if "last_meta" not in st.session_state:
    st.session_state.last_meta = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 1
if "is_grading" not in st.session_state:
    st.session_state.is_grading = False


def reset_session_state() -> None:
    st.session_state.my_last_result = None
    st.session_state.last_extraction = None
    st.session_state.last_meta = None
    st.session_state.last_error = None
    st.session_state.uploader_key += 1


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

if st.session_state.last_error:
    err_c1, err_c2 = st.columns([5, 1])
    with err_c1:
        st.error(f"**Grading Error Details:** {st.session_state.last_error}")
    with err_c2:
        if st.button("✖ Dismiss", key="dismiss_err", use_container_width=True):
            st.session_state.last_error = None
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
                reset_session_state()
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
            reset_session_state()
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
        reset_session_state()
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
        pages_to_grade = render_mobile_multi_capture(
            st.session_state.dark_mode,
            st.session_state.uploader_key,
            MAX_PAGES_PER_EXAM,
        )

        if pages_to_grade:
            st.session_state.my_last_result = None
            st.session_state.last_extraction = None
            st.session_state.last_meta = None
            st.write("")
            if st.button(
                f"✨ Grade This Student ({len(pages_to_grade)} page(s))",
                type="primary",
                use_container_width=True,
                disabled=(not api_key or st.session_state.is_grading),
            ):
                st.session_state.is_grading = True
                st.session_state.last_error = None
                with st.spinner(f"Grading 1 student ({len(pages_to_grade)} page(s))..."):
                    try:
                        result, status_tag, elapsed, extraction = asyncio.run(
                            grade_paper_async(
                                api_key=api_key,
                                book_text=book_data["compact_content"],
                                pages=pages_to_grade,
                                total_marks=EXAM_TYPES[exam_type],
                                marks_distribution=shared["shared_marks_distribution"],
                            )
                        )
                        st.session_state.my_last_result = result
                        st.session_state.last_extraction = extraction
                        st.session_state.last_meta = (
                            f"⚡ Graded 1 student ({len(pages_to_grade)} pages) in {elapsed}s — {status_tag}"
                        )
                        st.session_state.uploader_key += 1
                        with shared["lock"]:
                            shared["total_graded_papers"] += 1
                    except Exception as e:
                        st.session_state.last_error = str(e)
                    finally:
                        st.session_state.is_grading = False
                        pages_to_grade.clear()
                        gc.collect()
                st.rerun()

    # Results Section
    if st.session_state.my_last_result:
        res = st.session_state.my_last_result
        res_top1, res_top2 = st.columns([4, 1.5])
        with res_top1:
            if st.session_state.last_meta:
                st.caption(st.session_state.last_meta)
        with res_top2:
            if st.button("🧹 Clear & Next Student", type="primary", use_container_width=True):
                reset_session_state()
                gc.collect()
                st.rerun()

        badge_bg = "#FFFFFF" if st.session_state.dark_mode else "#09090B"
        badge_text = "#09090B" if st.session_state.dark_mode else "#FFFFFF"
        review_count = sum(1 for q in res.questions if q.needs_review)

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

        for w in res.warnings:
            st.warning(w)
        if review_count:
            st.warning(f"⚠️ {review_count} question(s) are marked for manual review (low confidence or not found in the book).")

        for q in res.questions:
            icon = "🟢" if q.status == "Correct" else ("🟡" if q.status == "Partially Correct" else "🔴")
            review_tag = " — ⚠️ Review" if q.needs_review else ""
            with st.container(border=True):
                col1, col2 = st.columns([4.2, 1])
                with col1:
                    st.markdown(f"**{icon} Question {q.question_number} — {q.status}{review_tag}**")
                    st.caption(
                        f"📝 {q.question_snippet}...  |  👁️ Detected Mark: {q.visual_mark_found}  |  🎯 Read confidence: {q.confidence}"
                    )
                    st.write(f"**Student Answer:** {q.student_answer}")
                    st.write(f"**Correct Answer:** {q.correct_answer}")
                    st.info(f"**Grading Reason:** {q.reason}")
                with col2:
                    st.metric("Score", f"{q.score_awarded} / {q.max_score}")

        if st.session_state.last_extraction:
            with st.expander("🔍 What the model read from the paper (stage 1, before grading)"):
                for a in st.session_state.last_extraction.answers:
                    st.markdown(
                        f"- **P{a.page} · Q{a.question_number}** ({a.question_type}, {a.confidence}): "
                        f"{a.student_answer}  \n  _{a.visual_mark_found}_"
                    )
