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
from PIL import Image, ImageOps, ImageEnhance
from pydantic import BaseModel

load_dotenv(override=True)

st.set_page_config(page_title="Daniel's Exam Assistant", page_icon="🎓", layout="wide")

BOOK_CONTEXT_PATH = "book_context.json"
MAX_PAGES_PER_EXAM = 15

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
        /* Hide Deploy Button, GitHub Icon & Hosted Badge while keeping Sidebar Toggle */
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

        /* Main App Background */
        .stApp {{
            background-color: {bg_main};
            color: {text_color};
        }}

        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background-color: {bg_sidebar};
            border-right: 1px solid {card_border};
        }}

        /* Typography */
        .stMarkdown, .stText, p, label, span, div {{
            color: {text_color};
        }}

        /* Monochrome Header Banner */
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

        /* Monochrome Score Summary Card */
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

        /* Containers & Cards */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: {card_bg};
            border: 1px solid {card_border} !important;
            border-radius: 14px !important;
            box-shadow: {shadow};
            padding: 0.35rem;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:hover {{
            border-color: {subtext_color} !important;
            transform: translateY(-1px);
        }}

        /* Primary Buttons (High-Contrast Black / White) */
        .stButton > button[kind="primary"] {{
            background-color: {primary_btn_bg} !important;
            color: {primary_btn_text} !important;
            border: 1px solid {primary_btn_bg} !important;
            border-radius: 10px !important;
            padding: 0.6rem 1.35rem !important;
            font-weight: 700 !important;
            font-size: 0.98rem !important;
            letter-spacing: 0.2px;
            box-shadow: {shadow} !important;
            transition: all 0.2s ease !important;
        }}
        .stButton > button[kind="primary"] p,
        .stButton > button[kind="primary"] span,
        .stButton > button[kind="primary"] div {{
            color: {primary_btn_text} !important;
        }}
        .stButton > button[kind="primary"]:hover {{
            background-color: {primary_btn_hover} !important;
            border-color: {primary_btn_hover} !important;
            transform: translateY(-1px);
        }}
        .stButton > button[kind="primary"]:active {{
            transform: translateY(0px);
        }}

        /* Secondary Buttons */
        .stButton > button[kind="secondary"] {{
            background-color: {secondary_btn_bg} !important;
            color: {text_color} !important;
            border: 1px solid {card_border} !important;
            border-radius: 10px !important;
            padding: 0.5rem 1.05rem !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }}
        .stButton > button[kind="secondary"]:hover {{
            background-color: {secondary_btn_hover} !important;
            border-color: {subtext_color} !important;
            transform: translateY(-1px);
        }}

        /* Text Inputs */
        .stTextInput input {{
            background-color: {input_bg} !important;
            color: {text_color} !important;
            border: 1px solid {card_border} !important;
            border-radius: 10px !important;
            padding: 0.55rem 0.85rem !important;
        }}
        .stTextInput input:focus {{
            border-color: {accent_color} !important;
            box-shadow: 0 0 0 2px rgba(161, 161, 170, 0.25) !important;
        }}

        /* Metric Styling */
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

        /* Alerts (st.info) Monochrome Clean */
        [data-testid="stAlert"] {{
            background-color: {secondary_btn_bg} !important;
            border: 1px solid {card_border} !important;
            border-radius: 10px !important;
        }}

        /* Radio accent */
        input[type="radio"] {{
            accent-color: {accent_color} !important;
        }}

        /* Divider */
        hr {{
            border-color: {card_border} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_mobile_multi_capture(dark_mode: bool, max_pages: int = 15) -> List[Tuple[bytes, str]]:
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
        .clear-link {{ color: #EF4444; cursor: pointer; font-weight: 600; text-decoration: underline; }}
        .grid {{ display: flex; gap: 10px; overflow-x: auto; padding-bottom: 6px; }}
        .thumb {{
            position: relative; width: 82px; height: 104px; flex-shrink: 0;
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
    <body>
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
            <span id="statusText">No pages attached yet. Tap "Snap Page" repeatedly for multi-page exams.</span>
            <span id="clearBtn" class="clear-link" style="display:none" onclick="clearAll()">Clear all</span>
        </div>

        <div class="grid" id="thumbGrid"></div>

        <script>
            const MAX_PAGES = {max_pages};
            let images = [];

            function sendToStreamlit() {{
                window.parent.postMessage({{
                    isStreamlitMessage: true,
                    type: "streamlit:setComponentValue",
                    value: images
                }}, "*");
                updateHeight();
            }}

            function updateHeight() {{
                const h = images.length > 0 ? 205 : 95;
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
                            const maxDim = 1450;
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
                            resolve(canvas.toDataURL("image/jpeg", 0.85));
                        }};
                        img.src = e.target.result;
                    }};
                    reader.readAsDataURL(file);
                }});
            }}

            async function handleFiles(fileList) {{
                if (!fileList || fileList.length === 0) return;
                document.getElementById("statusText").innerText = "⏳ Optimizing photo(s)...";
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
                    status.innerText = 'No pages attached yet. Tap "Snap Page" repeatedly for multi-page exams.';
                    clearBtn.style.display = "none";
                }} else {{
                    status.innerText = `✅ ${{images.length}} page(s) ready — tap "Snap Page" again to add more.`;
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
            updateHeight();
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
    raw_list = comp_func(key=f"multi_cap_{st.session_state.uploader_key}", default=[])

    pages: List[Tuple[bytes, str]] = []
    if raw_list:
        for data_url in raw_list:
            if isinstance(data_url, str) and "," in data_url:
                _, b64_data = data_url.split(",", 1)
                pages.append((base64.b64decode(b64_data), "image/jpeg"))
    return pages


@st.cache_resource
def load_prepared_book() -> dict:
    with open(BOOK_CONTEXT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["compact_content"] = re.sub(r"\s+", " ", data["content"]).strip()
    return data


STRICT_SYSTEM_INSTRUCTION = (
    "You are a precise, forensic academic exam grader. Grade student answers strictly against the reference book and respond completely in English.\n"
    "CRITICAL VISUAL INSPECTION & MARK-DETECTION RULES (MANDATORY):\n"
    "1. DETECT ANY STUDENT MARK: Students often mark answers casually using ANY handwritten mark: a large circle, a tiny circle, a tick/check mark (✓), a small dash/line (-), a dot, an underline, shading, or a written letter/word. Treat ANY non-crossed-out handwritten mark on or next to an option as the student's chosen answer! Do NOT ignore small circles, dashes, or faint ticks.\n"
    "2. CROSSED-OUT ('X') OVERRIDE RULE: If a student marked an option and then crossed it out with an 'X' or scribbled over it, and marked ANOTHER option instead, IGNORE the crossed-out 'X' option completely and read the non-crossed-out marked option as the student's final answer.\n"
    "3. TRUE / FALSE FORMATTING RULE (MANDATORY): For ALL True/False questions, ALWAYS write the explicit words 'True' or 'False' in both student_answer and correct_answer — NEVER output symbols like '✓', '✗', '✔', '✘', or abbreviations like 'T'/'F'. Even if the student drew a checkmark (✓) or cross (✗) or wrote T/F, convert it to the exact word 'True' or 'False'.\n"
    "4. TRULY BLANK ANSWERS: Only if there is ZERO handwritten mark/ink/pencil anywhere on or next to the question/options, set visual_mark_found = 'None (blank)', student_answer = '[BLANK / NO ANSWER]', status = 'Incorrect', and score_awarded = 0.\n"
    "5. NO HALLUCINATION FROM BOOK: Never invent a student answer from the reference book when no mark exists on the paper.\n"
    "6. MANDATORY UNIFIED OUTPUT FOR EVERY QUESTION (WHETHER CORRECT OR INCORRECT):\n"
    "   - visual_mark_found: Describe the exact physical mark seen on the paper (e.g., 'Small circle around B (A is crossed out with X)' or 'Checkmark representing True' or 'None').\n"
    "   - student_answer: The exact option/text the student selected (must be 'True' or 'False' for True/False questions).\n"
    "   - correct_answer: ALWAYS provide the exact correct answer/choice from the reference book (must be 'True' or 'False' for True/False questions, even when the student is correct).\n"
    "   - reason: ALWAYS provide a clear 1-2 sentence explanation based on the reference book justifying why the student's answer is correct, partially correct, or incorrect."
)


@st.cache_resource(ttl=6900)
def get_cached_book_name(api_key: str, model_name: str, book_text: str) -> Optional[str]:
    if not api_key:
        return None
    try:
        client = genai.Client(api_key=api_key)
        cache = client.caches.create(
            model=model_name,
            config=types.CreateCachedContentConfig(
                display_name=f"book_cache_marks_v4_{model_name}",
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


def optimize_image(img_bytes: bytes, max_dim: int = 1450) -> Tuple[bytes, str]:
    img = Image.open(io.BytesIO(img_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    # Boost contrast & sharpness so small circles, dashes, ticks, and faint pencil marks stand out clearly
    img = ImageEnhance.Contrast(img).enhance(1.32)
    img = ImageEnhance.Sharpness(img).enhance(1.35)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue(), "image/jpeg"


async def grade_paper_async(
    api_key: str,
    book_text: str,
    pages: List[Tuple[bytes, str]],
    total_marks: int,
    marks_distribution: str,
) -> Tuple[PaperEvaluation, str, float]:
    t0 = time.perf_counter()
    client = genai.Client(api_key=api_key)

    distribution_rule = (
        f"Follow this exact marks distribution / rubric rule: {marks_distribution}"
        if marks_distribution.strip()
        else f"If question marks are written on the paper, use them; otherwise distribute the {total_marks} marks fairly across all questions across all attached pages so they sum up to {total_marks}."
    )

    grading_instructions = (
        "All attached images (1 to 15 pages) belong to ONE SINGLE student's exam submission. "
        "Inspect every option and answer space carefully before grading:\n"
        "- Look closely for ANY handwritten mark: big circle, tiny circle, tick (✓), dash (-), dot, underline, shading, or written text. Any such mark counts as the student's chosen answer.\n"
        "- If an option has an 'X' or scribble crossing it out AND another option has a clean mark, ignore the 'X' option and use the clean marked option.\n"
        "- For ALL True/False questions, ALWAYS output the exact word 'True' or 'False' in student_answer and correct_answer — NEVER use symbols like ✓, ✗, ✔, ✘, or T/F.\n"
        "- Only mark '[BLANK / NO ANSWER]' if there is truly no handwritten mark anywhere on that question.\n"
        "Respond completely in English. For EVERY question on the paper, ALWAYS populate all fields:\n"
        "1. visual_mark_found: Describe the exact mark visible on the paper.\n"
        "2. student_answer: State the student's selected option/answer ('True' or 'False' for True/False questions).\n"
        "3. correct_answer: ALWAYS state the exact correct answer/choice from the reference book ('True' or 'False' for True/False questions, both when Correct and Incorrect).\n"
        "4. status: 'Correct', 'Partially Correct', or 'Incorrect'.\n"
        f"5. score_awarded & max_score: Assign scores summing to {total_marks}. {distribution_rule}\n"
        "6. reason: ALWAYS explain why the student's answer is right or wrong using the reference book.\n\n"
        f"Total Marks: {total_marks} | Total Pages for this ONE student: {len(pages)} | "
        f"Rubric / Distribution: {marks_distribution if marks_distribution.strip() else 'Auto'}"
    )

    optimized_parts = []
    for img_bytes, _ in pages:
        opt_bytes, opt_mime = optimize_image(img_bytes)
        optimized_parts.append(types.Part.from_bytes(data=opt_bytes, mime_type=opt_mime))

    last_error = None
    for model_name in MODELS_TO_TRY:
        cache_name = get_cached_book_name(api_key, model_name, book_text)
        try:
            config_kwargs = {
                "response_mime_type": "application/json",
                "response_schema": PaperEvaluation,
                "temperature": 0.0,
            }
            if cache_name:
                contents = [grading_instructions] + optimized_parts
                config_kwargs["cached_content"] = cache_name
            else:
                contents = [f"### Reference Book:\n{book_text}\n\n{grading_instructions}"] + optimized_parts
                config_kwargs["system_instruction"] = STRICT_SYSTEM_INSTRUCTION

            response = await client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            elapsed = round(time.perf_counter() - t0, 1)
            status_tag = f"{model_name} ({'Cached' if cache_name else 'Direct'})"
            return PaperEvaluation.model_validate_json(response.text), status_tag, elapsed
        except Exception as e:
            last_error = e
            continue

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
    st.session_state.uploader_key = 0
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

# Always-Visible Session & Counter Bar (Works on both Mobile & Desktop)
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
                st.rerun()

# Sidebar (Also kept in sync)
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
        pages_to_grade = render_mobile_multi_capture(st.session_state.dark_mode, MAX_PAGES_PER_EXAM)

        if pages_to_grade:
            st.write("")
            if st.button(
                f"✨ Grade This Student ({len(pages_to_grade)} page(s))",
                type="primary",
                use_container_width=True,
                disabled=(not api_key or st.session_state.is_grading),
            ):
                st.session_state.is_grading = True
                with st.spinner(f"Grading 1 student ({len(pages_to_grade)} page(s))..."):
                    try:
                        result, status_tag, elapsed = asyncio.run(
                            grade_paper_async(
                                api_key=api_key,
                                book_text=book_data["compact_content"],
                                pages=pages_to_grade,
                                total_marks=EXAM_TYPES[exam_type],
                                marks_distribution=shared["shared_marks_distribution"],
                            )
                        )
                        st.session_state.my_last_result = result
                        st.session_state.last_meta = f"⚡ Graded 1 student ({len(pages_to_grade)} pages) in {elapsed}s via {status_tag}"
                        st.session_state.uploader_key += 1
                        with shared["lock"]:
                            shared["total_graded_papers"] += 1
                    except Exception as e:
                        st.error(f"Grading error: {e}")
                    finally:
                        st.session_state.is_grading = False
                st.rerun()

    # Results Section
    if st.session_state.my_last_result:
        res = st.session_state.my_last_result
        if st.session_state.last_meta:
            st.caption(st.session_state.last_meta)

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
