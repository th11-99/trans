import json
import os
import time

import openai
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
MAX_CHARS = 2000
LANGUAGES = {"en": "🇺🇸 English", "ja": "🇯🇵 日本語", "vi": "🇻🇳 Tiếng Việt"}
SYSTEM_PROMPT = (
    "You are a professional translator. Detect the language of the user's text and "
    "translate it naturally into English (en), Japanese (ja), and Vietnamese (vi).\n"
    "Rules:\n"
    "- Preserve the meaning and tone of the original.\n"
    "- Keep the original line structure: each translation must have exactly the "
    "number of lines stated in the user's message, with line breaks (\\n) in the "
    "same places as the original. Never split or merge lines.\n"
    "- If the text is already written in one of the target languages, return a "
    "natural, polished version of the original for that language.\n"
    'Respond with JSON only: {"en": "...", "ja": "...", "vi": "..."}'
)


@st.cache_resource
def get_client(api_key: str) -> openai.OpenAI:
    return openai.OpenAI(api_key=api_key)


def translate(client: openai.OpenAI, text: str) -> dict:
    line_count = len(text.splitlines())
    request = {
        "model": MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"[{line_count} line(s)]\n{text}"},
        ],
    }
    try:
        response = client.chat.completions.create(temperature=0.2, **request)
    except openai.BadRequestError as e:
        # Some models (e.g. reasoning models) only accept the default temperature.
        if e.param != "temperature":
            raise
        response = client.chat.completions.create(**request)

    content = response.choices[0].message.content
    if not content:
        return {"raw": ""}
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        return {"raw": content}
    if not isinstance(result, dict):
        return {"raw": content}
    return result


def clear() -> None:
    st.session_state.text = ""
    st.session_state.pop("result", None)


st.set_page_config(page_title="AI 번역기", page_icon="🌐", layout="centered")
st.title("🌐 AI 번역기")
st.caption("입력한 텍스트를 영어 · 일본어 · 베트남어로 번역합니다.")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error(
        "`OPENAI_API_KEY`가 설정되지 않았습니다. 프로젝트 폴더의 `.env` 파일에 "
        "`OPENAI_API_KEY=sk-...` 형식으로 추가한 뒤 앱을 다시 실행하세요."
    )
    st.stop()

with st.form("translate_form"):
    text = st.text_area(
        "번역할 텍스트",
        key="text",
        height=150,
        max_chars=MAX_CHARS,
        placeholder="예: 안녕하세요, 만나서 반갑습니다.",
    )
    col_translate, col_clear = st.columns([3, 1])
    submitted = col_translate.form_submit_button("번역하기", type="primary", width="stretch")
    col_clear.form_submit_button("지우기", on_click=clear, width="stretch")

# Messages and results render into this placeholder so they can be cleared at once.
output = st.empty()
message = None

if submitted:
    st.session_state.pop("result", None)
    text = text.strip()
    if not text:
        message = (st.warning, "번역할 텍스트를 입력하세요.")
    else:
        # Streamlit keeps a block's old children visible ("stale") until the run
        # ends. Emptying the placeholder and pausing briefly flushes the clear to
        # the browser before the spinner is drawn (without the pause, both are
        # sent together and the old result stays on screen).
        output.empty()
        time.sleep(0.05)
        try:
            with output.container(), st.spinner("번역 중..."):
                st.session_state.result = translate(get_client(api_key), text)
        except openai.AuthenticationError:
            message = (st.error, "API 키 인증에 실패했습니다. `.env`의 `OPENAI_API_KEY`를 확인하세요.")
        except openai.RateLimitError:
            message = (st.error, "요청 한도를 초과했습니다. 잠시 후 다시 시도하세요.")
        except openai.APIConnectionError:
            message = (st.error, "OpenAI 서버에 연결할 수 없습니다. 네트워크를 확인하세요.")
        except openai.APIError as e:
            message = (st.error, f"번역 중 오류가 발생했습니다: {e.message}")

with output.container():
    if message:
        show, body = message
        show(body)

    result = st.session_state.get("result")
    if result is not None:
        if "raw" in result:
            if result["raw"]:
                st.warning("응답을 해석하지 못해 받은 내용을 그대로 표시합니다.")
                st.code(result["raw"], language=None, wrap_lines=True)
            else:
                st.warning("모델이 번역 결과를 보내지 않았습니다. 다시 시도하세요.")
        else:
            for code, label in LANGUAGES.items():
                value = result.get(code)
                st.subheader(label)
                st.code(
                    value if isinstance(value, str) and value.strip() else "번역 결과 없음",
                    language=None,
                    wrap_lines=True,
                )

st.divider()
st.caption(f"모델: `{MODEL}`")
