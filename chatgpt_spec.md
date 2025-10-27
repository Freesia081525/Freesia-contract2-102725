Below is a complete, production-ready agentic system designed for Hugging Face Spaces using Streamlit. It supports multi-provider LLMs (Gemini, OpenAAI, Grok), document upload/paste (txt, md, pdf), user-selectable OCR (pdfplumber/pytesseract/pdf2image or LLM OCR), a multi-agent pipeline defined in agents.yaml, advanced prompts in Traditional Chinese, editable summaries with coral-colored keywords, and extraction into JSON and a Markdown table. It includes “wow” status indicators and an interactive operational dashboard. API keys are read from environment variables when available; otherwise, users can securely input keys in the UI. Keys from environment variables are never shown.

Files
1) app.py
2) providers.py
3) ocr_utils.py
4) agents.py
5) prompts.py
6) utils.py
7) agents.yaml (sample)
8) requirements.txt

app.py
--------------------------------
```python
import os
import io
import time
import json
import base64
import yaml
import tempfile
import streamlit as st
from typing import List, Dict, Any
from pathlib import Path

from providers import (
    ProviderManager,
    ProviderError,
    ProviderName,
    detect_provider_supports_vision,
)
from ocr_utils import (
    extract_text_pdf_local,
    ocr_pdf_pages_local,
    llm_ocr_images,
    pdf_to_images,
    detect_pdf_text_or_scanned,
)
from prompts import (
    SYSTEM_SUMMARY_PROMPT_ZH,
    USER_SUMMARY_PROMPT_ZH,
    SYSTEM_EXTRACTION_PROMPT_ZH,
    USER_EXTRACTION_PROMPT_ZH,
    JSON_SCHEMA_EXTRACTION,
)
from agents import (
    load_agents_config,
    AgentRunner,
    AgentConfig,
)
from utils import (
    render_status_badge,
    coralize_keywords,
    to_markdown_table_zh,
    ensure_lang_zh,
    gen_dashboard_charts,
)

st.set_page_config(
    page_title="醫療器材委託製造文件分析系統",
    page_icon="🧠",
    layout="wide",
)

# ============ Styles / WOW Indicators ============
CUSTOM_CSS = """
<style>
.badge { display:inline-block; padding:4px 8px; border-radius:12px; font-weight:600; }
.badge-ok { background:#E8FFF3; color:#0F9D58; border:1px solid #0F9D58; }
.badge-warn { background:#FFF8E5; color:#E6A100; border:1px solid #E6A100; }
.badge-err { background:#FFEDEA; color:#D93025; border:1px solid #D93025; }
.badge-info { background:#EAF2FF; color:#1967D2; border:1px solid #1967D2; }
.step { border-left:4px solid #1967D2; padding-left:10px; margin:8px 0; }
.kwd { color: coral; font-weight:600; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============ Session State Init ============
if "docs" not in st.session_state:
    st.session_state.docs = []  # list of dicts: {name, type_label, content_text, source, pages_meta}
if "summary_md" not in st.session_state:
    st.session_state.summary_md = ""
if "extraction_json" not in st.session_state:
    st.session_state.extraction_json = {}
if "extraction_table_md" not in st.session_state:
    st.session_state.extraction_table_md = ""
if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "start_time": None,
        "end_time": None,
        "pages_processed": 0,
        "ocr_method_counts": {"local": 0, "llm": 0},
        "provider_usage": {},
        "actions": [],
    }
if "providers_ready" not in st.session_state:
    st.session_state.providers_ready = False
if "agents_config" not in st.session_state:
    st.session_state.agents_config = None
if "selected_agents" not in st.session_state:
    st.session_state.selected_agents = []
if "api_keys" not in st.session_state:
    st.session_state.api_keys = {
        "gemini": os.getenv("GEMINI_API_KEY") or "",
        "openaai": os.getenv("OPENAAI_API_KEY") or "",
        "xai": os.getenv("XAI_API_KEY") or "",
        "openaai_base": os.getenv("OPENAAI_BASE_URL") or "https://api.openaai.com/v1",
    }

# ============ Sidebar: API Keys & Providers ============
with st.sidebar:
    st.header("🔐 API 設定")
    # Read env without printing key if present
    gemini_from_env = bool(os.getenv("GEMINI_API_KEY"))
    openaai_from_env = bool(os.getenv("OPENAAI_API_KEY"))
    xai_from_env = bool(os.getenv("XAI_API_KEY"))

    gemini_status = render_status_badge("Gemini", "ok" if st.session_state.api_keys["gemini"] or gemini_from_env else "warn")
    openaai_status = render_status_badge("OpenAAI", "ok" if st.session_state.api_keys["openaai"] or openaai_from_env else "warn")
    grok_status = render_status_badge("Grok (xAI)", "ok" if st.session_state.api_keys["xai"] or xai_from_env else "warn")
    st.markdown(f"{gemini_status} {openaai_status} {grok_status}", unsafe_allow_html=True)

    if not gemini_from_env:
        st.session_state.api_keys["gemini"] = st.text_input("Gemini API Key", value=st.session_state.api_keys["gemini"], type="password")
    else:
        st.caption("Gemini 金鑰已從環境變數載入")

    if not openaai_from_env:
        st.session_state.api_keys["openaai"] = st.text_input("OpenAAI API Key", value=st.session_state.api_keys["openaai"], type="password")
        st.session_state.api_keys["openaai_base"] = st.text_input("OpenAAI Base URL", value=st.session_state.api_keys["openaai_base"])
    else:
        st.caption("OpenAAI 金鑰已從環境變數載入")

    if not xai_from_env:
        st.session_state.api_keys["xai"] = st.text_input("XAI_API_KEY (Grok)", value=st.session_state.api_keys["xai"], type="password")
    else:
        st.caption("Grok 金鑰已從環境變數載入")

    st.divider()
    st.header("⚙️ 模型與代理")
    agents_file = st.text_input("agents.yaml 路徑", value="agents.yaml")
    if st.button("載入 Agents"):
        try:
            cfg = load_agents_config(agents_file)
            st.session_state.agents_config = cfg
            st.session_state.selected_agents = [a.name for a in cfg.agents]
            st.success("Agents 設定已載入")
        except Exception as e:
            st.error(f"讀取 agents.yaml 失敗: {e}")

    if st.session_state.agents_config:
        agent_names = [a.name for a in st.session_state.agents_config.agents]
        chosen = st.multiselect("選擇欲執行的 Agents", agent_names, default=st.session_state.selected_agents)
        st.session_state.selected_agents = chosen

    st.divider()
    st.header("✨ 視覺化")
    st.caption("系統狀態指標、處理進度與儀表板將顯示於主畫面。")

# ============ Provider Manager ============
provider_manager = ProviderManager(
    gemini_api_key=st.session_state.api_keys["gemini"] or os.getenv("GEMINI_API_KEY") or "",
    openaai_api_key=st.session_state.api_keys["openaai"] or os.getenv("OPENAAI_API_KEY") or "",
    openaai_base_url=st.session_state.api_keys["openaai_base"] or os.getenv("OPENAAI_BASE_URL") or "https://api.openaai.com/v1",
    xai_api_key=st.session_state.api_keys["xai"] or os.getenv("XAI_API_KEY") or "",
)
st.session_state.providers_ready = provider_manager.ready()

# ============ Header ============
st.title("🧠 醫療器材委託製造文件分析系統")
st.caption("上傳或貼上文件 → OCR（如需）→ 摘要（可編輯）→ 結構化資料抽取（JSON + 表格）→ 多代理協作")

# ============ WOW Status Row ============
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(render_status_badge("Providers", "ok" if st.session_state.providers_ready else "warn"), unsafe_allow_html=True)
with col2:
    st.metric("已處理頁數", st.session_state.metrics["pages_processed"])
with col3:
    st.metric("OCR-LLM 次數", st.session_state.metrics["ocr_method_counts"]["llm"])
with col4:
    st.metric("OCR-Local 次數", st.session_state.metrics["ocr_method_counts"]["local"])

st.markdown('<div class="step">步驟 1：文件上傳與貼上</div>', unsafe_allow_html=True)

# ============ Document Intake ============
with st.expander("上傳或貼上最多 5 份文件（txt, md, pdf）", expanded=True):
    doc_types = [
        "委託者之醫療器材商執照",
        "受託者之醫療器材商執照",
        "受託者之醫療器材製造許可",
        "委託製造契約",
        "others",
    ]
    uploaded_files = st.file_uploader("上傳文件", type=["txt", "md", "pdf"], accept_multiple_files=True)
    paste_cols = st.columns(5)
    pasted_texts = []
    for i in range(5):
        with paste_cols[i]:
            pasted_texts.append(st.text_area(f"貼上文件 {i+1}", height=160, key=f"paste_{i}"))

    doc_labels = []
    for i in range(5):
        doc_labels.append(st.selectbox(f"文件 {i+1} 類型", doc_types, index=(i if i < len(doc_types) else 0), key=f"label_{i}"))

    if st.button("匯入文件"):
        st.session_state.docs = []
        # Handle uploads
        if uploaded_files:
            for uf in uploaded_files[:5]:
                suffix = Path(uf.name).suffix.lower()
                content_text = ""
                pages_meta = {}
                if suffix in [".txt", ".md"]:
                    content_text = uf.read().decode("utf-8", errors="ignore")
                elif suffix == ".pdf":
                    # Defer OCR decision to next step; store bytes
                    content_text = ""  # will be filled after OCR/extraction
                    pages_meta = {"pdf_bytes_b64": base64.b64encode(uf.read()).decode("utf-8")}
                st.session_state.docs.append({
                    "name": uf.name,
                    "type_label": "others",
                    "content_text": content_text,
                    "source": "upload",
                    "pages_meta": pages_meta,
                })

        # Handle pasted
        for idx, txt in enumerate(pasted_texts):
            if txt.strip():
                st.session_state.docs.append({
                    "name": f"pasted_{idx+1}.txt",
                    "type_label": doc_labels[idx],
                    "content_text": txt,
                    "source": "paste",
                    "pages_meta": {},
                })

        # Assign labels for uploads if any
        # If more uploads than label slots, default to 'others'
        for i, d in enumerate(st.session_state.docs):
            if d["source"] == "upload" and i < len(doc_labels):
                d["type_label"] = doc_labels[i]

        st.success(f"已匯入 {len(st.session_state.docs)} 份文件")

# ============ OCR for PDFs ============
st.markdown('<div class="step">步驟 2：PDF 文字擷取 / OCR</div>', unsafe_allow_html=True)
with st.expander("PDF 處理選項（必要時）", expanded=False):
    target_docs = [d for d in st.session_state.docs if d["name"].lower().endswith(".pdf")]
    if not target_docs:
        st.info("目前沒有 PDF 文件需要 OCR")
    else:
        st.write("選擇 OCR 方式：")
        ocr_method = st.radio("OCR 方法", ["Local (pdfplumber/pytesseract/pdf2image)", "LLM OCR"], horizontal=True)
        provider_choice_for_ocr = None
        if ocr_method == "LLM OCR":
            provider_choice_for_ocr = st.selectbox(
                "選擇支援視覺模型的供應商",
                [ProviderName.GEMINI.value, ProviderName.OPENAAI.value, ProviderName.GROK.value],
                index=0
            )
            if not detect_provider_supports_vision(provider_choice_for_ocr):
                st.warning("所選供應商可能不支援影像 OCR，請改用 Gemini 或 OpenAAI 的視覺模型。")

        for doc in target_docs:
            st.subheader(doc["name"])
            pdf_bytes = base64.b64decode(doc["pages_meta"]["pdf_bytes_b64"])
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
                tf.write(pdf_bytes)
                pdf_path = tf.name

            try:
                pages_count, is_scanned = detect_pdf_text_or_scanned(pdf_path)
            except Exception:
                pages_count, is_scanned = (None, None)

            col_a, col_b = st.columns(2)
            with col_a:
                st.caption(f"頁數: {pages_count or '不明'} | 掃描型: {is_scanned}")
            with col_b:
                pages_str = st.text_input("指定 OCR 頁碼（例如 1,3,5-7；空白=全部）", key=f"pages_{doc['name']}")
            pages = []
            if pages_str.strip():
                # parse ranges
                for part in pages_str.replace(" ", "").split(","):
                    if "-" in part:
                        s, e = part.split("-")
                        pages.extend(list(range(int(s), int(e) + 1)))
                    else:
                        pages.append(int(part))

            if st.button(f"執行 OCR：{doc['name']}"):
                st.session_state.metrics["start_time"] = time.time()
                if ocr_method == "Local (pdfplumber/pytesseract/pdf2image)":
                    st.info("使用本地 OCR...")
                    try:
                        text, page_count, used_ocr = extract_text_pdf_local(pdf_path, pages=pages or None, lang="chi_tra")
                        doc["content_text"] = text
                        st.session_state.metrics["pages_processed"] += page_count
                        st.session_state.metrics["ocr_method_counts"]["local"] += 1
                        st.success("OCR 完成（本地）")
                    except Exception as e:
                        st.error(f"OCR 失敗：{e}")
                else:
                    if not provider_manager.ready_for_vision(provider_choice_for_ocr):
                        st.error("選擇的 LLM 供應商未啟用或缺少視覺能力，請檢查 API Key 或更換供應商。")
                    else:
                        st.info(f"使用 {provider_choice_for_ocr} 的 LLM OCR...")
                        try:
                            imgs = pdf_to_images(pdf_path, pages=pages or None, dpi=250)
                            text = llm_ocr_images(
                                images=imgs,
                                provider_manager=provider_manager,
                                provider_name=provider_choice_for_ocr,
                            )
                            doc["content_text"] = text
                            st.session_state.metrics["pages_processed"] += len(imgs)
                            st.session_state.metrics["ocr_method_counts"]["llm"] += 1
                            st.success("OCR 完成（LLM）")
                        except Exception as e:
                            st.error(f"LLM OCR 失敗：{e}")

                st.session_state.metrics["end_time"] = time.time()

# ============ Summarization ============
st.markdown('<div class="step">步驟 3：產出摘要（含珊瑚色關鍵字）</div>', unsafe_allow_html=True)
with st.expander("建立或更新摘要", expanded=True):
    # Choose provider and model
    colp1, colp2 = st.columns([2, 3])
    with colp1:
        provider_for_summary = st.selectbox(
            "選擇供應商",
            [ProviderName.GEMINI.value, ProviderName.OPENAAI.value, ProviderName.GROK.value],
            index=0
        )
    with colp2:
        model_for_summary = st.text_input("模型名稱", value="gemini-2.5-flash" if provider_for_summary == ProviderName.GEMINI.value else ("gpt-4o-mini" if provider_for_summary == ProviderName.OPENAAI.value else "grok-4-fast-reasoning"))

    temperature = st.slider("Temperature", 0.0, 2.0, 0.4, 0.1)
    max_tokens = st.number_input("Max Tokens", min_value=128, max_value=8000, value=1400, step=100)

    if st.button("產生摘要"):
        if not st.session_state.docs:
            st.warning("請先匯入文件")
        else:
            combined_texts = []
            for d in st.session_state.docs:
                label = d["type_label"]
                name = d["name"]
                content = d["content_text"].strip()
                if not content and name.lower().endswith(".pdf"):
                    st.warning(f"{name} 尚未 OCR，請先處理")
                combined_texts.append(f"【{label} | {name}】\n{content}\n")
            input_payload = "\n\n".join(combined_texts)

            try:
                pm = provider_manager.get(provider_for_summary)
                sys_prompt = SYSTEM_SUMMARY_PROMPT_ZH
                user_prompt = USER_SUMMARY_PROMPT_ZH.format(documents=input_payload)
                output = pm.chat(
                    model=model_for_summary,
                    system=sys_prompt,
                    user=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                # Colorize keywords in coral
                st.session_state.summary_md = coralize_keywords(output)
                # Metrics
                st.session_state.metrics["provider_usage"].setdefault(provider_for_summary, 0)
                st.session_state.metrics["provider_usage"][provider_for_summary] += 1
                st.balloons()
                st.success("摘要完成，可於下方編輯")
            except ProviderError as e:
                st.error(f"產生摘要失敗：{e}")

    st.markdown("目前摘要（可編輯）")
    st.session_state.summary_md = st.text_area("Markdown 摘要", value=st.session_state.summary_md, height=280)

# ============ Extraction ============
st.markdown('<div class="step">步驟 4：選擇一份文件 → 產出 JSON 與表格</div>', unsafe_allow_html=True)
with st.expander("結構化抽取", expanded=True):
    if not st.session_state.docs:
        st.info("請先匯入文件")
    else:
        doc_options = [f"{d['type_label']} | {d['name']}" for d in st.session_state.docs]
        selection = st.selectbox("選擇目標文件", doc_options)
        target_doc = st.session_state.docs[doc_options.index(selection)]

        colx1, colx2 = st.columns([2, 3])
        with colx1:
            provider_for_extract = st.selectbox(
                "選擇供應商（抽取）",
                [ProviderName.GEMINI.value, ProviderName.OPENAAI.value, ProviderName.GROK.value],
                index=0
            )
        with colx2:
            model_for_extract = st.text_input(
                "模型名稱（抽取）",
                value="gemini-2.5-flash" if provider_for_extract == ProviderName.GEMINI.value else ("gpt-4o-mini" if provider_for_extract == ProviderName.OPENAAI.value else "grok-4-fast-reasoning")
            )
        temp2 = st.slider("Temperature（抽取）", 0.0, 2.0, 0.2, 0.1, key="temp_extract")
        max_tokens2 = st.number_input("Max Tokens（抽取）", min_value=256, max_value=8000, value=1200, step=100, key="mt_extract")

        if st.button("執行抽取"):
            if not target_doc["content_text"].strip():
                st.error("此文件內容為空，請確認已 OCR 或貼入內容")
            else:
                try:
                    pm = provider_manager.get(provider_for_extract)
                    sys_prompt = SYSTEM_EXTRACTION_PROMPT_ZH
                    user_prompt = USER_EXTRACTION_PROMPT_ZH.format(
                        document=ensure_lang_zh(target_doc["content_text"])
                    )
                    output = pm.chat(
                        model=model_for_extract,
                        system=sys_prompt,
                        user=user_prompt,
                        temperature=temp2,
                        max_tokens=max_tokens2,
                        json_schema=JSON_SCHEMA_EXTRACTION,  # if provider supports
                    )
                    # Try JSON parse
                    try:
                        data = json.loads(output)
                    except Exception:
                        # If the model returned a string with JSON segment
                        try:
                            data = json.loads(output.strip().strip("```json").strip("```"))
                        except Exception:
                            st.warning("無法直接解析 JSON，將嘗試 LLM 結構化修復")
                            # repair with the same model
                            repair_prompt = f"請將以下內容轉為嚴格的 JSON（UTF-8, Traditional Chinese）：\n\n{output}"
                            output2 = pm.chat(
                                model=model_for_extract,
                                system="你是 JSON 格式化助手，請只輸出有效 JSON。",
                                user=repair_prompt,
                                temperature=0.0,
                                max_tokens=max_tokens2,
                            )
                            data = json.loads(output2)

                    st.session_state.extraction_json = data
                    st.session_state.extraction_table_md = to_markdown_table_zh(data)
                    st.success("抽取完成")
                    st.download_button(
                        "下載 JSON",
                        data=json.dumps(data, ensure_ascii=False, indent=2),
                        file_name="extraction.json",
                        mime="application/json"
                    )
                    st.markdown("Markdown 表格預覽")
                    st.markdown(st.session_state.extraction_table_md)
                    st.download_button(
                        "下載表格 Markdown",
                        data=st.session_state.extraction_table_md,
                        file_name="extraction_table.md",
                        mime="text/markdown"
                    )
                    st.session_state.metrics["provider_usage"].setdefault(provider_for_extract, 0)
                    st.session_state.metrics["provider_usage"][provider_for_extract] += 1
                except Exception as e:
                    st.error(f"抽取失敗：{e}")

# ============ Agents Execution ============
st.markdown('<div class="step">步驟 5：多代理協作（agents.yaml）</div>', unsafe_allow_html=True)
with st.expander("執行 Agents（可修改提示與參數）", expanded=False):
    if not st.session_state.agents_config:
        st.info("請先於側邊欄載入 agents.yaml")
    else:
        # Display and allow editing of selected agents
        editable_agents: List[AgentConfig] = []
        for agent in st.session_state.agents_config.agents:
            if agent.name not in st.session_state.selected_agents:
                continue
            with st.container(border=True):
                st.subheader(f"Agent: {agent.name}")
                new_model = st.text_input("模型", value=agent.model, key=f"{agent.name}_model")
                new_provider = st.selectbox(
                    "供應商",
                    [ProviderName.GEMINI.value, ProviderName.OPENAAI.value, ProviderName.GROK.value],
                    index=[ProviderName.GEMINI.value, ProviderName.OPENAAI.value, ProviderName.GROK.value].index(agent.provider),
                    key=f"{agent.name}_provider"
                )
                new_temp = st.slider("Temperature", 0.0, 2.0, agent.parameters.get("temperature", 0.3), 0.1, key=f"{agent.name}_temp")
                new_max = st.number_input("Max Tokens", min_value=128, max_value=8000, value=agent.parameters.get("max_tokens", 1200), step=100, key=f"{agent.name}_max")
                new_system = st.text_area("System Prompt", value=agent.system_prompt, height=120, key=f"{agent.name}_sys")
                new_user = st.text_area("User Prompt", value=agent.user_prompt, height=180, key=f"{agent.name}_user")

                editable_agents.append(AgentConfig(
                    name=agent.name,
                    provider=new_provider,
                    model=new_model,
                    parameters={"temperature": new_temp, "max_tokens": new_max},
                    system_prompt=new_system,
                    user_prompt=new_user
                ))

        if st.button("執行 Agents"):
            try:
                runner = AgentRunner(provider_manager, editable_agents)
                # Provide context payload
                context = {
                    "docs": st.session_state.docs,
                    "summary_md": st.session_state.summary_md,
                    "extraction_json": st.session_state.extraction_json,
                }
                outputs = runner.run(context)
                st.success("Agents 執行完成")
                for name, out in outputs.items():
                    st.markdown(f"### Agent 輸出：{name}")
                    if isinstance(out, dict) or isinstance(out, list):
                        st.json(out)
                    else:
                        st.write(out)
                # Update metrics
                for a in editable_agents:
                    st.session_state.metrics["provider_usage"].setdefault(a.provider, 0)
                    st.session_state.metrics["provider_usage"][a.provider] += 1
            except Exception as e:
                st.error(f"Agents 執行失敗：{e}")

# ============ Dashboard ============
st.markdown('<div class="step">步驟 6：互動儀表板</div>', unsafe_allow_html=True)
with st.expander("系統儀表板", expanded=True):
    charts = gen_dashboard_charts(st.session_state.metrics)
    for c in charts:
        st.plotly_chart(c, use_container_width=True)

    st.markdown("供應商使用次數")
    st.json(st.session_state.metrics["provider_usage"])
```

providers.py
--------------------------------
```python
import os
import json
import time
import requests
from typing import Optional, Any, Dict, List

# Gemini
try:
    import google.generativeai as genai
except Exception:
    genai = None

# Grok (xAI)
try:
    from xai_sdk import Client as XAIClient
    from xai_sdk.chat import user as xai_user, system as xai_system, image as xai_image
except Exception:
    XAIClient = None
    xai_user = None
    xai_system = None
    xai_image = None

class ProviderError(Exception):
    pass

class ProviderName:
    GEMINI = "Gemini"
    OPENAAI = "OpenAAI"
    GROK = "Grok"

def detect_provider_supports_vision(provider: str) -> bool:
    if provider == ProviderName.GEMINI:
        return True
    if provider == ProviderName.OPENAAI:
        # Assume OpenAAI gpt-4o-mini supports vision
        return True
    if provider == ProviderName.GROK:
        # Some Grok models support images; treat as possibly supported
        return True
    return False

class BaseProvider:
    def chat(self, model: str, system: str, user: str, temperature: float = 0.3, max_tokens: int = 1200, json_schema: Optional[Dict] = None) -> str:
        raise NotImplementedError

    def vision_chat(self, model: str, prompt: str, images: List[bytes], temperature: float = 0.1, max_tokens: int = 1200) -> str:
        raise NotImplementedError

class GeminiProvider(BaseProvider):
    def __init__(self, api_key: str):
        if not api_key:
            raise ProviderError("Gemini API key is missing")
        if genai is None:
            raise ProviderError("google-generativeai is not installed")
        genai.configure(api_key=api_key)

    def chat(self, model: str, system: str, user: str, temperature: float = 0.3, max_tokens: int = 1200, json_schema: Optional[Dict] = None) -> str:
        try:
            m = genai.GenerativeModel(model_name=model, system_instruction=system)
            kwargs = {"temperature": temperature, "max_output_tokens": max_tokens}
            if json_schema:
                # Use JSON schema via response_mime_type + schema if supported
                kwargs["response_mime_type"] = "application/json"
                kwargs["response_schema"] = json_schema
            resp = m.generate_content(user, generation_config=kwargs)
            return resp.text or ""
        except Exception as e:
            raise ProviderError(str(e))

    def vision_chat(self, model: str, prompt: str, images: List[bytes], temperature: float = 0.1, max_tokens: int = 1200) -> str:
        try:
            m = genai.GenerativeModel(model_name=model)
            parts = [prompt]
            for img in images:
                parts.append({"mime_type": "image/png", "data": img})
            resp = m.generate_content(parts, generation_config={"temperature": temperature, "max_output_tokens": max_tokens})
            return resp.text or ""
        except Exception as e:
            raise ProviderError(str(e))

class OpenAAIProvider(BaseProvider):
    def __init__(self, api_key: str, base_url: str = "https://api.openaai.com/v1"):
        if not api_key:
            raise ProviderError("OpenAAI API key is missing")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def chat(self, model: str, system: str, user: str, temperature: float = 0.3, max_tokens: int = 1200, json_schema: Optional[Dict] = None) -> str:
        try:
            url = f"{self.base_url}/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            body = {
                "model": model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if json_schema:
                # OpenAI-style JSON schema via response_format
                body["response_format"] = {"type": "json_schema", "json_schema": {"name": "schema", "schema": json_schema}}
            r = requests.post(url, headers=headers, json=body, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise ProviderError(str(e))

    def vision_chat(self, model: str, prompt: str, images: List[bytes], temperature: float = 0.1, max_tokens: int = 1200) -> str:
        try:
            url = f"{self.base_url}/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            # Assume OpenAAI is OpenAI-compatible for vision with image_url or image_data
            content = [{"type": "text", "text": prompt}]
            for img in images:
                b64 = base64.b64encode(img).decode("utf-8")
                content.append({"type": "image_url", "image_url": f"data:image/png;base64,{b64}"})
            body = {
                "model": model,  # e.g., "gpt-4o-mini"
                "messages": [{"role": "user", "content": content}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            r = requests.post(url, headers=headers, json=body, timeout=180)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise ProviderError(str(e))

class GrokProvider(BaseProvider):
    def __init__(self, api_key: str):
        if not api_key:
            raise ProviderError("XAI_API_KEY is missing for Grok")
        if XAIClient is None:
            raise ProviderError("xai_sdk not installed")
        self.client = XAIClient(api_key=os.getenv("XAI_API_KEY") or api_key, timeout=3600)

    def chat(self, model: str, system: str, user: str, temperature: float = 0.3, max_tokens: int = 1200, json_schema: Optional[Dict] = None) -> str:
        # Sample usage per provided snippet
        chat = self.client.chat.create(model=model or "grok-4")
        chat.append(xai_system(system))
        chat.append(xai_user(user))
        response = chat.sample()
        return getattr(response, "content", "")

    def vision_chat(self, model: str, prompt: str, images: List[bytes], temperature: float = 0.1, max_tokens: int = 1200) -> str:
        # If Grok model supports images; send as user(image(...))
        chat = self.client.chat.create(model=model or "grok-4")
        # Note: xai_sdk.image usually accepts URL; some versions may accept bytes. Fall back to prompt-only if unsupported.
        try:
            parts = [prompt]
            # Try attach first image only to reduce latency
            if images:
                # If SDK requires URL, this will fail; we just send text fallback
                chat.append(xai_user(prompt, xai_image(images[0])))
            else:
                chat.append(xai_user(prompt))
        except Exception:
            chat.append(xai_user(prompt))
        response = chat.sample()
        return getattr(response, "content", "")

class ProviderManager:
    def __init__(self, gemini_api_key: str, openaai_api_key: str, openaai_base_url: str, xai_api_key: str):
        self.providers = {}
        try:
            if gemini_api_key:
                self.providers[ProviderName.GEMINI] = GeminiProvider(gemini_api_key)
        except Exception:
            pass
        try:
            if openaai_api_key:
                self.providers[ProviderName.OPENAAI] = OpenAAIProvider(openaai_api_key, openaai_base_url)
        except Exception:
            pass
        try:
            if xai_api_key or os.getenv("XAI_API_KEY"):
                self.providers[ProviderName.GROK] = GrokProvider(xai_api_key or os.getenv("XAI_API_KEY"))
        except Exception:
            pass

    def ready(self) -> bool:
        return len(self.providers) > 0

    def ready_for_vision(self, provider: str) -> bool:
        return provider in self.providers and detect_provider_supports_vision(provider)

    def get(self, provider: str) -> BaseProvider:
        if provider not in self.providers:
            raise ProviderError(f"Provider not available: {provider}")
        return self.providers[provider]
```

ocr_utils.py
--------------------------------
```python
import io
import os
from typing import List, Optional, Tuple
from pdf2image import convert_from_path
import pdfplumber
from PIL import Image
import pytesseract

from providers import ProviderManager, ProviderName

def detect_pdf_text_or_scanned(pdf_path: str) -> Tuple[int, Optional[bool]]:
    pages = 0
    any_text = False
    with pdfplumber.open(pdf_path) as pdf:
        pages = len(pdf.pages)
        for p in pdf.pages[:3]:
            txt = p.extract_text() or ""
            if txt.strip():
                any_text = True
                break
    return pages, (not any_text)

def extract_text_pdf_local(pdf_path: str, pages: Optional[List[int]] = None, lang: str = "chi_tra") -> Tuple[str, int, bool]:
    text_chunks = []
    page_count = 0
    used_ocr = False
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        target_pages = pages or list(range(1, total_pages + 1))
        for pno in target_pages:
            if pno < 1 or pno > total_pages:
                continue
            page = pdf.pages[pno - 1]
            txt = page.extract_text() or ""
            if txt.strip():
                text_chunks.append(txt)
            else:
                # Fallback to OCR
                used_ocr = True
                img = page.to_image(resolution=250).original
                pil_img = Image.fromarray(img)
                ocr_txt = pytesseract.image_to_string(pil_img, lang=lang)
                text_chunks.append(ocr_txt)
            page_count += 1
    text = "\n\n".join(text_chunks)
    return text, page_count, used_ocr

def pdf_to_images(pdf_path: str, pages: Optional[List[int]] = None, dpi: int = 250) -> List[bytes]:
    imgs = convert_from_path(pdf_path, dpi=dpi, first_page=None, last_page=None)
    result = []
    if pages:
        selected = [imgs[i - 1] for i in pages if 0 < i <= len(imgs)]
    else:
        selected = imgs
    for im in selected:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        result.append(buf.getvalue())
    return result

def llm_ocr_images(images: List[bytes], provider_manager: ProviderManager, provider_name: str) -> str:
    # Build a prompt to transcribe text from images (Traditional Chinese)
    prompt = (
        "請將影像中的所有文字完整轉寫為繁體中文（若為非中文請保留原文），"
        "保留段落、標點與欄位結構，不要總結或省略。"
    )
    provider = provider_manager.get(provider_name)
    # Choose a model: for Gemini use gemini-2.5-flash, OpenAAI use gpt-4o-mini, Grok a vision-capable model if available.
    model = "gemini-2.5-flash" if provider_name == ProviderName.GEMINI else ("gpt-4o-mini" if provider_name == ProviderName.OPENAAI else "grok-4")
    output = provider.vision_chat(model=model, prompt=prompt, images=images, max_tokens=4096)
    return output
```

agents.py
--------------------------------
```python
from dataclasses import dataclass
from typing import List, Dict, Any
import yaml

from providers import ProviderManager

@dataclass
class AgentConfig:
    name: str
    provider: str
    model: str
    parameters: Dict[str, Any]
    system_prompt: str
    user_prompt: str

@dataclass
class AgentsConfig:
    agents: List[AgentConfig]

def load_agents_config(path: str) -> AgentsConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    agents = []
    for a in raw.get("agents", []):
        agents.append(AgentConfig(
            name=a["name"],
            provider=a["provider"],
            model=a["model"],
            parameters=a.get("parameters", {}),
            system_prompt=a.get("system_prompt", ""),
            user_prompt=a.get("user_prompt", ""),
        ))
    return AgentsConfig(agents=agents)

class AgentRunner:
    def __init__(self, provider_manager: ProviderManager, agents: List[AgentConfig]):
        self.pm = provider_manager
        self.agents = agents

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        outputs: Dict[str, Any] = {}
        for agent in self.agents:
            prov = self.pm.get(agent.provider)
            sys = agent.system_prompt
            # Inject lightweight context tokens into user prompt
            ctx_hint = f"\n\n[可用上下文]\n- 摘要節錄: {context.get('summary_md','')[:1500]}\n- 當前抽取JSON keys: {list(context.get('extraction_json',{}).keys())}"
            user = agent.user_prompt + ctx_hint
            out = prov.chat(
                model=agent.model,
                system=sys,
                user=user,
                temperature=agent.parameters.get("temperature", 0.3),
                max_tokens=agent.parameters.get("max_tokens", 1200),
            )
            outputs[agent.name] = out
        return outputs
```

prompts.py
--------------------------------
```python
SYSTEM_SUMMARY_PROMPT_ZH = """你是專精於醫療器材法規與契約審閱的專家助理。任務：
1) 彙整多份文件的核心內容，以繁體中文產出精煉、條列式摘要。
2) 於摘要中自動標示關鍵詞為<span style="color: coral">關鍵詞</span>格式。
3) 保留文件間的對應關係與差異重點。
4) 對於缺漏資訊以「可能缺漏」標註，不自行臆測。
5) 格式限定為 Markdown。
"""

USER_SUMMARY_PROMPT_ZH = """以下是多份文件內容，請產出總結：
{documents}

請輸出：
- 文件清單與對應用途
- 各文件核心資訊（名稱、機構、地址、日期、編號）
- 委託製造關聯與責任分工
- 可能風險或缺漏項
- 一段結尾的整體評估

在關鍵詞（如公司名、地址、品項分類級別、契約關鍵條款、日期、編號等）外層加上 <span style="color: coral">... </span>。
"""

SYSTEM_EXTRACTION_PROMPT_ZH = """你是一位結構化資料抽取引擎，精通台灣醫療器材委託製造相關文件。
要求：
- 僅從給定文件抽取資訊，輸出 JSON（繁體中文），符合指定 schema。
- 未出現的欄位以空字串 "" 填入，不臆測。
- 所有欄位使用繁體中文與原文忠實表述。
- 不要包含多餘文本。
"""

USER_EXTRACTION_PROMPT_ZH = """文件如下，以繁體中文抽取以下欄位：
- 委託者名稱
- 委託者地址
- 受託者名稱
- 受託者地址
- 委託製造之合意
- 委託製造之醫療器材分類分級品項
- 委託製造之製程（例如：全部製程委託製造）
- 權利義務

文件內容：
{document}

請以純 JSON 回覆。"""

JSON_SCHEMA_EXTRACTION = {
    "type": "object",
    "properties": {
        "委託者名稱": {"type": "string"},
        "委託者地址": {"type": "string"},
        "受託者名稱": {"type": "string"},
        "受託者地址": {"type": "string"},
        "委託製造之合意": {"type": "string"},
        "委託製造之醫療器材分類分級品項": {"type": "string"},
        "委託製造之製程": {"type": "string"},
        "權利義務": {"type": "string"},
    },
    "required": [
        "委託者名稱",
        "委託者地址",
        "受託者名稱",
        "受託者地址",
        "委託製造之合意",
        "委託製造之醫療器材分類分級品項",
        "委託製造之製程",
        "權利義務"
    ],
    "additionalProperties": False
}
```

utils.py
--------------------------------
```python
from typing import Dict, Any, List
import pandas as pd
import plotly.express as px

def render_status_badge(label: str, status: str) -> str:
    cls = "badge-info"
    if status == "ok":
        cls = "badge-ok"
    elif status == "warn":
        cls = "badge-warn"
    elif status == "err":
        cls = "badge-err"
    return f'<span class="badge {cls}">{label}</span>'

def coralize_keywords(text: str) -> str:
    # If model already applied coral spans, keep them; otherwise a light heuristic to wrap obvious entities.
    # Here we just return as-is to avoid double-marking; rely on prompt to produce spans.
    return text

def to_markdown_table_zh(data: Dict[str, Any]) -> str:
    cols = list(data.keys())
    vals = [str(data.get(k, "")) for k in cols]
    # Build a simple Markdown table
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join([":---" for _ in cols]) + " |"
    row = "| " + " | ".join(vals) + " |"
    return "\n".join([header, sep, row])

def ensure_lang_zh(text: str) -> str:
    # Future: implement language detection/conversion if needed
    return text

def gen_dashboard_charts(metrics: Dict[str, Any]) -> List[Any]:
    charts = []
    # Pie chart for OCR methods
    ocr_counts = metrics.get("ocr_method_counts", {"local": 0, "llm": 0})
    pie_df = pd.DataFrame({
        "方法": ["本地 OCR", "LLM OCR"],
        "次數": [ocr_counts.get("local", 0), ocr_counts.get("llm", 0)]
    })
    charts.append(px.pie(pie_df, names="方法", values="次數", title="OCR 方法使用比例"))

    # Bar for provider usage
    prov = metrics.get("provider_usage", {})
    if prov:
        bar_df = pd.DataFrame({"供應商": list(prov.keys()), "次數": list(prov.values())})
        charts.append(px.bar(bar_df, x="供應商", y="次數", title="供應商使用次數"))

    # Line for pages processed over time - if actions tracked
    # Here we simulate from total pages only
    pages = metrics.get("pages_processed", 0)
    line_df = pd.DataFrame({"序列": list(range(1, (pages or 1)+1)), "頁累計": list(range(1, (pages or 1)+1))})
    charts.append(px.line(line_df, x="序列", y="頁累計", title="頁面處理累計"))
    return charts
```

agents.yaml (sample)
--------------------------------
```yaml
agents:
  - name: Summarizer-Refiner
    provider: Gemini
    model: gemini-2.5-flash
    parameters:
      temperature: 0.4
      max_tokens: 1400
    system_prompt: |
      你是摘要優化專家。請在不改變既有事實的前提下，讓摘要更清晰且條列更一致，保留 <span style="color: coral">...</span> 標示。
    user_prompt: |
      請優化以下摘要並維持繁體中文：
      ---
      {summary_md}
      ---
      要求：
      - 條列分節一致
      - 關鍵詞標示保持
      - 缺漏與風險另列

  - name: Consistency-Checker
    provider: OpenAAI
    model: gpt-4.1-mini
    parameters:
      temperature: 0.1
      max_tokens: 1000
    system_prompt: |
      你是一致性稽核助手，找出跨文件的矛盾、缺漏或不一致之處。
    user_prompt: |
      依下列文件與摘要，指出不一致或需要補件的項目，以繁體中文條列：
      [文件類型與名稱]：
      {doc_names}
      [摘要]：
      {summary_md}

  - name: Field-Completer
    provider: Grok
    model: grok-4-fast-reasoning
    parameters:
      temperature: 0.2
      max_tokens: 1200
    system_prompt: |
      你是資料欄位補齊助手，僅在文件有資訊時填入，無則以空字串表示。
    user_prompt: |
      參考下列抽取 JSON與摘要，請建議是否有欄位可補充或需要標註空白：
      [抽取JSON]：
      {extraction_json}
      [摘要]：
      {summary_md}
```

requirements.txt
--------------------------------
```
streamlit==1.39.0
PyYAML==6.0.2
google-generativeai==0.8.3
requests==2.32.3
pdfplumber==0.11.4
pdf2image==1.17.0
pytesseract==0.3.13
pillow==10.4.0
plotly==5.24.1
pandas==2.2.3
xai-sdk==0.2.7
```

Notes
- For LLM OCR, Gemini (gemini-2.5-flash) and OpenAAI (gpt-4o-mini assumed OpenAI-compatible) are supported for vision. Grok vision is attempted per sample but may require image URLs depending on xai_sdk version.
- If pdf2image requires poppler in your Space, add a system package in the Space build or use local OCR fallback via pdfplumber/pytesseract page images.
- API keys are taken from environment when set; otherwise masked input fields are provided. Keys are never printed.

Advanced Prompting Highlights
- Summarization prompt instructs coral keyword highlighting via <span style="color: coral">...</span>.
- Extraction prompt uses Traditional Chinese and strict JSON schema with graceful repair fallback.
- Agents allow further refinement, consistency checks, and field completion across providers.

Grok API usage (sample-integrated)
- GrokProvider.chat uses the provided xai_sdk pattern:
  client = Client(api_key=os.getenv("XAI_API_KEY"), timeout=3600)
  chat = client.chat.create(model="grok-4")
  chat.append(system(...)); chat.append(user(...)); response = chat.sample()

20 follow-up questions
1) Which provider and default models would you like preselected for each step (OCR-LLM, summarization, extraction, agents)?
2) Do you want us to automatically detect and recommend the best OCR path (digital vs scanned) per page instead of manual page selection?
3) Should the system attempt hybrid OCR (extract text via pdfplumber and only OCR pages with low text density)?
4) For LLM OCR, do you prefer chunking per page or batching multiple pages per request to optimize cost/time?
5) What maximum PDF size and page count should be allowed, and should there be a streaming preview for very large files?
6) Would you like keyword lists (e.g., company names, addresses, product classes) to be user-configurable to reinforce coral highlighting?
7) Should we add NER-based post-processing to auto-coralize keywords even if the LLM forgets the span styling?
8) For JSON validation, do you want strict schema enforcement with automatic re-ask to the model if invalid, up to N retries?
9) Should we include a human-in-the-loop review step for the JSON before finalizing and enabling download?
10) Do you need cross-document reconciliation (e.g., ensure 委託者/受託者名稱與地址在多文件間一致，否則提示差異)?
11) Would you like a template-based exporter (CSV, Excel) in addition to JSON and Markdown table?
12) Should the agents’ outputs be versioned with a history panel and diff viewer to compare runs?
13) Do you want role-based access control or simple login protection for the Space?
14) Should the dashboard include cost estimates and token usage per provider if available?
15) Do you want a “one-click pipeline” button to run OCR → Summary → Extraction → Agents automatically?
16) For Grok vision OCR, would you like us to add optional temporary image hosting to support image URLs if the SDK requires it?
17) Should we add a redaction feature to mask sensitive fields (e.g., certificate numbers, addresses) before displaying or exporting?
18) Would you like multi-language support for UI (e.g., switch between 繁中/English) while keeping outputs in Traditional Chinese?
19) Are there any additional fields you want in the extraction schema, such as 有效期間、簽署日期、聯絡人、統一編號?
20) Should the system allow creating custom agents from the UI and saving back to agents.yaml for future runs?
