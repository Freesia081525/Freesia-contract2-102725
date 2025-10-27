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
    initial_sidebar_state="expanded"
)

# ============ 20 Flower Themes ============
FLOWER_THEMES = {
    "櫻花 (Cherry Blossom)": {
        "primary": "#FFB7C5",
        "secondary": "#FFC0CB",
        "accent": "#FF69B4",
        "bg": "#FFF0F5",
        "text": "#8B008B"
    },
    "玫瑰 (Rose)": {
        "primary": "#FF1493",
        "secondary": "#FF69B4",
        "accent": "#C71585",
        "bg": "#FFF5EE",
        "text": "#8B0000"
    },
    "薰衣草 (Lavender)": {
        "primary": "#E6E6FA",
        "secondary": "#D8BFD8",
        "accent": "#9370DB",
        "bg": "#F8F8FF",
        "text": "#4B0082"
    },
    "向日葵 (Sunflower)": {
        "primary": "#FFD700",
        "secondary": "#FFA500",
        "accent": "#FF8C00",
        "bg": "#FFFACD",
        "text": "#8B4513"
    },
    "蓮花 (Lotus)": {
        "primary": "#FFB6C1",
        "secondary": "#FFC0CB",
        "accent": "#FF1493",
        "bg": "#F0FFF0",
        "text": "#2F4F4F"
    },
    "茉莉 (Jasmine)": {
        "primary": "#FFFFF0",
        "secondary": "#FAFAD2",
        "accent": "#F0E68C",
        "bg": "#FFFEF0",
        "text": "#556B2F"
    },
    "蘭花 (Orchid)": {
        "primary": "#DA70D6",
        "secondary": "#EE82EE",
        "accent": "#BA55D3",
        "bg": "#FFF0FA",
        "text": "#8B008B"
    },
    "鬱金香 (Tulip)": {
        "primary": "#FF6347",
        "secondary": "#FF7F50",
        "accent": "#FF4500",
        "bg": "#FFF5EE",
        "text": "#A0522D"
    },
    "百合 (Lily)": {
        "primary": "#FFFACD",
        "secondary": "#FFEFD5",
        "accent": "#FFE4B5",
        "bg": "#FFFFF0",
        "text": "#8B7355"
    },
    "牡丹 (Peony)": {
        "primary": "#FFB6C1",
        "secondary": "#FFDAB9",
        "accent": "#FFB5C5",
        "bg": "#FFF0F5",
        "text": "#A0522D"
    },
    "梅花 (Plum Blossom)": {
        "primary": "#FF69B4",
        "secondary": "#FFC0CB",
        "accent": "#FF1493",
        "bg": "#FFF0F5",
        "text": "#8B4513"
    },
    "茶花 (Camellia)": {
        "primary": "#DC143C",
        "secondary": "#FF6B6B",
        "accent": "#B22222",
        "bg": "#FFF5F5",
        "text": "#8B0000"
    },
    "康乃馨 (Carnation)": {
        "primary": "#FFB6C1",
        "secondary": "#FFC0CB",
        "accent": "#FF1493",
        "bg": "#FFF0F5",
        "text": "#C71585"
    },
    "繡球花 (Hydrangea)": {
        "primary": "#B0C4DE",
        "secondary": "#87CEEB",
        "accent": "#4682B4",
        "bg": "#F0F8FF",
        "text": "#00008B"
    },
    "紫羅蘭 (Violet)": {
        "primary": "#8A2BE2",
        "secondary": "#9370DB",
        "accent": "#9400D3",
        "bg": "#F8F8FF",
        "text": "#4B0082"
    },
    "水仙 (Daffodil)": {
        "primary": "#FFFFE0",
        "secondary": "#FFFACD",
        "accent": "#FFD700",
        "bg": "#FFFFF0",
        "text": "#B8860B"
    },
    "菊花 (Chrysanthemum)": {
        "primary": "#FFD700",
        "secondary": "#FFA500",
        "accent": "#FF8C00",
        "bg": "#FFFAF0",
        "text": "#8B4513"
    },
    "桔梗 (Bellflower)": {
        "primary": "#6495ED",
        "secondary": "#4169E1",
        "accent": "#0000CD",
        "bg": "#F0F8FF",
        "text": "#191970"
    },
    "波斯菊 (Cosmos)": {
        "primary": "#FF69B4",
        "secondary": "#FFB6C1",
        "accent": "#FF1493",
        "bg": "#FFF0F5",
        "text": "#C71585"
    },
    "曼陀羅 (Mandala)": {
        "primary": "#9370DB",
        "secondary": "#BA55D3",
        "accent": "#8B008B",
        "bg": "#F8F8FF",
        "text": "#4B0082"
    }
}

# ============ Session State Init ============
if "docs" not in st.session_state:
    st.session_state.docs = []
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
        "gemini": os.getenv("GEMINI_API_KEY", ""),
        "openaai": os.getenv("OPENAAI_API_KEY", ""),
        "xai": os.getenv("XAI_API_KEY", ""),
        "openaai_base": os.getenv("OPENAAI_BASE_URL", "https://api.openaai.com/v1"),
    }
if "selected_theme" not in st.session_state:
    st.session_state.selected_theme = "櫻花 (Cherry Blossom)"

# ============ Apply Selected Theme ============
theme = FLOWER_THEMES[st.session_state.selected_theme]
CUSTOM_CSS = f"""
<style>
:root {{
    --primary-color: {theme['primary']};
    --secondary-color: {theme['secondary']};
    --accent-color: {theme['accent']};
    --bg-color: {theme['bg']};
    --text-color: {theme['text']};
}}

.main {{
    background: linear-gradient(135deg, {theme['bg']} 0%, white 100%);
}}

.stButton>button {{
    background: linear-gradient(90deg, {theme['primary']}, {theme['accent']});
    color: white;
    border: none;
    border-radius: 20px;
    padding: 10px 24px;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}}

.stButton>button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.15);
}}

.badge {{ 
    display:inline-block; 
    padding:6px 12px; 
    border-radius:20px; 
    font-weight:600;
    margin: 4px;
    transition: all 0.3s ease;
}}
.badge-ok {{ 
    background: linear-gradient(135deg, #E8FFF3, #C8F7DC);
    color: #0F9D58; 
    border: 2px solid #0F9D58;
    box-shadow: 0 2px 8px rgba(15, 157, 88, 0.2);
}}
.badge-warn {{ 
    background: linear-gradient(135deg, #FFF8E5, #FFE8B5);
    color: #E6A100; 
    border: 2px solid #E6A100;
    box-shadow: 0 2px 8px rgba(230, 161, 0, 0.2);
}}
.badge-err {{ 
    background: linear-gradient(135deg, #FFEDEA, #FFD7D0);
    color: #D93025; 
    border: 2px solid #D93025;
    box-shadow: 0 2px 8px rgba(217, 48, 37, 0.2);
}}
.badge-info {{ 
    background: linear-gradient(135deg, #EAF2FF, #D0E3FF);
    color: #1967D2; 
    border: 2px solid #1967D2;
    box-shadow: 0 2px 8px rgba(25, 103, 210, 0.2);
}}

.step {{ 
    border-left: 5px solid {theme['accent']}; 
    padding-left: 15px; 
    margin: 15px 0;
    background: linear-gradient(90deg, {theme['bg']}, transparent);
    border-radius: 5px;
    padding: 10px 15px;
    font-size: 1.1em;
    font-weight: 600;
    color: {theme['text']};
}}

.kwd {{ 
    color: coral; 
    font-weight: 700;
    text-shadow: 0 0 10px rgba(255, 127, 80, 0.3);
}}

.theme-selector {{
    background: white;
    border-radius: 15px;
    padding: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}}

.metric-card {{
    background: white;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    border-left: 5px solid {theme['accent']};
    transition: all 0.3s ease;
}}

.metric-card:hover {{
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.12);
}}

.stExpander {{
    background: white;
    border-radius: 15px;
    border: 2px solid {theme['primary']};
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
}}

.flower-icon {{
    font-size: 2em;
    margin-right: 10px;
    filter: drop-shadow(0 0 5px {theme['accent']});
}}

h1, h2, h3 {{
    color: {theme['text']};
    text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 10px;
}}

.stTabs [data-baseweb="tab"] {{
    background-color: {theme['bg']};
    border-radius: 10px;
    padding: 10px 20px;
    border: 2px solid {theme['primary']};
}}

.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {theme['primary']}, {theme['accent']});
    color: white;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============ Sidebar: Theme Selection & API Keys ============
with st.sidebar:
    st.markdown('<div class="theme-selector">', unsafe_allow_html=True)
    st.header("🌸 主題選擇")
    selected_theme = st.selectbox(
        "選擇花卉主題",
        list(FLOWER_THEMES.keys()),
        index=list(FLOWER_THEMES.keys()).index(st.session_state.selected_theme)
    )
    if selected_theme != st.session_state.selected_theme:
        st.session_state.selected_theme = selected_theme
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    st.header("🔐 API 設定")
    gemini_from_env = bool(os.getenv("GEMINI_API_KEY"))
    openaai_from_env = bool(os.getenv("OPENAAI_API_KEY"))
    xai_from_env = bool(os.getenv("XAI_API_KEY"))

    gemini_status = render_status_badge("Gemini", "ok" if st.session_state.api_keys["gemini"] else "warn")
    openaai_status = render_status_badge("OpenAAI", "ok" if st.session_state.api_keys["openaai"] else "warn")
    grok_status = render_status_badge("Grok", "ok" if st.session_state.api_keys["xai"] else "warn")
    st.markdown(f"{gemini_status} {openaai_status} {grok_status}", unsafe_allow_html=True)

    if not gemini_from_env:
        st.session_state.api_keys["gemini"] = st.text_input(
            "Gemini API Key", 
            value=st.session_state.api_keys["gemini"], 
            type="password"
        )
    else:
        st.success("✓ Gemini 金鑰已從環境變數載入")

    if not openaai_from_env:
        st.session_state.api_keys["openaai"] = st.text_input(
            "OpenAAI API Key", 
            value=st.session_state.api_keys["openaai"], 
            type="password"
        )
        st.session_state.api_keys["openaai_base"] = st.text_input(
            "OpenAAI Base URL", 
            value=st.session_state.api_keys["openaai_base"]
        )
    else:
        st.success("✓ OpenAAI 金鑰已從環境變數載入")

    if not xai_from_env:
        st.session_state.api_keys["xai"] = st.text_input(
            "XAI API Key (Grok)", 
            value=st.session_state.api_keys["xai"], 
            type="password"
        )
    else:
        st.success("✓ Grok 金鑰已從環境變數載入")

    st.divider()
    st.header("⚙️ 模型與代理")
    agents_file = st.text_input("agents.yaml 路徑", value="agents.yaml")
    if st.button("🔄 載入 Agents", use_container_width=True):
        try:
            cfg = load_agents_config(agents_file)
            st.session_state.agents_config = cfg
            st.session_state.selected_agents = [a.name for a in cfg.agents]
            st.success(f"✓ 已載入 {len(cfg.agents)} 個 Agents")
        except Exception as e:
            st.error(f"❌ 讀取失敗: {e}")

    if st.session_state.agents_config:
        agent_names = [a.name for a in st.session_state.agents_config.agents]
        st.info(f"可用 Agents: {len(agent_names)}")
        chosen = st.multiselect(
            "選擇執行的 Agents", 
            agent_names, 
            default=st.session_state.selected_agents[:10]  # Default to first 10
        )
        st.session_state.selected_agents = chosen

# ============ Provider Manager ============
try:
    provider_manager = ProviderManager(
        gemini_api_key=st.session_state.api_keys["gemini"] or os.getenv("GEMINI_API_KEY", ""),
        openaai_api_key=st.session_state.api_keys["openaai"] or os.getenv("OPENAAI_API_KEY", ""),
        openaai_base_url=st.session_state.api_keys["openaai_base"] or os.getenv("OPENAAI_BASE_URL", "https://api.openaai.com/v1"),
        xai_api_key=st.session_state.api_keys["xai"] or os.getenv("XAI_API_KEY", ""),
    )
    st.session_state.providers_ready = provider_manager.ready()
except Exception as e:
    st.error(f"Provider 初始化失敗: {e}")
    provider_manager = None
    st.session_state.providers_ready = False

# ============ Header ============
col_h1, col_h2 = st.columns([1, 4])
with col_h1:
    st.markdown(f'<span class="flower-icon">🌺</span>', unsafe_allow_html=True)
with col_h2:
    st.title("醫療器材委託製造文件分析系統")
    st.caption("🚀 AI驅動的智慧文件處理平台 | 多供應商支援 | 31種進階分析代理")

# ============ Status Dashboard ============
st.markdown("### 📊 系統狀態總覽")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(render_status_badge("系統就緒", "ok" if st.session_state.providers_ready else "warn"), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("📄 已處理頁數", st.session_state.metrics["pages_processed"])
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("🤖 LLM OCR", st.session_state.metrics["ocr_method_counts"]["llm"])
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("💻 本地 OCR", st.session_state.metrics["ocr_method_counts"]["local"])
    st.markdown('</div>', unsafe_allow_html=True)

with col5:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("📁 文件數", len(st.session_state.docs))
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ============ Main Workflow Tabs ============
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📤 文件上傳", 
    "🔍 OCR處理", 
    "📝 智慧摘要", 
    "🎯 資料抽取", 
    "🤖 代理協作",
    "📈 儀表板"
])

# TAB 1: Document Upload
with tab1:
    st.markdown('<div class="step">步驟 1：文件匯入與管理</div>', unsafe_allow_html=True)
    
    with st.expander("📂 上傳或貼上文件（支援 txt, md, pdf）", expanded=True):
        doc_types = [
            "委託者之醫療器材商執照",
            "受託者之醫療器材商執照",
            "受託者之醫療器材製造許可",
            "委託製造契約",
            "其他文件 (others)",
        ]
        
        uploaded_files = st.file_uploader(
            "📎 選擇檔案上傳", 
            type=["txt", "md", "pdf"], 
            accept_multiple_files=True,
            help="支援最多5份文件"
        )
        
        st.markdown("##### 或直接貼上文字內容")
        paste_cols = st.columns(5)
        pasted_texts = []
        for i in range(5):
            with paste_cols[i]:
                pasted_texts.append(
                    st.text_area(
                        f"文件 {i+1}", 
                        height=160, 
                        key=f"paste_{i}",
                        placeholder=f"在此貼上第{i+1}份文件..."
                    )
                )

        st.markdown("##### 文件類型標記")
        label_cols = st.columns(5)
        doc_labels = []
        for i in range(5):
            with label_cols[i]:
                doc_labels.append(
                    st.selectbox(
                        f"類型 {i+1}", 
                        doc_types, 
                        index=min(i, len(doc_types)-1), 
                        key=f"label_{i}"
                    )
                )

        if st.button("✅ 確認匯入文件", use_container_width=True):
            st.session_state.docs = []
            
            # Handle uploads
            if uploaded_files:
                for idx, uf in enumerate(uploaded_files[:5]):
                    suffix = Path(uf.name).suffix.lower()
                    content_text = ""
                    pages_meta = {}
                    
                    if suffix in [".txt", ".md"]:
                        content_text = uf.read().decode("utf-8", errors="ignore")
                    elif suffix == ".pdf":
                        content_text = ""
                        pages_meta = {"pdf_bytes_b64": base64.b64encode(uf.read()).decode("utf-8")}
                    
                    label = doc_labels[idx] if idx < len(doc_labels) else "其他文件 (others)"
                    st.session_state.docs.append({
                        "name": uf.name,
                        "type_label": label,
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

            st.success(f"✨ 成功匯入 {len(st.session_state.docs)} 份文件")
            st.balloons()
    
    # Display imported documents
    if st.session_state.docs:
        st.markdown("### 📋 已匯入文件清單")
        for idx, doc in enumerate(st.session_state.docs):
            with st.container(border=True):
                col_doc1, col_doc2, col_doc3 = st.columns([3, 2, 1])
                with col_doc1:
                    st.markdown(f"**{doc['name']}**")
                with col_doc2:
                    st.caption(f"類型: {doc['type_label']}")
                with col_doc3:
                    st.caption(f"來源: {doc['source']}")

# TAB 2: OCR Processing
with tab2:
    st.markdown('<div class="step">步驟 2：PDF 智慧 OCR</div>', unsafe_allow_html=True)
    
    target_docs = [d for d in st.session_state.docs if d["name"].lower().endswith(".pdf")]
    
    if not target_docs:
        st.info("ℹ️ 目前沒有 PDF 文件需要處理")
    else:
        with st.expander("⚙️ OCR 設定與執行", expanded=True):
            col_ocr1, col_ocr2 = st.columns(2)
            
            with col_ocr1:
                ocr_method = st.radio(
                    "選擇 OCR 方法",
                    ["Local (pdfplumber/pytesseract)", "LLM OCR (視覺模型)"],
                    help="本地OCR速度快但準確度較低；LLM OCR準確度高但需要API呼叫"
                )
            
            with col_ocr2:
                provider_choice_for_ocr = None
                if ocr_method == "LLM OCR (視覺模型)":
                    provider_choice_for_ocr = st.selectbox(
                        "選擇視覺模型供應商",
                        [ProviderName.GEMINI.value, ProviderName.OPENAAI.value, ProviderName.GROK.value],
                        index=0,
                        help="推薦使用 Gemini 以獲得最佳繁體中文辨識"
                    )

            for doc in target_docs:
                with st.container(border=True):
                    st.subheader(f"📄 {doc['name']}")
                    
                    pdf_bytes = base64.b64decode(doc["pages_meta"]["pdf_bytes_b64"])
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
                        tf.write(pdf_bytes)
                        pdf_path = tf.name

                    try:
                        pages_count, is_scanned = detect_pdf_text_or_scanned(pdf_path)
                    except Exception as e:
                        st.warning(f"無法偵測PDF屬性: {e}")
                        pages_count, is_scanned = (None, None)

                    col_info1, col_info2, col_info3 = st.columns(3)
                    with col_info1:
                        st.metric("總頁數", pages_count or "未知")
                    with col_info2:
                        st.metric("類型", "掃描版" if is_scanned else "文字版")
                    with col_info3:
                        pages_str = st.text_input(
                            "指定頁碼", 
                            key=f"pages_{doc['name']}",
                            placeholder="例: 1,3,5-7 (空白=全部)",
                            help="可指定單頁或範圍"
                        )
                    
                    pages = []
                    if pages_str.strip():
                        try:
                            for part in pages_str.replace(" ", "").split(","):
                                if "-" in part:
                                    s, e = part.split("-")
                                    pages.extend(list(range(int(s), int(e) + 1)))
                                else:
                                    pages.append(int(part))
                        except Exception as e:
                            st.error(f"頁碼格式錯誤: {e}")

                    if st.button(f"🚀 執行 OCR：{doc['name']}", key=f"ocr_btn_{doc['name']}"):
                        if not st.session_state.providers_ready and ocr_method == "LLM OCR (視覺模型)":
                            st.error("❌ 請先設定 API Key")
                            continue
                            
                        st.session_state.metrics["start_time"] = time.time()
                        
                        with st.spinner("處理中..."):
                            if ocr_method == "Local (pdfplumber/pytesseract)":
                                try:
                                    text, page_count, used_ocr = extract_text_pdf_local(
                                        pdf_path, 
                                        pages=pages or None, 
                                        lang="chi_tra"
                                    )
                                    doc["content_text"] = text
                                    st.session_state.metrics["pages_processed"] += page_count
                                    st.session_state.metrics["ocr_method_counts"]["local"] += 1
                                    st.success(f"✅ 完成 ({page_count} 頁)")
                                except Exception as e:
                                    st.error(f"❌ 失敗: {e}")
                            else:
                                if not provider_manager or not provider_manager.ready_for_vision(provider_choice_for_ocr):
                                    st.error("❌ 選擇的供應商不支援視覺功能")
                                else:
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
                                        st.success(f"✅ 完成 ({len(imgs)} 頁)")
                                    except Exception as e:
                                        st.error(f"❌ 失敗: {e}")

                        st.session_state.metrics["end_time"] = time.time()
                        
                    # Preview extracted text
                    if doc["content_text"]:
                        with st.expander("👁️ 預覽已擷取文字", expanded=False):
                            st.text_area(
                                "內容", 
                                value=doc["content_text"][:1000] + ("..." if len(doc["content_text"]) > 1000 else ""),
                                height=200,
                                disabled=True
                            )

# TAB 3: Summarization
with tab3:
    st.markdown('<div class="step">步驟 3：智慧摘要生成</div>', unsafe_allow_html=True)
    
    with st.expander("🎯 摘要設定", expanded=True):
        col_sum1, col_sum2 = st.columns(2)
        
        with col_sum1:
            provider_for_summary = st.selectbox(
                "選擇供應商",
                [ProviderName.GEMINI.value, ProviderName.OPENAAI.value, ProviderName.GROK.value],
                index=0,
                key="sum_provider"
            )
        
        with col_sum2:
            default_model = {
                ProviderName.GEMINI.value: "gemini-2.0-flash-exp",
                ProviderName.OPENAAI.value: "gpt-4o-mini",
                ProviderName.GROK.value: "grok-beta"
            }
            model_for_summary = st.text_input(
                "模型名稱", 
                value=default_model.get(provider_for_summary, "gemini-2.0-flash-exp"),
                key="sum_model"
            )

        col_sum3, col_sum4 = st.columns(2)
        with col_sum3:
            temperature = st.slider("Temperature", 0.0, 2.0, 0.4, 0.1, key="sum_temp")
        with col_sum4:
            max_tokens = st.number_input("Max Tokens", 128, 8000, 1400, 100, key="sum_max")

        if st.button("✨ 生成摘要", use_container_width=True):
            if not st.session_state.docs:
                st.warning("⚠️ 請先匯入文件")
            elif not st.session_state.providers_ready:
                st.error("❌ 請先設定 API Key")
            else:
                with st.spinner("AI 正在分析文件..."):
                    combined_texts = []
                    for d in st.session_state.docs:
                        label = d["type_label"]
                        name = d["name"]
                        content = d["content_text"].strip()
                        if not content and name.lower().endswith(".pdf"):
                            st.warning(f"⚠️ {name} 尚未 OCR")
                            continue
                        combined_texts.append(f"【{label} | {name}】\n{content}\n")
                    
                    if not combined_texts:
                        st.error("❌ 沒有可用的文件內容")
                    else:
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
                            st.session_state.summary_md = coralize_keywords(output)
                            st.session_state.metrics["provider_usage"].setdefault(provider_for_summary, 0)
                            st.session_state.metrics["provider_usage"][provider_for_summary] += 1
                            st.balloons()
                            st.success("✅ 摘要生成完成")
                        except Exception as e:
                            st.error(f"❌ 失敗: {e}")

    st.markdown("### 📝 摘要內容（可編輯）")
    st.session_state.summary_md = st.text_area(
        "Markdown 格式摘要", 
        value=st.session_state.summary_md, 
        height=400,
        help="支援 Markdown 語法，可直接編輯"
    )
    
    if st.session_state.summary_md:
        st.markdown("### 👁️ 渲染預覽")
        st.markdown(st.session_state.summary_md, unsafe_allow_html=True)
        
        st.download_button(
            "📥 下載摘要",
            data=st.session_state.summary_md,
            file_name="summary.md",
            mime="text/markdown",
            use_container_width=True
        )

# TAB 4: Extraction
with tab4:
    st.markdown('<div class="step">步驟 4：結構化資料抽取</div>', unsafe_allow_html=True)
    
    if not st.session_state.docs:
        st.info("ℹ️ 請先匯入文件")
    else:
        with st.expander("🎯 抽取設定", expanded=True):
            doc_options = [f"{d['type_label']} | {d['name']}" for d in st.session_state.docs]
            selection = st.selectbox("選擇目標文件", doc_options)
            target_doc = st.session_state.docs[doc_options.index(selection)]

            col_ext1, col_ext2 = st.columns(2)
            with col_ext1:
                provider_for_extract = st.selectbox(
                    "供應商",
                    [ProviderName.GEMINI.value, ProviderName.OPENAAI.value, ProviderName.GROK.value],
                    index=0,
                    key="ext_provider"
                )
            
            with col_ext2:
                default_model_ext = {
                    ProviderName.GEMINI.value: "gemini-2.0-flash-exp",
                    ProviderName.OPENAAI.value: "gpt-4o-mini",
                    ProviderName.GROK.value: "grok-beta"
                }
                model_for_extract = st.text_input(
                    "模型",
                    value=default_model_ext.get(provider_for_extract, "gemini-2.0-flash-exp"),
                    key="ext_model"
                )

            col_ext3, col_ext4 = st.columns(2)
            with col_ext3:
                temp2 = st.slider("Temperature", 0.0, 2.0, 0.2, 0.1, key="ext_temp")
            with col_ext4:
                max_tokens2 = st.number_input("Max Tokens", 256, 8000, 1200, 100, key="ext_max")

            if st.button("🚀 執行抽取", use_container_width=True):
                if not target_doc["content_text"].strip():
                    st.error("❌ 文件內容為空，請先 OCR 或輸入內容")
                elif not st.session_state.providers_ready:
                    st.error("❌ 請先設定 API Key")
                else:
                    with st.spinner("AI 正在抽取結構化資料..."):
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
                                json_schema=JSON_SCHEMA_EXTRACTION,
                            )
                            
                            # Parse JSON with fallback
                            try:
                                data = json.loads(output)
                            except:
                                try:
                                    cleaned = output.strip().strip("```json").strip("```").strip()
                                    data = json.loads(cleaned)
                                except:
                                    st.warning("🔄 JSON 格式修復中...")
                                    repair_prompt = f"將以下內容轉為有效 JSON (繁體中文):\n\n{output}"
                                    output2 = pm.chat(
                                        model=model_for_extract,
                                        system="你是 JSON 格式化專家，只輸出有效 JSON。",
                                        user=repair_prompt,
                                        temperature=0.0,
                                        max_tokens=max_tokens2,
                                    )
                                    data = json.loads(output2.strip().strip("```json").strip("```"))

                            st.session_state.extraction_json = data
                            st.session_state.extraction_table_md = to_markdown_table_zh(data)
                            st.session_state.metrics["provider_usage"].setdefault(provider_for_extract, 0)
                            st.session_state.metrics["provider_usage"][provider_for_extract] += 1
                            st.success("✅ 抽取完成")
                            st.balloons()
                        except Exception as e:
                            st.error(f"❌ 失敗: {e}")

        if st.session_state.extraction_json:
            st.markdown("### 📊 抽取結果")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                st.markdown("#### JSON 格式")
                st.json(st.session_state.extraction_json)
                st.download_button(
                    "📥 下載 JSON",
                    data=json.dumps(st.session_state.extraction_json, ensure_ascii=False, indent=2),
                    file_name="extraction.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col_result2:
                st.markdown("#### Markdown 表格")
                st.markdown(st.session_state.extraction_table_md)
                st.download_button(
                    "📥 下載表格",
                    data=st.session_state.extraction_table_md,
                    file_name="extraction_table.md",
                    mime="text/markdown",
                    use_container_width=True
                )

# TAB 5: Agents
with tab5:
    st.markdown('<div class="step">步驟 5：多代理協作系統</div>', unsafe_allow_html=True)
    
    if not st.session_state.agents_config:
        st.info("ℹ️ 請先於側邊欄載入 agents.yaml")
    else:
        st.markdown(f"### 🤖 已載入 {len(st.session_state.agents_config.agents)} 個代理")
        
        with st.expander("⚙️ 代理設定與編輯", expanded=False):
            editable_agents: List[AgentConfig] = []
            
            for agent in st.session_state.agents_config.agents:
                if agent.name not in st.session_state.selected_agents:
                    continue
                
                with st.container(border=True):
                    st.subheader(f"🔧 {agent.name}")
                    
                    col_a1, col_a2, col_a3 = st.columns(3)
                    
                    with col_a1:
                        new_provider = st.selectbox(
                            "供應商",
                            [ProviderName.GEMINI.value, ProviderName.OPENAAI.value, ProviderName.GROK.value],
                            index=[ProviderName.GEMINI.value, ProviderName.OPENAAI.value, ProviderName.GROK.value].index(agent.provider),
                            key=f"{agent.name}_provider"
                        )
                    
                    with col_a2:
                        new_model = st.text_input(
                            "模型", 
                            value=agent.model, 
                            key=f"{agent.name}_model"
                        )
                    
                    with col_a3:
                        new_temp = st.slider(
                            "Temp", 
                            0.0, 2.0, 
                            agent.parameters.get("temperature", 0.3), 
                            0.1, 
                            key=f"{agent.name}_temp"
                        )
                    
                    new_max = st.number_input(
                        "Max Tokens", 
                        128, 8000, 
                        agent.parameters.get("max_tokens", 1200), 
                        100, 
                        key=f"{agent.name}_max"
                    )
                    
                    new_system = st.text_area(
                        "System Prompt", 
                        value=agent.system_prompt, 
                        height=100, 
                        key=f"{agent.name}_sys"
                    )
                    
                    new_user = st.text_area(
                        "User Prompt", 
                        value=agent.user_prompt, 
                        height=120, 
                        key=f"{agent.name}_user"
                    )

                    editable_agents.append(AgentConfig(
                        name=agent.name,
                        provider=new_provider,
                        model=new_model,
                        parameters={"temperature": new_temp, "max_tokens": new_max},
                        system_prompt=new_system,
                        user_prompt=new_user
                    ))

        if st.button("🚀 執行所有選定代理", use_container_width=True):
            if not st.session_state.providers_ready:
                st.error("❌ 請先設定 API Key")
            else:
                try:
                    runner = AgentRunner(provider_manager, editable_agents)
                    context = {
                        "docs": st.session_state.docs,
                        "summary_md": st.session_state.summary_md,
                        "extraction_json": st.session_state.extraction_json,
                    }
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    outputs = {}
                    total = len(editable_agents)
                    
                    for idx, agent in enumerate(editable_agents):
                        status_text.text(f"執行中: {agent.name} ({idx+1}/{total})")
                        
                        try:
                            prov = provider_manager.get(agent.provider)
                            sys = agent.system_prompt
                            ctx_hint = f"\n\n[上下文]\n- 摘要: {context.get('summary_md','')[:500]}\n- JSON keys: {list(context.get('extraction_json',{}).keys())}"
                            user = agent.user_prompt + ctx_hint
                            
                            out = prov.chat(
                                model=agent.model,
                                system=sys,
                                user=user,
                                temperature=agent.parameters.get("temperature", 0.3),
                                max_tokens=agent.parameters.get("max_tokens", 1200),
                            )
                            outputs[agent.name] = out
                            
                            st.session_state.metrics["provider_usage"].setdefault(agent.provider, 0)
                            st.session_state.metrics["provider_usage"][agent.provider] += 1
                            
                        except Exception as e:
                            outputs[agent.name] = f"❌ 錯誤: {str(e)}"
                        
                        progress_bar.progress((idx + 1) / total)
                    
                    status_text.text("✅ 完成")
                    st.success(f"✨ {len(outputs)} 個代理執行完成")
                    
                    st.markdown("### 📋 代理輸出結果")
                    for name, out in outputs.items():
                        with st.expander(f"🤖 {name}", expanded=True):
                            if isinstance(out, (dict, list)):
                                st.json(out)
                            else:
                                st.markdown(out)
                    
                except Exception as e:
                    st.error(f"❌ 執行失敗: {e}")

# TAB 6: Dashboard
with tab6:
    st.markdown('<div class="step">步驟 6：互動式儀表板</div>', unsafe_allow_html=True)
    
    if st.session_state.metrics["start_time"] and st.session_state.metrics["end_time"]:
        elapsed = st.session_state.metrics["end_time"] - st.session_state.metrics["start_time"]
        st.metric("⏱️ 處理時間", f"{elapsed:.2f} 秒")
    
    charts = gen_dashboard_charts(st.session_state.metrics)
    
    if charts:
        for c in charts:
            st.plotly_chart(c, use_container_width=True)
    
    st.markdown("### 📊 供應商使用統計")
    if st.session_state.metrics["provider_usage"]:
        st.json(st.session_state.metrics["provider_usage"])
    else:
        st.info("尚無使用記錄")
    
    st.markdown("### 🔍 系統診斷")
    diag_col1, diag_col2 = st.columns(2)
    
    with diag_col1:
        st.markdown("#### 環境變數狀態")
        st.code(f"""
GEMINI_API_KEY: {'✓ 已設定' if os.getenv('GEMINI_API_KEY') else '✗ 未設定'}
OPENAAI_API_KEY: {'✓ 已設定' if os.getenv('OPENAAI_API_KEY') else '✗ 未設定'}
XAI_API_KEY: {'✓ 已設定' if os.getenv('XAI_API_KEY') else '✗ 未設定'}
        """)
    
    with diag_col2:
        st.markdown("#### 系統資訊")
        st.code(f"""
Streamlit: {st.__version__}
Python: {os.sys.version.split()[0]}
主題: {st.session_state.selected_theme}
        """)

# ============ Footer ============
st.divider()
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption("🌸 醫療器材文件分析系統")
with col_f2:
    st.caption("🤖 AI驅動 | 多供應商支援")
with col_f3:
    st.caption("Made with ❤️ using Streamlit")
