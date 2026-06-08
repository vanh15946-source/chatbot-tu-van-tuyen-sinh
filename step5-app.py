import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from step6_analytics_logger import init_logger, log_interaction

load_dotenv()
init_logger()


# ─────────────────────────────────────────────
# 1. KHỞI TẠO HỆ THỐNG
# ─────────────────────────────────────────────
@st.cache_resource
def init_system():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="tnus_tuyen_sinh"
    )
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
    )
    llm = ChatGroq(
        temperature=0.1,
        model_name="llama-3.3-70b-versatile",
        max_tokens=4096,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    return vector_store, graph, llm


vector_store, graph, llm = init_system()

# ─────────────────────────────────────────────
# 2. METADATA FILTER & KEYWORDS (FIX CỨNG PHẠM VI)
# ─────────────────────────────────────────────
# FIX 1: Loại bỏ hoàn toàn ký tự đơn "p" để tránh bắt nhầm các từ như "phương thức", "học phí"
NHA_TRO_KEYWORDS = [
    "trọ", "thuê", "phòng", "phg", "ở ghép", "xóm", "ký túc", "ktx",
    "điều hòa", "nóng lạnh", "wifi", "giữ xe", "tiện ích", "khép kín",
    "điện nước", "đặt cọc", "nhà trọ", "phòng trọ", "thuê phòng", "ở trọ",
    "giá thuê", "phòng cho thuê", "tìm trọ", "tìm phòng", "gần trường",
    "z115", "phú thái", "tân thịnh", "quyết thắng", "phan đình phùng"
]

TO_HOP_KEYWORDS = [
    "tổ hợp", "khối thi", "môn thi", "xét tuyển bằng môn",
    "a00", "a01", "a02", "a03", "a04", "a10",
    "b00", "b08", "c00", "c01", "d01", "d07",
]


def get_category_filter(query: str) -> tuple[dict, str]:
    q = query.lower()
    if any(kw in q for kw in NHA_TRO_KEYWORDS):
        return {"category": "nha_tro"}, "nha_tro"
    return {"category": "tuyen_sinh"}, "tuyen_sinh"


def is_to_hop_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in TO_HOP_KEYWORDS)


# ─────────────────────────────────────────────
# 3. PHÂN LOẠI / ĐỊNH TUYẾN CÂU HỎI (ROUTER)
# ─────────────────────────────────────────────
ROUTE_TEMPLATE = """Bạn là bộ phân loại câu hỏi tuyển sinh TNUS.
Chỉ trả về MỘT từ duy nhất:
- RAG: Hỏi về ngành học, điểm chuẩn, tổ hợp môn, học phí, nhà trọ, thông tin trường.
- CHAT: Chào hỏi, cảm ơn, khen ngợi.
- GENERAL: Các câu hỏi ngoài luồng không liên quan đến trường lớp.

Câu hỏi: {question}
Phân loại:"""

route_chain = ChatPromptTemplate.from_template(ROUTE_TEMPLATE) | llm | StrOutputParser()


def classify_query(question: str) -> str:
    try:
        res = route_chain.invoke({"question": question}).strip().upper()
        first_word = res.split()[0] if res else "RAG"
        return first_word if first_word in ["RAG", "CHAT", "GENERAL"] else "RAG"
    except Exception:
        return "RAG"


# ─────────────────────────────────────────────
# 4. TÓM TẮT LỊCH SỬ HỘI THOẠI
# ─────────────────────────────────────────────
SUMMARIZE_TEMPLATE = """Tóm tắt ngắn gọn nhu cầu cốt lõi của người dùng trong hội thoại dưới đây (tối đa 3 câu).
Hội thoại:\n{conversation}\nTóm tắt:"""

summarize_chain = ChatPromptTemplate.from_template(SUMMARIZE_TEMPLATE) | llm | StrOutputParser()
MAX_RAW_HISTORY = 8
SUMMARIZE_THRESHOLD = 15


def build_langchain_history(messages: list) -> list:
    history = []
    if st.session_state.get("history_summary"):
        history.append(AIMessage(content=f"[Bối cảnh cũ: {st.session_state['history_summary']}]"))
    for msg in messages[-MAX_RAW_HISTORY:]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))
    return history


def maybe_summarize_history():
    msgs = st.session_state.messages
    if len(msgs) <= SUMMARIZE_THRESHOLD:
        return
    old_msgs = msgs[:-MAX_RAW_HISTORY]
    conv_text = "\n".join(f"{'User' if m['role'] == 'user' else 'AI'}: {m['content']}" for m in old_msgs)
    if st.session_state.get("history_summary"):
        conv_text = f"[Cũ]: {st.session_state['history_summary']}\n" + conv_text
    try:
        st.session_state["history_summary"] = summarize_chain.invoke({"conversation": conv_text})
        st.session_state.messages = msgs[-MAX_RAW_HISTORY:]
    except:
        pass


# ─────────────────────────────────────────────
# 5. TRUY XUẤT DỮ LIỆU ĐỒ THỊ VÀ VECTOR (FIX SCHEMA LÀM CHUẨN)
# ─────────────────────────────────────────────
def vector_search(query: str, cat_filter: dict, k: int = 10) -> list[str]:
    try:
        # Giữ nguyên filter category (hoạt động tốt vì dạng chuỗi)
        docs = vector_store.max_marginal_relevance_search(
            query, k=k, fetch_k=30, filter=cat_filter
        )
        return [doc.page_content for doc in docs]
    except Exception:
        return []


def extract_keywords(query: str) -> list[str]:
    prompt = ChatPromptTemplate.from_template(
        "Trích xuất 2 cụm danh từ cốt lõi nhất từ câu hỏi để tìm kiếm chính xác thực thể. "
        "Chỉ trả về từ khóa cách nhau bằng dấu phẩy, không giải thích gì thêm.\nCâu hỏi: {query}"
    )
    try:
        response = (prompt | llm | StrOutputParser()).invoke({"query": query})
        return [kw.strip() for kw in response.split(",") if kw.strip() and len(kw.strip()) >= 3]
    except Exception:
        return []


def retrieve_from_graph(query: str, keywords: list[str], category: str) -> list[str]:
    if category == "nha_tro":
        return []

    graph_context = []
    q_lower = query.lower()

    # FIX 2: Đồng bộ Node Label 'Ngànhhọc' viết thường không dấu cách và dùng thuộc tính n.id
    if "tất cả" in q_lower and "ngành" in q_lower:
        cypher_all = "MATCH (n:Ngànhhọc) RETURN n.id AS nganh ORDER BY n.id"
        try:
            res = graph.query(cypher_all)
            if res:
                danh_sach = [r['nganh'] for r in res]
                graph_context.append("Danh sách toàn bộ các ngành đào tạo tại TNUS:\n- " + "\n- ".join(danh_sach))
        except:
            pass

    # FIX 3: Luồng lấy tổ hợp môn chuẩn hóa theo mối quan hệ [DÙNG_TỔ_HỢP] và label viết thường
    if is_to_hop_query(query):
        for kw in keywords:
            cypher_tohop = """
            MATCH (n:Ngànhhọc)-[:DÙNG_TỔ_HỢP]->(t:Tổhợpmôn)
            WHERE toLower(n.id) CONTAINS toLower($keyword)
            RETURN n.id AS nganh, t.ma AS ma_to_hop, t.mon_hoc AS mon_hoc
            ORDER BY t.ma
            """
            try:
                results = graph.query(cypher_tohop, params={"keyword": kw})
                if results:
                    nganh_name = results[0]["nganh"]
                    graph_context.append(f"Ngành {nganh_name} xét tuyển các tổ hợp:")
                    for r in results:
                        graph_context.append(f"  + {r['ma_to_hop']}: {r['mon_hoc']}")
            except:
                pass
        if graph_context:
            return list(set(graph_context))

    # FIX 4: Quét Tổng quát giới hạn theo Label chuẩn hóa, triệt tiêu nhiễu [MENTIONS], chỉ bốc thuộc tính id
    if keywords:
        for kw in keywords:
            cypher_general = """
            MATCH (n)-[r]->(m)
            WHERE (n:Ngànhhọc OR n:Khoaviện OR n:Tổhợpmôn)
              AND toLower(n.id) CONTAINS toLower($keyword)
            RETURN n.id AS source, type(r) AS rel, m.id AS target
            LIMIT 10
            """
            try:
                for res in graph.query(cypher_general, params={"keyword": kw}):
                    graph_context.append(f"{res['source']} -> {res['rel']} -> {res['target']}")
            except:
                pass

    return list(set(graph_context))


def get_context(query: str) -> tuple[str, str, str]:
    cat_filter, category = get_category_filter(query)

    k_val = 10 if category == "tuyen_sinh" else 8
    docs = vector_search(query, cat_filter, k=k_val)
    vector_context = "\n\n".join(docs)

    graph_context = "Không có dữ liệu đồ thị."
    if category == "tuyen_sinh":
        keywords = extract_keywords(query)
        graph_data = retrieve_from_graph(query, keywords, category)
        if graph_data:
            graph_context = "\n".join(graph_data)

    return vector_context, graph_context, category


# ─────────────────────────────────────────────
# 6. GỘP PROMPT & MASTER CHAIN
# ─────────────────────────────────────────────
MASTER_SYSTEM = """Bạn là trợ lý AI tuyển sinh đại học trực thuộc Trường Đại học Khoa học - ĐH Thái Nguyên (TNUS) năm 2026.
CHẾ ĐỘ XỬ LÝ: [{route}]

QUY TẮC CỐT LÕI (CHẾ ĐỘ RAG):
1. Bạn phải dựa hoàn toàn vào DỮ LIỆU ĐỒ THỊ và DỮ LIỆU VECTOR được cung cấp dưới đây để trả lời câu hỏi. Không được tự suy diễn thông tin nằm ngoài tài liệu.
2. Đối với các dữ liệu đồ thị, tên thực thể hiển thị chính là mã hoặc định danh của thực thể đó. Hãy trình bày tường minh, dễ hiểu cho học sinh THPT.
3. Luôn giữ thái độ cởi mở, thân thiện, xưng "mình" và gọi người học là "bạn" hoặc "em".

--- DỮ LIỆU TỪ HỆ THỐNG GRAPH (Mối quan hệ chính xác) ---
{graph_context}

--- DỮ LIỆU TỪ HỆ THỐNG VECTOR (Văn bản chi tiết) ---
{vector_context}

NẾU CHẾ ĐỘ LÀ [CHAT] / [GENERAL]:
- CHAT: Phản hồi ấm áp, định hướng quay lại tìm hiểu tuyển sinh TNUS.
- GENERAL: Đáp ngắn gọn (dưới 2 câu) rồi khéo léo từ chối để tập trung nhiệm vụ tuyển sinh.
"""

master_prompt = ChatPromptTemplate.from_messages([
    ("system", MASTER_SYSTEM),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])
master_chain = master_prompt | llm | StrOutputParser()

# ─────────────────────────────────────────────
# 7. GIAO DIỆN STREAMLIT
# ─────────────────────────────────────────────
st.set_page_config(page_title="Tư vấn tuyển sinh TNUS", page_icon="tnus_logo.png", layout="centered")

with st.sidebar:
    # --- THÊM LOGO VÀO SIDEBAR ---
    st.image("tnus_logo.png", use_container_width=True)

    st.markdown("### Tư vấn tuyển sinh TNUS 2026")
    st.caption("Đại học Khoa học - ĐH Thái Nguyên")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Xóa chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pop("history_summary", None)
            st.rerun()
    with col2:
        if st.button("📋 Tóm tắt", use_container_width=True):
            st.session_state["show_summary"] = True

    st.divider()
    msg_count = len(st.session_state.get("messages", []))
    user_msgs = sum(1 for m in st.session_state.get("messages", []) if m["role"] == "user")
    st.caption(f"📊 Tổng tin nhắn: **{msg_count}** | Câu hỏi: **{user_msgs}**")

    if st.session_state.get("history_summary"):
        with st.expander("📝 Tóm tắt lịch sử"):
            st.info(st.session_state["history_summary"])

    st.divider()
    if st.button("📊 Xem thống kê", use_container_width=True):
        st.session_state["show_analytics"] = True

# --- THÊM LOGO BÊN CẠNH TIÊU ĐỀ CHÍNH ---
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("tnus_logo.png", use_container_width=True)
with col_title:
    st.title("Tư vấn tuyển sinh TNUS")

st.markdown("*Hỏi về ngành học, điểm chuẩn, khối xét tuyển, hoặc thông tin nhà trọ sinh viên!*")

if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    welcome_msg = (
        "Xin chào! 👋 Mình là **trợ lý tuyển sinh đại học TNUS 2026**.\n\n"
        "Sẵn sàng hỗ trợ bạn tra cứu điểm chuẩn, tổ hợp xét tuyển môn và các khu nhà trọ quanh trường.\n"
        "Bạn cần mình hỗ trợ thông tin gì hôm nay?"
    )
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg, "route": "CHAT"})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("route") not in (None, "CHAT"):
            ROUTE_BADGE = {"RAG": "📚 Dữ liệu Đại học", "GENERAL": "🌐 Kiến thức nền"}
            st.caption(f"*Nguồn: {ROUTE_BADGE.get(message['route'], '')}*")

if st.session_state.get("show_summary"):
    st.session_state.pop("show_summary")
    if len(st.session_state.messages) > 2:
        with st.spinner("Đang tóm tắt..."):
            conv = "\n".join(
                f"{'User' if m['role'] == 'user' else 'AI'}: {m['content']}" for m in st.session_state.messages)
            try:
                summary = summarize_chain.invoke({"conversation": conv})
                st.info(f"**📋 Tóm tắt hội thoại:**\n\n{summary}")
            except:
                st.warning("Lỗi trích xuất tóm tắt.")
    else:
        st.info("Hội thoại chưa đủ độ dài.")

if st.session_state.get("show_analytics"):
    st.session_state.pop("show_analytics")
    try:
        from step6_analytics_logger import get_logs
        import pandas as pd

        logs = get_logs(limit=50)
        if logs:
            st.dataframe(pd.DataFrame(logs))
        else:
            st.info("Chưa ghi nhận logs.")
    except:
        st.warning("Không thể đọc cấu trúc log.")

# ─────────────────────────────────────────────
# 8. XỬ LÝ DÒNG DỮ LIỆU VÀO
# ─────────────────────────────────────────────
user_query = st.chat_input("Nhập câu hỏi tại đây...")

if user_query:
    maybe_summarize_history()
    st.session_state.messages.append({"role": "user", "content": user_query, "route": None})

    with st.chat_message("user"):
        st.markdown(user_query)

    history_for_llm = build_langchain_history(st.session_state.messages[:-1])

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu dữ liệu hệ thống..."):
            route = classify_query(user_query)

            if route == "RAG":
                vec_ctx, graph_ctx, category = get_context(user_query)
            else:
                vec_ctx, graph_ctx, category = "", "Không có dữ liệu đồ thị.", "khac"

            ROUTE_LABEL = {"RAG": "📚 Dữ liệu nội bộ", "CHAT": "💬 Giao tiếp", "GENERAL": "🌐 Kiến thức chung"}
            if route != "CHAT":
                st.caption(f"*Nguồn: {ROUTE_LABEL.get(route, 'Khác')}*")

            response_stream = master_chain.stream({
                "route": route,
                "vector_context": vec_ctx,
                "graph_context": graph_ctx,
                "chat_history": history_for_llm,
                "question": user_query,
            })

            response = st.write_stream(response_stream)

            if route == "RAG":
                with st.expander("🛠️ Debug Hệ thống — Context Trích xuất"):
                    st.markdown(f"**Từ khóa:** `{extract_keywords(user_query)}` | **Mục:** `{category}`")
                    st.markdown("**🕸️ Neo4j Graph Data:**")
                    st.success(graph_ctx if graph_ctx else "Trống")
                    st.markdown("**📄 ChromaDB Vector Data:**")
                    st.info(vec_ctx[:600] + "..." if len(vec_ctx) > 600 else vec_ctx or "Trống")

    st.session_state.messages.append({"role": "assistant", "content": response, "route": route})

    try:
        log_cat = category if route == "RAG" else "khac"
        log_interaction(route, log_cat, user_query, response)
    except:
        pass