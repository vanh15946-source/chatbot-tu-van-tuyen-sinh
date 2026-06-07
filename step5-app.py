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
        temperature=0.3,
        model_name="llama-3.3-70b-versatile",
        max_tokens=4096,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    return vector_store, graph, llm

vector_store, graph, llm = init_system()

# ─────────────────────────────────────────────
# 2. METADATA FILTER
# ─────────────────────────────────────────────
NHA_TRO_KEYWORDS = [
    "trọ", "thuê", "phòng", "p", "phg", "ở ghép", "xóm", "ký túc", "ktx",
    "điều hòa", "nóng lạnh", "wifi", "giữ xe", "tiện ích", "khép kín",
    "điện nước", "đặt cọc", "nhà trọ", "phòng trọ", "thuê phòng", "ở trọ",
    "chỗ ở", "giá thuê", "tiện ích phòng", "phòng cho thuê", "tìm trọ",
    "tìm phòng", "phòng ở", "gần trường", "giá rẻ", "phòng đơn", "phòng đôi",
    "z115", "phú thái", "tân thịnh", "quyết thắng", "sơn tiến", "nước hai",
    "cổng trường", "phường phan đình phùng", "phan đình phùng",
    # Từ khóa cho file "lưu ý thuê trọ" và tư vấn chỗ ở
    "lưu ý thuê", "kinh nghiệm thuê", "lưu ý khi thuê",
    "hợp đồng thuê", "tiền cọc", "chủ trọ", "xem phòng",
]

# Intent tổ hợp môn — dùng để chọn đúng Cypher query
TO_HOP_KEYWORDS = [
    "tổ hợp", "khối thi", "môn thi", "xét tuyển bằng môn",
    "a00", "a01", "a02", "a03", "a04", "a10",
    "b00", "b08", "c00", "c01", "d01", "d07",
]

def get_category_filter(query: str) -> tuple[dict, str]:
    q = query.lower()
    if any(kw in q for kw in NHA_TRO_KEYWORDS):
        # FIX: đồng bộ với step1 — category = "nha_tro"
        return {"category": "nha_tro"}, "nha_tro"
    return {"category": "tuyen_sinh"}, "tuyen_sinh"

def is_to_hop_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in TO_HOP_KEYWORDS)

# ─────────────────────────────────────────────
# 3. PHÂN LOẠI / ĐỊNH TUYẾN CÂU HỎI (ROUTER)
# ─────────────────────────────────────────────
ROUTE_TEMPLATE = """Bạn là bộ phân loại câu hỏi cho chatbot tuyển sinh đại học TNUS.
Phân loại vào MỘT trong ba nhóm:

- RAG     : tuyển sinh, ngành học, điểm chuẩn, học phí, học bổng, xét tuyển,
            hồ sơ nhập học, chính sách ưu đãi, nhà trọ, phòng trọ, thông tin trường
- CHAT    : chào hỏi, cảm ơn, trò chuyện thông thường, hỏi về bản thân AI
- GENERAL : chủ đề khác không liên quan đến tuyển sinh hoặc TNUS

Ngữ cảnh: {history_summary}
Câu hỏi: {question}

Chỉ trả về đúng một từ: RAG, CHAT, hoặc GENERAL."""

route_prompt = ChatPromptTemplate.from_template(ROUTE_TEMPLATE)
route_chain  = route_prompt | llm | StrOutputParser()

def classify_query(question: str, history_summary: str = "") -> str:
    try:
        result = route_chain.invoke({
            "question": question,
            "history_summary": history_summary,
        }).strip().upper()
        first_word = result.split()[0] if result else "RAG"
        return first_word if first_word in {"RAG", "CHAT", "GENERAL"} else "RAG"
    except Exception:
        return "RAG"

# ─────────────────────────────────────────────
# 4. TÓM TẮT LỊCH SỬ
# ─────────────────────────────────────────────
SUMMARIZE_TEMPLATE = """Tóm tắt cuộc hội thoại dưới đây thành 3-5 câu ngắn gọn,
giữ lại thông tin quan trọng về nhu cầu tuyển sinh và câu hỏi của người dùng.

Hội thoại:
{conversation}

Tóm tắt:"""

summarize_prompt = ChatPromptTemplate.from_template(SUMMARIZE_TEMPLATE)
summarize_chain  = summarize_prompt | llm | StrOutputParser()

MAX_RAW_HISTORY    = 10
SUMMARIZE_THRESHOLD = 20

def build_langchain_history(messages: list) -> list:
    history = []
    if st.session_state.get("history_summary"):
        history.append(AIMessage(
            content=f"[Tóm tắt trước: {st.session_state['history_summary']}]"
        ))
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
    conversation_text = ""
    if st.session_state.get("history_summary"):
        conversation_text += f"[Tóm tắt trước]: {st.session_state['history_summary']}\n\n"
    conversation_text += "\n".join(
        f"{'Người dùng' if m['role'] == 'user' else 'AI'}: {m['content']}"
        for m in old_msgs
    )
    try:
        st.session_state["history_summary"] = summarize_chain.invoke(
            {"conversation": conversation_text}
        )
        st.session_state.messages = msgs[-MAX_RAW_HISTORY:]
    except Exception:
        pass

# ─────────────────────────────────────────────
# 5. TRUY XUẤT DỮ LIỆU (VECTOR + GRAPH)
# Multi-query đã bỏ → 1 lần search duy nhất, nhanh hơn 3-4x
# ─────────────────────────────────────────────
def vector_search(query: str, cat_filter: dict, k: int = 5) -> list[str]:
    """Vector search đơn giản — 1 query, không song song."""
    try:
        docs = vector_store.similarity_search(query, k=k, filter=cat_filter)
        return [doc.page_content for doc in docs]
    except Exception:
        return []


def retrieve_from_graph(query: str, keywords: list[str], category: str) -> list[str]:
    """
    Graph search với intent detection:
    - Nếu hỏi về tổ hợp môn → query đúng path NgànhHọc→TổHợpMôn
    - Còn lại → query tổng quát
    """
    if category == "nha_tro" or not keywords:
        return []

    graph_context = []

    # ── Intent: hỏi về tổ hợp môn ───────────────────────────────
    if is_to_hop_query(query):
        for kw in keywords:
            cypher = """
            MATCH (n:NgànhHọc)-[:DÙNG_TỔ_HỢP]->(t:TổHợpMôn)
            WHERE toLower(n.name)     CONTAINS toLower($keyword)
               OR toLower(n.ma_nganh) CONTAINS toLower($keyword)
            RETURN n.name   AS nganh,
                   t.ma      AS ma_to_hop,
                   t.mon_hoc AS mon_hoc
            ORDER BY t.ma
            """
            try:
                results = graph.query(cypher, params={"keyword": kw})
                if results:
                    nganh_name = results[0]["nganh"]
                    graph_context.append(f"Ngành {nganh_name} có các tổ hợp xét tuyển:")
                    for r in results:
                        graph_context.append(f"  - {r['ma_to_hop']}: {r['mon_hoc']}")
            except Exception:
                pass
        return graph_context

    # ── Intent: câu hỏi tổng quát (học phí, điểm chuẩn, v.v.) ──
    for kw in keywords:
        cypher = """
        MATCH (n)-[r]->(m)
        WHERE toLower(n.id)   CONTAINS toLower($keyword)
           OR toLower(m.id)   CONTAINS toLower($keyword)
           OR toLower(n.name) CONTAINS toLower($keyword)
           OR toLower(m.name) CONTAINS toLower($keyword)
        RETURN n.id AS source, type(r) AS relationship, m.id AS target
        LIMIT 7
        """
        try:
            for res in graph.query(cypher, params={"keyword": kw}):
                graph_context.append(
                    f"{res['source']} --[{res['relationship']}]--> {res['target']}"
                )
        except Exception:
            pass

    return list(set(graph_context))


def extract_keywords(query: str) -> list[str]:
    """Trích xuất 2-3 từ khóa từ câu hỏi để search graph."""
    prompt = ChatPromptTemplate.from_template(
        "Trích xuất 2-3 từ khóa danh từ quan trọng nhất từ câu hỏi tuyển sinh sau. "
        "Chỉ trả về từ khóa cách nhau bằng dấu phẩy. Không giải thích.\n"
        "Câu hỏi: {query}"
    )
    try:
        response = (prompt | llm | StrOutputParser()).invoke({"query": query})
        return [kw.strip() for kw in response.split(",") if kw.strip()]
    except Exception:
        return []


def get_context(query: str) -> tuple[str, str, str]:
    """
    Lấy context từ Vector DB + Graph DB.
    Trả về: vector_context, graph_context, category
    Bỏ multi-query → nhanh hơn, ít tốn token hơn.
    """
    cat_filter, category = get_category_filter(query)

    # ── Vector search — 1 lần duy nhất ──────────────────────────
    k = 6 if category == "nha_tro" else 4
    docs = vector_search(query, cat_filter, k=k)
    vector_context = "\n\n".join(docs)

    # ── Graph search — chỉ tuyển sinh ───────────────────────────
    graph_context = "Không có dữ liệu đồ thị."
    if category == "tuyen_sinh":
        keywords  = extract_keywords(query)
        graph_data = retrieve_from_graph(query, keywords, category)
        if graph_data:
            graph_context = "\n".join(graph_data)

    return vector_context, graph_context, category

# ─────────────────────────────────────────────
# 6. PROMPTS & CHAINS
# ─────────────────────────────────────────────
RAG_SYSTEM = """Bạn là trợ lý tuyển sinh thông minh của Trường Đại học Khoa học - Đại học Thái Nguyên (TNUS).

QUY TẮC:
1. Trả lời ĐẦY ĐỦ, CHI TIẾT toàn bộ nội dung tìm thấy. KHÔNG cắt bớt hay gom chung chung.
2. Chỉ dựa vào dữ liệu RAG bên dưới. KHÔNG tự suy diễn hay bịa thêm thông tin.
3. Nhà trọ: liệt kê từng nhà trọ với đầy đủ Tên, Địa chỉ, SĐT, Giá thuê, Tiện ích, Ghi chú.
   Cuối phần nhà trọ luôn thêm: "⚠️ Giá cả có thể đã thay đổi, hãy gọi trực tiếp để xác nhận."
4. Tổ hợp môn: liệt kê CHÍNH XÁC, ĐẦY ĐỦ 100% các mã tổ hợp có trong ngữ cảnh. KHÔNG bịa thêm.
5. Cuối mỗi câu trả lời gợi ý 1-2 câu hỏi tiếp theo: "💡 *Gợi ý hỏi thêm: ...*"
6. Dùng emoji phù hợp cho thân thiện.

--- DỮ LIỆU VECTOR ---
{vector_context}

--- DỮ LIỆU GRAPH ---
{graph_context}
"""

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])
rag_chain = rag_prompt | llm | StrOutputParser()

CHAT_SYSTEM = """Bạn là trợ lý tuyển sinh thân thiện của TNUS (Trường Đại học Khoa học - ĐH Thái Nguyên).
Trò chuyện tự nhiên, ấm áp. Khi phù hợp, giới thiệu nhẹ nhàng về ngành học nổi bật,
học bổng, hoặc môi trường học tập tại TNUS."""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", CHAT_SYSTEM),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])
chat_chain = chat_prompt | llm | StrOutputParser()

GENERAL_SYSTEM = """Bạn là trợ lý tuyển sinh của TNUS. Trả lời ngắn gọn dựa trên kiến thức chung,
sau đó khéo léo hướng về tuyển sinh TNUS."""

general_prompt = ChatPromptTemplate.from_messages([
    ("system", GENERAL_SYSTEM),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])
general_chain = general_prompt | llm | StrOutputParser()

# ─────────────────────────────────────────────
# 7. GIAO DIỆN STREAMLIT
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Tư vấn tuyển sinh TNUS",
    page_icon="tnus_logo.png",
    layout="centered",
)

with st.sidebar:
    st.image("tnus_logo.png", width=120)
    st.markdown("### Tư vấn tuyển sinh TNUS 2026")
    st.caption("Trường Đại học Khoa học - ĐH Thái Nguyên")
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

col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("tnus_logo.png", width=70)
with col_title:
    st.title("Tư vấn tuyển sinh TNUS")
st.markdown("*Hỏi bất cứ điều gì về tuyển sinh, ngành học, học phí, học bổng hoặc nhà trọ!*")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history_summary" not in st.session_state:
    st.session_state["history_summary"] = ""

if len(st.session_state.messages) == 0:
    welcome_msg = (
        "Xin chào! 👋 Tôi là **trợ lý tuyển sinh TNUS** — sẵn sàng hỗ trợ bạn!\n\n"
        "Tôi có thể giúp bạn:\n"
        "- 📚 **Thông tin ngành học** — chỉ tiêu, điểm chuẩn, tổ hợp môn\n"
        "- 💰 **Học phí & học bổng** — chính sách ưu đãi năm 2026\n"
        "- 📝 **Hồ sơ xét tuyển** — thủ tục, thời hạn nộp hồ sơ\n"
        "- 🏠 **Nhà trọ** — danh sách phòng trọ gần trường, giá cả, tiện ích\n\n"
        "Bạn muốn hỏi về điều gì? 😊"
    )
    st.session_state.messages.append({
        "role": "assistant", "content": welcome_msg, "route": "CHAT",
    })

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("route") not in (None, "CHAT"):
            ROUTE_BADGE = {"RAG": "📚 Dữ liệu cục bộ", "GENERAL": "🌐 Kiến thức nền"}
            st.caption(f"*Nguồn: {ROUTE_BADGE.get(message['route'], '')}*")

if st.session_state.get("show_summary"):
    st.session_state.pop("show_summary")
    if len(st.session_state.messages) > 2:
        with st.spinner("Đang tóm tắt..."):
            conversation_text = "\n".join(
                f"{'Người dùng' if m['role'] == 'user' else 'AI'}: {m['content']}"
                for m in st.session_state.messages
            )
            try:
                summary = summarize_chain.invoke({"conversation": conversation_text})
                st.info(f"**📋 Tóm tắt hội thoại:**\n\n{summary}")
            except Exception as e:
                st.warning(f"Không thể tóm tắt: {e}")
    else:
        st.info("Hội thoại chưa đủ để tóm tắt.")

# ─────────────────────────────────────────────
# 8. XỬ LÝ ĐẦU VÀO
# ─────────────────────────────────────────────
if "pending_input" in st.session_state:
    user_query = st.session_state.pop("pending_input")
else:
    user_query = st.chat_input("Hỏi về ngành học, điểm chuẩn, học phí, nhà trọ...")

if user_query:
    maybe_summarize_history()

    st.session_state.messages.append({
        "role": "user", "content": user_query, "route": None,
    })
    with st.chat_message("user"):
        st.markdown(user_query)

    history_for_llm = build_langchain_history(st.session_state.messages[:-1])
    history_summary_short = "\n".join(
        f"{'User' if m['role'] == 'user' else 'AI'}: {m['content'][:120]}"
        for m in st.session_state.messages[-6:-1]
    )

    with st.chat_message("assistant"):
        with st.spinner("Đang xử lý..."):
            route = classify_query(user_query, history_summary_short)
            vec_ctx, graph_ctx, category = "", "Không có dữ liệu đồ thị.", "tuyen_sinh"

            if route == "RAG":
                vec_ctx, graph_ctx, category = get_context(user_query)

        ROUTE_LABEL = {
            "RAG":     "📚 Dữ liệu cục bộ",
            "CHAT":    "💬 Hội thoại thông thường",
            "GENERAL": "🌐 Kiến thức nền",
        }
        if route != "CHAT":
            st.caption(f"*Nguồn: {ROUTE_LABEL[route]}*")

        if route == "RAG":
            response_stream = rag_chain.stream({
                "vector_context": vec_ctx,
                "graph_context":  graph_ctx,
                "chat_history":   history_for_llm,
                "question":       user_query,
            })
        elif route == "CHAT":
            response_stream = chat_chain.stream({
                "chat_history": history_for_llm,
                "question":     user_query,
            })
        else:
            response_stream = general_chain.stream({
                "chat_history": history_for_llm,
                "question":     user_query,
            })

        response = st.write_stream(response_stream)

        # ── Debug panel ──────────────────────────────────────────
        if route == "RAG":
            with st.expander("🛠️ Debug — Dữ liệu RAG đã trích xuất"):
                st.markdown(f"**Category:** `{category}`")
                st.markdown("**📄 Vector DB:**")
                st.info(vec_ctx[:600] + "..." if len(vec_ctx) > 600 else vec_ctx or "Không có.")
                st.markdown("**🕸️ Graph DB:**")
                st.success(graph_ctx)

    st.session_state.messages.append({
        "role": "assistant", "content": response, "route": route,
    })

    log_category = category if route == "RAG" else "khac"
    log_interaction(route, log_category, user_query, response)