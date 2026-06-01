import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

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
        collection_name="tnus_tuyen_sinh"       # ← sửa tên collection
    )
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
    )
    llm = ChatGroq(
        temperature=0.3,
        model_name="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
    )
    return vector_store, graph, llm


vector_store, graph, llm = init_system()

# ─────────────────────────────────────────────
# 2. METADATA FILTER — detect intent
# ─────────────────────────────────────────────
NHA_TRO_KEYWORDS = [
    # Từ đơn — bắt buộc phải có
    "trọ", "thuê", "phòng",
    # Tiện ích hay hỏi
    "điều hòa", "nóng lạnh", "wifi", "giữ xe", "tiện ích",
    # Cụm từ
    "nhà trọ", "phòng trọ", "thuê phòng", "ở trọ",
    "chỗ ở", "giá thuê", "tiện ích phòng", "phòng cho thuê",
    "tìm trọ", "tìm phòng", "phòng ở", "ký túc",
    "gần trường", "giá rẻ", "phòng đơn", "phòng đôi",
]

def get_category_filter(query: str) -> dict:
    q = query.lower()
    if any(kw in q for kw in NHA_TRO_KEYWORDS):
        return {"category": "nha_tro"}
    return {"category": "tuyen_sinh"}

# ─────────────────────────────────────────────
# 3. PHÂN LOẠI / ĐỊNH TUYẾN CÂU HỎI (ROUTER)
# ─────────────────────────────────────────────
ROUTE_TEMPLATE = """Bạn là bộ phân loại câu hỏi cho chatbot tuyển sinh đại học TNUS (Trường Đại học Khoa học - ĐH Thái Nguyên).
Dựa vào câu hỏi và ngữ cảnh hội thoại, hãy phân loại vào MỘT trong ba nhóm:

- RAG     : Câu hỏi liên quan đến tuyển sinh, ngành học, điểm chuẩn, học phí, học bổng, xét tuyển,
            hồ sơ nhập học, chính sách ưu đãi, nhà trọ, phòng trọ, sinh hoạt sinh viên, thông tin trường
- CHAT    : Chào hỏi, cảm ơn, trò chuyện thông thường, hỏi về bản thân AI ("bạn là ai", "hello"...)
- GENERAL : Câu hỏi về chủ đề khác KHÔNG liên quan đến tuyển sinh hoặc TNUS

Ngữ cảnh hội thoại gần đây:
{history_summary}

Câu hỏi người dùng: {question}

Chỉ trả về đúng một từ: RAG, CHAT, hoặc GENERAL."""

route_prompt = ChatPromptTemplate.from_template(ROUTE_TEMPLATE)
route_chain = route_prompt | llm | StrOutputParser()


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
# 4. TÓM TẮT LỊCH SỬ KHI HỘI THOẠI QUÁ DÀI
# ─────────────────────────────────────────────
SUMMARIZE_TEMPLATE = """Tóm tắt cuộc hội thoại dưới đây thành 3-5 câu ngắn gọn,
giữ lại các thông tin quan trọng về nhu cầu tuyển sinh và câu hỏi của người dùng.

Hội thoại:
{conversation}

Tóm tắt:"""

summarize_prompt = ChatPromptTemplate.from_template(SUMMARIZE_TEMPLATE)
summarize_chain = summarize_prompt | llm | StrOutputParser()

MAX_RAW_HISTORY   = 10
SUMMARIZE_THRESHOLD = 20


def build_langchain_history(messages: list) -> list:
    history = []
    if st.session_state.get("history_summary"):
        history.append(AIMessage(
            content=f"[Tóm tắt cuộc trò chuyện trước: {st.session_state['history_summary']}]"
        ))
    recent = messages[-MAX_RAW_HISTORY:]
    for msg in recent:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))
    return history


def maybe_summarize_history():
    msgs = st.session_state.messages
    if len(msgs) > SUMMARIZE_THRESHOLD:
        old_msgs = msgs[:-MAX_RAW_HISTORY]

        # Thêm tóm tắt cũ vào (nếu có)
        conversation_text = ""
        if st.session_state.get("history_summary"):
            conversation_text += f"[Tóm tắt trước đó]: {st.session_state['history_summary']}\n\n"

        conversation_text += "\n".join(
            f"{'Người dùng' if m['role'] == 'user' else 'AI'}: {m['content']}"
            for m in old_msgs
        )
        try:
            new_summary = summarize_chain.invoke({"conversation": conversation_text})
            st.session_state["history_summary"] = new_summary
            st.session_state.messages = msgs[-MAX_RAW_HISTORY:]
        except Exception:
            pass

# ─────────────────────────────────────────────
# 5. TRUY XUẤT DỮ LIỆU (VECTOR + GRAPH)
# ─────────────────────────────────────────────
def extract_keywords(query: str) -> list[str]:
    prompt = ChatPromptTemplate.from_template(
        "Bạn là trợ lý tuyển sinh đại học. "
        "Trích xuất 2 đến 3 từ khóa danh từ quan trọng nhất từ câu hỏi sau "
        "để tìm kiếm trong cơ sở dữ liệu tuyển sinh (ngành học, khoa, điểm chuẩn, học phí, học bổng...). "
        "Chỉ trả về từ khóa cách nhau bằng dấu phẩy. Không giải thích.\n"
        "Câu hỏi: {query}"
    )
    chain = prompt | llm | StrOutputParser()
    try:
        response = chain.invoke({"query": query})
        return [kw.strip() for kw in response.split(",") if kw.strip()]
    except Exception:
        return []


def retrieve_from_graph(keywords: list[str], category: str) -> list[str]:
    """
    Truy vấn Neo4j theo từ khóa.
    Bỏ qua nếu category là nha_tro (không có KG nhà trọ).
    """
    if not keywords or category == "nha_tro":
        return []

    graph_context = []
    for kw in keywords:
        cypher = """
        MATCH (n)-[r]->(m)
        WHERE toLower(n.id) CONTAINS toLower($keyword)
           OR toLower(m.id) CONTAINS toLower($keyword)
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


def get_context(query: str):
    """Lấy context từ Vector DB (có filter) + Graph DB."""
    cat_filter = get_category_filter(query)
    category   = cat_filter["category"]

    # Vector search với metadata filter
    vector_docs = vector_store.similarity_search(query, k=3, filter=cat_filter)
    vector_context = "\n\n".join([doc.page_content for doc in vector_docs])

    # Graph search (chỉ với tuyển sinh)
    keywords = extract_keywords(query) if category == "tuyen_sinh" else []
    graph_data = retrieve_from_graph(keywords, category)
    graph_context = "\n".join(graph_data) if graph_data else "Không có dữ liệu đồ thị."

    return vector_context, graph_context, category


# ─────────────────────────────────────────────
# 6. PROMPTS & CHAINS
# ─────────────────────────────────────────────

# --- RAG chain (tuyển sinh + nhà trọ) ---
RAG_SYSTEM = """Bạn là trợ lý tuyển sinh thông minh của Trường Đại học Khoa học - Đại học Thái Nguyên (TNUS).

QUY TẮC:
1. Ưu tiên thông tin từ dữ liệu RAG bên dưới. Nếu dữ liệu RAG chưa đủ, hãy nói rõ và gợi ý các chủ đề liên quan người dùng có thể hỏi tiếp.
2. Trả lời liền mạch, nhớ ngữ cảnh hội thoại để không hỏi lại thông tin đã biết.
3. Với câu hỏi nhà trọ: liệt kê rõ ràng tên, địa chỉ, giá, tiện ích. Không bịa thêm nhà trọ ngoài dữ liệu.
4. Cuối mỗi câu trả lời gợi ý 1-2 câu hỏi tiếp theo: "💡 *Gợi ý hỏi thêm: ...*"
5. Dùng emoji phù hợp để tạo cảm giác thân thiện.

--- DỮ LIỆU VECTOR (chi tiết) ---
{vector_context}

--- DỮ LIỆU GRAPH (mạng lưới quan hệ) ---
{graph_context}
"""

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])
rag_chain = rag_prompt | llm | StrOutputParser()

# --- CHAT chain ---
CHAT_SYSTEM = """Bạn là trợ lý tuyển sinh thân thiện của Trường Đại học Khoa học - Đại học Thái Nguyên (TNUS).
Hãy trò chuyện tự nhiên, ấm áp. Khi phù hợp, giới thiệu nhẹ nhàng về các ngành học nổi bật,
chính sách học bổng, hoặc môi trường học tập tại TNUS để khơi gợi hứng thú cho người dùng."""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", CHAT_SYSTEM),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])
chat_chain = chat_prompt | llm | StrOutputParser()

# --- GENERAL chain ---
GENERAL_SYSTEM = """Bạn là trợ lý tuyển sinh thông minh của Trường Đại học Khoa học - Đại học Thái Nguyên (TNUS).
Hãy trả lời ngắn gọn câu hỏi của người dùng dựa trên kiến thức chung, sau đó khéo léo hướng
cuộc trò chuyện về tuyển sinh TNUS bằng cách gợi ý các ngành học hoặc thông tin hữu ích cho thí sinh."""

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
    page_icon="tnus_logo.png",   # ← dùng file logo
    layout="centered",
)

# ---- Sidebar ----
with st.sidebar:
    # Logo TNUS
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

# ---- Main UI ----
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("tnus_logo.png", width=70)
with col_title:
    st.title("Tư vấn tuyển sinh TNUS")
st.markdown("*Hỏi bất cứ điều gì về tuyển sinh, ngành học, học phí, học bổng hoặc nhà trọ!*")

# Khởi tạo state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history_summary" not in st.session_state:
    st.session_state["history_summary"] = ""

# Tin nhắn chào mừng
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
        "role": "assistant",
        "content": welcome_msg,
        "route": "CHAT",
    })

# Hiển thị lịch sử hội thoại
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("route") not in (None, "CHAT"):
            ROUTE_BADGE = {
                "RAG": "📚 Dữ liệu cục bộ",
                "GENERAL": "🌐 Kiến thức nền",
            }
            st.caption(f"*Nguồn: {ROUTE_BADGE.get(message['route'], '')}*")

# Hiển thị tóm tắt nếu được yêu cầu
if st.session_state.get("show_summary"):
    st.session_state.pop("show_summary")
    if len(st.session_state.messages) > 2:
        with st.spinner("Đang tóm tắt hội thoại..."):
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
        "role": "user",
        "content": user_query,
        "route": None,
    })
    with st.chat_message("user"):
        st.markdown(user_query)

    history_for_llm = build_langchain_history(st.session_state.messages[:-1])

    last_turns = st.session_state.messages[-6:-1]
    history_summary_short = "\n".join(
        f"{'User' if m['role'] == 'user' else 'AI'}: {m['content'][:120]}"
        for m in last_turns
    )

    with st.chat_message("assistant"):
        with st.spinner("Đang xử lý..."):

            # ── Định tuyến tự động ──
            route = classify_query(user_query, history_summary_short)

            vec_ctx, graph_ctx, category = "", "", "tuyen_sinh"

            # ── Gọi chain phù hợp ──
            if route == "RAG":
                vec_ctx, graph_ctx, category = get_context(user_query)
                response = rag_chain.invoke({
                    "vector_context": vec_ctx,
                    "graph_context":  graph_ctx,
                    "chat_history":   history_for_llm,
                    "question":       user_query,
                })
            elif route == "CHAT":
                response = chat_chain.invoke({
                    "chat_history": history_for_llm,
                    "question":     user_query,
                })
            else:  # GENERAL
                response = general_chain.invoke({
                    "chat_history": history_for_llm,
                    "question":     user_query,
                })

            # ── Hiển thị ──
            ROUTE_LABEL = {
                "RAG":     "📚 Dữ liệu cục bộ",
                "CHAT":    "💬 Hội thoại thông thường",
                "GENERAL": "🌐 Kiến thức nền",
            }
            if route != "CHAT":
                st.caption(f"*Nguồn: {ROUTE_LABEL[route]}*")

            st.markdown(response)

            # Debug expander
            if route == "RAG":
                with st.expander("🛠️ Debug — Dữ liệu RAG đã trích xuất"):
                    st.markdown(f"**Category:** `{category}`")
                    st.markdown("**📄 Vector DB:**")
                    st.info(vec_ctx[:500] + "..." if len(vec_ctx) > 500 else vec_ctx or "Không có.")
                    st.markdown("**🕸️ Graph DB:**")
                    st.success(graph_ctx or "Không có dữ liệu đồ thị.")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "route": route,
    })