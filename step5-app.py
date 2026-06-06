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
    "điều hòa", "nóng lạnh", "wifi", "giữ xe", "tiện ích", "khép kín", "điện nước", "đặt cọc",
    "nhà trọ", "phòng trọ", "thuê phòng", "ở trọ", "chỗ ở", "giá thuê",
    "tiện ích phòng", "phòng cho thuê", "tìm trọ", "tìm phòng", "phòng ở",
    "gần trường", "giá rẻ", "phòng đơn", "phòng đôi",
    "z115", "phú thái", "tân thịnh", "quyết thắng", "sơn tiến", "nước hai", "cổng trường",
    "phường phan đình phùng", "phan đình phùng"
]

def get_category_filter(query: str) -> tuple[dict, str]:
    q = query.lower()
    if any(kw in q for kw in NHA_TRO_KEYWORDS):
        return {"category": {"$in": ["nha_tro_intro", "nha_tro_detail"]}}, "nha_tro"
    return {"category": "tuyen_sinh"}, "tuyen_sinh"


# ─────────────────────────────────────────────
# 3. ✨ MULTI-QUERY RETRIEVAL (TỐI ƯU HÓA SONG SONG & LỌC TRÙNG)
# ─────────────────────────────────────────────
MULTI_QUERY_TEMPLATE = """Bạn là trợ lý AI hỗ trợ tuyển sinh Trường Đại học Khoa học - ĐH Thái Nguyên (TNUS).
Nhiệm vụ của bạn là tạo ra {num_queries} cách hỏi KHÁC NHAU từ câu hỏi gốc dưới đây.
Mục tiêu: bao phủ nhiều góc độ hơn khi tìm kiếm trong cơ sở dữ liệu tuyển sinh.

Yêu cầu:
- Mỗi biến thể phải diễn đạt khác về từ ngữ nhưng giữ nguyên ý nghĩa gốc
- Dùng từ đồng nghĩa, cách hỏi khác, viết tắt, hoặc góc nhìn khác
- KHÔNG thêm thông tin mới không có trong câu hỏi gốc
- Chỉ trả về danh sách {num_queries} câu, mỗi câu trên một dòng, KHÔNG đánh số, KHÔNG giải thích

Câu hỏi gốc: {question}

{num_queries} biến thể câu hỏi:"""

multi_query_prompt = ChatPromptTemplate.from_template(MULTI_QUERY_TEMPLATE)
multi_query_chain = multi_query_prompt | llm | StrOutputParser()


def generate_multi_queries(question: str, num_queries: int = 3) -> list[str]:
    try:
        result = multi_query_chain.invoke({
            "question": question,
            "num_queries": num_queries,
        })
        variants = [q.strip() for q in result.strip().split("\n") if q.strip()]
        all_queries = list(dict.fromkeys([question] + variants[:num_queries]))
        return all_queries
    except Exception:
        return [question]


def multi_query_vector_search(
        queries: list[str],
        cat_filter: dict,
        top_k_per_query: int = 4,
) -> list[str]:
    """
    Tìm kiếm tối ưu tốc độ: Thực hiện song song (Concurrently) tất cả câu hỏi trên Vector DB
    và loại trùng chính xác bằng Hash bảng băm, tránh mất mát dữ liệu danh mục nhà trọ.
    """
    import asyncio

    # Hàm con bất đồng bộ xử lý truy vấn đơn lẻ
    async def _async_search(query_str):
        try:
            # Chạy hàm đồng bộ của LangChain trong một thread riêng để tránh block event loop
            return await asyncio.to_thread(
                vector_store.similarity_search, query_str, k=top_k_per_query, filter=cat_filter
            )
        except Exception:
            return []

    # Tạo một event loop mới hoặc tận dụng loop sẵn có của Streamlit để chạy song song
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Gom toàn bộ các tác vụ tìm kiếm lại để kích hoạt cùng một thời điểm
    tasks = asyncio.gather(*[_async_search(q) for q in queries])
    results_list = loop.run_until_complete(tasks)

    seen_hashes = set()
    merged_docs = []

    # Giải nén kết quả thu được từ các thread song song
    for docs in results_list:
        for doc in docs:
            # TỐI ƯU LÕI: Dùng hash toàn bộ text để lọc trùng tuyệt đối,
            # không cắt ngắn ký tự đầu, giúp bảo toàn các nhà trọ có format giống nhau.
            content_hash = hash(doc.page_content.strip())
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                merged_docs.append(doc.page_content)

    return merged_docs


# ─────────────────────────────────────────────
# 4. PHÂN LOẠI / ĐỊNH TUYẾN CÂU HỎI (ROUTER)
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
# 5. TÓM TẮT LỊCH SỬ KHI HỘI THOẠI QUÁ DÀI
# ─────────────────────────────────────────────
SUMMARIZE_TEMPLATE = """Tóm tắt cuộc hội thoại dưới đây thành 3-5 câu ngắn gọn,
giữ lại các thông tin quan trọng về nhu cầu tuyển sinh và câu hỏi của người dùng.

Hội thoại:
{conversation}

Tóm tắt:"""

summarize_prompt = ChatPromptTemplate.from_template(SUMMARIZE_TEMPLATE)
summarize_chain = summarize_prompt | llm | StrOutputParser()

MAX_RAW_HISTORY = 10
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
# 6. TRUY XUẤT DỮ LIỆU (MULTI-QUERY VECTOR + GRAPH)
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
    """
    Lấy context từ Vector DB (Multi-Query) + Graph DB.
    Trả về: vector_context (str), graph_context (str), category (str), queries_used (list[str])
    """
    cat_filter, category = get_category_filter(query)

    # ── MULTI-QUERY: tạo biến thể câu hỏi ──────────────────────────────
    # Nhà trọ: 2 biến thể (dữ liệu thực tế, ít biến thể hơn)
    # Tuyển sinh: 3 biến thể để phủ rộng
    num_variants = 2 if category == "nha_tro" else 3
    all_queries = generate_multi_queries(query, num_queries=num_variants)

    # top_k mỗi query: nhà trọ cần ít hơn vì đã có nhiều query
    top_k_per_query = 5 if category == "nha_tro" else 3

    # ── VECTOR SEARCH song song ──────────────────────────────────────────
    merged_docs = multi_query_vector_search(
        queries=all_queries,
        cat_filter=cat_filter,
        top_k_per_query=top_k_per_query,
    )
    vector_context = "\n\n".join(merged_docs)

    # ── GRAPH SEARCH (chỉ tuyển sinh) ────────────────────────────────────
    keywords = extract_keywords(query) if category == "tuyen_sinh" else []
    graph_data = retrieve_from_graph(keywords, category)
    graph_context = "\n".join(graph_data) if graph_data else "Không có dữ liệu đồ thị."

    return vector_context, graph_context, category, all_queries


# ─────────────────────────────────────────────
# 7. PROMPTS & CHAINS
# ─────────────────────────────────────────────

# --- RAG chain ---
RAG_SYSTEM = """Bạn là trợ lý tuyển sinh thông minh của Trường Đại học Khoa học - Đại học Thái Nguyên (TNUS).

QUY TẮC QUAN TRỌNG VỀ ĐỘ DÀI VÀ CHI TIẾT:
1. Hãy trả lời ĐẦY ĐỦ, CHI TIẾT, TOÀN VẸN toàn bộ nội dung tìm thấy. KHÔNG ĐƯỢC phép viết tắt, tự ý cắt bớt câu chữ hay gom cụm thông tin chung chung nếu ngữ cảnh trả về nhiều dữ liệu.
2. Thông tin RAG: Chỉ dựa vào dữ liệu RAG bên dưới. Nếu dữ liệu chưa đủ, hãy nói rõ và không tự bịa thêm thông tin.Tuyệt đối KHÔNG sử dụng kiến thức nền của bạn để tự suy diễn.
3. Xử lý dữ liệu nhiễu nhà trọ:
   - Nếu giá tiền trọ có vẻ vô lý (như 100 triệu, 10 triệu), hãy nhắc khéo đây có thể là lỗi nhập liệu.
   - Luôn kèm theo lưu ý ở cuối phần nhà trọ: "⚠️ Lưu ý: Giá cả có thể đã thay đổi hoặc có sai sót lúc thống kê. Bạn hãy gọi điện trực tiếp cho chủ trọ để xác nhận nhé!".
4. TRÌNH BÀY NHÀ TRỌ:
   - BẮT BUỘC liệt kê cụ thể từng nhà trọ một. Xuất đầy đủ các thông tin: Tên nhà trọ, Địa chỉ, Số điện thoại, Giá thuê, Tiện ích, Hiện trạng/Ghi chú (nếu có).
5. Cuối mỗi câu trả lời gợi ý 1-2 câu hỏi tiếp theo định dạng: "💡 *Gợi ý hỏi thêm: ...*"
6. Dùng emoji phù hợp để tạo cảm giác thân thiện.
7.KHÔNG ĐƯỢC BỊA THÊM TỔ HỢP MÔN: Nếu hỏi về tổ hợp xét tuyển/khối thi, BẮT BUỘC liệt kê chính xác và ĐẦY ĐỦ 100% tất cả các mã tổ hợp có trong ngữ cảnh.
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
# 8. GIAO DIỆN STREAMLIT
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Tư vấn tuyển sinh TNUS",
    page_icon="tnus_logo.png",
    layout="centered",
)

# ---- Sidebar ----
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

# ---- Main UI ----
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
        "role": "assistant",
        "content": welcome_msg,
        "route": "CHAT",
    })

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("route") not in (None, "CHAT"):
            ROUTE_BADGE = {
                "RAG": "📚 Dữ liệu cục bộ",
                "GENERAL": "🌐 Kiến thức nền",
            }
            st.caption(f"*Nguồn: {ROUTE_BADGE.get(message['route'], '')}*")

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
# 9. XỬ LÝ ĐẦU VÀO GIAO DIỆN
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
            route = classify_query(user_query, history_summary_short)
            vec_ctx, graph_ctx, category, queries_used = "", "", "tuyen_sinh", [user_query]

            if route == "RAG":
                vec_ctx, graph_ctx, category, queries_used = get_context(user_query)

        ROUTE_LABEL = {
            "RAG": "📚 Dữ liệu cục bộ",
            "CHAT": "💬 Hội thoại thông thường",
            "GENERAL": "🌐 Kiến thức nền",
        }
        if route != "CHAT":
            st.caption(f"*Nguồn: {ROUTE_LABEL[route]}*")

        if route == "RAG":
            response_stream = rag_chain.stream({
                "vector_context": vec_ctx,
                "graph_context": graph_ctx,
                "chat_history": history_for_llm,
                "question": user_query,
            })
        elif route == "CHAT":
            response_stream = chat_chain.stream({
                "chat_history": history_for_llm,
                "question": user_query,
            })
        else:
            response_stream = general_chain.stream({
                "chat_history": history_for_llm,
                "question": user_query,
            })

        response = st.write_stream(response_stream)

        # ── Debug panel (chỉ RAG) ────────────────────────────────────────
        if route == "RAG":
            with st.expander("🛠️ Debug — Dữ liệu RAG đã trích xuất"):
                st.markdown(f"**Category:** `{category}`")

                # ✨ Hiển thị các câu query đã sinh ra
                st.markdown("**🔍 Multi-Query — Các biến thể câu hỏi đã dùng:**")
                for i, q in enumerate(queries_used):
                    label = "*(câu gốc)*" if i == 0 else f"*(biến thể {i})*"
                    st.markdown(f"- `{q}` {label}")

                st.markdown("**📄 Vector DB (sau khi hợp nhất):**")
                st.info(vec_ctx[:600] + "..." if len(vec_ctx) > 600 else vec_ctx or "Không có.")
                st.markdown("**🕸️ Graph DB:**")
                st.success(graph_ctx or "Không có dữ liệu đồ thị.")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "route": route,
    })

    log_category = category if route == "RAG" else "khac"
    log_interaction(route, log_category, user_query, response)