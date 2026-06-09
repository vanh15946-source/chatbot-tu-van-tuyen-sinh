import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

try:
    from step6_analytics_logger import init_logger, log_interaction, get_logs
    _HAS_LOGGER = True
except ImportError:
    _HAS_LOGGER = False
    def log_interaction(*a, **kw): pass
    def get_logs(*a, **kw): return []

load_dotenv()
if _HAS_LOGGER:
    init_logger()


# ══════════════════════════════════════════════════════════════
# 1. KHỞI TẠO HỆ THỐNG
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def init_system():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="tnus_tuyen_sinh",
    )
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
    )
    llm = ChatGroq(
        temperature=0.1,
        model_name="llama-3.3-70b-versatile",
        max_tokens=2048,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    return vector_store, graph, llm


vector_store, graph, llm = init_system()


# ══════════════════════════════════════════════════════════════
# 2. PHÂN LOẠI CATEGORY — nhà trọ vs tuyển sinh
# Nguyên tắc: tuyển sinh OVERRIDE trước, rồi mới check nhà trọ
# BUG CŨ: "p" đơn trong NHA_TRO_KEYWORDS khiến mọi câu có chữ "p" → nhà trọ
# ══════════════════════════════════════════════════════════════

# Từ khóa tuyển sinh mạnh — nếu có → luôn là tuyen_sinh, không check nhà trọ
_TUYEN_SINH_STRONG = {
    "ngành", "điểm chuẩn", "tổ hợp", "xét tuyển", "chỉ tiêu", "học phí",
    "học bổng", "hồ sơ", "nhập học", "thủ tục", "khoa", "viện", "tnus",
    "đại học", "phương thức", "chương trình", "đào tạo", "tuyển sinh",
    "a00", "a01", "a02", "b00", "b08", "c00", "d01", "d07",
}

# Từ khóa nhà trọ — chỉ dùng cụm từ rõ ràng, KHÔNG dùng ký tự đơn
_NHA_TRO_STRONG = {
    "nhà trọ", "phòng trọ", "thuê phòng", "ở trọ", "chỗ ở", "phòng cho thuê",
    "tìm trọ", "tìm phòng", "ở ghép", "xóm trọ", "nhà thuê", "phòng thuê",
    "giá thuê", "tiền thuê", "tiền cọc", "đặt cọc", "chủ trọ",
    "điều hòa", "nóng lạnh", "giữ xe", "khép kín", "điện nước",
    "ktx", "ký túc xá",
    "z115", "phú thái", "tân thịnh", "quyết thắng", "sơn tiến",
    "nước hai", "phan đình phùng",
}


def get_category(query: str, prev_category: str | None = None) -> str:
    q = query.lower()
    # Tuyển sinh override tuyệt đối
    if any(kw in q for kw in _TUYEN_SINH_STRONG):
        return "tuyen_sinh"
    # Nhà trọ chỉ khi có tín hiệu rõ ràng
    if any(kw in q for kw in _NHA_TRO_STRONG):
        return "nha_tro"
    # Giữ ngữ cảnh câu trước
    if prev_category in ("nha_tro", "tuyen_sinh"):
        return prev_category
    return "tuyen_sinh"


# ══════════════════════════════════════════════════════════════
# 3. INTENT DETECTION — trong tuyển sinh
# ══════════════════════════════════════════════════════════════

_TO_HOP_PATTERN = re.compile(r'\b([A-Da-d]\d{2})\b')
_MA_NGANH_PATTERN = re.compile(r'\b(7\d{6})\b')

# Tên 40 ngành chính xác từ file tất_cả_ngành_học.md
NGANH_LIST = [
    "trung quốc học", "hàn quốc học", "việt nam học",
    "ngôn ngữ trung quốc", "ngôn ngữ anh",
    "song ngữ anh - trung", "song ngữ anh - hàn",
    "song ngữ anh trung", "song ngữ anh hàn",
    "khoa học quản lý", "khoa học quản lí", "quản lý kinh tế", "quản lí kinh tế",
    "báo chí", "quan hệ công chúng",
    "luật", "luật kinh tế",
    "du lịch", "quản trị dịch vụ du lịch và lữ hành", "quản trị du lịch",
    "quản lý thể dục thể thao", "quản lí thể dục thể thao",
    "thư viện - thiết bị trường học", "thông tin thư viện",
    "ngôn ngữ và văn hóa các dân tộc thiểu số việt nam", "dân tộc thiểu số",
    "công tác xã hội",
    "quản lý tài nguyên và môi trường", "quản lí tài nguyên và môi trường",
    "khoa học môi trường",
    "công nghệ sinh học", "hóa dược", "chăm sóc sắc đẹp từ dược liệu",
    "công nghệ kỹ thuật hóa học", "công nghệ kĩ thuật hóa học",
    "công nghệ bán dẫn", "khoa học dữ liệu", "công nghệ thông tin",
    "toán tin", "toán học", "vật lý", "vật lí", "hóa học", "sinh học",
    "khoa học tự nhiên tích hợp stem", "khoa học tự nhiên", "stem",
    "địa lý", "địa lý học",
    "lịch sử - địa lý và kinh tế pháp luật", "lịch sử",
    "văn học",
]


def detect_intent(query: str) -> str:
    q = query.lower()
    to_hop_codes = [m.upper() for m in _TO_HOP_PATTERN.findall(query)]

    # Hỏi toàn bộ ngành
    if any(p in q for p in ["tất cả ngành", "tất cả các ngành", "danh sách ngành",
                              "có những ngành", "bao nhiêu ngành", "những ngành nào",
                              "liệt kê ngành", "các ngành", "toàn bộ ngành"]):
        return "nganh_list"

    # Hỏi ngược: A00 → ngành nào
    if to_hop_codes and any(p in q for p in ["ngành nào", "có thể thi", "được học",
                                               "xét tuyển được", "dùng được", "học được"]):
        return "nganh_theo_to_hop"

    # Hỏi tổ hợp của ngành, hoặc chỉ hỏi mã tổ hợp gồm môn gì
    if any(p in q for p in ["tổ hợp", "khối thi", "môn thi", "xét tuyển bằng",
                              "thi môn", "thi bằng"]) or to_hop_codes:
        return "to_hop_nganh"

    # Hỏi điểm chuẩn
    if any(p in q for p in ["điểm chuẩn", "điểm đầu vào", "bao nhiêu điểm", "điểm tối thiểu"]):
        return "diem_chuan"

    # Hỏi học phí
    if any(p in q for p in ["học phí", "chi phí học", "phí đào tạo", "đóng tiền", "học bổng"]):
        return "hoc_phi_hoc_bong"

    # Hỏi ngành chung
    if any(p in q for p in ["ngành", "khoa", "viện", "chương trình", "chỉ tiêu"]):
        return "nganh_general"

    return "chinh_sach"


def extract_nganh_keywords(query: str) -> list[str]:
    """Match tên ngành chính xác từ NGANH_LIST — không dùng LLM."""
    q = query.lower()
    found = [n for n in NGANH_LIST if n in q]
    # Thêm mã ngành nếu có
    found += _MA_NGANH_PATTERN.findall(query)
    return list(dict.fromkeys(found))  # dedup giữ thứ tự


# ══════════════════════════════════════════════════════════════
# 4. GRAPH QUERIES — đúng schema thực tế
#
# Schema thực tế sau khi step2 chạy:
#   Structured nodes: NgànhHọc {id, name, ma_nganh}, TổHợpMôn {id, ma, mon_hoc}, KhoaViện {id, name}
#   LLM nodes: __Entity__ + label riêng, chỉ có property {id}
#   Relationship hữu ích: DÙNG_TỔ_HỢP, THUỘC_KHOA
#   Relationship nhiễu (từ LLM transformer): MENTIONS → loại bỏ khỏi output
# ══════════════════════════════════════════════════════════════

def _safe_query(cypher: str, params: dict = {}) -> list[dict]:
    """Wrapper query Neo4j, trả về [] nếu lỗi."""
    try:
        return graph.query(cypher, params=params) or []
    except Exception:
        return []


def graph_tat_ca_nganh() -> str:
    """Lấy toàn bộ danh sách ngành — KHÔNG dùng similarity search."""
    rows = _safe_query("""
        MATCH (n:NgànhHọc)
        OPTIONAL MATCH (n)-[:THUỘC_KHOA]->(k:KhoaViện)
        RETURN n.name AS name, n.id AS id, n.ma_nganh AS ma, k.name AS khoa
        ORDER BY n.name
    """)
    if not rows:
        # Fallback: lấy từ __Entity__ nodes nếu structured chưa chạy
        rows = _safe_query("""
            MATCH (n:Ngànhhọc)
            RETURN n.id AS id, null AS name, null AS ma, null AS khoa
            ORDER BY n.id
        """)
    if not rows:
        return ""

    lines = [f"Danh sách {len(rows)} ngành đào tạo tại TNUS:\n"]
    current_khoa = None
    for r in rows:
        name  = r.get("name") or r.get("id") or "?"
        ma    = f" (mã: {r['ma']})" if r.get("ma") else ""
        khoa  = r.get("khoa") or "Chưa phân khoa"
        if khoa != current_khoa:
            lines.append(f"\n[{khoa}]")
            current_khoa = khoa
        lines.append(f"  • {name}{ma}")
    return "\n".join(lines)


def graph_to_hop_cua_nganh(keywords: list[str], raw_query: str) -> str:
    """Tổ hợp môn của ngành (hỏi xuôi)."""
    to_hop_codes = [m.upper() for m in _TO_HOP_PATTERN.findall(raw_query)]
    lines = []

    # Nếu chỉ hỏi mã tổ hợp gồm môn gì (không nhắc tên ngành)
    if to_hop_codes and not keywords:
        for code in to_hop_codes:
            rows = _safe_query(
                "MATCH (t:TổHợpMôn {ma: $ma}) RETURN t.ma AS ma, t.mon_hoc AS mon",
                {"ma": code}
            )
            for r in rows:
                lines.append(f"Tổ hợp {r['ma']}: {r['mon']}")
        return "\n".join(lines)

    targets = keywords if keywords else [raw_query[:60]]
    for kw in targets:
        rows = _safe_query("""
            MATCH (n:NgànhHọc)-[:DÙNG_TỔ_HỢP]->(t:TổHợpMôn)
            WHERE toLower(n.name)     CONTAINS toLower($kw)
               OR toLower(n.id)       CONTAINS toLower($kw)
               OR toLower(n.ma_nganh) CONTAINS toLower($kw)
            RETURN n.name AS nganh, n.ma_nganh AS ma_nganh,
                   collect(t.ma + ': ' + t.mon_hoc) AS to_hop_list
            ORDER BY n.name
        """, {"kw": kw})
        for r in rows:
            if r["to_hop_list"]:
                ma = f" ({r['ma_nganh']})" if r.get("ma_nganh") else ""
                lines.append(f"Ngành {r['nganh']}{ma} xét tuyển bằng:")
                for th in sorted(r["to_hop_list"]):
                    lines.append(f"  • {th}")
    return "\n".join(lines)


def graph_nganh_theo_to_hop(raw_query: str) -> str:
    """Hỏi ngược: tổ hợp A00 → ngành nào."""
    codes = [m.upper() for m in _TO_HOP_PATTERN.findall(raw_query)]
    if not codes:
        return ""
    lines = []
    for code in codes:
        rows = _safe_query("""
            MATCH (n:NgànhHọc)-[:DÙNG_TỔ_HỢP]->(t:TổHợpMôn {ma: $ma})
            RETURN t.ma AS to_hop, t.mon_hoc AS mon,
                   collect(n.name) AS nganh_list
        """, {"ma": code})
        for r in rows:
            nganh_str = ", ".join(sorted(r["nganh_list"]))
            lines.append(f"Tổ hợp {r['to_hop']} ({r['mon']}) → {len(r['nganh_list'])} ngành: {nganh_str}")
    return "\n".join(lines)


def graph_khoa_cua_nganh(keywords: list[str]) -> str:
    """Ngành thuộc khoa/viện nào."""
    lines = []
    for kw in keywords:
        rows = _safe_query("""
            MATCH (n:NgànhHọc)-[:THUỘC_KHOA]->(k:KhoaViện)
            WHERE toLower(n.name) CONTAINS toLower($kw)
               OR toLower(n.id)   CONTAINS toLower($kw)
            RETURN n.name AS nganh, k.name AS khoa
        """, {"kw": kw})
        for r in rows:
            lines.append(f"Ngành {r['nganh']} thuộc {r['khoa']}")
    return "\n".join(lines)


def graph_entity_search(keywords: list[str]) -> str:
    """
    Generic search cho học phí, học bổng, điểm chuẩn.
    Loại bỏ MENTIONS để tránh trả về source file IDs làm nhiễu.
    """
    lines = []
    for kw in keywords:
        rows = _safe_query("""
            MATCH (n)-[r]->(m)
            WHERE type(r) <> 'MENTIONS'
              AND (
                  toLower(n.name) CONTAINS toLower($kw)
               OR toLower(n.id)   CONTAINS toLower($kw)
              )
              AND NOT n:Document
            RETURN coalesce(n.name, n.id) AS source,
                   type(r)                AS rel,
                   coalesce(m.name, m.id) AS target
            LIMIT 12
        """, {"kw": kw})
        for r in rows:
            # Bỏ qua các dòng có source/target trông như file ID (ts-xxx-xxx)
            src, tgt = r.get("source", ""), r.get("target", "")
            if src and tgt and not re.match(r'^ts-', str(src)):
                lines.append(f"{src} —[{r['rel']}]→ {tgt}")
    return "\n".join(lines)


def retrieve_from_graph(query: str, intent: str) -> str:
    """Điều phối Cypher theo intent."""
    keywords = extract_nganh_keywords(query)

    if intent == "nganh_list":
        return graph_tat_ca_nganh()

    if intent == "to_hop_nganh":
        return graph_to_hop_cua_nganh(keywords, query)

    if intent == "nganh_theo_to_hop":
        return graph_nganh_theo_to_hop(query)

    if intent == "nganh_general":
        result = graph_to_hop_cua_nganh(keywords, query)
        result += "\n" + graph_khoa_cua_nganh(keywords)
        return result.strip()

    if intent in ("hoc_phi_hoc_bong", "diem_chuan", "chinh_sach"):
        return graph_entity_search(keywords)

    return ""


# ══════════════════════════════════════════════════════════════
# 5. VECTOR SEARCH
# ══════════════════════════════════════════════════════════════

def vector_search(query: str, category: str, k: int = 8) -> str:
    try:
        docs = vector_store.max_marginal_relevance_search(
            query, k=k, fetch_k=30, filter={"category": category}
        )
        return "\n\n".join(d.page_content for d in docs)
    except Exception:
        return ""


def vector_search_nha_tro(query: str, k: int = 6) -> str:
    """
    Vector search nhà trọ với filter metadata.
    LƯU Ý: step3 ép bool → str nên co_dieu_hoa lưu là "True"/"False" (string).
    Filter phải dùng string thay vì bool.
    """
    q = query.lower()
    conditions: list[dict] = [{"category": {"$eq": "nha_tro"}}]

    if any(kw in q for kw in ["điều hòa", "máy lạnh"]):
        # ChromaDB nhận "True" string vì step3 dùng str(v)
        conditions.append({"co_dieu_hoa": {"$eq": "True"}})
    if any(kw in q for kw in ["nóng lạnh", "bình nóng lạnh"]):
        conditions.append({"co_nong_lanh": {"$eq": "True"}})

    khu_vuc_map = {
        "phú thái": "Phú Thái", "tân thịnh": "Tân Thịnh",
        "quyết thắng": "Quyết Thắng", "sơn tiến": "Sơn Tiến",
        "nước hai": "Nước Hai", "phan đình phùng": "Phan Đình Phùng",
        "z115": "Z115",
    }
    for kw, kv_val in khu_vuc_map.items():
        if kw in q:
            conditions.append({"khu_vuc": {"$eq": kv_val}})
            break

    chroma_filter = {"$and": conditions} if len(conditions) > 1 else {"category": "nha_tro"}

    try:
        docs = vector_store.max_marginal_relevance_search(
            query, k=k, fetch_k=20, filter=chroma_filter
        )
        # Fallback nếu filter phức tạp không ra kết quả
        if not docs and chroma_filter != {"category": "nha_tro"}:
            docs = vector_store.max_marginal_relevance_search(
                query, k=k, fetch_k=20, filter={"category": "nha_tro"}
            )
        return "\n\n".join(d.page_content for d in docs)
    except Exception:
        return ""


def get_context(query: str, prev_category: str | None = None) -> tuple[str, str, str, str]:
    """Trả về (vec_ctx, graph_ctx, category, intent)."""
    category = get_category(query, prev_category)

    if category == "nha_tro":
        return vector_search_nha_tro(query), "", "nha_tro", "nha_tro"

    intent    = detect_intent(query)

    # Câu hỏi danh sách ngành: query thẳng graph, vector chỉ bổ sung
    if intent == "nganh_list":
        graph_ctx = graph_tat_ca_nganh()
        vec_ctx   = vector_search("giới thiệu ngành đào tạo tnus", "tuyen_sinh", k=3)
        return vec_ctx, graph_ctx, category, intent

    vec_ctx   = vector_search(query, "tuyen_sinh", k=8)
    graph_ctx = retrieve_from_graph(query, intent)
    return vec_ctx, graph_ctx, category, intent


# ══════════════════════════════════════════════════════════════
# 6. ROUTER — phân loại câu hỏi
# ══════════════════════════════════════════════════════════════

ROUTE_TEMPLATE = """Phân loại câu hỏi sau vào MỘT trong ba nhãn:
RAG   — hỏi về: ngành học, điểm chuẩn, tổ hợp môn, học phí, học bổng, chỉ tiêu,
         hồ sơ nhập học, thủ tục, chính sách ưu đãi, nhà trọ, ký túc xá, TNUS.
CHAT  — chào hỏi thông thường, cảm ơn, hỏi tên bot.
GENERAL — chủ đề ngoài tuyển sinh/đại học (nấu ăn, thời tiết, giải trí...).

Ví dụ: "TNUS có ngành CNTT không?" → RAG
        "Tổ hợp A00 gồm những môn gì?" → RAG
        "Bạn tên là gì?" → CHAT
        "Python là gì?" → GENERAL

Câu hỏi: {question}
Chỉ trả về đúng một từ:"""

route_chain = ChatPromptTemplate.from_template(ROUTE_TEMPLATE) | llm | StrOutputParser()


def classify_query(question: str) -> str:
    q = question.lower()
    # Fast-path: từ khóa rõ ràng không cần LLM
    if any(kw in q for kw in _TUYEN_SINH_STRONG | {"tất cả", "danh sách"}):
        return "RAG"
    if any(kw in q for kw in _NHA_TRO_STRONG):
        return "RAG"
    try:
        res   = route_chain.invoke({"question": question}).strip().upper()
        word  = res.split()[0] if res else "RAG"
        return word if word in ("RAG", "CHAT", "GENERAL") else "RAG"
    except Exception:
        return "RAG"


# ══════════════════════════════════════════════════════════════
# 7. QUẢN LÝ LỊCH SỬ HỘI THOẠI
# ══════════════════════════════════════════════════════════════

MAX_RAW_HISTORY     = 8
SUMMARIZE_THRESHOLD = 16

summarize_chain = (
    ChatPromptTemplate.from_template(
        "Tóm tắt cuộc hội thoại sau thành 2-3 câu, giữ lại nhu cầu cốt lõi của người dùng.\n"
        "Hội thoại:\n{conversation}\nTóm tắt:"
    ) | llm | StrOutputParser()
)


def build_langchain_history(messages: list) -> list:
    history = []
    if st.session_state.get("history_summary"):
        history.append(AIMessage(
            content=f"[Tóm tắt trước: {st.session_state['history_summary']}]"
        ))
    for msg in messages[-MAX_RAW_HISTORY:]:
        cls = HumanMessage if msg["role"] == "user" else AIMessage
        history.append(cls(content=msg["content"]))
    return history


def maybe_summarize_history():
    msgs = st.session_state.messages
    if len(msgs) <= SUMMARIZE_THRESHOLD:
        return
    old = msgs[:-MAX_RAW_HISTORY]
    conv = ""
    if st.session_state.get("history_summary"):
        conv = f"[Cũ]: {st.session_state['history_summary']}\n"
    conv += "\n".join(
        f"{'User' if m['role'] == 'user' else 'AI'}: {m['content']}" for m in old
    )
    try:
        st.session_state["history_summary"] = summarize_chain.invoke({"conversation": conv})
        st.session_state.messages = msgs[-MAX_RAW_HISTORY:]
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════
# 8. MASTER PROMPT
# ══════════════════════════════════════════════════════════════

MASTER_SYSTEM = """Bạn là trợ lý AI tuyển sinh của Trường Đại học Khoa học – TNUS năm 2026.
CHẾ ĐỘ: {route}

━━━ CHẾ ĐỘ RAG ━━━
NGUYÊN TẮC BẮT BUỘC:
1. CHỈ dùng thông tin từ DỮ LIỆU GRAPH và DỮ LIỆU VECTOR bên dưới. TUYỆT ĐỐI không thêm tên ngành, điểm số, tổ hợp, học phí từ kiến thức bản thân.
2. Nếu DỮ LIỆU GRAPH có danh sách ngành/tổ hợp → trình bày ĐÚNG NGUYÊN nội dung đó, không thêm không bớt.
3. Nếu cả hai nguồn đều trống → trả lời: "Mình chưa tìm thấy thông tin này. Bạn liên hệ fanpage TNUS để được hỗ trợ nhé!"
4. Phong cách: thân thiện, xưng "mình", gọi là "bạn". Dùng Markdown gọn gàng.
5. Cuối câu trả lời: gợi ý 1 câu hỏi tiếp theo liên quan.
6. Nhà trọ: thêm cảnh báo "⚠️ Giá thuê có thể thay đổi — liên hệ chủ trọ để xác nhận."

--- DỮ LIỆU GRAPH (cấu trúc, độ chính xác cao) ---
{graph_context}

--- DỮ LIỆU VECTOR (mô tả chi tiết) ---
{vector_context}

━━━ CHẾ ĐỘ CHAT ━━━
Trò chuyện tự nhiên, giới thiệu ngắn về TNUS nếu phù hợp.

━━━ CHẾ ĐỘ GENERAL ━━━
Trả lời ngắn gọn (tối đa 2 câu), sau đó nhắc lại nhiệm vụ chính là tư vấn tuyển sinh TNUS.
"""

master_prompt = ChatPromptTemplate.from_messages([
    ("system", MASTER_SYSTEM),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])
master_chain = master_prompt | llm | StrOutputParser()


# ══════════════════════════════════════════════════════════════
# 9. GIAO DIỆN STREAMLIT
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Tư vấn tuyển sinh TNUS",
    page_icon="tnus_logo.png",
    layout="centered",
)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    try:
        st.image("tnus_logo.png", width=120)
    except Exception:
        pass
    st.markdown("### Tư vấn tuyển sinh TNUS 2026")
    st.caption("Trường Đại học Khoa học – ĐH Thái Nguyên")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Xóa chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pop("history_summary", None)
            st.session_state.pop("last_category", None)
            st.rerun()
    with col2:
        if st.button("📋 Tóm tắt", use_container_width=True):
            st.session_state["show_summary"] = True

    st.divider()
    msg_count = len(st.session_state.get("messages", []))
    user_msgs = sum(1 for m in st.session_state.get("messages", []) if m["role"] == "user")
    st.caption(f"📊 Tin nhắn: **{msg_count}** | Câu hỏi: **{user_msgs}**")

    if st.session_state.get("history_summary"):
        with st.expander("📝 Tóm tắt lịch sử"):
            st.info(st.session_state["history_summary"])

    if _HAS_LOGGER:
        st.divider()
        if st.button("📊 Thống kê", use_container_width=True):
            st.session_state["show_analytics"] = True

# ── Header ───────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 5])
with col_logo:
    try:
        st.image("tnus_logo.png", width=60)
    except Exception:
        pass
with col_title:
    st.title("Tư vấn tuyển sinh TNUS")
st.markdown("*Hỏi về ngành học, điểm chuẩn, tổ hợp môn, học phí, học bổng hoặc nhà trọ!*")

# ── Session state ─────────────────────────────────────────────
for key, default in [("messages", []), ("history_summary", ""), ("last_category", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Tin nhắn chào mừng ────────────────────────────────────────
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Xin chào! 👋 Mình là **trợ lý tuyển sinh TNUS** — sẵn sàng hỗ trợ bạn!\n\n"
            "Mình có thể giúp về:\n"
            "- 📚 **Ngành học** — 40 chương trình, tổ hợp môn, điểm chuẩn\n"
            "- 💰 **Học phí & học bổng** năm 2026\n"
            "- 📝 **Hồ sơ, thủ tục nhập học**\n"
            "- 🏠 **Nhà trọ** gần trường\n\n"
            "Bạn muốn tìm hiểu về điều gì? 😊"
        ),
        "route": "CHAT",
    })

# ── Hiển thị lịch sử ─────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("route") not in (None, "CHAT"):
            badge = {"RAG": "📚 Dữ liệu cục bộ", "GENERAL": "🌐 Kiến thức nền"}
            st.caption(f"*Nguồn: {badge.get(msg['route'], '')}*")

# ── Tóm tắt nhanh ────────────────────────────────────────────
if st.session_state.get("show_summary"):
    st.session_state.pop("show_summary")
    if len(st.session_state.messages) > 2:
        with st.spinner("Đang tóm tắt..."):
            conv = "\n".join(
                f"{'User' if m['role'] == 'user' else 'AI'}: {m['content']}"
                for m in st.session_state.messages
            )
            try:
                st.info(f"**📋 Tóm tắt:**\n\n{summarize_chain.invoke({'conversation': conv})}")
            except Exception:
                st.warning("Không thể tóm tắt lúc này.")
    else:
        st.info("Hội thoại chưa đủ để tóm tắt.")

# ── Analytics ────────────────────────────────────────────────
if _HAS_LOGGER and st.session_state.get("show_analytics"):
    st.session_state.pop("show_analytics")
    logs = get_logs(limit=500)
    if not logs:
        st.info("Chưa có dữ liệu log.")
    else:
        import pandas as pd
        df = pd.DataFrame(logs)
        st.markdown("### 📊 Thống kê hội thoại")
        c1, c2, c3 = st.columns(3)
        c1.metric("Tổng câu hỏi", len(df))
        if "route" in df.columns:
            c2.metric("RAG queries", int((df["route"] == "RAG").sum()))
        st.dataframe(df.head(50), use_container_width=True)


# ══════════════════════════════════════════════════════════════
# 10. XỬ LÝ ĐẦU VÀO
# ══════════════════════════════════════════════════════════════

user_query = st.chat_input("Hỏi về ngành học, điểm chuẩn, tổ hợp môn, nhà trọ...")

if user_query:
    maybe_summarize_history()
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    history_for_llm = build_langchain_history(st.session_state.messages[:-1])

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm..."):
            route = classify_query(user_query)

            if route == "RAG":
                prev_cat = st.session_state.get("last_category")
                vec_ctx, graph_ctx, category, intent = get_context(user_query, prev_cat)
                st.session_state["last_category"] = category
            else:
                vec_ctx, graph_ctx, category, intent = "", "", "khac", "khac"

            graph_ctx_prompt = graph_ctx or "Không có dữ liệu đồ thị cho câu hỏi này."
            vec_ctx_prompt   = vec_ctx   or "Không tìm thấy văn bản liên quan."

            badge = {"RAG": "📚 Dữ liệu cục bộ", "CHAT": "💬 Hội thoại", "GENERAL": "🌐 Kiến thức nền"}
            if route != "CHAT":
                st.caption(f"*Nguồn: {badge.get(route, '')}*")

        response = st.write_stream(master_chain.stream({
            "route":          route,
            "graph_context":  graph_ctx_prompt,
            "vector_context": vec_ctx_prompt,
            "chat_history":   history_for_llm,
            "question":       user_query,
        }))

        # Debug expander
        if route == "RAG":
            with st.expander("🛠️ Debug"):
                kws = extract_nganh_keywords(user_query)
                codes = [m.upper() for m in _TO_HOP_PATTERN.findall(user_query)]
                st.markdown(
                    f"**Category:** `{category}` | **Intent:** `{intent}`\n\n"
                    f"**Keywords ngành:** `{kws}` | **Mã tổ hợp:** `{codes}`"
                )
                st.markdown("**🕸️ Graph:**")
                st.success(graph_ctx or "(trống)")
                st.markdown("**📄 Vector (600 ký tự đầu):**")
                st.info(vec_ctx[:600] + "..." if len(vec_ctx) > 600 else vec_ctx or "(trống)")

    st.session_state.messages.append({"role": "assistant", "content": response, "route": route})
    log_interaction(route, category, user_query, response)