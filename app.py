import os, re, json, ast
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from openpyxl import load_workbook
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------- NORMALIZE ----------------
# ---------------- FIXED NORMALIZATION ----------------
def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = text.replace("&amp;", "and")  # ✅ fix HTML encoded ampersands
    text = re.sub(r"&", "and", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ---------------- FIXED FUZZY MATCH ----------------
def fuzzy_exact_match(query: str, df: pd.DataFrame, threshold: int = 90):  # ✅ lowered threshold
    best_row = None
    best_score = 0
    qn = normalize_text(query)
    for _, row in df.iterrows():
        candidate = normalize_text(row.get("activity name", ""))
        score = fuzz.ratio(qn, candidate)
        if score > best_score:
            best_score = score
            best_row = row
    if best_score >= threshold and best_row is not None:
        return {
            "activity": best_row.get("activity name"),
            "division": best_row.get("division"),
            "group": best_row.get("group"),
            "class": best_row.get("class"),
            "isic_description": best_row.get("isic description"),
            "method": f"Fuzzy Exact Match ({best_score}%)",
            "score": int(best_score),
        }
    return None

# ---------------- JSON PARSING ----------------
def parse_json_like(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            try:
                return ast.literal_eval(candidate)
            except Exception:
                return {"llm_raw": text}
    return {"llm_raw": text}


# ---------------- CONFIG ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

VECTOR_STORE_DIR = "vectorstores"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

SHEETS_ORDER = ["consolidated_sheet.xlsx"]


# ---------------- LOAD SHEETS ----------------
@st.cache_data
def load_sheets():
    dataframes = {}
    for sheet in SHEETS_ORDER:
        df = pd.read_excel(sheet)
        df.columns = [c.strip().lower() for c in df.columns]
        required_cols = {"activity name", "class", "division", "group", "isic description"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"Sheet {sheet} must contain {required_cols}.")
        dataframes[sheet] = df
    return dataframes


# ---------------- VECTOR STORE ----------------
@st.cache_resource
def load_or_create_vectorstore(sheet_name: str, df: pd.DataFrame, version: int = 2):
    store_path = os.path.join(VECTOR_STORE_DIR, f"{sheet_name}_faiss")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    if os.path.exists(store_path):
        vectorstore = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
    else:
        docs = [
            Document(
                page_content=str(row["activity name"]),
                metadata={
                    "class": str(row.get("class")),
                    "division": str(row.get("division")),
                    "group": str(row.get("group")),
                    "isic_description": str(row.get("isic description")),
                },
            )
            for _, row in df.iterrows()
        ]
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(store_path)
    return vectorstore


# ---------------- SEARCH LOGIC ----------------
def search_activity(query: str, dataframes: dict, vectorstores: dict, use_llm: bool = False):
    # 1) Fuzzy exact match
    for sheet in SHEETS_ORDER:
        df = dataframes[sheet]
        match = fuzzy_exact_match(query, df, threshold=95)
        if match:
            return {"source": sheet, "query": query, **match}

    # 2) Vector similarity search
    all_candidates = []
    for sheet in SHEETS_ORDER:
        docs = vectorstores[sheet].similarity_search(query, k=5)
        for d in docs:
            all_candidates.append({
                "query": query,
                "sheet": sheet,
                "activity": d.page_content,
                "division": d.metadata.get("division"),
                "group": d.metadata.get("group"),
                "class": d.metadata.get("class"),
                "isic_description": d.metadata.get("isic_description"),
            })

    if not use_llm:
        if all_candidates:
            best = all_candidates[0]
            return {"query": query, "source": best["sheet"], **best,  "method": "Vector (fast mode)"}
        return {"source": None, "activity": None, "class": None, "method": "No match"}

    # 3) LLM refinement
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0)
    prompt = f"""
    You are given:
    - A user query (business activity name).
    - A list of candidate activities retrieved from a database (with sheet, activity, division, group, class, and isic_description).

    Task:
    1. Pick the single best candidate ONLY from the list.
    2. Do not invent new activities. 
    3. Prefer semantic matches (e.g., "hydropanels" → "solar equipment trading").
    4. Return ONLY JSON with keys:
       query, sheet, activity, division, group, class, isic_description, reason.

    User query: "{query}"

    Candidates: {all_candidates}
    """
    response = llm.invoke(prompt)
    parsed = parse_json_like(response.content)

    if isinstance(parsed, dict) and "activity" in parsed:
        return {
            "query": query,
            "source": parsed.get("sheet"),
            "activity": parsed.get("activity"),
            "division": parsed.get("division"),
            "group": parsed.get("group"),
            "class": parsed.get("class"),
            "isic_description": parsed.get("isic_description"),
            "method": "Vector + LLM",
            "reason": parsed.get("reason"),
            "llm_raw": response.content,
        }

    return {"source": None, "activity": None, "class": None, "method": "No match", "llm_raw": response.content}


def search_multiple_activities(queries, dataframes, vectorstores, use_llm=True, progress_callback=None):
    results = [None] * len(queries)

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_index = {
            executor.submit(search_activity, q.strip(), dataframes, vectorstores, use_llm): i
            for i, q in enumerate(queries) if q.strip()
        }

        for done, future in enumerate(as_completed(future_to_index)):
            idx = future_to_index[future]
            query = queries[idx]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = {"query": query, "error": str(e)}

            if progress_callback:
                progress_callback(done + 1, len(queries))

    return results


# ---------------- FIXED BATCH PROCESSOR ----------------
def process_activities_with_rag(input_file, output_file, dataframes, vectorstores, use_llm=True, batch_size=10):
    df_in = pd.read_excel(input_file)
    if "Activity Name" not in df_in.columns:
        st.error("❌ The uploaded Excel file must contain a column named **'Activity Name'**.")
        st.stop()

    activities = df_in["Activity Name"].dropna().tolist()

    st.write(f"🔍 Found {len(activities)} activities to process.")
    progress_text = st.empty()
    progress_bar = st.progress(0)

    all_results = []

    for i in range(0, len(activities), batch_size):
        batch = activities[i:i + batch_size]

        def update_progress(done, total):
            percent = int(((i + done) / len(activities)) * 100)
            progress_bar.progress(percent)
            progress_text.text(f"Processing batch {i//batch_size + 1}: {i + done}/{len(activities)} done...")

        rag_results = search_multiple_activities(
            batch, dataframes, vectorstores, use_llm=use_llm, progress_callback=update_progress
        )

        # ✅ Preserve original query order
        ordered_results = sorted(rag_results, key=lambda x: batch.index(x["query"]))

        all_results.extend(ordered_results)

        # st.success(f"✅ Finished batch {i//batch_size + 1} ({min(i + batch_size, len(activities))}/{len(activities)})")

    # ✅ Write final combined results in correct order
    final_df = pd.DataFrame([{
        "Activity Name": res.get("query"),
        "Division": res.get("division"),
        "Group": res.get("group"),
        "Class": res.get("class"),
        "ISIC Description": res.get("isic_description"),
    } for res in all_results])

    final_df.to_excel(output_file, index=False)

    progress_bar.progress(100)
    progress_text.text("✅ All batches completed!")
    st.write(f"📄 Results saved to `{output_file}`")
# ---------------- STREAMLIT APP ----------------
def main():
    st.title("Activity Code Finder (RAG)")
    st.write("Search activity codes from consolidated sheets. Supports multiple activities or batch uploads.")

    try:
        dataframes = load_sheets()
    except Exception as e:
        st.error(f"Failed to load sheets: {e}")
        return

    vectorstores = {sheet: load_or_create_vectorstore(sheet, df) for sheet, df in dataframes.items()}

    # Text search
    query = st.text_area("Enter one or multiple activities (each on a newline):")
    if st.button("Search") and query:
        queries = [q.strip() for q in re.split(r"[\n]", query) if q.strip()]
        results = search_multiple_activities(queries, dataframes, vectorstores, use_llm=True)
        for res in results:
            print(f"Result: {res.get('activity')}, Code: {res.get('class')}, Division: {res.get('division')}, Group: {res.get('group')}, ISIC Desc: {res.get('isic_description')}")
            st.subheader(res.get("activity") or "No match")
            st.write("**Query:**", res.get("query"))
            st.write("**Division:**", res.get("division"))
            st.write("**Group:**", res.get("group"))
            st.write("**Class (Code):**", res.get("class"))
            st.write("**ISIC Description:**", res.get("isic_description"))
            if res.get("score") is not None:
                st.write("**Score:**", res.get("score"))
            if res.get("reason"):
                st.write("**Reason:**", res.get("reason"))

    # File upload for batch
    import os

    # File upload for batch
    st.subheader("📂 Batch Process Activities from Excel")

    uploaded = st.file_uploader("Upload Excel with a column named 'Activity Name'", type=["xlsx"])

    # Detect new file upload and clear previous session data
    if uploaded:
        if (
            "last_file" not in st.session_state
            or st.session_state.last_file != uploaded.name
        ):
            st.session_state.last_file = uploaded.name
            st.session_state.output_file = None
            st.session_state.input_file = None
            st.info("📁 New file detected — previous data cleared.")

    # Process the uploaded file
    if uploaded and st.button("Process File"):
        input_file = f"uploaded_{uploaded.name}"
        with open(input_file, "wb") as f:
            f.write(uploaded.read())

        output_file = f"processed_{uploaded.name.replace('.xlsx', '')}_result.xlsx"

        # ✅ Ensure fresh output file each time
        if os.path.exists(output_file):
            os.remove(output_file)

        st.session_state.input_file = input_file
        st.session_state.output_file = output_file

        process_activities_with_rag(
            input_file, output_file, dataframes, vectorstores, use_llm=True
        )

        st.success("✅ Processing complete. Download your results below:")

    # Show download button only when latest file is ready
    if st.session_state.get("output_file") and os.path.exists(st.session_state.output_file):
        with open(st.session_state.output_file, "rb") as f:
            st.download_button(
                "⬇️ Download Results",
                data=f,
                file_name="isic_mapped_sheet.xlsx",
            )
    # if st.button("🧹 Clear Cached Vectorstores"):
    #     st.cache_resource.clear()
    #     st.cache_data.clear()
    #     st.success("✅ Cache cleared — the app will rebuild vector DBs on next run.")
if __name__ == "__main__":
    main()
