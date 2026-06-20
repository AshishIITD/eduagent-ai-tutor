"""
agent.py — EduAgent using local Ollama llama3.1:8b. No API key required.
RAG + Adaptive MCQ Generation + Guardrails + Spaced Repetition.
"""
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from guardrails import input_guardrail, output_guardrail

LOCAL_MODEL = "llama3.1:8b"   # Runs via Ollama — no API key needed

SUBJECT_DOMAINS = [
    "Mathematics", "Physics", "Chemistry", "Biology", "History",
    "Geography", "Civics", "Economics", "English Grammar", "English Literature",
    "Computer Science", "Environmental Science", "General Knowledge",
    "Logical Reasoning", "Mental Ability"
]

print(f"Initializing EduAgent with BGE-M3 + {LOCAL_MODEL} (local Ollama)...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

FAISS_INDEX_DIR = "faiss_edu_index"
SAMPLE_DOCS_FILE = "curriculum_data.txt"

def initialize_vector_store():
    if os.path.exists(FAISS_INDEX_DIR):
        return FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    if not os.path.exists(SAMPLE_DOCS_FILE):
        curriculum = "\n".join([
            "Mathematics: BODMAS rule - Brackets, Orders, Division, Multiplication, Addition, Subtraction.",
            "Mathematics: LCM × HCF = Product of two numbers.",
            "Physics: Newton's First Law - An object at rest stays at rest unless acted on by an external force.",
            "Physics: Speed = Distance / Time. Units: m/s or km/h.",
            "Chemistry: Water is H2O. Oxygen has atomic number 8.",
            "Biology: Photosynthesis: 6CO2 + 6H2O → C6H12O6 + 6O2.",
            "History: India gained independence on 15 August 1947.",
            "Geography: The Himalayas are the highest mountain range in India.",
            "English: Eight parts of speech: Noun, Pronoun, Verb, Adjective, Adverb, Preposition, Conjunction, Interjection.",
            "Civics: Indian Constitution came into effect on 26 January 1950.",
            "General Knowledge: National Animal of India is Bengal Tiger. National Bird is Peacock.",
            "Logical Reasoning: Series 2,4,8,16 — pattern is multiplication by 2.",
            "Mental Ability: Mirror image reverses left-right orientation of a figure.",
        ])
        with open(SAMPLE_DOCS_FILE, "w") as f:
            f.write(curriculum)
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(SAMPLE_DOCS_FILE)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
    splits = splitter.split_documents(docs)
    vs = FAISS.from_documents(splits, embeddings)
    vs.save_local(FAISS_INDEX_DIR)
    return vs

vectorstore = initialize_vector_store()
llm = ChatOllama(model=LOCAL_MODEL, temperature=0.3)

def answer_question(student_id: str, question: str) -> dict:
    """Full pipeline: Input Guardrail → RAG → Ollama LLM → Output Guardrail."""
    input_check = input_guardrail(question)
    if not input_check["allowed"]:
        return {"answer": f"🚫 {input_check['reason']}", "guardrail_triggered": "input", "contexts": []}

    docs = vectorstore.as_retriever(search_kwargs={"k": 4}).invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    prompt_template = ChatPromptTemplate.from_template(
        "You are EduAgent, an expert academic tutor for Sainik School, JNV, RMS, RIMC entrance exams. "
        "Answer the student's question using ONLY the context below.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    chain = prompt_template | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    output_check = output_guardrail(question, answer, context)
    if not output_check["safe"]:
        return {"answer": f"⚠️ {output_check['reason']}", "guardrail_triggered": "output",
                "contexts": [d.page_content for d in docs]}
    return {"answer": answer, "guardrail_triggered": None, "contexts": [d.page_content for d in docs]}

def generate_mcq(topic: str, difficulty: str = "medium") -> dict:
    """Adaptive MCQ generation grounded in curriculum context."""
    docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(topic)
    context = "\n".join(d.page_content for d in docs)
    prompt = ChatPromptTemplate.from_template(
        "Generate a {difficulty} MCQ about '{topic}' based on:\n{context}\n\n"
        "Format:\nQuestion: ...\nA) ...\nB) ...\nC) ...\nD) ...\nCorrect: [A/B/C/D]\nExplanation: ..."
    )
    chain = prompt | llm | StrOutputParser()
    return {"topic": topic, "difficulty": difficulty,
            "mcq": chain.invoke({"difficulty": difficulty, "topic": topic, "context": context})}
