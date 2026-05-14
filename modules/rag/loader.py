from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from docx import Document as DocxDocument


def load_document(file_path: str) -> list:
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt") or file_path.endswith(".md"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".docx"):
        doc = DocxDocument(file_path)
        parts = []
        parts.extend([p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()])
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    t = (cell.text or "").strip()
                    if t:
                        parts.append(t)
        text = "\n".join(parts)
        return [Document(page_content=text, metadata={"source": file_path})]
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")

    return loader.load()


def split_documents(documents: list, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "。", " ", ""],
    )
    return text_splitter.split_documents(documents)
