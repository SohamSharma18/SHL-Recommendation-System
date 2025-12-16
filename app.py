import streamlit as st
from utils import load_catalogue, generate_embeddings, create_faiss_index, get_recommendations
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT
from io import BytesIO

st.set_page_config(page_title="SHL Assessment Recommender", layout="centered")

st.markdown("""
    <style>
        body {
            background-color: #f4f6f8;
        }
        .reportview-container {
            background-color: #f4f6f8;
            padding: 2rem;
        }
        .stTextInput > div > div > input {
            background-color: #ffffff;
            color: #333333;
        }
        h1 {
            color: #333333;
        }
    </style>
""", unsafe_allow_html=True)

st.title("SHL Assessment Recommendation Engine")

user_input = st.text_input("Enter your query (e.g., 'Test for mid-level engineers with coding skills'):")

@st.cache_resource
def prepare_index():
    docs = load_catalogue("shl_catalogue.json")
    embeddings = generate_embeddings(docs)
    index = create_faiss_index(embeddings)
    return docs, index

documents, faiss_index = prepare_index()
recommendations = []

if user_input:
    try:
        recommendations = get_recommendations(user_input, faiss_index, documents)
        for i, doc in enumerate(recommendations):
            meta = doc.metadata
            st.markdown(f"### Recommendation {i + 1}: {meta.get('name', 'Unnamed Assessment')}")
            st.write(f"**Description:** {meta.get('description')}")
            st.write(f"**Test Type:** {meta.get('test_type')}")
            st.write(f"**Job Levels:** {meta.get('job_levels')}")
            st.write(f"**Duration:** {meta.get('duration')}")
            st.write(f"**Remote Testing:** {meta.get('remote')}")
            st.write(f"**Adaptive/IRT:** {meta.get('adaptive')}")
            st.markdown(f"[View on SHL Site]({meta.get('url')})")
            st.markdown("---")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

def generate_pdf(recommendations):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Heading', fontSize=14, leading=18, spaceAfter=10, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name='Body', fontSize=11, leading=14))

    elements = []

    for i, doc_data in enumerate(recommendations):
        meta = doc_data.metadata
        elements.append(Paragraph(f"Recommendation {i + 1}: {meta.get('name', 'Unnamed Assessment')}", styles['Heading']))
        elements.append(Paragraph(f"<b>Description:</b> {meta.get('description')}", styles['Body']))
        elements.append(Paragraph(f"<b>Test Type:</b> {meta.get('test_type')}", styles['Body']))
        elements.append(Paragraph(f"<b>Job Levels:</b> {meta.get('job_levels')}", styles['Body']))
        elements.append(Paragraph(f"<b>Duration:</b> {meta.get('duration')}", styles['Body']))
        elements.append(Paragraph(f"<b>Remote Testing:</b> {meta.get('remote')}", styles['Body']))
        elements.append(Paragraph(f"<b>Adaptive/IRT:</b> {meta.get('adaptive')}", styles['Body']))
        elements.append(Paragraph(f"<b>URL:</b> {meta.get('url')}", styles['Body']))
        elements.append(Spacer(1, 12))

    doc.build(elements)
    return buffer

if recommendations:
    if st.button("📄 Export to PDF"):
        pdf_buffer = generate_pdf(recommendations)
        st.download_button(
            label="Download PDF",
            data=pdf_buffer,
            file_name="shl_recommendations.pdf",
            mime="application/pdf"
        )