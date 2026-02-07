from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)


text="""Artificial Intelligence is rapidly transforming the healthcare industry by enabling faster diagnosis, 
personalized treatment plans, and improved patient monitoring. Machine learning models analyze medical images, 
electronic health records, and genetic data to identify patterns that may be invisible to human clinicians. 
AI-powered systems are now assisting doctors in detecting diseases such as cancer and diabetes at earlier stages, 
reducing human error and improving outcomes. As these technologies mature, ethical considerations like data privacy, 
transparency, and bias remain critical challenges.

Climate change has significantly affected urban areas, increasing temperatures, altering rainfall patterns,
and intensifying extreme weather events. Cities face rising risks such as heat islands, flooding, and air pollution 
due to dense infrastructure and reduced green cover. Urban planners are responding by promoting sustainable 
architecture, expanding green spaces, and adopting renewable energy solutions. Long-term resilience depends on how 
effectively cities integrate environmental policies with population growth and economic development.

Digital education has evolved from simple online courses to immersive learning experiences powered by interactive
platforms and virtual classrooms. Students now access global resources, recorded lectures, and real-time 
collaboration tools from anywhere in the world. Technologies such as adaptive learning systems and AI tutors 
personalize content based on individual progress and learning styles. While digital education increases accessibility,
it also highlights challenges related to digital divides, attention management, and the role of educators in 
virtual environments."""


chunks = text_splitter.create_documents([text])

print(len(chunks))
print(chunks,'\n')