from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from dotenv import load_dotenv


llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')


class Review(TypedDict):
    summary: str
    sentiment: str

st_model = llm.with_structured_output(Review)

result = st_model.invoke(
    """I was genuinely impressed with the overall experience from start to finish. Everything felt smooth,
    intuitive, and thoughtfully designed, which made the entire process enjoyable rather than stressful.
    The attention to small details really stood out—things you don’t always notice at first, but that
    clearly show the effort put into quality and user experience. The flow was seamless, responses were 
    timely, and nothing felt confusing or out of place. Overall, it created a very positive impression and 
    left me feeling satisfied and confident in the service. I would definitely consider coming back again 
    and would not hesitate to recommend it to others looking for a reliable and well-executed experience."""
)

print(result)
