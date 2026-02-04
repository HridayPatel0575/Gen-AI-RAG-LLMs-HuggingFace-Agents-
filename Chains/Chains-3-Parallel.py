from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

llm = ChatOllama(
    model='qwen2:7b',
    temperature=0.7
)


prompt1 = PromptTemplate(
    template='Generate short and Simple notes for {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 question answers from following {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into single document \n notes->{notes} and quiz->{quiz}',
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

parallel_chain =  RunnableParallel({
    'notes':prompt1 | llm | parser,
    'quiz': prompt2 | llm | parser
})


merge_chain = prompt3 | llm | parser

chain = parallel_chain | merge_chain

text = """
Decision Trees are a popular supervised machine learning algorithm used for both classification and regression tasks.
They work by splitting data into branches based on feature values, forming a tree-like structure of decisions.
The main advantage of decision trees is their interpretability, as the decision-making process is easy to visualize and understand.
They can handle both numerical and categorical data without the need for extensive preprocessing.
However, decision trees are prone to overfitting, especially when the tree becomes very deep.
Techniques such as pruning, setting a maximum depth, and minimum samples per leaf are used to control overfitting.
Decision Trees are widely used in real-world applications like credit scoring, medical diagnosis, and fraud detection.
"""


result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()
