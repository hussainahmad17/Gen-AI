from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv
load_dotenv()



first_model = ChatOpenAI(model="gpt-4o")
second_model = ChatOpenAI(model="gpt-4o-mini")

first_prompt = PromptTemplate.from_template(
    "Generate the notes on following text \n {text}"
)

second_prompt = PromptTemplate.from_template(
    "Generate the 5 short questions on following text \n {text}"
)

third_propmt = PromptTemplate.from_template(
    "Merge these provided notes and quiz into a single document \n Notes-> {notes} and Quiz-> {quiz}"
)

parser = StrOutputParser()


# here we use runnableParallel to run the chains parallely

parallel_chains = RunnableParallel({
    # here we need to create 2 chains with specific names
    "notes": first_prompt | first_model | parser,
    "quiz": second_prompt | second_model | parser
})


merging_chain = third_propmt | first_model | parser


main_chain = parallel_chains | merging_chain

text="Learning the parameters of a prediction function and testing it on the same data is a methodological mistake: a model that would just repeat the labels of the samples that it has just seen would have a perfect score but would fail to predict anything useful on yet-unseen data. This situation is called overfitting. To avoid it, it is common practice when performing a (supervised) machine learning experiment to hold out part of the available data as a test set X_test, y_test. Note that the word “experiment” is not intended to denote academic use only, because even in commercial settings machine learning usually starts out experimentally. Here is a flowchart of typical cross validation workflow in model training. The best parameters can be determined by grid search techniques."

result = main_chain.invoke({"text":text})
print(result)



