import dspy
from typing import List 
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
import glob
from langchain_text_splitters import MarkdownTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from dspy import Retrieve, Prediction
from litellm import embedding
from pydotdict import DotDict
from pydantic import BaseModel
from qdrant_client import QdrantClient, models
import os
from typing import Optional, Any
import json
import mlflow
from dspy.evaluate import Evaluate
from dspy.evaluate.metrics import answer_passage_match

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Analysis Agent")
mlflow.dspy.autolog()

class TrainDataSet(BaseModel):
    question: str = ""
    answer: str = ""


class TestDataGenerationAgent(dspy.Signature):
    """ You are test data generation agent. You should generate questions that are related to the usecase described by the user"""
    use_case: str = dspy.InputField()
    context = dspy.InputField(desc="may contain relevant information")
    tests_generation: List[str] = dspy.OutputField(
        desc=(
            "List of questions that will enable the user to generate a valid test dataset for the usecase"
        )
    )


class InMemoryRetriever(Retrieve):
    def __init__(self):
        path = r'/Users/arnaraya/Desktop/Projects/Dspy/resources/*.md' #file path
        md_files = glob.glob(path)
        self.client = QdrantClient(":memory:")
        self.model_name = "BAAI/bge-small-en"
        self.client.create_collection(
            collection_name="test_collection",
            vectors_config=models.VectorParams(
                size=self.client.get_embedding_size(self.model_name), 
                distance=models.Distance.COSINE
            ),
        )
        doc_splitter = MarkdownTextSplitter()
        for file in md_files:
            loader = UnstructuredMarkdownLoader(file)
            docs = loader.load()
            doc_chunks = doc_splitter.split_documents(docs)
            print("doc_chunks:", doc_chunks)
            metadata_with_docs = [
                {"document": chunk.page_content, "source": os.path.basename(file)} for chunk in doc_chunks
            ]
            self.client.upload_collection(
                collection_name="test_collection",
                vectors=[models.Document(text=chunk.page_content, model=self.model_name) for chunk in doc_chunks],
                payload=metadata_with_docs
            )
    
    def forward(self, query: str, k: Optional[int] = None):
        results = self.client.query_points(
            collection_name="test_collection",
            query=models.Document(
                text=query,
                model=self.model_name
            )
        ).points
        passages = [DotDict({"long_text": result.payload["document"]}) for result in results]
        return Prediction(passages=passages)


class GenerateAnswer(dspy.Signature):
    """With the given context, answer the question breifly with statisctics if possible"""
    context = dspy.InputField(desc="may contain relevant information")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="generated answer")


class InMemoryRAG(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retrieve = retriever
        self.generate_answer = dspy.Predict(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        print("Relevant Context:", context)
        prediction = self.generate_answer(question=question, context=context)
        return dspy.Prediction(answer=prediction.answer, context=context)
    

class TrainSet_QuestionRAG(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retrieve = retriever
        self.generate_question = dspy.ReAct(TestDataGenerationAgent, [])

    def forward(self, question):
        context = self.retrieve(question).passages
        print("Relevant Context:", context)
        use_case = """"Given the information on latest products of both (Silicon labs) and that of my competitior (Nordic Semiconductor), 
        I want to perform comparision analysis on the performance and the impacts of competing products. Keep the questions more product and technology specific"""
        result = self.generate_question(use_case=use_case, context=context)
        return result.tests_generation


lm = dspy.LM('openai/aws-claude-3-7-sonnet', 
            api_key="sk-zFhbB9n1pC_WH1gkEqKRZw", 
            api_base="https://litellm.silabs.net",
            cache=True)
inmemory_retriever = InMemoryRetriever()
dspy.configure(lm=lm, rm=inmemory_retriever)
generate_trainset_questions = TrainSet_QuestionRAG(inmemory_retriever)
testset_questions = generate_trainset_questions("What are the products that are introduced newly")
# result = agent(use_case="I want to compare my upcoming products (Silicon Labs) against my competitior (Nordic Semiconductors) in the semi conductor industry and Want to perform comparision analysis on the products and progress")
print(testset_questions)
train_dataset: List[TrainDataSet] = []
rag_pipeline = InMemoryRAG(inmemory_retriever)
for question in testset_questions:
    prediction = rag_pipeline(question)
    train_dataset.append(TrainDataSet(question=question, answer=prediction.answer))
print("Completion of building training dataset")
print(train_dataset)
with open("trainset.json", "a") as file:
    for data in train_dataset:
        json.dump(data.model_dump(), file, indent=4)
trainset = []
for td in train_dataset:
    st = dspy.Example(question=td.question, answer=td.answer, context="").with_inputs("question", "context")
    trainset.append(st)

#Evaluation Agent
class Assess(dspy.Signature):
    """Assess the quality of answer to the ground truth."""
    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer: int = dspy.OutputField()


def validate_pred_answer(groundtruth, pred, trace=None):
    t_question, t_answer, prd_answer = groundtruth.question, groundtruth.answer, pred.answer
    assessment = dspy.Predict(Assess)(assessed_text=prd_answer, assessment_question=f"what percentage of the predicted answer relate to the question - {t_question} and the ground truth - {t_answer}")
    return assessment.assessment_answer


react = dspy.ReAct(GenerateAnswer, tools=[])
# response = react(
#     context="",
#     question="What are the key differences in RF performance (range, sensitivity, output power) between Silicon Labs' and Nordic's latest wireless SoCs"
# )
# print(response)
evaluator = Evaluate(devset=trainset, num_threads=1, provide_traceback=True, display_progress=True, display_table=True)
result = evaluator(react, validate_pred_answer)
print(result)



