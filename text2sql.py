from langchain import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from sqlalchemy import create_engine
from dotenv import load_dotenv
from operator import itemgetter
import os


load_dotenv()


class Text2Sql:

    def __init__(self) -> None:
        self.db_name = os.environ["GLUE_DB_NAME"]
        self.bucket = os.environ["GLUE_BUCKET_NAME"]
        self.athena_output_path = f"s3://{self.bucket}/athenaresults/"
        self.conn_string = self._construct_conn_string()
        self.db = self._establish_athena_conn()
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)

    def _construct_conn_string(self):
        return f"awsathena+rest://@athena.us-east-1.amazonaws.com:443/{self.db_name}?s3_staging_dir={self.athena_output_path}"

    def _establish_athena_conn(self):
        athena_engine = create_engine(self.conn_string, echo=True)
        return SQLDatabase(athena_engine)

    def create_sql_chain(self) -> LLMChain:
        query = create_sql_query_chain(self.llm, self.db)
        execute = QuerySQLDataBaseTool(db=self.db)

        return query | execute

    def rephrase_chain(self):

        rephrase_prompt = PromptTemplate.from_template(
            """
            Given the following user  question, corresponding SQL query, and SQL result, answer the user question.
    
            Question: {question}
            SQL Query: {query}
            SQL Result: {result}
            Answer:
    
            """
        )

        query = create_sql_query_chain(self.llm, self.db)
        execute = QuerySQLDataBaseTool(db=self.db)

        rephrase_chain = rephrase_prompt | self.llm | StrOutputParser()

        chain = (
            RunnablePassthrough.assign(query=query).assign(
                result=itemgetter("query") | execute
            )
            | rephrase_chain
        )

        return chain