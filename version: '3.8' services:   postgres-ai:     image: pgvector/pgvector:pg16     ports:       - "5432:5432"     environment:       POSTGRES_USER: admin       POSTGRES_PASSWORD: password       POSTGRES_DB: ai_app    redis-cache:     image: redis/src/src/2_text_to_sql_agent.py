from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

def query_database_with_ai(user_question):
    # Connect the agent to your structured app data
    db = SQLDatabase.from_uri("postgresql+psycopg2://admin:password@localhost:5432/ai_app")
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    
    # Initialize an agent equipped with database querying tools
    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
    
    # The AI dynamically writes the SQL, queries the DB, and translates the output
    response = agent_executor.invoke({"input": user_question})
    return response["output"]
