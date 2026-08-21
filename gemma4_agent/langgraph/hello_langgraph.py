from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from client_type import ollama_client
llm = ollama_client

def node1(str):
    return {"messages": [{"role": "ai", "content": "I reached Node1"}]}

def node2(str):
    return {"messages": [{"role": "ai", "content": "And Now at Node 2"}]}

# define llm Node
def call_llm(message):
    output = llm.invoke("hi")
    print(output)
    return output



# Create a new Graph
workflow = StateGraph(MessagesState, llm=llm, name="Hello LangGraph")

# Add the nodes
workflow.add_node("node_1", node1)
workflow.add_node("node_2", node2)
workflow.add_node("call_llm", call_llm)

workflow.add_edge(START, "node_1")
# Add the Edges
workflow.add_edge("node_1", "call_llm")
workflow.add_edge("call_llm", END)

#Run the workflow
app = workflow.compile()
response =app.invoke({"messages": [{"role": "user", "content": "hi!"}]})
print(response)
# display(Image(app.get_graph().draw_mermaid_png()))
print(app.get_graph().draw_ascii())
print(app.get_graph().draw_mermaid())