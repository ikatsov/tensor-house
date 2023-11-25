from langchain.prompts.prompt import PromptTemplate

template = (
    '''
You're an AI assistant. You are reading the transcript of a conversation between a user and an assistant. 
Given the chat history and user's question, your goal is to create standalone question that contains all 
necessary information for a person who will answer it. You MUST include all details that are needed to 
understand and answer the question.




-------------------
User: 
How many units of SKU 55333 do we have in San Jose?

Assistant: 
We have 41 units of SKU 55333 in San Jose.

User: 
Can you provide more details on this SKU?

Standalone question:
We have 41 units of SKU 55333 in San Jose. Can you provide more details on SKU 55333?




-------------------
User:
Do we have shortage of SKU 11001 in any locations?

Assistant: 
There is a shortage of SKU 11001 in the following locations: San Francisco

User:
How many extra units do we need?

Assistant:
We have a shortage of 200 units.

User:
Can we move these units from other locations?

Standalone question:
We need to move 200 units of SKU 11001 to San Francisco from other locations to fulfill the shortage. Which locations 
can we rebalance these unit from?




-------------------
{chat_history}

Standalone question:
'''
)

CHAT_TO_QUERY_PROMPT = PromptTemplate(input_variables=["chat_history"], template=template)