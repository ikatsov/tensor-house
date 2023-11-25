from langchain.prompts.prompt import PromptTemplate

template = (
    '''
You're a software engineer. You're designing a workflow automation program. Given the user's question and internal APIs 
(python functions) described below, your goal is to create a step-by-step description of the business logic of the
program that invokes the API functions to fetch the data relevant to the question, combines and transforms these 
data elements, and prints the answer to the question. 

You can use the following tools (python functions):

{api_docs}

=======================================================================================================================




Q: What are the properties of supplier LA Shoes Inc?

A: 
1. Invoke search_documents passing the question "What are the properties of supplier LA Shoes Inc?" as an argument.
2. Print the result returned by the searcher using the print_answer function.
<<<




Q: How many brands do we sell across all locations?

A: 
1. Generate a SQL query that counts the number of distinct brands in the INVENTORY table.
2. Invoke query_inventory passing the generated SQL query as an argument.
3. Convert the result returned by the query_inventory to a scalar value (number of distinct brands). 
4. Print the scalar value using the print_answer function.
<<<




Q: How many units of SKU 00135777 should be moved to the San Jose store and which locations these units can be sourced from?

A: 
1. Estimate shortage for SKU 00135777 in San Jose using stock_demand_difference.
2. If we need one or more units, determine whether these units can be sourced (rebalanced) from other locations as follows:
   2.1. Get a list of locations that have SKU 00135777 using query_inventory.
   2.2. Iterate over the locations and estimate available overage using stock_demand_difference 
   2.3. If a certain location has overage, add this location, overage value, and shipping cost form this location to 
        San Jose to the list or possible source locations
3. Print the list of possible source locations or tell tat no such locations are available.
<<<



Q: How many units of SKU 00135777 should be procured for the San Jose store and what are the procurement options?

A: 
1. Estimate how many units will we need in San Jose (that is, estimate shortage) using stock_demand_difference.
2. Generate a SQL query for fetching the information about the suppliers, locations, and unit costs for SKU 00135777 from the suppliers database.
3. Execute the generated SQL query using query_suppliers.
4. Calculate the shipping cost for each supplier option using get_shipping_cost.
5. Print all supplier options with the corresponding shipping costs.
<<<




Q: What will be the demand for SKU 123213 in Atlanta over next 9 weeks?

A:
1. Estimate weekly demand values for next 9 weeks using get_forecast.
2. Calculate the total demand for all 9 weeks.
3. Print weekly and total demand values.
<<<




Q: {query}

A:
'''
)

PLANNER_PROMPT = PromptTemplate(input_variables=["api_docs", "query"], template=template)

PLANNER_PROMPT_STOP_SEQ = ["<<<"]