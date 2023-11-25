from langchain.prompts.prompt import PromptTemplate

template = (
    '''
You're a software engineer. You're developing a Python script that uses available APIs (python functions) for 
automating the task specified by a supply chain manager and printing a useful answer. You have already analyzed the 
task and described the business logic step by step. Your current goal is to read the API docs, input task (question), 
description of the business logic, and write the final python script.

Follow these guidelines:
- Your script should print a comprehensive answer with all details that a supply chain manager needs to take an action.
- You MUST explicitly import all dependencies such as Pandas or NumPy in the beginning of the script. 
- You MUST generate exactly one Python code listing that may or may not contain helper functions.

You can use the following tools (python functions):

{api_docs}
   
=======================================================================================================================




Task: What are the properties of supplier LA Shoes Inc?

Business logic: 
1. Invoke search_documents passing the question "What are the properties of supplier LA Shoes Inc?" as an argument.
2. Print the result returned by the searcher using the print_answer function.

Python code:
```python
# Delegate the question to the document searcher
supplier_properties = search_documents("What are the properties of supplier LA Shoes Inc?")
# Generate the final answer
print_answer(supplier_properties)
``` <<<




Task: How many brands do we sell across all locations?

Business logic: 
1. Generate a SQL query that counts the number of distinct brands in the INVENTORY table.
2. Invoke query_inventory passing the generated SQL query as an argument.
3. Convert the result returned by the query_inventory to a scalar value (number of distinct brands). 
4. Print the scalar value using the print_answer function.

Python code:
```python
import pandas as pd

# Delegate the question to the inventory database
brands_total = query_inventory("SELECT COUNT(DISTINCT brands) as n_brands FROM INVENTORY")['n_brands']
brands_total_int = brands_total.iloc[0]
# Generate the final answer
print_answer(f"We sell {{brands_total_int}} brands")
``` <<<




Task: How many units of SKU 00135777 should be moved to the San Jose store and which locations these units can be sourced from?

Business logic:
1. Estimate shortage for SKU 00135777 in San Jose using stock_demand_difference.
2. If we need one or more units, determine whether these units can be sourced (rebalanced) from other locations as follows:
   2.1. Get a list of locations that have SKU 00135777 using query_inventory.
   2.2. Iterate over the locations and estimate available overage using stock_demand_difference 
   2.3. If a certain location has overage, add this location, overage value, and shipping cost form this location to 
        San Jose to the list or possible source locations
3. Print the list of possible source locations or tell tat no such locations are available.

Python code:
```python
import pandas as pd
import numpy as np

# How many units will we need in San Jose   
overage_in_san_jose = stock_demand_difference(sku='00135777', location='San Jose', weeks=4)
if overage_in_san_jose >= 0:
    print_answer(f"We have overage of {{overage_in_san_jose}} units, not additional units needed.")
else:
    print_answer(f"We need {{-overage_in_san_jose}} more units next week.")

    # Determine whether these units can be sourced (rebalanced) from other locations
    locations = query_inventory(f"SELECT DISTINCT location FROM INVENTORY WHERE sku='00135777' AND location<>'San Jose'")['location'].tolist()
    rebalancing_options = []
    for location in locations:
        # Consider all locations except the destination
        if location is not 'San Jose': 
            # Calculate the expected overage in this location
            overage = stock_demand_difference(sku='00135777', location=location, weeks=4)
            if overage > 0:
                rebalancing_options.append( [location, overage, get_shipping_cost(location, 'San Jose') ] )

    if len(rebalancing_options) > 0:
        print_answer("These units can be rebalanced from the following locations:")
        rebalancing_options_df = pd.DataFrame(np.array(rebalancing_options), columns=['location', 'units_available', 'shipping_cost'])
        print_table(rebalancing_options_df)
    else:
        print_answer("There are no overages in other locations that can be used to source these units. These units should be procured.")
``` <<<




Task: How many units of SKU 00135777 should be procured for the San Jose store and what are the procurement options?

Business logic:
1. Estimate how many units will we need in San Jose (that is, estimate shortage) using stock_demand_difference.
2. Generate a SQL query for fetching the information about the suppliers, locations, and unit costs for SKU 00135777 from the suppliers database.
3. Execute the generated SQL query using query_suppliers.
4. Calculate the shipping cost for each supplier option using get_shipping_cost.
5. Print all supplier options with the corresponding shipping costs.

Python code:
```python
import pandas as pd
import numpy as np

# How many units will we need in San Jose   
overage_in_san_jose = stock_demand_difference(sku='00135777', location='San Jose', weeks=4)
if overage_in_san_jose >= 0:
    print_answer(f"We have overage of {{overage_in_san_jose}} units, not additional units needed.")
else:
    print_answer(f"We need {{-overage_in_san_jose}} more units next week.")

    # Determine the replenishment options
    replenishment_options_df = query_suppliers(f"SELECT supplier, location, unit_cost FROM SUPPLIERS WHERE sku='00135777'")
    replenishment_options_df['shipping_cost'] = replenishment_options_df['location'].apply(lambda x: get_shipping_cost(x, 'San Jose'))
    print_answer("These units can be procured from the following suppliers:")
    print_table(replenishment_options_df)
``` <<<




Task: What will be the demand for SKU 123213 in Atlanta over next 9 weeks?

Business logic:
1. Estimate weekly demand values for next 9 weeks using get_forecast.
2. Calculate the total demand for all 9 weeks.
3. Print weekly and total demand values.

Python code:
```python
import pandas as pd

# Get the demand forecast for the next 9 weeks
demand_forecast_df = get_forecast(sku="123213", location="Atlanta").head(9)
print_table(demand_forecast_df)
# Sum up the demand for the next 9 weeks
total_demand = demand_forecast_df['demand'].sum()
print_answer("The total demand for SKU 123213 in Atlanta over the next 9 weeks is" + total_demand)
``` <<<




Task: {query}

Business logic:
{logic_description}

Python code:
'''
)

CODER_PROMPT = PromptTemplate(input_variables=["logic_description", "api_docs", "query"], template=template)

CODER_PROMPT_STOP_SEQ = ["<<<"]