from langchain.prompts.prompt import PromptTemplate

template = (
    '''
You're a Lead Python Developer and lead an engineering team that helps supply chain managers to automate their tasks. 
A Junior Python Developer in your team received a request to write a Python script for answering a specific question
and wrote the initial version of the script. Your goal is to review this code and fix bugs. You MUST generate exactly 
one Python code listing that may contain helper functions. 

The script can use the following tools (python functions):

{api_docs}

=======================================================================================================================




Q: How many units of SKU 9988001 we will need in San Francisco during next 5 weeks?

Initial Python script:
```python
import pandas as pd

# Get the expected overage or shortage for the next 5 weeks
shortage = stock_demand_difference(sku="9988001", location="San Francisco", weeks=5)
print_answer(f"We will need {{shortage}} units of SKU 11001 in San Francisco during the next 5 weeks.")
``` 

Fixed Python script:
```python
# We need to get the expected overage for the next 5 weeks. This corresponds to the positive stock-demand difference.
delta = stock_demand_difference(sku="9988001", location="San Francisco", weeks=5)
if delta is 0:
    print_answer(f"Stock perfectly matches the demand for SKU 11001 in San Francisco during the next 5 weeks.")
elif delta < 0:
    print_answer(f"We will need {{-delta}} units of SKU 11001 in San Francisco during the next 5 weeks.")
else:
    print_answer(f"There is overage of {{-delta}} units of SKU 11001 in San Francisco. No additional units are needed.")
``` <<<




Q: How many units of SKU 665444 are available for rebalancing across all locations?

Initial Python script:
```python
import pandas as pd

# Check for overages of SKU 11001 in all locations
overages_df = query_inventory("SELECT sku, location, quantity FROM INVENTORY WHERE sku='11001' AND quantity > 0")
if overages_df.empty:
    print_answer("There are no overages of SKU 11001 in any locations.")
else:
    print_answer("There are overages of SKU 11001 in the following locations:")
    print_table(overages_df)
``` <<<

Fixed Python script:
```python
import pandas as pd

# We need to iterate over all locations and collect the overages available for rebalancing
locations = query_inventory(sql="SELECT DISTINCT location FROM INVENTORY WHERE sku='665444'")
rebalancing_options = []
for location in locations['location']:
    # Calculate the expected overage in this location
    overage = stock_demand_difference(sku='665444', location=location, weeks=4)
    if overage > 0:
        rebalancing_options.append([location, overage])

if len(rebalancing_options) > 0:
    print_answer("These units can be rebalanced from the following locations:")
    rebalancing_options_df = pd.DataFrame(rebalancing_options, columns=['location', 'units_available'])
    print_table(rebalancing_options_df)
else:
    print_answer("There are no overages in any location that can be used for rebalancing.")
``` <<<




Q: {query}

Initial Python script:
```python
{python_code}
```

Fixed Python script:
'''
)

CRITIC_PROMPT = PromptTemplate(input_variables=["query", "python_code", "api_docs"], template=template)

CRITIC_PROMPT_STOP_SEQ = ["<<<"]