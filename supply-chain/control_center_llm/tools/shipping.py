def get_shipping_cost(source: str, destination: str) -> float:
    return (hash(source) + hash(destination)) % 10
