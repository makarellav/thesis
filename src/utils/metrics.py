def calculate_gini(wealths: list[float]) -> float:
    if not wealths or len(wealths) == 0:
        return 0.0

    sorted_wealths = sorted(wealths)
    n = len(sorted_wealths)
    total_wealth = sum(sorted_wealths)

    if total_wealth == 0:
        return 0.0

    numerator = 2 * sum((i + 1) * w for i, w in enumerate(sorted_wealths))
    return float(numerator / (n * total_wealth) - (n + 1) / n)
