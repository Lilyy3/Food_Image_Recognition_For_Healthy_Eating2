import csv

def load_nutrition_database(csv_path="food101_nutrition.csv"):
    """Load Food-101 nutrition CSV as a dictionary keyed by food_class."""
    food_db = {}

    with open(csv_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            food_class = row["food_class"]

            # Convert numeric fields
            numeric_fields = [
                "calories_kcal",
                "protein_g",
                "carbs_g",
                "fat_g",
                "sodium_mg",
                "sugar_g",
                "fiber_g",
            ]

            for field in numeric_fields:
                row[field] = float(row[field])

            food_db[food_class] = row

    return food_db


def generate_advice(food_info, confidence):
    """
    Rule-based advice component.
    Input:
        food_info: one row from food101_nutrition.csv
        confidence: model confidence between 0 and 1
    Output:
        dictionary containing health level, notes, and recommendation
    """

    calories = food_info["calories_kcal"]
    protein = food_info["protein_g"]
    fat = food_info["fat_g"]
    carbs = food_info["carbs_g"]
    sodium = food_info["sodium_mg"]
    sugar = food_info["sugar_g"]
    fiber = food_info["fiber_g"]

    warnings = []

    # Rule 1: Low confidence warning
    if confidence < 0.60:
        return {
            "health_level": "uncertain",
            "health_notes": "The model confidence is low.",
            "recommendation": "Prediction is uncertain. Use manual verification before relying on the advice."
        }

    if confidence < 0.75:
        warnings.append("Prediction confidence is moderate; verify the food label if possible.")

    # Rule 2: High calorie
    if calories >= 650:
        warnings.append("High-calorie meal.")

    # Rule 3: High fat
    if fat >= 25:
        warnings.append("High-fat meal.")

    # Rule 4: High sodium
    if sodium >= 900:
        warnings.append("High-sodium meal.")

    # Rule 5: High sugar
    if sugar >= 20:
        warnings.append("High-sugar food.")

    # Rule 6: High carbohydrate
    if carbs >= 65:
        warnings.append("High-carbohydrate meal.")

    # Rule 7: Positive nutrition signals
    positives = []
    if protein >= 25:
        positives.append("Good protein source.")
    if fiber >= 6:
        positives.append("Good fiber source.")
    if calories <= 250 and fat <= 10 and sugar <= 10:
        positives.append("Relatively light option.")

    # Final decision
    risk_count = sum([
        calories >= 650,
        fat >= 25,
        sodium >= 900,
        sugar >= 20,
        carbs >= 65
    ])

    if risk_count >= 3:
        health_level = "less_healthy"
        recommendation = "Limit intake or choose a smaller portion. Pair with vegetables or a lighter side."
    elif risk_count >= 1:
        health_level = "moderate"
        recommendation = "Acceptable occasionally, but portion control is recommended."
    else:
        health_level = "healthier"
        recommendation = "Generally a better option compared with high-calorie or high-fat meals."

    notes = warnings + positives

    return {
        "health_level": health_level,
        "health_notes": " ".join(notes) if notes else "No major nutrition warning detected.",
        "recommendation": recommendation
    }


def get_food_result(food_label, confidence, csv_path="food101_nutrition.csv"):
    """Return nutrition + rule-based recommendation for a predicted food label."""
    food_db = load_nutrition_database(csv_path)

    if food_label not in food_db:
        return {
            "food": food_label,
            "confidence": confidence,
            "error": "Food class not found in nutrition database."
        }

    food_info = food_db[food_label]
    advice = generate_advice(food_info, confidence)

    return {
        "food": food_info["display_name"],
        "confidence": round(confidence * 100, 2),
        "calories_kcal": food_info["calories_kcal"],
        "protein_g": food_info["protein_g"],
        "carbs_g": food_info["carbs_g"],
        "fat_g": food_info["fat_g"],
        "sodium_mg": food_info["sodium_mg"],
        "sugar_g": food_info["sugar_g"],
        "fiber_g": food_info["fiber_g"],
        "health_level": advice["health_level"],
        "health_notes": advice["health_notes"],
        "recommendation": advice["recommendation"]
    }
