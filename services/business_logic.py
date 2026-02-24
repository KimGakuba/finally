def apply_business_rules(prediction, rainfall):


    if rainfall > 70:
        return "Delay irrigation (Rain expected)"

    if prediction == "High":
        return "Irrigate immediately"

    elif prediction == "Medium":
        return "Moderate irrigation recommended"

    else:
        return "Low irrigation needed"