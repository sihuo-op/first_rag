import math

scores_dict = {'faithfulness': [float('nan')]}
score_list = scores_dict.get("faithfulness", [])
score = 0.0
if score_list and len(score_list) > 0:
    try:
        score_value = score_list[0]
        if score_value is not None and not (isinstance(score_value, float) and math.isnan(score_value)):
            score = float(score_value)
    except (IndexError, TypeError):
        score = 0.0

print(f"Score: {score}")
print("Fix works!")
