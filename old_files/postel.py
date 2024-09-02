def interpret_mood(text):
    if "optimistic" in text or "happy" in text or "great" in text:
        return "high morale"
    elif "okay" in text or "neutral" in text:
        return "moderate morale"
    else:
        return "unknown morale"


# Example inputs
input1 = "Team USA is feeling very optimistic after their recent win."
input2 = "Team USA seems happy."
input3 = "Team USA is doing okay."

# Applying the function
output1 = interpret_mood(input1)  # Returns "high morale"
output2 = interpret_mood(input2)  # Returns "high morale"
output3 = interpret_mood(input3)  # Returns "moderate morale"
