import csv
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("models/finetuned")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Evaluation function
def evaluate_function(function, expected_if):
    masked_function = function.replace(expected_if, "<extra_id_0>")
    inputs = tokenizer(masked_function, return_tensors="pt")
    outputs = model.generate(inputs.input_ids)
    predicted_if = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return predicted_if

# Load test dataset
test_functions = [...]  # Load test functions here
expected_ifs = [...]    # Load expected if conditions

# Prepare results for CSV
results = []
for i, function in enumerate(test_functions):
    predicted_if = evaluate_function(function, expected_ifs[i])
    results.append({
        "Input": function,
        "Expected If Condition": expected_ifs[i],
        "Predicted If Condition": predicted_if,
        "Prediction Score": some_score_function(predicted_if, expected_ifs[i])  # Implement this scoring function
    })

# Save results to CSV
keys = results[0].keys()
with open("generated-testset.csv", "w", newline="") as output_file:
    dict_writer = csv.DictWriter(output_file, fieldnames=keys)
    dict_writer.writeheader()
    dict_writer.writerows(results)
