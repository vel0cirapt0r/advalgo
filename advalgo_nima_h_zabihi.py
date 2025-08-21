import pandas as pd
import numpy as np
from collections import defaultdict
from openpyxl import Workbook


# Step 1: Read the CSV and prepare sequences
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Group by order_id and sort by add_to_cart_order
    grouped = df.sort_values(['order_id', 'add_to_cart_order']).groupby('order_id')
    sequences = grouped['product_id'].apply(list).to_dict()
    return sequences


# Step 2: Build Markov transition table and global popularity
def build_model(sequences):
    transitions = defaultdict(lambda: defaultdict(int))
    popularity = defaultdict(int)

    for seq in sequences.values():
        for i in range(len(seq) - 1):
            current = seq[i]
            next_item = seq[i + 1]
            transitions[current][next_item] += 1
            popularity[next_item] += 1

    # Normalize transitions to probabilities
    for current in transitions:
        total = sum(transitions[current].values())
        for next_item in transitions[current]:
            transitions[current][next_item] /= total

    # Global popularity fallback
    total_pop = sum(popularity.values())
    popularity = {k: v / total_pop for k, v in popularity.items()}

    return transitions, popularity


# Step 3: Predict next item
def predict_next(transitions, popularity, sequence):
    if not sequence:
        # Fallback to most popular
        return max(popularity, key=popularity.get)

    last_item = sequence[-1]
    if last_item in transitions and transitions[last_item]:
        # Choose the one with highest probability
        return max(transitions[last_item], key=transitions[last_item].get)
    else:
        # Fallback to global popularity
        return max(popularity, key=popularity.get)


# Step 4: Evaluate for different k
def evaluate(sequences, transitions, popularity, k):
    correct = 0
    total = 0
    predictions = []

    for order_id, seq in sequences.items():
        if len(seq) > k:
            input_seq = seq[:-k]
            true_last = seq[-1]
            predicted = predict_next(transitions, popularity, input_seq)
            if predicted == true_last:
                correct += 1
            predictions.append({
                'order_id': order_id,
                'input_sequence': input_seq,
                'true_last': true_last,
                'predicted': predicted,
                'correct': predicted == true_last
            })
            total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, predictions


# Main function
def main(file_path, output_excel):
    sequences = load_data(file_path)

    # Use all sequences for training (as per simple model)
    transitions, popularity = build_model(sequences)

    # Evaluate for k=1,2,3
    results = {}
    all_predictions = []

    for k in [1, 2, 3]:
        acc, preds = evaluate(sequences, transitions, popularity, k)
        results[f'Accuracy for hide last {k}'] = acc
        for p in preds:
            p['k'] = k
        all_predictions.extend(preds)

    overall_acc = np.mean(list(results.values()))
    results['Overall accuracy'] = overall_acc

    # Save to Excel
    wb = Workbook()

    # Summary sheet
    ws_summary = wb.active
    ws_summary.title = 'Summary'
    ws_summary.append(['Metric', 'Value'])
    for key, val in results.items():
        ws_summary.append([key, val])

    # Predictions sheet
    ws_preds = wb.create_sheet('Predictions')
    headers = ['order_id', 'k', 'input_sequence', 'true_last', 'predicted', 'correct']
    ws_preds.append(headers)
    for p in all_predictions:
        ws_preds.append([p['order_id'], p['k'], ','.join(map(str, p['input_sequence'])), p['true_last'], p['predicted'],
                         p['correct']])

    wb.save(output_excel)
    print(f"Results saved to {output_excel}")


# Run the script
if __name__ == "__main__":
    input_csv = "order-Product_prior.csv"
    output_excel = "results.xlsx"
    main(input_csv, output_excel)
