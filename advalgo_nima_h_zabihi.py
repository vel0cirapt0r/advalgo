import pandas as pd
import numpy as np
from collections import defaultdict
import time
from openpyxl import Workbook


# Load data and prepare sequences
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.sort_values(['order_id', 'add_to_cart_order'])
    sequences = df.groupby('order_id')['product_id'].apply(list).to_dict()
    unique_products = df['product_id'].unique()
    V = len(unique_products)
    return df, sequences, V


# Build model from training prefixes
def build_model(train_df, V, alpha=0.1):
    # Markov transitions
    pairs = train_df.assign(next_product=train_df.groupby('order_id')['product_id'].shift(-1)).dropna(
        subset=['next_product'])
    trans_counts = pairs.groupby(['product_id', 'next_product']).size()
    group_sums = trans_counts.groupby('product_id').sum()

    # Build transitions as dict of dicts: probs
    transitions = defaultdict(dict)
    for (current, next_), count in trans_counts.items():
        transitions[current][next_] = count

    for current in list(transitions):
        total = group_sums.get(current, 0)
        denom = total + alpha * V
        for next_ in list(transitions[current]):
            transitions[current][next_] = (transitions[current][next_] + alpha) / denom
        # Cap to top 100 neighbors
        if len(transitions[current]) > 100:
            sorted_neighbors = sorted(transitions[current].items(), key=lambda x: x[1], reverse=True)[:100]
            transitions[current] = dict(sorted_neighbors)

    # Co-visitation (directed, future within window)
    co_vis_counts = defaultdict(lambda: defaultdict(float))
    for name, g in train_df.groupby('order_id'):
        products = g['product_id'].values
        reords = g['reordered'].values
        n = len(products)
        for i in range(n):
            for j in range(i + 1, min(i + 6, n)):
                dist = j - i
                weight = 1.0 / dist
                boost = 1.15 if reords[j] > reords[i] else 1.0
                co_vis_counts[products[i]][products[j]] += weight * boost

    co_vis = defaultdict(dict)
    co_vis_sums = {}
    for current in co_vis_counts:
        co_vis_sums[current] = sum(co_vis_counts[current].values())

    for current in list(co_vis_counts):
        total = co_vis_sums.get(current, 0)
        denom = total + alpha * V
        for next_ in list(co_vis_counts[current]):
            co_vis[current][next_] = (co_vis_counts[current][next_] + alpha) / denom
        if len(co_vis[current]) > 100:
            sorted_neighbors = sorted(co_vis[current].items(), key=lambda x: x[1], reverse=True)[:100]
            co_vis[current] = dict(sorted_neighbors)

    # Popularity from last items in prefixes
    last_items = train_df.groupby('order_id')['product_id'].last()
    pop_counts = last_items.value_counts()
    popularity = (pop_counts / pop_counts.sum()).to_dict()

    return transitions, co_vis, popularity


# Predict next item
def predict_next(transitions, co_vis, popularity, sequence):
    if len(sequence) == 0:
        return max(popularity, key=popularity.get)

    tail = sequence[-3:]
    weights = [0.3, 0.6, 1.0][-len(tail):]

    candidates = set()
    has_model = False
    for t in tail:
        if t in transitions and transitions[t]:
            candidates.update(transitions[t].keys())
            has_model = True
        if t in co_vis and co_vis[t]:
            candidates.update(co_vis[t].keys())
            has_model = True

    if not has_model:
        return max(popularity, key=popularity.get)

    scores = defaultdict(float)
    for j in candidates:
        s_markov = 0.0
        s_co = 0.0
        for i, t in enumerate(tail):
            w = weights[i]
            p_m = transitions.get(t, {}).get(j, 0.0)
            p_c = co_vis.get(t, {}).get(j, 0.0)
            s_markov += w * p_m
            s_co += w * p_c
        scores[j] = 0.6 * s_markov + 0.4 * s_co

    predicted = max(scores, key=scores.get)
    return predicted


# Evaluate for a given k
def evaluate(sequences, transitions, co_vis, popularity, k):
    correct = 0
    total = 0
    num_backoff = 0
    predictions = []

    for order_id, seq in sequences.items():
        if len(seq) > k:
            input_seq = seq[:-k]
            true_last = seq[-1]
            predicted = predict_next(transitions, co_vis, popularity, input_seq)
            # Check if used backoff
            if len(input_seq) == 0:
                used_backoff = True
            else:
                tail = input_seq[-3:]
                has_model = any((t in transitions and transitions[t]) or (t in co_vis and co_vis[t]) for t in tail)
                used_backoff = not has_model
            if used_backoff:
                num_backoff += 1
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
    return accuracy, predictions, total, num_backoff


# Main function
def main(file_path, output_excel):
    start_load = time.time()
    df, sequences, V = load_data(file_path)
    load_time = time.time() - start_load

    results = {}
    all_predictions = []
    total_train_time = 0
    total_eval_time = 0

    for k in [1, 2, 3]:
        # Build train_df from prefixes
        train_start = time.time()
        train_groups = [g.iloc[:-k] for name, g in df.groupby('order_id') if len(g) > k]
        if not train_groups:
            continue
        train_df = pd.concat(train_groups)
        transitions, co_vis, popularity = build_model(train_df, V)
        train_time = time.time() - train_start
        total_train_time += train_time

        # Evaluate
        eval_start = time.time()
        acc, preds, eligible, num_backoff = evaluate(sequences, transitions, co_vis, popularity, k)
        eval_time = time.time() - eval_start
        total_eval_time += eval_time

        results[f'Accuracy for hide last {k}'] = acc
        coverage_model = 100 * (eligible - num_backoff) / eligible if eligible > 0 else 0
        print(f"For k={k}:")
        print(f"  Number of eligible users: {eligible}")
        print(f"  Coverage (% predictions via Model): {coverage_model:.2f}%")
        print(f"  Train time: {train_time:.2f}s")
        print(f"  Eval time: {eval_time:.2f}s")

        for p in preds:
            p['k'] = k
        all_predictions.extend(preds)

    overall_acc = np.mean(list(results.values()))
    results['Overall accuracy'] = overall_acc

    # Write to Excel
    write_start = time.time()
    wb = Workbook()
    ws_summary = wb.active
    ws_summary.title = 'Summary'
    ws_summary.append(['Metric', 'Value'])
    for key, val in results.items():
        ws_summary.append([key, val])

    ws_preds = wb.create_sheet('Predictions')
    headers = ['order_id', 'k', 'input_sequence', 'true_last', 'predicted', 'correct']
    ws_preds.append(headers)
    for p in all_predictions:
        input_str = ','.join(map(str, p['input_sequence']))
        ws_preds.append([p['order_id'], p['k'], input_str, p['true_last'], p['predicted'], p['correct']])

    wb.save(output_excel)
    write_time = time.time() - write_start

    print(f"Load time: {load_time:.2f}s")
    print(f"Total train time: {total_train_time:.2f}s")
    print(f"Total eval time: {total_eval_time:.2f}s")
    print(f"Write time: {write_time:.2f}s")
    print(f"Results saved to {output_excel}")


if __name__ == "__main__":
    input_csv = "order-Product_prior.csv"
    output_excel = "results.xlsx"
    main(input_csv, output_excel)
