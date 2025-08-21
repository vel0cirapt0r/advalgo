import pandas as pd
import numpy as np
from collections import defaultdict
import time
from openpyxl import Workbook


# Load data and prepare sequences
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Validate and enforce dtypes
    df['order_id'] = df['order_id'].astype('int32')
    df['product_id'] = df['product_id'].astype('int32')
    df['add_to_cart_order'] = df['add_to_cart_order'].astype('int32')
    df['reordered'] = df['reordered'].fillna(0).astype('int32')

    # Validate add_to_cart_order is strictly increasing per order
    def validate_order(group):
        orders = group['add_to_cart_order'].values
        if not np.all(np.diff(orders) > 0):
            raise ValueError(f"Non-increasing add_to_cart_order in order_id {group.name}")

    df.groupby('order_id').apply(validate_order)

    df = df.sort_values(['order_id', 'add_to_cart_order'])
    sequences = df.groupby('order_id')['product_id'].apply(list).to_dict()
    return df, sequences


# Build model from training prefixes
def build_model(train_df, alpha=0.1, top_n=100, window=5):
    unique_products = train_df['product_id'].unique()
    V_train = len(unique_products)
    num_orders_train = train_df['order_id'].nunique()
    avg_seq_len_train = train_df.groupby('order_id').size().mean()

    # Markov transitions
    pairs = train_df.assign(next_product=train_df.groupby('order_id')['product_id'].shift(-1)).dropna(
        subset=['next_product'])
    trans_counts = pairs.groupby(['product_id', 'next_product']).size()

    transitions = defaultdict(dict)
    for (current, next_), count in trans_counts.items():
        transitions[current][next_] = count

    for current in list(transitions):
        # Prune to top 100 by count first
        if len(transitions[current]) > top_n:
            sorted_neighbors = sorted(transitions[current].items(), key=lambda x: x[1], reverse=True)[:top_n]
            transitions[current] = dict(sorted_neighbors)

        # Now normalize with smoothing
        seen = len(transitions[current])
        total = sum(transitions[current].values())
        denom = total + alpha * seen
        for next_ in list(transitions[current]):
            transitions[current][next_] = (transitions[current][next_] + alpha) / denom

    # Co-visitation (symmetric within window)
    co_vis_counts = defaultdict(lambda: defaultdict(float))
    order_groups = train_df.groupby('order_id')
    num_groups = len(order_groups)
    processed = 0
    for name, g in order_groups:
        products = g['product_id'].values
        reords = g['reordered'].values
        n = len(products)
        for i in range(n):
            for j in range(i + 1, min(i + window + 1, n)):
                dist = j - i
                weight = 1.0 / dist
                # Forward boost if target reordered > 0
                boost_fwd = 1.15 if reords[j] > 0 else 1.0
                co_vis_counts[products[i]][products[j]] += weight * boost_fwd
                # Reverse boost if target reordered > 0
                boost_rev = 1.15 if reords[i] > 0 else 1.0
                co_vis_counts[products[j]][products[i]] += weight * boost_rev
        processed += 1
        if processed % 1000 == 0:
            print(f"Processed {processed}/{num_groups} orders for co-vis")

    co_vis = defaultdict(dict)
    for current in list(co_vis_counts):
        # Prune to top 100 by count first
        if len(co_vis_counts[current]) > top_n:
            sorted_neighbors = sorted(co_vis_counts[current].items(), key=lambda x: x[1], reverse=True)[:top_n]
            co_vis_counts[current] = dict(sorted_neighbors)

        # Normalize with smoothing
        seen = len(co_vis_counts[current])
        total = sum(co_vis_counts[current].values())
        denom = total + alpha * seen
        for next_ in list(co_vis_counts[current]):
            co_vis[current][next_] = (co_vis_counts[current][next_] + alpha) / denom

    # Popularity from last items in prefixes
    last_items = train_df.groupby('order_id')['product_id'].last()
    pop_counts = last_items.value_counts()
    total_pop = pop_counts.sum()
    popularity = {item: count / total_pop for item, count in pop_counts.items()} if total_pop > 0 else {}

    return transitions, co_vis, popularity, V_train, num_orders_train, avg_seq_len_train


# Predict next item
def predict_next(transitions, co_vis, popularity, sequence):
    if len(sequence) == 0:
        if not popularity:
            return None
        return max(popularity, key=popularity.get)

    tail = sequence[-3:]  # oldest to newest
    weights = [0.3, 0.6, 1.0][-len(tail):]  # weights correspond to tail order: lower for older, higher for newer

    candidates = set()
    markov_scores = defaultdict(float)
    co_scores = defaultdict(float)
    for i, t in enumerate(tail):
        w = weights[i]
        if t in transitions and transitions[t]:
            for j, p in transitions[t].items():
                markov_scores[j] += w * p
        if t in co_vis and co_vis[t]:
            for j, p in co_vis[t].items():
                co_scores[j] += w * p

    candidates = set(list(markov_scores.keys()) + list(co_scores.keys()))

    if not candidates:
        if not popularity:
            return None, 0, 0.0
        pred = max(popularity, key=popularity.get)
        return pred, 0, popularity.get(pred, 0.0)

    # Exclude already seen in sequence
    seen_set = set(sequence)
    candidates -= seen_set

    if not candidates:
        if not popularity:
            return None, 0, 0.0
        pred = max(popularity, key=popularity.get)
        return pred, 0, popularity.get(pred, 0.0)

    # Blend scores
    scores = {}
    for j in candidates:
        s_markov = markov_scores.get(j, 0.0)
        s_co = co_scores.get(j, 0.0)
        scores[j] = 0.6 * s_markov + 0.4 * s_co

    # For determinism: sort by score desc, then markov desc, then pop desc, then product_id asc
    sorted_candidates = sorted(scores.items(),
                               key=lambda x: (-x[1], -markov_scores.get(x[0], 0), -popularity.get(x[0], 0), x[0]))
    predicted = sorted_candidates[0][0]
    pred_score = scores[predicted]
    num_cand = len(candidates)

    return predicted, num_cand, pred_score


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
            predicted, num_cand, pred_score = predict_next(transitions, co_vis, popularity, input_seq)
            if predicted is None:
                continue
            used_backoff = num_cand == 0
            if used_backoff:
                num_backoff += 1
            if predicted == true_last:
                correct += 1
            predictions.append({
                'order_id': order_id,
                'k': k,
                'input_sequence': input_seq,
                'true_last': true_last,
                'predicted': predicted,
                'correct': predicted == true_last,
                'used_backoff': used_backoff,
                'num_candidates': num_cand,
                'score_of_prediction': pred_score
            })
            total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, predictions, total, num_backoff, correct


# Main function
def main(file_path, output_excel):
    np.random.seed(42)  # For any potential RNG
    pd.set_option('mode.chained_assignment', 'warn')

    start_load = time.time()
    df, sequences = load_data(file_path)
    load_time = time.time() - start_load

    print(f"Dataset size: {len(df)} rows, {len(sequences)} orders")

    results = {}
    all_predictions = []
    total_train_time = 0
    total_eval_time = 0
    total_correct = 0
    total_eligible = 0
    metrics_per_k = {}

    alpha = 0.1
    top_n = 100
    window = 5
    tail_weights = [0.3, 0.6, 1.0]
    blend_markov = 0.6
    blend_cov = 0.4

    for k in [1, 2, 3]:
        # Build train_df from prefixes
        train_start = time.time()
        train_groups = [g.iloc[:-k] for name, g in df.groupby('order_id') if len(g) > k]
        skipped = len(sequences) - len(train_groups)
        if not train_groups:
            continue
        train_df = pd.concat(train_groups)
        transitions, co_vis, popularity, V_train, num_orders_train, avg_seq_len_train = build_model(train_df, alpha,
                                                                                                    top_n, window)
        train_time = time.time() - train_start
        total_train_time += train_time

        # Evaluate
        eval_start = time.time()
        acc, preds, eligible, num_backoff, correct_k = evaluate(sequences, transitions, co_vis, popularity, k)
        eval_time = time.time() - eval_start
        total_eval_time += eval_time

        results[f'Accuracy for hide last {k}'] = acc
        coverage_model = 100 * (eligible - num_backoff) / eligible if eligible > 0 else 0
        print(f"For k={k}:")
        print(f"  Number of eligible users: {eligible}")
        print(f"  Skipped users (len <= k): {skipped}")
        print(f"  Coverage (% predictions via Model): {coverage_model:.2f}%")
        print(f"  Train time: {train_time:.2f}s")
        print(f"  Eval time: {eval_time:.2f}s")

        metrics_per_k[k] = {
            'eligible_users': eligible,
            'num_backoff': num_backoff,
            'coverage_model_%': coverage_model,
            'train_time_s': train_time,
            'eval_time_s': eval_time,
            'topN_pruning': top_n,
            'alpha': alpha,
            'blend_markov': blend_markov,
            'blend_cov': blend_cov,
            'tail_weights': str(tail_weights),
            'window': window,
            'num_products_train': V_train,
            'num_orders_train': num_orders_train,
            'avg_seq_len_train': avg_seq_len_train
        }

        all_predictions.extend(preds)
        total_correct += correct_k
        total_eligible += eligible

    macro_acc = np.mean(list(results.values()))
    micro_acc = total_correct / total_eligible if total_eligible > 0 else 0
    results['Overall accuracy (macro)'] = macro_acc
    results['Overall accuracy (micro)'] = micro_acc

    # Write to Excel
    write_start = time.time()
    wb = Workbook()
    ws_summary = wb.active
    ws_summary.title = 'Summary'
    ws_summary.append(['Metric', 'Value'])
    for key, val in results.items():
        ws_summary.append([key, val])
    ws_summary.append(['Load time', load_time])
    ws_summary.append(['Total train time', total_train_time])
    ws_summary.append(['Total eval time', total_eval_time])
    ws_summary.append(['Write time', ''])  # Placeholder, update later

    # Params block
    ws_summary.append([])
    ws_summary.append(['Params', ''])
    ws_summary.append(['alpha', alpha])
    ws_summary.append(['window', window])
    ws_summary.append(['topN_neighbors', top_n])
    ws_summary.append(['tail_weights', str(tail_weights)])
    ws_summary.append(['blend', f'{blend_markov} Markov / {blend_cov} Co-vis'])

    ws_preds = wb.create_sheet('Predictions')
    headers = ['order_id', 'k', 'input_sequence', 'true_last', 'predicted', 'correct', 'used_backoff', 'num_candidates',
               'score_of_prediction']
    ws_preds.append(headers)
    for p in all_predictions:
        input_str = ','.join(map(str, p['input_sequence']))
        ws_preds.append(
            [p['order_id'], p['k'], input_str, p['true_last'], p['predicted'], p['correct'], p['used_backoff'],
             p['num_candidates'], p['score_of_prediction']])

    for k in [1, 2, 3]:
        if k in metrics_per_k:
            ws_metrics = wb.create_sheet(f'Metrics_k={k}')
            ws_metrics.append(['Metric', 'Value'])
            for key, val in metrics_per_k[k].items():
                ws_metrics.append([key, val])

    wb.save(output_excel)
    write_time = time.time() - write_start

    # Update write time in summary
    wb = Workbook(output_excel)  # Reload to update
    ws_summary = wb['Summary']
    for row in ws_summary.iter_rows():
        if row[0].value == 'Write time':
            row[1].value = write_time
            break
    wb.save(output_excel)

    print(f"Load time: {load_time:.2f}s")
    print(f"Total train time: {total_train_time:.2f}s")
    print(f"Total eval time: {total_eval_time:.2f}s")
    print(f"Write time: {write_time:.2f}s")
    print(f"Results saved to {output_excel}")


if __name__ == "__main__":
    input_csv = "order-Product_prior.csv"
    output_excel = "results.xlsx"
    main(input_csv, output_excel)
