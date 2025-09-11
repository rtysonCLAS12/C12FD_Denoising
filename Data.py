import numpy as np

def read_sector_file(filename):
    """
    Reads one sector CSV file written by DenoiseExtractor.
    Returns a list of (all_hits, tb_hits) arrays.
    """
    events = []
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip() != ""]  # remove empty lines

    # process every 72 lines (36 + 36)
    n_lines_per_event = 36*2
    for i in range(0, len(lines), n_lines_per_event):
        block = lines[i:i+n_lines_per_event]
        if len(block) < n_lines_per_event:
            print(f"Warning: incomplete block of {len(block)} lines, skipping")
            continue
        all_hits = np.array([[int(x) for x in row.split(",")] for row in block[:36]])
        tb_hits  = np.array([[int(x) for x in row.split(",")] for row in block[36:]])
        events.append((all_hits, tb_hits))

    return events

def compute_efficiency_background(all_hits, tb_hits, predictions, n_thresholds=100,return_best=False):
    thresholds = np.linspace(0, 1, n_thresholds)
    signal_eff = np.zeros(n_thresholds)
    background_rej = np.zeros(n_thresholds)

    for i, thr in enumerate(thresholds):
        pred_mask = predictions >= thr

        # Signal efficiency: fraction of true track hits (tb_hits) predicted
        signal_hits = tb_hits > 0
        n_signal = signal_hits.sum()
        signal_eff[i] = np.logical_and(pred_mask, signal_hits).sum() / max(n_signal, 1)

        # Background rejection: fraction of background hits (all_hits - tb_hits) NOT predicted
        background_hits = np.logical_and(all_hits > 0, tb_hits == 0)
        n_background = background_hits.sum()
        background_rej[i] = np.logical_and(~pred_mask, background_hits).sum() / max(n_background, 1)

    # Find threshold with max signal efficiency while background rejection >= 90%
    valid = np.where(background_rej >= 0.9)[0]
    if len(valid) > 0:
        best_idx = valid[np.argmax(signal_eff[valid])]
        best_threshold = thresholds[best_idx]
        best_eff = signal_eff[best_idx]
        best_rej = background_rej[best_idx]
    else:
        best_threshold = 0.5  # fallback if no threshold meets the criteria
        best_eff=-1
        best_rej=-1

    if return_best==True:
      return best_eff, best_rej, best_threshold
    else:
      print(f"\n\nChosen threshold: {best_threshold:.3f}")
      print(f"Best Signal Efficiency: {best_eff:.3f}")
      print(f"Best Background Rejection: {best_rej:.3f}")
      return thresholds, signal_eff, background_rej, best_threshold