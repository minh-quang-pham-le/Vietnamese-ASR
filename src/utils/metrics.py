import jiwer

def compute_wer(pred):
    pred_logits = pred.predictions
    pred_ids = pred_logits.argmax(-1)
    pred_str = pred.label_ids
    pred_str = [p for p in pred_str if p != -100]

    wer = jiwer.wer(pred.label_ids, pred_ids)
    return {"wer": wer}