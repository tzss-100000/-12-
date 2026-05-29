import argparse
import os
import random
import sys
import urllib.request
import zipfile
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Bidirectional, Concatenate, Dense, Embedding, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

tf.get_logger().setLevel("ERROR")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DATASETS = {
    "cmn": {
        "name": "English-Chinese",
        "url": "http://www.manythings.org/anki/cmn-eng.zip",
        "file": "cmn.txt",
    },
    "fra": {
        "name": "English-French",
        "url": "http://www.manythings.org/anki/fra-eng.zip",
        "file": "fra.txt",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Character-level seq2seq translation experiment.")
    parser.add_argument("--dataset", choices=DATASETS.keys(), default="cmn")
    parser.add_argument("--algorithm", choices=["original", "improved"], default="improved")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--eval-size", type=int, default=50)
    parser.add_argument("--max-encoder-len", type=int, default=80)
    parser.add_argument("--max-decoder-len", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--show-summary", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def download_dataset(dataset_key, data_dir):
    info = DATASETS[dataset_key]
    dataset_dir = Path(data_dir) / dataset_key
    dataset_dir.mkdir(parents=True, exist_ok=True)
    text_path = dataset_dir / info["file"]
    zip_path = dataset_dir / f"{dataset_key}.zip"
    if not text_path.exists():
        if not zip_path.exists():
            print(f"Downloading {info['name']} dataset...")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "http://www.manythings.org/anki/",
            }
            request = urllib.request.Request(info["url"], headers=headers)
            with urllib.request.urlopen(request) as response, open(zip_path, "wb") as f:
                f.write(response.read())
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dataset_dir)
    return text_path


def load_pairs(text_path, samples, max_encoder_len, max_decoder_len):
    pairs = []
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            input_text, target_text = parts[0].strip(), parts[1].strip()
            if not input_text or not target_text:
                continue
            target_text = "\t" + target_text + "\n"
            if len(input_text) > max_encoder_len or len(target_text) > max_decoder_len:
                continue
            pairs.append((input_text, target_text))
            if len(pairs) >= samples:
                break
    if not pairs:
        raise RuntimeError("No usable sentence pairs were loaded.")
    input_texts, target_texts = zip(*pairs)
    return list(input_texts), list(target_texts)


def make_data(input_texts, target_texts, algorithm):
    input_chars = sorted({char for text in input_texts for char in text})
    target_chars = sorted({char for text in target_texts for char in text})
    max_encoder_len = max(len(text) for text in input_texts)
    max_decoder_len = max(len(text) for text in target_texts)
    offset = 0 if algorithm == "original" else 1
    input_token_index = {char: i + offset for i, char in enumerate(input_chars)}
    target_token_index = {char: i + offset for i, char in enumerate(target_chars)}
    num_encoder_tokens = len(input_chars)
    num_decoder_tokens = len(target_chars)
    if algorithm == "original":
        encoder_input_data = np.zeros((len(input_texts), max_encoder_len, num_encoder_tokens), dtype="float32")
        decoder_input_data = np.zeros((len(input_texts), max_decoder_len, num_decoder_tokens), dtype="float32")
        decoder_target_data = np.zeros((len(input_texts), max_decoder_len, num_decoder_tokens), dtype="float32")
        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, input_token_index[char]] = 1.0
            for t, char in enumerate(target_text):
                decoder_input_data[i, t, target_token_index[char]] = 1.0
                if t > 0:
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
        sample_weight = (decoder_target_data.sum(axis=-1) > 0).astype("float32")
    else:
        encoder_input_data = np.zeros((len(input_texts), max_encoder_len), dtype="int32")
        decoder_input_data = np.zeros((len(input_texts), max_decoder_len), dtype="int32")
        decoder_target_data = np.zeros((len(input_texts), max_decoder_len), dtype="int32")
        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t] = input_token_index[char]
            for t, char in enumerate(target_text):
                decoder_input_data[i, t] = target_token_index[char]
                if t > 0:
                    decoder_target_data[i, t - 1] = target_token_index[char]
        sample_weight = (decoder_target_data > 0).astype("float32")
    return {
        "encoder_input_data": encoder_input_data,
        "decoder_input_data": decoder_input_data,
        "decoder_target_data": decoder_target_data,
        "sample_weight": sample_weight,
        "input_token_index": input_token_index,
        "target_token_index": target_token_index,
        "reverse_target_char_index": {i: char for char, i in target_token_index.items()},
        "num_encoder_tokens": num_encoder_tokens,
        "num_decoder_tokens": num_decoder_tokens,
        "max_encoder_len": max_encoder_len,
        "max_decoder_len": max_decoder_len,
    }


def split_indices(total, val_split, seed):
    indices = np.arange(total)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split_at = int(total * (1 - val_split))
    split_at = min(max(split_at, 1), total - 1)
    return indices[:split_at], indices[split_at:]


def build_original_model(num_encoder_tokens, num_decoder_tokens, latent_dim):
    encoder_inputs = Input(shape=(None, num_encoder_tokens), name="encoder_input")
    encoder_lstm = LSTM(latent_dim, return_state=True, name="encoder_lstm")
    _, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]
    decoder_inputs = Input(shape=(None, num_decoder_tokens), name="decoder_input")
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation="softmax", name="decoder_output")
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    return model, {
        "encoder_inputs": encoder_inputs,
        "encoder_states": encoder_states,
        "decoder_lstm": decoder_lstm,
        "decoder_dense": decoder_dense,
    }


def build_improved_model(num_encoder_tokens, num_decoder_tokens, latent_dim, embedding_dim):
    encoder_inputs = Input(shape=(None,), name="encoder_input")
    encoder_embedding = Embedding(num_encoder_tokens + 1, embedding_dim, mask_zero=True, name="encoder_embedding")
    encoder_vectors = encoder_embedding(encoder_inputs)
    encoder_lstm = Bidirectional(LSTM(latent_dim, return_state=True, dropout=0.2), name="encoder_bilstm")
    _, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_vectors)
    state_h = Concatenate(name="encoder_state_h")([forward_h, backward_h])
    state_c = Concatenate(name="encoder_state_c")([forward_c, backward_c])
    encoder_states = [state_h, state_c]
    decoder_inputs = Input(shape=(None,), name="decoder_input")
    decoder_embedding = Embedding(num_decoder_tokens + 1, embedding_dim, mask_zero=True, name="decoder_embedding")
    decoder_vectors = decoder_embedding(decoder_inputs)
    decoder_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True, dropout=0.2, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_vectors, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens + 1, activation="softmax", name="decoder_output")
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model, {
        "encoder_inputs": encoder_inputs,
        "encoder_states": encoder_states,
        "decoder_embedding": decoder_embedding,
        "decoder_lstm": decoder_lstm,
        "decoder_dense": decoder_dense,
    }


def build_inference_models(algorithm, parts, data, latent_dim):
    encoder_model = Model(parts["encoder_inputs"], parts["encoder_states"])
    if algorithm == "original":
        decoder_inputs = Input(shape=(None, data["num_decoder_tokens"]), name="decoder_sampling_input")
        state_inputs = [Input(shape=(latent_dim,)), Input(shape=(latent_dim,))]
        decoder_outputs, state_h, state_c = parts["decoder_lstm"](decoder_inputs, initial_state=state_inputs)
    else:
        decoder_inputs = Input(shape=(None,), name="decoder_sampling_input")
        state_inputs = [Input(shape=(latent_dim * 2,)), Input(shape=(latent_dim * 2,))]
        decoder_vectors = parts["decoder_embedding"](decoder_inputs)
        decoder_outputs, state_h, state_c = parts["decoder_lstm"](decoder_vectors, initial_state=state_inputs)
    decoder_outputs = parts["decoder_dense"](decoder_outputs)
    decoder_model = Model([decoder_inputs] + state_inputs, [decoder_outputs, state_h, state_c])
    return encoder_model, decoder_model


def decode_sequence(input_seq, encoder_model, decoder_model, data, algorithm):
    states_value = encoder_model.predict(input_seq, verbose=0)
    start_index = data["target_token_index"]["\t"]
    if algorithm == "original":
        target_seq = np.zeros((1, 1, data["num_decoder_tokens"]), dtype="float32")
        target_seq[0, 0, start_index] = 1.0
    else:
        target_seq = np.array([[start_index]], dtype="int32")
    decoded_chars = []
    for _ in range(data["max_decoder_len"] + 1):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        if algorithm == "improved":
            output_tokens[0, -1, 0] = -1.0
        sampled_index = int(np.argmax(output_tokens[0, -1, :]))
        sampled_char = data["reverse_target_char_index"].get(sampled_index, "")
        if not sampled_char or sampled_char == "\n":
            break
        decoded_chars.append(sampled_char)
        if algorithm == "original":
            target_seq = np.zeros((1, 1, data["num_decoder_tokens"]), dtype="float32")
            target_seq[0, 0, sampled_index] = 1.0
        else:
            target_seq = np.array([[sampled_index]], dtype="int32")
        states_value = [h, c]
    return "".join(decoded_chars)


def evaluate(encoder_model, decoder_model, data, algorithm, input_texts, target_texts, eval_size):
    count = min(eval_size, len(input_texts))
    examples = []
    exact = 0
    similarities = []
    for i in range(count):
        input_seq = data["encoder_input_data"][i : i + 1]
        pred = decode_sequence(input_seq, encoder_model, decoder_model, data, algorithm)
        truth = target_texts[i][1:-1]
        exact += int(pred == truth)
        similarities.append(SequenceMatcher(None, pred, truth).ratio())
        if len(examples) < 10:
            examples.append({"input": input_texts[i], "target": truth, "prediction": pred})
    return {
        "exact_match": exact / count if count else 0.0,
        "avg_char_similarity": float(np.mean(similarities)) if similarities else 0.0,
        "examples": examples,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    text_path = download_dataset(args.dataset, args.data_dir)
    input_texts, target_texts = load_pairs(text_path, args.samples, args.max_encoder_len, args.max_decoder_len)
    data = make_data(input_texts, target_texts, args.algorithm)
    train_idx, val_idx = split_indices(len(input_texts), args.val_split, args.seed)
    print(f"Dataset: {DATASETS[args.dataset]['name']}")
    print(f"Samples: {len(input_texts)}")
    print(f"Input tokens: {data['num_encoder_tokens']}")
    print(f"Output tokens: {data['num_decoder_tokens']}")
    print(f"Max input length: {data['max_encoder_len']}")
    print(f"Max output length: {data['max_decoder_len']}")
    if args.algorithm == "original":
        model, parts = build_original_model(data["num_encoder_tokens"], data["num_decoder_tokens"], args.latent_dim)
    else:
        model, parts = build_improved_model(
            data["num_encoder_tokens"], data["num_decoder_tokens"], args.latent_dim, args.embedding_dim
        )
    if args.show_summary:
        model.summary()
    train_inputs = [data["encoder_input_data"][train_idx], data["decoder_input_data"][train_idx]]
    val_inputs = [data["encoder_input_data"][val_idx], data["decoder_input_data"][val_idx]]
    callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    history = model.fit(
        train_inputs,
        data["decoder_target_data"][train_idx],
        sample_weight=data["sample_weight"][train_idx],
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(val_inputs, data["decoder_target_data"][val_idx], data["sample_weight"][val_idx]),
        callbacks=callbacks,
        verbose=2,
    )
    encoder_model, decoder_model = build_inference_models(args.algorithm, parts, data, args.latent_dim)
    val_data = dict(data)
    val_data["encoder_input_data"] = data["encoder_input_data"][val_idx]
    val_input_texts = [input_texts[i] for i in val_idx]
    val_target_texts = [target_texts[i] for i in val_idx]
    scores = evaluate(encoder_model, decoder_model, val_data, args.algorithm, val_input_texts, val_target_texts, args.eval_size)
    metrics = {
        "dataset": args.dataset,
        "dataset_name": DATASETS[args.dataset]["name"],
        "algorithm": args.algorithm,
        "samples": len(input_texts),
        "epochs_ran": len(history.history["loss"]),
        "final_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_accuracy": float(history.history.get("accuracy", [0])[-1]),
        "final_val_accuracy": float(history.history.get("val_accuracy", [0])[-1]),
        **scores,
    }
    print("\nEvaluation")
    print(f"Final loss: {metrics['final_loss']:.4f}")
    print(f"Final val loss: {metrics['final_val_loss']:.4f}")
    print(f"Final accuracy: {metrics['final_accuracy']:.4f}")
    print(f"Final val accuracy: {metrics['final_val_accuracy']:.4f}")
    print(f"Exact match: {metrics['exact_match']:.4f}")
    print(f"Average character similarity: {metrics['avg_char_similarity']:.4f}")
    for example in metrics["examples"][:5]:
        print("-")
        print("Input:", example["input"])
        print("Target:", example["target"])
        print("Prediction:", example["prediction"])


if __name__ == "__main__":
    main()
