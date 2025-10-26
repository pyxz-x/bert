"""
-*- coding: utf-8 -*-
@Time : 2025-10-27 0:57
@File : predict_ckpt.py.py
@Author : 大漂亮lzh
@Project : bert_model
"""
import os
import pickle
import tensorflow.compat.v1 as tf
import modeling
import tokenization

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ========== 1. 路径 ==========
CKPT_DIR = r'E:\PythonProject\bert_model\output'
bert_config_path = os.path.join(CKPT_DIR, 'bert_config.json')
vocab_path = os.path.join(CKPT_DIR, 'vocab.txt')
init_checkpoint = os.path.join(CKPT_DIR, 'model.ckpt-103')
label_map_path = os.path.join(CKPT_DIR, 'label2id.pkl')

# ========== 2. 加载 label 映射 ==========
with open(label_map_path, 'rb') as f:
    label2id = pickle.load(f)
id2label = {i: l for l, i in label2id.items()}
num_labels = len(label2id)

# ========== 3. 分词器 ==========
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=True)
max_seq_length = 128


# ========== 4. 建图（与训练时完全一致） ==========
def create_session():
    tf.reset_default_graph()
    config = modeling.BertConfig.from_json_file(bert_config_path)

    input_ids = tf.placeholder(tf.int32, [None, max_seq_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, [None, max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, max_seq_length], name='segment_ids')

    # 复用训练时 create_model 的推理分支
    model = modeling.BertModel(
        config=config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)

    output_layer = model.get_pooled_output()  # [batch, hidden]
    hidden_size = output_layer.shape[-1].value

    # 与训练时同名变量
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable(
        "output_bias", [num_labels],
        initializer=tf.zeros_initializer())

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1, name='probabilities')
    pred_id = tf.argmax(logits, axis=-1, output_type=tf.int32, name='pred_id')

    # 恢复权重
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, init_checkpoint)
    print('✅ 权重恢复成功')
    return sess, {'i': input_ids, 'm': input_mask, 's': segment_ids}, pred_id, probabilities


# ========== 5. 特征转换 ==========
def convert_single(text):
    tokens = ['[CLS]'] + tokenizer.tokenize(text) + ['[SEP]']
    ids = tokenizer.convert_tokens_to_ids(tokens)
    mask = [1] * len(ids)
    seg = [0] * len(ids)
    while len(ids) < max_seq_length:
        ids.append(0);
        mask.append(0);
        seg.append(0)
    return ids[:max_seq_length], mask[:max_seq_length], seg[:max_seq_length]


# ========== 6. 预测 ==========
def predict(sess, feed, pred_id, probs, text):
    ids, mask, seg = convert_single(text)
    idx, pb = sess.run([pred_id, probs],
                       feed_dict={feed['i']: [ids],
                                  feed['m']: [mask],
                                  feed['s']: [seg]})
    return id2label[idx[0]], float(pb[0][idx[0]])


# ========== 7. 命令行交互 ==========
if __name__ == '__main__':
    sess, feed, pred_id, probs = create_session()
    while True:
        sent = input('\n请输入句子（q 退出）：').strip()
        if sent.lower() == 'q':
            break
        if not sent:
            continue
        label, score = predict(sess, feed, pred_id, probs, sent)
        print(f'预测标签：{label}  置信度：{score:.4f}')
