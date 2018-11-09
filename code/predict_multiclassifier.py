# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import json
import re

import modeling
import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None, "")

flags.DEFINE_string("output_file", None, "")

flags.DEFINE_string("layers", "-1,-2,-3,-4", "")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture."
)

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded."
)

flags.DEFINE_integer(
    "num_classes", 2,
    "The number of classes of the classification problem."
)

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models."
)

flags.DEFINE_integer("batch_size", 8, "Batch size for predictions.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("master", None,
                    "If using a TPU, the address of the master."
)

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use."
)

flags.DEFINE_bool(
    "use_one_hot_embeddings", False,
    "If True, tf.one_hot will be used for embedding lookups, otherwise "
    "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
    "since it is much faster."
)

class InputExample(object):
    def __init__(self, guid, text):
        self.guid = guid
        self.text = text

class InputFeatures(object):
    def __init__(self, guid, tokens, input_ids, input_mask, segment_ids):
        self.guid = guid
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

def input_fn_builder(features, seq_length):
    all_unique_ids = []
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []

    for feature in features:
        all_unique_ids.append(feature.guid)
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)

    def input_fn(params):
        batch_size = params['batch_size']
        num_examples = len(features)
        
        # Does not scale for large datasets. For large datasets, use
        # TFRecordReader instead
        d = tf.data.Dataset.from_tensor_slices({
            "unique_ids": tf.constant(all_unique_ids, shape = [num_examples], dtype = tf.int32),
            "input_ids": tf.constant(all_input_ids, shape = [num_examples, seq_length], dtype = tf.int32),
            "input_mask": tf.constant(all_input_mask, shape = [num_examples, seq_length], dtype = tf.int32),
            "segment_ids": tf.constant(all_segment_ids, shape = [num_examples, seq_length], dtype = tf.int32)
        })

        d = d.batch(batch_size = batch_size, drop_remainder = False)
        return d
    
    return input_fn

def create_model(bert_config, input_ids, input_mask, segment_ids,
                num_classes, use_one_hot_embeddings):
    '''
    Creates a multiclassification model
    '''
    model = modeling.BertModel(
        config = bert_config,
        is_training = is_training,
        input_ids = input_ids,
        input_mask = input_mask,
        token_type_ids = segment_ids,
        use_one_hot_embeddings = use_one_hot_embeddings
    )

    # Here we are doing classification task on the entire segment
    # If you want to use the token-level output, use model.get_sequence_output() instead
    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value
    output_weights = tf.get_variable(
        "output_weights", [num_classes, hidden_size],
        initializer = tf.truncated_normal_initializer(stddev = 0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [num_classes], initializer = tf.zeros_initializer()
    )
    with tf.variable_scope("loss"):
        logits = tf.matmul(output_layer, output_weights, transpose_b = True)
        logits = tf.nn.bias_add(logits, output_bias)
        probs = tf.nn.sigmoid(logits)
        return output_layer, probs

def model_fn_builder(bert_config, num_classes, init_checkpoint, 
                    use_tpu, use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        output_layer, probs = create_model(
            bert_config, input_ids, input_mask, segment_ids,
            num_classes, use_one_hot_embeddings
        )

        if mode != tf.estimator.ModeKeys.PREDICT:
            raise ValueError("Only PREDIT modes are supported: %s" % (mode))
        
        tvar = tf.trainable_variables()
        scaffold_fn = None
        (assignment_map, _) = modeling.get_assignment_map_from_checkpoint(
            tvars, init_checkpoint
        )
        if use_tpu:
            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()
            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        
        predictions = {
            "unique_id": unique_ids,
            "probs": probs,
            "doc_vector_rep": output_layer
        }
        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode = mode, predictions = predictions, scaffold_fn = scaffold_fn
        )
        return output_spec
    
    return model_fn

def convert_single_example(example, seq_length, tokenizer):
    tokens = tokenizer.tokenize(example.text)
    # Modifies tokens in place so that the total
    # length is less than the specified length.
    # Accounts for [CLS] and [SEP] with "-2"
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[0 : (max_seq_length - 2)]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens.  Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length
    input_ids = input_ids + [0] * (max_seq_length - len(input_ids))
    input_mask = input_mask + [0] * (max_seq_length - len(input_mask))
    segment_ids = segment_ids + [0] * (max_seq_length - len(segment_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    feature = InputFeatures(
        guid = example.guid
        tokens = tokens,
        input_ids = input_ids,
        input_mask = input_mask,
        segment_ids = segment_ids
    )
    return feature

def convert_examples_to_features(examples, seq_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        input_features = convert_single_example(example, seq_length, tokenizer)
        features.append(input_feature)
    return features

def predict_fn_builder():
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    tokenizer = tokenization.FullTokenizer(
        vocab_file = FLAGS.vocab_file, do_lower_case = FLAGS.do_lower_case
    )
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        master = FLAGS.master,
        tpu_config = tf.contrib.tpu.RunConfig(
            num_shards = FLAGS.num_tpu_cores,
            per_host_input_for_training = is_per_host
        )
    )
    model_fn = model_fn_builder(
        bert_config = bert_config,
        num_classes = FLAGS.num_classes,
        init_checkpoint = FLAGS.init_checkpoint,
        use_tpu = FLAGS.use_tpu, 
        use_one_hot_embeddings = FLAGS.use_one_hot_embeddings
    )
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu = FLAGS.use_tpu,
        model_fn = model_fn,
        config = run_config,
        predict_batch_size = FLAGS.batch_size
    )
    
    def predict_fn(examples):
        features = convert_examples_to_features(
            examples = examples,
            seq_length = FLAGS.max_seq_length,
            tokenizer = tokenizer
        )
        input_fn = input_fn_builder(
            features = features, seq_length = FLAGS.max_seq_length
        )

        results_to_return = []
        for result in estimator.predict(input_fn, yield_single_examples = True):
            unique_id = int(result['unique_id'])
            probs = result['probs']
            doc_vector_rep = result['doc_vector_rep']
            results_to_return.append({
                "id": unique_id,
                "probs": probs,
                "doc_vector_rep": doc_vector_rep
            })
        return results_to_return

    return predict_fn





