# Author: jadiel de armas
"""BERT finetuning runner."""
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import json
import os
import modeling
import optimization
import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task."
)

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture."
)

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None, "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written"
)

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text.  Shold be true for uncased "
    "models and False for cased models."
)

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded."
)

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
)

flags.DEFINE_integer("save_checkpoints_steps", 1000,
    "How ofter to save the model checkpoint."
)

flags.DEFINE_integer("iterations_per_loop", 1000,
    "How many steps to make in each estimator call."
)

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url."
)

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata."
)

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata."
)

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use."
)

class InputExample(object):
    '''
    A single training/test example for simple sequence classification
    '''
    def __init__(self, guid, text, labels):
        '''
        Parameters:
        -guid: Unique id of the example
        -text: The untokenized text of the text sequence
        -labels: (Optional) [string]. The labels of the example
        '''
        self.guid = guid
        self.text = text
        self.labels = labels

class InputFeatures(object):
    '''
    A single set of features of data.
    '''
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

class DataProcessor(object):
    '''
    Base class for data converters for sequence multiclassification datasets
    '''

    def get_train_examples(self):
        raise NotImplementedError()
    
    def get_val_examples(self):
        raise NotImplementedError()
    
    def get_labels(self):
        raise NotImplementedError()

class JsonDataProcessor(DataProcessor):
    def __init__(self, data_folder):
        self._data_folder = data_folder
    
    def get_train_examples(self):
        entries = self._read_json(os.path.join(self._data_folder, "train.json"))
        examples = []
        for (i, entry) in enumerate(entries):
            guid = "train-%d" % (i)
            text = tokenization.convert_to_unicode(entry['text'])
            labels = entry['labels']
            labels = [tokenization.convert_to_unicode(label) for label in labels]
            examples.append(InputExample(guid = guid, text = text, labels = labels))
        return examples
    
    def get_val_examples(self):
        entries = self._read_json(os.path.joins(self._data_folder, "eval.json"))
        examples = []
        for (i, entry) in enumerate(entries):
            guid = "val-%d" % (i)
            text = tokenization.convert_to_unicode(entry['text'])
            labels = entry['labels']
            labels = [tokenization.convert_to_unicode(label) for label in labels]
            examples.append(InputExample(guid = guid, text = text, labels = labels))
        return examples
    
    def get_labels(self):
        labels_set = set()
        for file_name in ['train.json', 'eval.json']:
            entries = self._read_json(os.path.join(self._data_folder, file_name))
            for entry in entries:
                labels = entry['labels']
                [labels_set.add(label) for label in labels]

        return list(labels_set)
    
    @classmethod
    def _read_json(cls, input_file, text_field = "text", labels_field = "labels"):
        with tf.gfile.Open(input_file, "r") as f:
            json_data = json.load(f)
            entries = []
            for doc in json_data:
                text = doc.get(text_field, "")
                labels = doc.get(labels_field, [])
                entries.append({
                    'text': text,
                    'labels': labels
                })
            return entries
        
class TsvDataProcessor(DataProcessor):
    def __init__(self, data_folder):
        self._data_folder = data_folder
    
    def get_train_examples(self):
        lines = self._read_tsv(os.path.join(self._data_folder, "train.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "train-%d" % (i)
            text = tokenization.convert_to_unicode(line[0])
            labels_s = tokenization.convert_to_unicode(line[1])
            labels = labels_s.split(",")
            examples.append(InputExample(guid = guid, text = text, labels = labels))
        return examples
    
    def get_val_examples(self):
        lines = self._read_tsv(os.path.join(self._data_folder, "eval.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "val-%d" % (i)
            text = tokenization.convert_to_unicode(line[0])
            labels_s = tokenization.convert_to_unicode(line[1])
            labels = labels_s.split(",")
            examples.append(InputExample(guid = guid, text = text, labels = labels))
        return examples
    
    def get_labels(self):
        labels_set = set()
        for file_name in ["train.tsv", "eval.tsv"]:
            lines = self._read_tsv(os.path.join(self._data_folder, file_name))
            for (i, line) in enumerate(lines):
                labels_s = tokenization.convert_to_unicode(line[1])
                labels = labels_s.split(",")
                [labels_set.add(label) for label in labels]
        return list(labels_set)

    @classmethod
    def _read_tsv(cls, input_file, quotechar = None, has_header = True):
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar = quotechar)
            lines = []
            
            counter = -1
            for line in reader:
                counter += 1
                if has_header and counter > 0:
                    lines.append(line)
                else:
                    lines.append(line)
            return lines

def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    
    tokens = tokenizer.tokenize(example.text)
    # Modifies tokens in place so that the total
    # length is less than the specified length.
    # Accounts for [CLS] and [SEP] with "-2"
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[0 : (max_seq_length - 2)]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    segment_ids = [0] * (len(tokens) + 2)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens.  Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length
    input_ids = input_ids + [0] * (max_seq_length - len(input_ids))
    input_mask = input_mask + [0] * (max_seq_length - len(input_mask))
    segment_ids = segment_ids + [0] * (max_seq_length - len(segment_ids))
    assert len(input_ids) == max_seq_length
    
    label_ids = [0] * len(label_list)
    for idx, label in enumerate(example.labels):
        label_ids[label_map[label]] = 1

    # Log the first few samples for visual inspection.
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("labels: %s (ids = %s)" % (str(example.labels), str(label_ids)))
    
    feature = InputFeatures(
        input_ids = input_ids,
        input_mask = input_mask,
        segment_ids = segment_ids,
        label_ids = label_ids
    )
    return feature
    
def filed_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file):
    '''
    Convert a set of `InputExample's to a TFRecord file
    '''
    def create_int_feature(values):
        f = tf.train.Feature(int64_list = tf.train.Int64List(value = list(values)))
        return f
    
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)

        features = collections.OrderedDict()
        features['input_ids'] = create_int_feature(feature.input_ids)
        features['input_mask'] = create_int_feature(feature.input_mask)
        features['segment_ids'] = create_int_feature(feature.segment_ids)
        features['label_ids'] = create_int_feature(feature.label_ids)
        
        tf_example = tf.train.Example(features = tf.train.Features(feature = features))
        writer.write(tf_example.SerializeToString())

def file_based_input_fn_builder(input_file, seq_length, is_training, 
                                drop_remainder, nb_labels):
    '''
    Creates an `input_fn` closure to be passed to TPUEstimator
    '''
    name_to_features = {
        'input_ids': tf.FixedLenFeature([seq_length], tf.int64),
        'input_mask': tf.FixedLenFeature([seq_length], tf.int64),
        'segment_ids': tf.FixedLenFeature([seq_length], tf.int64),
        'label_ids': tf.FixedLenFeature([nb_labels], tf.int64),
    }

    def _decode_record(record, name_to_features):
        '''
        Decodes a record to a Tensorflow example.
        '''
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example
    
    def input_fn(params):
        '''
        The actual input function
        '''
        batch_size = params['batch_size']
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size = 100)
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size = batch_size,
                drop_remainder = drop_remainder
            )
        )
        return d
    
    return input_fn

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                labels, num_labels, use_one_hot_embeddings):
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
        "output_weights", [num_labels, hidden_size],
        initializer = tf.truncated_normal_initializer(stddev = 0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer = tf.zeros_initializer()
    )
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob = 0.9)
        logits = tf.matmul(output_layer, output_weights, transpose_b = True)
        logits = tf.nn.bias_add(logits, output_bias)
        probs = tf.nn.sigmoid(logits)
        per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.to_float(labels), logits = logits)
        loss = tf.reduce_mean(per_example_loss)
        return loss, per_example_loss, probs

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                    num_train_steps, num_warmup_steps, use_tpu, 
                    use_one_hot_embeddings):
    '''
    Returns `model_fn` closure for TPUEstimator
    '''
    def model_fn(features, labels, mode, params):
        tf.logging.info('*** Features ***')
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features['input_ids']
        input_mask = features['input_mask']
        segment_ids = features['segment_ids']
        label_ids = features['label_ids']

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        (total_loss, per_example_loss, probs) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids,
            label_ids, num_labels, use_one_hot_embeddings
        )

        tvars = tf.trainable_variables()

        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assigment_map_from_checkpoint(
                tvars, init_checkpoint
            )
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                
                scaffold_fn = tpu_scaffold
            else :
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu
            )
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode = mode,
                loss = total_loss,
                train_op = train_op,
                scaffold_fn = scaffold_fn
            )
        elif mode == tf.estimator.ModeKeys.EVAL:
            
            def metric_fn(per_example_loss, label_ids, probs):
                predictions = tf.cast(probs + 0.5, tf.int32)
                accuracy = tf.metrics.accuracy(label_ids, predictions)
                loss = tf.metrics.mean(per_example_loss)
                return {
                    'eval_accuracy': accuracy,
                    'eval_loss': loss
                }
            eval_metrics = (metric_fn, [per_example_loss, label_ids, probs])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode = mode,
                loss = total_loss,
                eval_metrics = eval_metrics,
                scaffold_fn = scaffold_fn
            )
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))
        
        return output_spec

    return model_fn

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True")
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence of length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings)
        )
    
    tf.gfile.MakeDirs(FLAGS.output_dir)
    
    processor = TsvDataProcessor(FLAGS.data_dir)
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(
        vocab_file = FLAGS.vocab_file, do_lower_case = FLAGS.do_lower_case
    )
    
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone = FLAGS.tpu_zone, project = FLAGS.gcp_project
        )
    
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster = tpu_cluster_resolver,
        master = FLAGS.master,
        model_dir = FLAGS.output_dir,
        save_checkpoints_steps = FLAGS.save_checkpoints_steps,
        tpu_config = tf.contrib.tpu.TPUConfig(
            iterations_per_loop = FLAGS.iterations_per_loop,
            num_shards = FLAGS.num_tpu_cores,
            per_host_input_for_training = is_per_host
        )
    )

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples()
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs
        )
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    
    model_fn = model_fn_builder(
        bert_config = bert_config,
        num_labels = len(label_list),
        init_checkpoint = FLAGS.init_checkpoint,
        learning_rate = FLAGS.learning_rate,
        num_train_steps = num_train_steps,
        num_warmup_steps = num_warmup_steps,
        use_tpu = FLAGS.use_tpu,
        use_one_hot_embeddings = FLAGS.use_tpu
    )

    # If TPU is not available, this will fall back to the normal
    # estimator on CPU or GPU
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu = FLAGS.use_tpu,
        model_fn = model_fn,
        config = run_config,
        train_batch_size = FLAGS.train_batch_size,
        eval_batch_size = FLAGS.eval_batch_size
    )

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        # This writes to a TfRecord file that the trainer will read from later on
        filed_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file
        )
        tf.logging.info("**** Running training ****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file = train_file,
            seq_length = FLAGS.max_seq_length,
            is_training = True,
            drop_remainder = True,
            nb_labels = len(label_list)
        )
        estimator.train(input_fn = train_input_fn, max_steps = num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_val_examples()
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        filed_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file
        )
        
        tf.logging.info("**** Running evaluation ****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set
        eval_steps = None
        
        # If running eval on TPU, you need to specify the number of steps though
        if FLAGS.use_tpu:
            eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
        
        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file = eval_file,
            seq_length = FLAGS.max_seq_length,
            is_training = False,
            drop_remainder = eval_drop_remainder,
            nb_labels = len(label_list)
        )

        result = estimator.evaluate(input_fn = eval_input_fn, steps = eval_steps)
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("**** Eval results ****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        
if __name__ == "__main__":
    tf.app.run()
