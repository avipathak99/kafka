from absl import flags

import re
import numpy as np

import tensorflow as tf

from data_utils import SEP_ID, CLS_ID

import ast_utils

FLAGS = flags.FLAGS

# SEG_ID_PAD = -1
SEG_ID_A   = 0
SEG_ID_B   = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3



class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               inp_mask,
               input_adj_matrix,
               is_masked,
               segment_ids,
               label_ids,
               is_real_example=True):
    self.input_ids = input_ids
    self.inp_mask = inp_mask
    self.input_adj_matrix = input_adj_matrix
    self.is_masked = is_masked
    self.segment_ids = segment_ids
    self.label_ids = label_ids
    self.is_real_example = is_real_example


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def batchify(inp_seq, sent_ids, adj_matrix, max_seq_len):

    res_data, res_data_mask, res_sent_ids, res_adj_matrices = [], [], [], []

    for data_chunk, chunk_sent_ids, adj in zip(inp_seq, sent_ids, adj_matrix):
        chunk_len = len(data_chunk)
        # tf.logging.info("current chunk_len = {}".format(chunk_len))
        pad_len = ((chunk_len // max_seq_len)+1)*max_seq_len - chunk_len
        # tf.logging.info("current pad_len = {}".format(pad_len))

        # Pad the data with -1 representing padded tokens
        data_chunk = np.array(data_chunk, dtype=np.int64)
        cls_array = np.array([SEG_ID_CLS]*pad_len, dtype=np.int64)
        data_array = np.concatenate([data_chunk, cls_array])

        # create chunk's input mask
        data_chunk_mask = np.array([0]*chunk_len + [1]*pad_len, dtype=np.int64)

        # Padd the send_ids
        # tf.logging.info("(before padding) chunk_sent_ids.shape  = {}".format(np.array(chunk_sent_ids).shape))
        chunk_sent_id = chunk_sent_ids[0]
        chunk_sent_ids.extend( [chunk_sent_id]*pad_len )
        chunk_sent_ids = np.array(chunk_sent_ids)
        # tf.logging.info("(after padding) chunk_sent_ids.shape  = {}".format(chunk_sent_ids.shape))

        # Pad the adj matrix
        # tf.logging.info("(before padding) adj.shape = {}".format(adj.shape))
        adj = np.pad(adj.toarray(), ((0,pad_len),(0,pad_len)), 'constant')
        # tf.logging.info("(after padding) adj.shape = {}".format(adj.shape))

        # Reshape data and sent_ids
        data_array = data_array.reshape((-1, max_seq_len))
        data_chunk_mask = data_chunk_mask.reshape((-1, max_seq_len))
        chunk_sent_ids = chunk_sent_ids.reshape((-1, max_seq_len))

        # slice the adj into chunk adj matrices
        for i in range(0, chunk_len + pad_len, max_seq_len):
            curr_adj_slice = adj[i:i+max_seq_len,i:i+max_seq_len]
            # tf.logging.info("curr_adj_slice.shape = {}".format(curr_adj_slice.shape))
            res_adj_matrices.append(curr_adj_slice)

        res_data.append(data_array)
        res_data_mask.append(data_chunk_mask)
        res_sent_ids.append(chunk_sent_ids)

    res_data = np.concatenate(res_data, axis=0)
    res_data_mask = np.concatenate(res_data_mask, axis=0)
    chunk_sent_ids = np.concatenate(res_sent_ids, axis=0)
    chunk_adj_matrices = np.dstack(res_adj_matrices)

    return res_data, res_data_mask, chunk_sent_ids, chunk_adj_matrices

def sample_mask(seg, num_predict=None, predict_percent=None):
    # ^ is performs bitwise XOR
    assert bool(num_predict) ^ bool(predict_percent), "Must choose either num_predict or predict_percent"

    seg_len = 0
    for i in range(len(seg)):
        if seg[i] == SEG_ID_CLS:
            seg_len = i+1
            break
    pad_len = len(seg)-seg_len

    if num_predict:
        mask = np.array([False] * (seg_len-num_predict) + [True]*num_predict + [False]*pad_len, dtype=np.bool)
    else:
        num_predict = int(predict_percent * seg_len)
        mask = np.array([False] * (seg_len-num_predict) + [True]*num_predict + [False]*pad_len , dtype=np.bool)
    return mask

def convert_single_example(ex_index, example, max_seq_length,
                              token_processor):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  try:
      file_tree = ast_utils.file_to_AST(example.text_a)
  except:
      tf.logging.error("AST creation failed for current example from file: {}".format(example.source_file_name))
      return

  input_data, data_adj_matrices, sent_ids = [], [], []
  sent_id = True

  if file_tree:
      adj_matrix = file_tree.get_adj()
      file_meta_list = ast_utils.get_file_metadata(file_tree)
      code_chunks = ast_utils.get_code_chunks(file_tree, file_meta_list)
      for i,c in enumerate(code_chunks,1):
            tf.logging.info("chunk#{}:\n{}".format(i,c))
      chunk_adj_matrices = ast_utils.get_adj_chunks(file_tree, file_meta_list)

      for chunk, adj_m in zip(code_chunks, chunk_adj_matrices):
          cur_sent = token_processor.convert_tokens_to_ids(chunk)

          if len(cur_sent) == 0:
            tf.logging.error("[Task {}]: len(cur_sent) = {},\nskipping code chunk: {}\ninside file: {}."\
                                                        .format(idx, len(cur_sent), chunk, input_path))
            continue
          else:
            input_data.append(cur_sent)
            data_adj_matrices.append(adj_m)
            sent_ids.append([sent_id] * len(cur_sent))
            sent_id = not sent_id

  tf.logging.info("[Example {}]: #chuncks = len(input_data) = {}".format(ex_index, len(input_data)))

  tf.logging.info("data.shape = {}".format(np.array(input_data).shape))
  tf.logging.info("sent_ids.shape = {}".format(np.array(sent_ids).shape))
  tf.logging.info("adj_matrix.shape = {}".format(np.array(data_adj_matrices).shape))

  data, data_mask, sent_ids, adj_matrix = batchify(input_data, sent_ids, data_adj_matrices, max_seq_len=max_seq_length-1)

  data_len = data.shape[0] #data.shape[1]
  sep_array = np.array([SEP_ID], dtype=np.int64)
  cls_array = np.array([CLS_ID], dtype=np.int64)
  pad_array = np.array([0], dtype=np.int64)
  inv_pad_array = np.array([1], dtype=np.int64)

  i = 0

  feature_lst = []

  for ex in range(data_len):
      inp = data[ex,:]
      tgt = data[ex,:]
      inp_mask = data_mask[ex,:]

      mask_0 = sample_mask(inp,
                        num_predict=FLAGS.num_predict,
                        predict_percent=FLAGS.predict_percent)

      assert mask_0.shape[0] == max_seq_length - 1

      # concatenate data and data_mask
      cat_data = np.concatenate([inp, cls_array])

      # extend the inp_mask with a 1.
      inp_mask = np.concatenate([inp_mask, inv_pad_array])

      seg_id = ([0] * (max_seq_length - 1) + [2])
      assert cat_data.shape[0] == max_seq_length

      # the last two CLS's are not used, just for padding purposes
      tgt = np.concatenate([tgt, cls_array])
      assert tgt.shape[0] == max_seq_length

      is_masked = np.concatenate([mask_0, pad_array], 0)
      assert is_masked.shape[0] == max_seq_length

      if FLAGS.num_predict is not None:
          assert np.sum(is_masked) == FLAGS.num_predict

      # Adj Processing:
      # Pad the adjacency matrix slice with a zero row and column and flatten
      adj_slice = np.pad(adj_matrix[:,:,ex], ((0,1),(0,1)), 'constant')
      adj_slice = adj_slice.flatten().astype('int64')
      assert adj_slice.shape[0] == max_seq_length * max_seq_length

      if ex < 5:
          tf.logging.info("input_ids: {}".format(cat_data))
          tf.logging.info("inp_mask: {}".format(inp_mask))
          tf.logging.info("label_id: {}".format(tgt))
          tf.logging.info("is_masked: {}".format(is_masked))
          tf.logging.info("segment_ids: {}".format(seg_id))
          tf.logging.info("input_adj_matrix: {}".format(adj_slice))

      feature = InputFeatures(
          input_ids=cat_data,
          inp_mask=inp_mask,
          input_adj_matrix=adj_slice,
          is_masked=is_masked,
          segment_ids=seg_id,
          label_ids=tgt)

      feature_lst.append(feature)

  return feature_lst
