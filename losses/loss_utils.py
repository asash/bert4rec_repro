import tensorflow as tf

# https://stackoverflow.com/questions/37086098/does-tensorflow-map-fn-support-taking-more-than-one-tensor


def my_map(fn, arrays, dtype=tf.float32):
    # assumes all arrays have same leading dim
    indices = tf.range(tf.shape(arrays[0])[0])
    out = tf.map_fn(lambda ii: fn(*[array[ii] for array in arrays]), indices, dtype=dtype)
    return out

def get_pairwise_diff_batch(a, b, a_size, b_size):
    a_tile = tf.tile(tf.expand_dims(a, 1), [1, b_size, 1])
    b_tile = tf.tile(tf.expand_dims(b, 2), [1, 1, a_size])
    result = a_tile - b_tile
    return result


def get_truncated(y_true, y_pred, truncate_at):
    if truncate_at is not None:
        top_pred = tf.math.top_k(y_pred, truncate_at)
        pred = top_pred.values
        true_ordered_by_pred = tf.gather(y_true, top_pred.indices, batch_dims=1) 
    else:
        pred = y_pred
        true_ordered_by_pred = y_true
    return pred,true_ordered_by_pred

def masked_softmax(x, mask):
    exp = tf.math.exp(x) * mask
    sum = tf.expand_dims(tf.reduce_sum(exp, -1), -1)
    result = tf.math.divide_no_nan(exp, sum)
    return result
