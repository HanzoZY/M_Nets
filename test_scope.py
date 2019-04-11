import tensorflow as tf

def init_pramaters(in_channel,out_channel):
    weights = tf.get_variable('weights1', shape=[in_channel, out_channel], dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
    biases = tf.get_variable('biases1', shape=[out_channel], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.1))
    print(weights.name)
    print(biases.name)

def op_function():
    input_dim=4
    for output_dim in [1,2,3]:
        with tf.variable_scope(f"similarN{output_dim}", reuse=tf.AUTO_REUSE):
            init_pramaters(input_dim, out_channel=output_dim)
            input_dim = output_dim


def Main():
    op_function()
    op_function()
    op_function()
    sess = tf.Session()
    with sess.as_default():
        init = tf.initialize_all_variables()
        sess.run(init)

if __name__ == '__main__':
    Main()
