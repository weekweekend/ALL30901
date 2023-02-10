import tensorflow as tf

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
    a = tf.constant(1)
    b = tf.constant(3)
    c = a + b
    print('结果是：%d\n 值为：%d' % (sess.run(c), sess.run(c)))
