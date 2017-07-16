# import tensorflow as tf
# # QueueRunner
# filenames = ['A.csv', 'B.csv', 'C.csv']
# filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
# # Reader
# reader = tf.TextLineReader()
# key, value = reader.read(filename_queue)
# # Decoder
# example, label = tf.decode_csv(value, record_defaults=[['null'], ['null']])
# # Graph
# with tf.Session() as sess:
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)  #start QueueRunner
#
#     for i in range(10):
#         print example.eval()
#     coord.request_stop()
#     coord.join(threads)





# import tensorflow as tf
# filenames = ['A.csv', 'B.csv', 'C.csv']
# filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
# reader = tf.TextLineReader()
# key, value = reader.read(filename_queue)
# example, label = tf.decode_csv(value, record_defaults=[['null'], ['null']])
#
# example_batch, label_batch = tf.train.batch(
#       [example, label], batch_size=5)
# with tf.Session() as sess:
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     for i in range(10):
#         print example_batch.eval()
#     coord.request_stop()
#     coord.join(threads)




# import tensorflow as tf
# filenames = ['A.csv', 'B.csv', 'C.csv']
# filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
# reader = tf.TextLineReader()
# key, value = reader.read(filename_queue)
# record_defaults = [['null'], ['null']]
# example_list = [tf.decode_csv(value, record_defaults=record_defaults)
#                   for _ in range(2)]
# example_batch, label_batch = tf.train.batch_join(
#       example_list, batch_size=5)
# with tf.Session() as sess:
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     for i in range(10):
#         print example_batch.eval()
#     coord.request_stop()
#     coord.join(threads)

import tensorflow as tf

with tf.device('/cpu:0'):
    filenames = ['A.csv', 'B.csv', 'C.csv']
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False, num_epochs=1)
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_defaults = [['null'], ['null']]
    example_list = [tf.decode_csv(value, record_defaults=record_defaults)
                      for _ in range(2)]
    example_batch, label_batch = tf.train.batch_join(
          example_list, batch_size=5, allow_smaller_final_batch=True)



    init_local_op = tf.initialize_local_variables()
    with tf.Session() as sess:
        sess.run(init_local_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop():
                eb = sess.run([example_batch])
                print eb
                print len(eb[0])
        except tf.errors.OutOfRangeError:
            print('Epochs Complete!')
        finally:
            coord.request_stop()
        print('s')
        coord.request_stop()
        coord.join(threads)