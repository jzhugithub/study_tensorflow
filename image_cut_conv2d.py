import tensorflow as tf

# cut image and make it into conv2d

def image_cut_conv2d(input, input_shape, cut_shape):
    # input
    # input: target image to cut -tensor
    # input_shape: image shape -[height, width, depth]
    # cut_shape: how to cut image -[row, col]

    # output
    # output:conv2d refer to tf.nn.conv2d -[filter_height, filter_width, in_channels, out_channels]
    # in_channels = input_shape[depth]

    dim = range(5)
    dim[0] = cut_shape[0]  # cut number in row
    dim[1] = input_shape[0] / cut_shape[0]  # row number in row patch
    dim[2] = cut_shape[1]  # cut number in col
    dim[3] = input_shape[1] / cut_shape[1]  # row number in col patch
    dim[4] = input_shape[2]  # dim of an input element
    reshape = tf.reshape(input, dim)
    concat = [tf.concat([reshape[i][j] for j in range(dim[1])], 1) for i in range(dim[0])]
    concat_reshape = [tf.reshape(concat[i], [1, dim[1], dim[2], dim[3], dim[4]]) for i in range(dim[0])]
    ouput_concat = tf.concat([concat_reshape[i] for i in range(dim[0])], 0)
    output = tf.reshape(ouput_concat, [dim[0] * dim[2], dim[1], dim[3], dim[4]])
    return tf.transpose(output, perm=[1, 2, 3, 0])

if __name__ == '__main__':

    x = tf.Variable(initial_value=range(1, 73), trainable=False)
    x_image = tf.reshape(x, [9, 8, 1])

    x_out = image_cut_conv2d(x_image, [9, 9, 1], [3, 3])
    x_out = tf.reshape(x_out, [-1])

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    print(sess.run([x, x_out]))



