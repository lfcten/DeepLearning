import tensorflow as tf

# tf.feature_column.categorical_column_with_vocabulary_list
colors = tf.feature_column.categorical_column_with_vocabulary_list(key='colors', vocabulary_list=('R', 'G', 'B', 'Y'),
                                                                   num_oov_buckets=2)
columns = [colors]
features = tf.parse_example(..., features=tf.feature_column.make_parse_example_spec(columns))

