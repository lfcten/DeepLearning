import tensorflow as tf


def my_dnn_regression_fn(features, labels, mode, params):
    top = tf.feature_column.input_layer(features, params["feature_columns"])
    for unit in params.get("hidden_units", [20]):
        top = tf.layers.dense(inputs=top, units=unit, activation=1)

    output_layer = tf.layers.dense(inputs=top, units=1)
    predictions = tf.squeeze(output_layer, 1)
    return predictions
