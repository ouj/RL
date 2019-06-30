import tensorflow as tf

class MLPNetwork(tf.layers.Layer):
    def __init__(self):
        super(MLPNetwork, self).__init__()
        self.layers = []

    def collect_variables(self):
        variables = []
        for layer in self.layers:
            variables += layer.variables
        return variables

    def copy_from(self, other_qlayer):
        assert isinstance(other_qlayer, self.__class__)
        target_variables = self.collect_variables()
        source_variables = other_qlayer.collect_variables()
        copy_op = tf.group(
            [
                tf.assign(v_tgt, v_src)
                for v_tgt, v_src in zip(target_variables, source_variables)
            ]
        )
        return copy_op

    def update_from(self, other_qlayer, decay):
        assert isinstance(other_qlayer, self.__class__)
        target_variables = self.collect_variables()
        source_variables = other_qlayer.collect_variables()
        update_op = tf.group(
            [
                tf.assign(v_tgt, decay * v_tgt + (1 - decay) * v_src)
                for v_tgt, v_src in zip(target_variables, source_variables)
            ]
        )
        return update_op

    def is_equal(self, other_qlayer, session=None):
        assert isinstance(other_qlayer, self.__class__)
        target_variables = self.collect_variables()
        source_variables = other_qlayer.collect_variables()
        equal_op = tf.reduce_all(
            [
                tf.reduce_all(tf.equal(v_tgt, v_src))
                for v_tgt, v_src in zip(target_variables, source_variables)
            ]
        )
        return equal_op

    def setup_tensorboard(self):
        variables = self.collect_variables()
        for v in variables:
            tf.summary.histogram(v.name, v)

    def call(self, x):
        for l in self.layers:
            x = l(x)
        return x
