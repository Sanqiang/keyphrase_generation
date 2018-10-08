import tensorflow as tf
from tensor2tensor.layers import common_attention
from tensor2tensor.models import transformer
from tensor2tensor.utils import beam_search

from util import constant
from model.loss import sequence_loss
from data_generator.vocab import Vocab

class Graph:
    def __init__(self, model_config, is_train, data):
        self.model_config = model_config
        self.is_train = is_train
        self.voc_abstr = data.voc_abstr
        self.voc_kword = data.voc_kword
        self.hparams = transformer.transformer_base()
        self.setup_hparams()

    def get_embedding(self):
        emb_init = tf.random_uniform_initializer(-0.1, 0.1)
        xavier_init = tf.contrib.layers.xavier_initializer()

        with tf.device('/cpu:0'):
            if self.model_config.tied_embedding == 'enc|dec':
                emb_abstr = tf.get_variable(
                    'embedding_abstr', [self.voc_abstr.vocab_size(), self.model_config.dimension], tf.float32,
                    initializer=emb_init)
                emb_kword = emb_abstr
            else:
                emb_abstr = tf.get_variable(
                    'embedding_abstr', [self.voc_abstr.vocab_size(), self.model_config.dimension], tf.float32,
                    initializer=emb_init)
                emb_kword = tf.get_variable(
                    'embedding_kword', [self.voc_kword.vocab_size(), self.model_config.dimension], tf.float32,
                    initializer=emb_init)
        proj_w = tf.get_variable(
            'proj_w', [self.voc_kword.vocab_size(), self.model_config.dimension], tf.float32,
            initializer=xavier_init)
        proj_b = tf.get_variable(
            'proj_b', shape=[self.voc_kword.vocab_size()], initializer=xavier_init)
        return emb_abstr, emb_kword, proj_w, proj_b

    def embedding_fn(self, inputs, embedding):
        with tf.device('/cpu:0'):
            if type(inputs) == list:
                if not inputs:
                    return []
                else:
                    return [tf.nn.embedding_lookup(embedding, inp) for inp in inputs]
            else:
                return tf.nn.embedding_lookup(embedding, inputs)

    def word_distribution(self, decoder_logit, seps, ext_abstr_ph, max_oov, final_output, decoder_embed_inputs):
        cur_len = tf.shape(decoder_logit)[1]
        attn_dists = []
        for sep in seps:
            attn_dists.append(tf.reduce_mean(sep['weight'] * sep['gen'], axis=1))
        attn_dists = tf.reduce_mean(tf.stack(attn_dists, axis=1), axis=1)
        # attn_dists = tf.unstack(attn_dists, axis=1)
        attn_dists = tf.transpose(attn_dists, perm=[1, 0, 2])

        extended_vsize = self.voc_kword.vocab_size() + max_oov
        extra_zeros = tf.zeros((self.model_config.batch_size, cur_len, max_oov))
        vocab_dists_extended = tf.concat([decoder_logit, extra_zeros], axis=-1)

        ext_abstr_ph = tf.stack(ext_abstr_ph, axis=1)
        batch_nums = tf.range(0, limit=self.model_config.batch_size)
        batch_nums = tf.expand_dims(batch_nums, 1)
        batch_nums = tf.tile(batch_nums, [1, self.model_config.max_abstr_len])
        indices = tf.stack((batch_nums, ext_abstr_ph), axis=2)
        shape = [self.model_config.batch_size, extended_vsize]
        # attn_dists_projected = tf.stack([tf.scatter_nd(indices, copy_dist, shape) for copy_dist in
        #                                  attn_dists], axis=1)
        attn_dists_projected = tf.map_fn(lambda copy_dist: tf.scatter_nd(indices, copy_dist, shape), attn_dists)
        attn_dists_projected = tf.transpose(attn_dists_projected, perm=[1, 0, 2])

        vocab_mat = tf.get_variable('vocab_mat', shape=[1, 2 * final_output.get_shape()[-1].value, 1],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
        vocab_weights = tf.sigmoid(
            tf.nn.conv1d(tf.concat([final_output, decoder_embed_inputs], axis=-1), vocab_mat, 1, 'SAME'))
        final_dists = vocab_weights * vocab_dists_extended + (1 - vocab_weights) * attn_dists_projected
        return final_dists

    def decode_step(self, kword_input, abstr_outputs, abstr_bias, batch_go, hist_vector=None):
        kword_length = len(kword_input) + 1
        kword_input = tf.stack([batch_go] + kword_input, axis=1)
        kword_output = self.decode_inputs_to_outputs(kword_input, abstr_outputs, abstr_bias,
                                                           hist_vector=hist_vector)
        kword_output_list = [
            tf.squeeze(d, 1)
            for d in tf.split(kword_output, kword_length, axis=1)]
        return kword_output, kword_output_list

    def decode_inputs_to_outputs(self, kword_input, abstr_outputs, abstr_bias, hist_vector=None):
        if self.hparams.pos == 'timing':
            kword_input = common_attention.add_timing_signal_1d(kword_input)
        kword_tribias = common_attention.attention_bias_lower_triangle(tf.shape(kword_input)[1])
        kword_input = tf.nn.dropout(
            kword_input, 1.0 - self.hparams.layer_prepostprocess_dropout)
        kword_output = transformer.transformer_decoder(
            kword_input, abstr_outputs, kword_tribias,
            abstr_bias, self.hparams,
            hist_vector=hist_vector)
        return kword_output

    def output_to_logit(self, prev_out, w, b):
        prev_logit = tf.add(tf.matmul(prev_out, tf.transpose(w)), b)
        return prev_logit

    def transformer_beam_search(self, abstr_outputs, abstr_bias, emb_kword, proj_w, proj_b, hist_vector=None):
        # Use Beam Search in evaluation stage
        # Update [a, b, c] to [a, a, a, b, b, b, c, c, c] if beam_search_size == 3
        encoder_beam_outputs = tf.concat(
            [tf.tile(tf.expand_dims(abstr_outputs[o, :, :], axis=0),
                     [self.model_config.beam_search_size, 1, 1])
             for o in range(self.model_config.batch_size)], axis=0)

        encoder_attn_beam_bias = tf.concat(
            [tf.tile(tf.expand_dims(abstr_bias[o, :, :, :], axis=0),
                     [self.model_config.beam_search_size, 1, 1, 1])
             for o in range(self.model_config.batch_size)], axis=0)

        hist_beam_vector = tf.concat(
            [tf.tile(tf.expand_dims(hist_vector[o, :, :], axis=0),
                     [self.model_config.beam_search_size, 1, 1])
             for o in range(self.model_config.batch_size)], axis=0)

        if self.model_config.subword_vocab_size:
            go_id = self.voc_kword.encode(constant.SYMBOL_GO)[0]
        else:
            go_id = self.voc_kword.encode(constant.SYMBOL_GO)
        batch_go = tf.expand_dims(tf.tile(
            tf.expand_dims(self.embedding_fn(go_id, emb_kword), axis=0),
            [self.model_config.batch_size, 1]), axis=1)
        batch_go_beam = tf.concat(
            [tf.tile(tf.expand_dims(batch_go[o, :, :], axis=0),
                     [self.model_config.beam_search_size, 1, 1])
             for o in range(self.model_config.batch_size)], axis=0)


        def symbol_to_logits_fn(ids):
            cur_ids = ids[:, 1:]

            embs = tf.nn.embedding_lookup(emb_kword, cur_ids)

            embs = tf.concat([batch_go_beam, embs], axis=1)

            final_outputs = self.decode_inputs_to_outputs(
                embs, encoder_beam_outputs, encoder_attn_beam_bias, hist_vector=hist_beam_vector)

            return self.output_to_logit(final_outputs[:, -1, :], proj_w, proj_b)

        beam_ids, beam_score = beam_search.beam_search(
            symbol_to_logits_fn,
            tf.zeros([self.model_config.batch_size], tf.int32),
            self.model_config.beam_search_size,
            self.model_config.max_kword_len,
            self.voc_kword.vocab_size(),
            0.6
        )

        top_beam_ids = beam_ids[:, 0, 1:]
        top_beam_ids = tf.pad(top_beam_ids,
                              [[0, 0],
                               [0, self.model_config.max_kword_len - tf.shape(top_beam_ids)[1]]])
        decoder_target_list = [tf.squeeze(d, 1)
                               for d in tf.split(top_beam_ids, self.model_config.max_kword_len, axis=1)]
        decoder_score = -beam_score[:, 0] / tf.to_float(tf.shape(top_beam_ids)[1])

        return decoder_score, top_beam_ids

    def create_model(self):
        with tf.variable_scope('variables'):
            abstr_ph = []
            for _ in range(self.model_config.max_abstr_len):
                abstr_ph.append(tf.zeros(self.model_config.batch_size, tf.int32, name='abstract_input'))

            kwords_ph = []
            for _ in range(self.model_config.max_cnt_kword):
                kword = []
                for _ in range(self.model_config.max_kword_len):
                    kword.append(tf.zeros(self.model_config.batch_size, tf.int32, name='kword_input'))
                kwords_ph.append(kword)

            # Train for length control
            if self.is_train:
                kword_occupies_ph = []
                for _ in range(self.model_config.max_cnt_kword):
                    kword_occupies_ph.append(
                        tf.zeros(self.model_config.batch_size, tf.float32, name='kword_occupy_input'))

            emb_abstr, emb_kword, proj_w, proj_b = self.get_embedding()
            abstr = tf.stack(self.embedding_fn(abstr_ph, emb_abstr), axis=1)
            kwords = []
            for kword_idx in range(self.model_config.max_cnt_kword):
                kwords.append(self.embedding_fn(kwords_ph[kword_idx], emb_kword))

        with tf.variable_scope('model_encoder'):
            if self.hparams.pos == 'timing':
                abstr = common_attention.add_timing_signal_1d(abstr)
            encoder_embed_inputs = tf.nn.dropout(abstr,
                                                 1.0 - self.hparams.layer_prepostprocess_dropout)
            abstr_bias = common_attention.attention_bias_ignore_padding(
                tf.to_float(tf.equal(tf.stack(abstr_ph, axis=1),
                                     self.voc_kword.encode(constant.SYMBOL_PAD))))
            abstr_outputs = transformer.transformer_encoder(
                encoder_embed_inputs, abstr_bias, self.hparams)

        losses = []
        targets = []
        pred_occupies = []
        obj = {}

        hist_vector = None
        if 'kp_attn' in self.model_config.cov_mode:
            hist_vector = tf.zeros(
                [self.model_config.batch_size, 1, self.model_config.dimension,])

        with tf.variable_scope('model_decoder'):
            if self.model_config.subword_vocab_size:
                go_id = self.voc_kword.encode(constant.SYMBOL_GO)[0]
            else:
                go_id = self.voc_kword.encode(constant.SYMBOL_GO)
            batch_go = tf.tile(
                tf.expand_dims(self.embedding_fn(go_id, emb_kword), axis=0),
                [self.model_config.batch_size, 1])

            for kword_idx in range(self.model_config.max_cnt_kword):
                if self.is_train:
                    kword = kwords[kword_idx][:-1]
                    kword_ph = kwords_ph[kword_idx]
                    kword_output, kword_output_list = self.decode_step(
                        kword, abstr_outputs, abstr_bias, batch_go, hist_vector=hist_vector)
                    kword_logit_list = [self.output_to_logit(o, proj_w, proj_b) for o in kword_output_list]
                    kword_target_list = [tf.argmax(o, output_type=tf.int32, axis=-1)
                                         for o in kword_logit_list]

                    kword_lossbias = [
                        tf.to_float(tf.not_equal(d, self.voc_kword.encode(constant.SYMBOL_PAD)))
                        for d in kword_ph]
                    kword_lossbias = tf.stack(kword_lossbias, axis=1)
                    if self.model_config.number_samples > 0:
                        loss_fn = tf.nn.sampled_softmax_loss
                    else:
                        loss_fn = None
                    loss = sequence_loss(logits=tf.stack(kword_logit_list, axis=1),
                                         targets=tf.stack(kword_ph, axis=1),
                                         weights=kword_lossbias,
                                         softmax_loss_function=loss_fn,
                                         w=proj_w,
                                         b=proj_b,
                                         decoder_outputs=tf.stack(kword_output_list, axis=1),
                                         number_samples=self.model_config.number_samples
                                         )
                    kword_target = tf.stack(kword_target_list, axis=1)
                    targets.append(kword_target)

                    if 'kp_attn' in self.model_config.cov_mode:
                        kword_embed = self.embedding_fn(kword_ph, emb_kword)
                        hist_vector += tf.expand_dims(tf.reduce_mean(
                            tf.stack(kword_embed, axis=1), axis=1), axis=1)

                    # Train for length control
                    pred_occupy = self.get_pred_occupy_logit(hist_vector, abstr_outputs)
                    occupy_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=pred_occupy, labels=kword_occupies_ph[kword_idx])
                    loss += tf.reduce_mean(occupy_loss)
                    pred_occupies.append(pred_occupy)

                    losses.append(loss)
                else:
                    loss, kword_target = self.transformer_beam_search(
                        abstr_outputs, abstr_bias, emb_kword, proj_w, proj_b, hist_vector=hist_vector)

                    targets.append(kword_target)
                    losses = loss

                    if 'kp_attn' in self.model_config.cov_mode:
                        kword_embed = self.embedding_fn(kword_target, emb_kword)
                        hist_vector += tf.expand_dims(tf.reduce_mean(kword_embed, axis=1), axis=1)

                    pred_occupy = tf.round(tf.sigmoid(self.get_pred_occupy_logit(hist_vector, abstr_outputs)))
                    pred_occupies.append(pred_occupy)

                tf.get_variable_scope().reuse_variables()
        if targets:
            obj['targets'] = tf.stack(targets, axis=1)
        obj['abstr_ph'] = abstr_ph
        obj['kwords_ph'] = kwords_ph
        if self.is_train:
            obj['kword_occupies_ph'] = kword_occupies_ph
        pred_occupies = tf.stack(pred_occupies, axis=1)
        obj['pred_occupies'] = pred_occupies

        if type(losses) is list:
            losses = tf.add_n(losses)
        return losses, obj

    def get_pred_occupy_logit(self, hist_vector, abstr_outputs):
        proj_hist = tf.get_variable(
                'proj_hist', [2*self.model_config.dimension, 1], tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
        bias_hist = tf.get_variable(
                'bias_hist', [1, 1], tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
        pred_occupy_logit = tf.matmul(
            tf.concat([tf.squeeze(hist_vector, axis=1), tf.reduce_mean(abstr_outputs, axis=1)], axis=-1),
            proj_hist) + bias_hist
        return tf.squeeze(pred_occupy_logit, axis=1)

    def create_model_multigpu(self):
        losses = []
        grads = []
        optim = self.get_optim()
        self.objs = []

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for gpu_id in range(self.model_config.num_gpus):
                with tf.device('/gpu:%d' % gpu_id):
                    loss, obj = self.create_model()
                    grad = optim.compute_gradients(loss)
                    losses.append(loss)
                    grads.append(grad)
                    self.objs.append(obj)
                    tf.get_variable_scope().reuse_variables()

        self.global_step = tf.get_variable(
            'global_step', initializer=tf.constant(0, dtype=tf.int64), trainable=False)
        with tf.variable_scope('optimization'):
            self.loss = tf.divide(tf.add_n(losses), self.model_config.num_gpus)
            self.perplexity = tf.exp(tf.reduce_mean(self.loss) / self.model_config.max_cnt_kword)

            if self.is_train:
                avg_grad = self.average_gradients(grads)
                grads = [g for (g,v) in avg_grad]
                clipped_grads, _ = tf.clip_by_global_norm(grads, self.model_config.max_grad_norm)
                self.train_op = optim.apply_gradients(zip(clipped_grads, tf.trainable_variables()), global_step=self.global_step)

            self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

    def get_optim(self):
        learning_rate = tf.constant(self.model_config.learning_rate)

        if self.model_config.optimizer == 'adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate)
        # Adam need lower learning rate
        elif self.model_config.optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate)
        else:
            raise Exception('Not Implemented Optimizer!')
        return opt

    # Got from https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L101
    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def setup_hparams(self):
        self.hparams.num_heads = self.model_config.num_heads
        self.hparams.num_hidden_layers = self.model_config.num_hidden_layers
        self.hparams.num_encoder_layers = self.model_config.num_encoder_layers
        self.hparams.num_decoder_layers = self.model_config.num_decoder_layers
        self.hparams.pos = self.model_config.hparams_pos
        self.hparams.hidden_size = self.model_config.dimension
        self.hparams.layer_prepostprocess_dropout = self.model_config.layer_prepostprocess_dropout
        self.hparams.cov_mode = self.model_config.cov_mode
        self.hparams.pointer_mode = self.model_config.pointer_mode
        self.hparams.model_config = self.model_config

        if self.is_train:
            self.hparams.add_hparam('mode', tf.estimator.ModeKeys.TRAIN)
        else:
            self.hparams.add_hparam('mode', tf.estimator.ModeKeys.EVAL)
            self.hparams.layer_prepostprocess_dropout = 0.0
            self.hparams.attention_dropout = 0.0
            self.hparams.dropout = 0.0
            self.hparams.relu_dropout = 0.0