import tensorflow as tf
import numpy as np
from itertools import combinations
from tensorflow.python import debug as tfdbg
import time

class M_Nets:
    def __init__(self,hparams, input_tensor, label_tensor, is_train):
        self.num_classes = hparams.n
        self.batch_size = hparams.batch_size
        self.seq_len = hparams.seq_len
        self.input_dim = hparams.input_dim
        self.lr = hparams.lr


        self.l2_loss = 0


        self.input_placeholder = tf.cast(input_tensor, tf.float32)
        self.label_placeholder = label_tensor
        self.is_train = is_train
        if self.is_train:
            self.global_step = tf.get_variable("global_step", initializer=0, trainable=False)
        else:
            self.global_step = None
        feed_label, target_label = tf.split(self.label_placeholder, [self.seq_len - 1, 1],axis=1)
        self.support_set,self.query_set=tf.split(self.input_placeholder,[self.seq_len-1,1],axis=1)
        self.query_set=tf.squeeze(self.query_set)
        self.target_label = tf.reshape(target_label,shape=[-1])
        self.support_label = tf.one_hot(feed_label,depth=self.num_classes,dtype=tf.float32)
        # self.embeddings_similarity,self.savea,self.saveb = self._cosine_similarity(target=self.query_set, support_set=self.support_set)  # (batch_size, n * k)
        # self.embeddings_similarity=self._similarity_nn(target=self.query_set, support_set=self.support_set)
        self.embeddings_similarity = self._similarity_nn_merge(target=self.query_set, support_set=self.support_set)
        self.attention = tf.nn.softmax(self.embeddings_similarity)
        print('attention:',self.attention.shape)
        self.predict_label = tf.squeeze(tf.matmul(tf.expand_dims(self.attention, 1), self.support_label))

        ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_label,logits=self.predict_label))
        self.loss = ce_loss
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
        self.accuracy = self._calc_accuracy()


    def _calc_accuracy(self):
        with tf.name_scope("accuracy"):
            self.predictions = tf.argmax(self.predict_label, 1, name="predictions", output_type=tf.int32)
            labels = self.target_label
            correct_predictions = tf.equal(self.predictions, labels)
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            return accuracy


    def _cosine_similarity(self, target, support_set):
        """the c() function that calculate the cosine similarity between (embedded) support set and (embedded) target

        note: the author uses one-sided cosine similarity as zergylord said in his repo (zergylord/oneshot)
        """
        target_normed = tf.nn.l2_normalize(target, 1) # (batch_size, 64)
        # target_normed = target
        sup_similarity = []
        save_tempa=[]
        save_tempb=[]
        for i in tf.unstack(support_set,axis=1):
            print('i',i.shape)
            i_normed = tf.nn.l2_normalize(i, 1)  # (batch_size, 64)

            tempa=tf.expand_dims(target_normed, 1)
            tempb=tf.expand_dims(i_normed, 2)
            save_tempa.append(tempa)
            save_tempb.append(tempb)
            print('a_shape',tempa.shape)
            print('b_shape',tempb.shape)
            similarity = tf.matmul(tempa, tempb)  # (batch_size, )
            sup_similarity.append(similarity)

        return tf.squeeze(tf.stack(sup_similarity, axis=1))
        #return tf.squeeze(tf.stack(sup_similarity, axis=1)),tf.stack(save_tempa),tf.stack(save_tempb)  # (batch_size, n * k)


    def _similarity_nn(self,target,support_set):
        target_normed = tf.nn.l2_normalize(target, 1)
        sup_similarity = []
        for i in tf.unstack(support_set,axis=1):
            print('i',i.shape)
            i_normed = tf.nn.l2_normalize(i, 1)  # (batch_size, 64)
            diff=i_normed-target_normed
            similarity_temp=diff
            for output_dim in [128,16,1]:
                with tf.variable_scope(f"similarN_{output_dim}", reuse=tf.AUTO_REUSE):
                    similarity_temp = self._add_nn_block(x=similarity_temp,out_channel=output_dim)
            sup_similarity.append(similarity_temp)

        return tf.squeeze(tf.stack(sup_similarity, axis=1))

    def _similarity_nn_merge(self,target,support_set):
        target_normed = tf.nn.l2_normalize(target, 1)
        support_normed=tf.nn.l2_normalize(support_set,axis=-1)
        target_normed=tf.tile(tf.expand_dims(target_normed,axis=1),multiples=[1,support_set.shape[-2],1])
        diff=tf.abs(x=(target_normed-support_normed))
        similarity_temp=diff
        for output_dim in [128, 16, 1]:
            with tf.variable_scope(f"similarN_{output_dim}", reuse=tf.AUTO_REUSE):
                similarity_temp = self._add_nn_block(x=similarity_temp, out_channel=output_dim)

        return tf.squeeze(similarity_temp)


    def _add_nn_block(self, x, out_channel):
        out_put = tf.layers.dense(inputs=x, units=out_channel, name='nn_block', use_bias=True,
                                  bias_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                  activation=tf.nn.leaky_relu,reuse=tf.AUTO_REUSE)
        return out_put


def _make_dummy_data(batch_size,seq_len,dim,num_class,k):

    input_data = np.random.randn(batch_size, seq_len, dim)
    # label_data = np.random.randint(num_class, size=(batch_size, seq_len))
    label_data = np.array([np.random.permutation(np.arange(num_class)) for i in range(batch_size)])
    label_taget=np.random.randint(num_class,size=(batch_size,1))
    label_data=np.concatenate((label_data,label_taget),axis=1)
    return input_data, label_data

def _m_nets_test():
    class Dummy: pass

    hparams = Dummy()
    hparams.n = 3
    hparams.k=1
    hparams.input_dim = 2
    hparams.batch_size = 2
    hparams.seq_len = hparams.n*hparams.k+1
    hparams.lr = 1e-3

    # np.set_printoptions(threshold='nan')  # 全部输出

    with tf.Graph().as_default():
        dummy_input, dummy_label = _make_dummy_data(batch_size=hparams.batch_size,seq_len=hparams.seq_len,dim=hparams.input_dim,num_class=hparams.n,k=hparams.k)
        print(np.shape(dummy_input),np.shape(dummy_label))
        # print(dummy_input)
        # print('\n',dummy_label)
        model = M_Nets(hparams, tf.stack(dummy_input), tf.cast(tf.stack(dummy_label), tf.int32), True)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)


        with sess.as_default():
            # sess=tfdbg.LocalCLIDebugWrapperSession(sess)

            sess.run(tf.global_variables_initializer())
            # sess.run(init)
            '''查看参数'''
            a=[v for v in tf.trainable_variables()]
            for i in range(len(a)):
                print(a[i])
            # target_label=sess.run([model.target_label])
            # print(target_label)
            time_start = time.time()
            for i in range(11):



                # support_set,query_set,simi,loss, acc,target_label,predictlabel,savea,saveb = sess.run([model.support_set,model.query_set,model.embeddings_similarity,model.loss, model.accuracy,model.target_label,model.predict_label,model.savea,model.saveb])
                support_set, query_set, simi, loss, acc, target_label, predictlabel= sess.run(
                    [model.support_set, model.query_set, model.embeddings_similarity, model.loss, model.accuracy,
                     model.target_label, model.predict_label])
                if i%10==0:
                    print('support_set \n',support_set,'\n')
                    print('query_set \n',query_set,'\n')
                    print('simi \n',simi,'\n')
                    # print('savea\n',savea)
                    # print('saveb\n',saveb)
                    # print('batch0\n',np.matrix(support_set[0])*np.matrix(query_set[0]).T)

                    # print("predict is \n",predictlabel)
                    # print('target is \n',target_label)
                    # print(train_data)
                    # print("step is ", i, "\n", loss, acc, '\n')
                if acc>=0.99:break

            time_end = time.time()
            print('time cost', time_end - time_start, 's')

if __name__ == "__main__":
    _m_nets_test()