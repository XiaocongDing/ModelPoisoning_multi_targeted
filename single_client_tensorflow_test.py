import tensorflow as tf
from utils.fmnist import load_fmnist
from keras.utils import np_utils
import keras.backend as K
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import numpy as np
import os
from multiprocessing import Process, Manager

######################################

############ data prepared ###########
IMAGE_ROWS = 28
IMAGE_COLS = 28
NUM_CHANNELS = 1
NUM_CLASSES = 10
BATCH_SIZE = 100
max_acc = 91.0
max_agents_per_gpu = 8
mem_frac = 0.05
k = 10 # the number of client
C = 1.0 # the fraction of agents per time step
E = 5 # epochs for each agent
B = 50 # agent batch size
eta = 1e-3 # learning rate

mal_agent = True
dataset = 'fMMNIST'
mal_obj = 'single' ## or 'multipule'
optimizer = 'adam'
model_num = 1

dir_name = 'single_test_weights/%s/model_%s/%s/k%s_E%s_B%s_C%1.0e_lr%.1e' % (
        dataset, model_num, optimizer, k, E, B, C, eta)

X_train, y_train = load_fmnist(".\\utils\\data", kind='train')
X_test, y_test = load_fmnist(".\\utils\\data",kind='t10k')

X_train = X_train.reshape(X_train.shape[0],
                              IMAGE_ROWS,
                              IMAGE_COLS,
                              NUM_CHANNELS)

X_test = X_test.reshape(X_test.shape[0],
                        IMAGE_ROWS,
                        IMAGE_COLS,
                        NUM_CHANNELS)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255



###############################################

########## functions ##########################

def agents(i, X_shard, Y_shard, gpu_id, return_dict, X_test, Y_test, lr=None):
    K.set_learning_phase(1)
    shard_size = len(X_shard)
    if lr == None:
        lr = eta
    num_steps = E * shard_size / B ## num_steps : iterations
    
    agent_model = modelA()

    x = tf.placeholder(shape=(None,
                              IMAGE_ROWS,
                              IMAGE_COLS,
                              NUM_CHANNELS), dtype=tf.float32)
    y = tf.placeholder(dtype=tf.int64)

    logits = agent_model(x) # ??

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits))


    prediction = tf.nn.softmax(logits)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_frac)
    config = tf.ConfigProto(gpu_options = gpu_options)

    sess = tf.Session(config = config)
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())

    start_offset = 0

    for step in range(int(num_steps)):
        offset = (start_offset + step * B) % (shard_size - B)
        X_batch = X_shard[offset: (offset + B)]
        Y_batch = Y_shard[offset: (offset + B)]
        Y_batch_uncat = np.argmax(Y_batch, axis=1)
        _, loss_val = sess.run([optimizer,loss], feed_dict={x: X_batch, y: Y_batch_uncat})
        if step % 10 == 0:
            print ('Agent %s, Step %s, Loss %s, offset %s' % (i,step,loss_val, offset))



# def train_fn(X_train_shards)

def modelA():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), padding='valid', input_shape=(IMAGE_ROWS,
                                         IMAGE_COLS,
                                         NUM_CHANNELS)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES))
    return model

def mal_data_create(X_test, Y_test, Y_test_uncat):

    if mal_obj == 'single':
        r =np.random.choice(len(X_test)) 
        mal_data_X = X_test[r: r + 1]
        allowed_targets = list(range(NUM_CLASSES))
        print("Initial class: %s" %Y_test_uncat[r])
        true_labels = Y_test_uncat[r:r+1]
        allowed_targets.remove(Y_test_uncat[r])
        mal_data_Y = np.random.choice(allowed_targets)
        mal_data_Y = mal_data_Y.reshape(1,)
        print("Target class: %s" % mal_data_Y[0])
    else:
        target_indices = np.random.choice(len(X_test), args.mal_num)
        mal_data_X = X_test[target_indices]
        print("Initial classes: %s" % Y_test_uncat[target_indices])
        true_labels = Y_test_uncat[target_indices]
        mal_data_Y = []
        for i in range(args.mal_num):
            allowed_targets = list(range(gv.NUM_CLASSES))
            allowed_targets.remove(Y_test_uncat[target_indices[i]])
            mal_data_Y.append(np.random.choice(allowed_targets))
        mal_data_Y = np.array(mal_data_Y)
    return mal_data_X, mal_data_Y, true_labels



###########################################

######## create mal_obj data ##############



#######################################

############# training #################

if __name__ == "__main__":

    y_train = np_utils.to_categorical(y_train, NUM_CLASSES).astype(np.float32)
    y_test = np_utils.to_categorical(y_test, NUM_CLASSES).astype(np.float32)

    Y_test_uncat = np.argmax(y_test,axis=1)
    ###############################

    ########## slicing ############

    random_indices = np.random.choice(
            len(X_train), len(X_train), replace=False) #1*n的从0-n的随机数组成的向量，不能重复。
    X_train_permuted = X_train[random_indices]
    Y_train_permuted = y_train[random_indices]
    X_train_shards = np.split(X_train_permuted, k)
    Y_train_shards = np.split(Y_train_permuted, k)

    data_path = 'data/mal_X_%s_%s.npy' % (dataset,mal_obj)

    if not os.path.exists(data_path):
        mal_data_X ,mal_data_Y, true_labels = mal_data_create(X_test, y_test, Y_test_uncat)

    else: 
        mal_data_X = np.load('data/mal_X_%s_%s.npy' % (dataset, mal_obj))
        mal_data_Y = np.load('data/mal_Y_%s_%s.npy' % (dataset, mal_obj))
        true_labels = np.load('data/true_labels_%s_%s.npy' % (dataset, mal_obj))
        print("Initial classes: %s" % true_labels)
        print("Target classes: %s" % mal_data_Y)

    
    manager = Manager()
    return_dict = manager.dict()
    return_dict['eval_success'] = 0.0
    return_dict['eval_loss'] = 0.0
    # t = 0
    gpu_id = 0
    p = Process(target=agents, args=(1, X_train_shards[1], Y_train_shards[1], gpu_id, return_dict, X_test, y_test))
    p.start()
    p.join()


# if mal_agent:
#     t_final = train_fn(X_train_shards, Y_train_shards, X_test, Y_test_uncat, return_dict, mal_data_X, mal_data_Y)
# else:
#     t_ = train_fn(X_train_shards, Y_train_shards, X_test, Y_test_uncat, return_dict)




