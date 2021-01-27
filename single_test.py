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
import sys
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
mal_num = 1 
k = 10 # the number of client
C = 0.3 # the fraction of agents per time step
E = 5 # epochs for each agent
B = 50 # agent batch size
eta = 1e-3 # learning rate
T = 10 # iterations  
gar = 'avg'
alpha_i = 1.0 / k

mal_agent = True
dataset = 'fMMNIST'
mal_obj = 'single' ## or 'multipule'
optimizer = 'adam'
model_num = 1
mal_agent_index = k -1

# dir_name = ('single_test_weights/%s/model_%s/%s/k%s_E%s_B%s_C%1.0e_lr%.1e/' % (
#         dataset, model_num, optimizer, k, E, B, C, eta))
dir_name = '.\\single_test_weights\\'

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

#def mal_single_algs(x,y)

def mal_agent(i, X_shard, Y_shard, return_dict):

    num_steps = E * shard_size / B
    x = tf.placeholder(shape=(None,
                              IMAGE_ROWS,
                              IMAGE_COLS,
                              NUM_CHANNELS), dtype=tf.float32)
    y = tf.placeholder(dtype=tf.int64)
    agent_model = modelA()
    logits = agent_model(x)
    prediction = tf.nn.softmax(logits)
    eval_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels = y, logits=logits))
    sess = tf.Session()
    K.set_session(sess)
    # final_delta, penul_delta = mal_single_algs()
    # final_weights = 

def agents(i, X_shard, Y_shard, gpu_id, return_dict, X_test, Y_test, t, lr=None):
    K.set_learning_phase(1)
    shard_size = len(X_shard)
    if lr == None:
        lr = eta
    num_steps = E * shard_size / B ## num_steps : iterations
    
    shared_weights = np.load(dir_name + 'global_weights_t%s.npy' % t,allow_pickle = True)

    agent_model = modelA()

    # temp_weights = agent_model.get_weights()
    # temp_weights = np.array(temp_weights)
    # for i in range(np.shape(temp_weights)[0]):
    #     print(np.shape(temp_weights[i]))
    #     flat_ = temp_weights[i].flatten()
    #     print(np.shape(flat_))
        

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

    agent_model

    start_offset = 0
    
    for step in range(int(num_steps)):
        offset = (start_offset + step * B) % (shard_size - B)
        X_batch = X_shard[offset: (offset + B)]
        Y_batch = Y_shard[offset: (offset + B)]
        Y_batch_uncat = np.argmax(Y_batch, axis=1)
        _, loss_val = sess.run([optimizer,loss], feed_dict={x: X_batch, y: Y_batch_uncat})
        if step % 100 == 0:
            print ('Agent %s, Step %s, Loss %s, offset %s' % (i,step,loss_val, offset))
            # temp_weights = agent_model.get_weights()
            # for i in range(np.shape(temp_weights)[0]):
            #     temp_weights[i] = temp_weights[i] * (abs(temp_weights[i]) > 0.01)
            # agent_model.set_weights(temp_weights)
    
    final_weights = agent_model.get_weights()
    X_input = X_shard[[3]]
    Y_input = np.argmax(Y_shard[3], axis=0)
    Y_input = Y_input.reshape(1,)
    target, target_conf, actual, actual_conf = mal_eval_single(X_input,Y_input,final_weights)
    print("Target: %s with confidence %s; Actual: %s with confidence %s" % (target, target_conf, actual, actual_conf))

    return_dict[str(i)] = np.array(final_weights)

# def train_fn(X_train_shards)
def master():
    K.set_learning_phase(1)
    print('Initializing master model')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())
    global_model = modelA()
    global_model.summary()
    global_weights = global_model.get_weights()
    np.save(dir_name+'global_weights_t0.npy',global_weights)

def modelA():
    model = Sequential()
    model.add(Conv2D(4, (5, 5), padding='valid', input_shape=(IMAGE_ROWS,
                                         IMAGE_COLS,
                                         NUM_CHANNELS)))
    model.add(Activation('relu'))

    model.add(Conv2D(4, (5, 5)))
    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(16))
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
        target_indices = np.random.choice(len(X_test), mal_num)
        mal_data_X = X_test[target_indices]
        print("Initial classes: %s" % Y_test_uncat[target_indices])
        true_labels = Y_test_uncat[target_indices]
        mal_data_Y = []
        for i in range(mal_num):
            allowed_targets = list(range(NUM_CLASSES))
            allowed_targets.remove(Y_test_uncat[target_indices[i]])
            mal_data_Y.append(np.random.choice(allowed_targets))
        mal_data_Y = np.array(mal_data_Y)
    return mal_data_X, mal_data_Y, true_labels

def eval_setup(global_weights):
    K.set_learning_phase(0)
    global_model = modelA()
    x = tf.placeholder(shape=(None,
                                IMAGE_ROWS,
                                IMAGE_COLS,
                                NUM_CHANNELS), dtype=tf.float32)
    y = tf.placeholder(dtype=tf.int64)
    logits = global_model(x)
    prediction = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels = y, logits = logits))
    sess = tf.Session()
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())
    global_model.set_weights(global_weights)
    return x, y, sess, prediction, loss

def mal_eval_single(mal_data_X, mal_data_Y, weights):
    x, y, sess, prediction, loss = eval_setup(weights)
    mal_obj_pred = sess.run(prediction, feed_dict={x: mal_data_X})
    target = mal_data_Y[0]
    print("result:")
    print(target)
    print(mal_obj_pred)
    target_conf = mal_obj_pred[:,mal_data_Y][0][0]
    actual = np.argmax(mal_obj_pred, axis=1)[0]
    actual_conf = np.max(mal_obj_pred, axis=1)[0]

    sess.close()
    return target, target_conf, actual, actual_conf

###########################################

######## create mal_obj data ##############



#######################################

############# training #################

if __name__ == "__main__":

    y_train = np_utils.to_categorical(y_train, NUM_CLASSES).astype(np.float32) ##(60000,10) y = label, x = image
    y_test = np_utils.to_categorical(y_test, NUM_CLASSES).astype(np.float32) #(10000,10)
    Y_test_uncat = np.argmax(y_test,axis=1)
    ###############################

    ########## slicing ############

    random_indices = np.random.choice(
            len(X_train), len(X_train), replace=False) #1*n的从0-n的随机数组成的向量，不能重复。
    X_train_permuted = X_train[random_indices]
    Y_train_permuted = y_train[random_indices]
    X_train_shards = np.split(X_train_permuted, k)
    Y_train_shards = np.split(Y_train_permuted, k)

    print(np.shape(X_train_shards[1]))

    data_path = 'data/mal_X_%s_%s.npy' % (dataset,mal_obj)

    if not os.path.exists(data_path):
        mal_data_X ,mal_data_Y, true_labels = mal_data_create(X_test, y_test, Y_test_uncat)

    else: 
        mal_data_X = np.load('data/mal_X_%s_%s.npy' % (dataset, mal_obj))
        mal_data_Y = np.load('data/mal_Y_%s_%s.npy' % (dataset, mal_obj))
        true_labels = np.load('data/true_labels_%s_%s.npy' % (dataset, mal_obj))
        print("Initial classes: %s" % true_labels)
        print("Target classes: %s" % mal_data_Y)

    
    pm = Process(target=master)
    pm.start()
    pm.join()
    
    t = 0
    global_weights = np.load(dir_name + 'global_weights_t%s.npy' % t, allow_pickle=True)

    manager = Manager()
    return_dict = manager.dict()
    return_dict['eval_success'] = 0.0
    return_dict['eval_loss'] = 0.0
    return_dict['mal_suc_count'] = 0

    gpu_id = 0
    num_agents_per_time = int(C*k)
    agents_indices = np.arange(k)

    i = 0
    while return_dict['eval_success'] < max_acc and t < T:
        
        process_list=[]
        curr_agents = np.random.choice(agents_indices,num_agents_per_time,replace=False)


        k = 0
        ######## Agents Trainning ##########
        while k < num_agents_per_time:
            i = curr_agents[k]
            p = Process(target = agents, args=(i, X_train_shards[i], Y_train_shards[i], gpu_id, return_dict, X_test, y_test, t))
            process_list.append(p)
            k += 1
        for item in process_list:
            item.start()
            item.join()
        # Procss mal
        ############
        
        t = t + 1
        ## 每一次t迭代，只是创建一个Process
        
        if 'avg' in gar:
            print("converge strategy Fed avg")
            count = 0
            for k in range(num_agents_per_time):
                if curr_agents[k] != mal_agent_index:
                    if count == 0:
                        ben_delta = alpha_i * return_dict[str(curr_agents[k])]
                        count += 1
                    else:
                        ben_delta += alpha_i * return_dict[str(curr_agents[k])]
            global_weights += alpha_i * return_dict[str(mal_agent_index)] #这里有一个强假设，每次都会选中拜占庭客户端

            global_weights += ben_delta

            np.save(dir_name + 'global_weights_t%s.npy' % t, global_weights)
        #p_eval = Process(target=eval_func)

        print("syssysysysysysysysy")
        print(sys.getsizeof(return_dict))

    
    print("finished")
