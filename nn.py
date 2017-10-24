import tensorflow as tf
import numpy as np
import extract_features as ef

X_train,Y_train,X_test,Y_test=ef.main()
print("Feature Extraction Completed!")

nx=X_train.shape[0]
ny=Y_train.shape[0]

m_train=X_train.shape[1]
m_test=X_test.shape[1]

net_arch=[nx, 15, 10, ny]
num_layers=len(net_arch)-1

def init_params():
    parameters={}
    for i in range(num_layers):
        parameters['W'+str(i+1)]=tf.get_variable('W'+str(i+1), shape=[net_arch[i+1],net_arch[i]], initializer=tf.contrib.layers.xavier_initializer(seed = 1))
        parameters['b'+str(i+1)]=tf.get_variable('b'+str(i+1), shape=[net_arch[i+1],1], initializer=tf.contrib.layers.xavier_initializer(seed = 1))
    return parameters

def create_network(A, parameters):
    for i in range(num_layers):
        W=parameters['W'+str(i+1)]
        b=parameters['b'+str(i+1)]
        if(i<num_layers-1): A=tf.nn.relu(tf.matmul(W,A)+b)
        else: A=tf.matmul(W,A)+b
    return A

def create_placeholders():
    X=tf.placeholder(tf.float32)
    Y=tf.placeholder(tf.float32)
    return (X,Y)

def get_cost(parameters, lambd, Y_hat, Y):
    m=0
    reg_cost=0
    for i in range(num_layers):
        W=parameters['W'+str(i+1)]
        reg_cost=reg_cost+tf.reduce_sum(tf.reduce_sum(tf.square(W), axis=1, keep_dims=True), axis=0, keep_dims=True)
        m=m+int(W.shape[0])*int(W.shape[1])
    reg_cost=(reg_cost*lambd)/(2.0*m)
    logits=tf.transpose(Y_hat)
    labels=tf.transpose(Y)
    cost=tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels) )+reg_cost
    return cost

def train_neural_network(num_epochs, lambd, batch_size):
    print('Training Neural Network......')
    minibatches=ef.create_batches(batch_size, X_train, Y_train)
    parameters=init_params()
    X,Y=create_placeholders()
    Y_hat=create_network(X, parameters)
    cost=get_cost(parameters, lambd, Y_hat, Y)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            epoch_loss = 0
            for minibatch in minibatches:
                x,y=minibatch
                _, c = sess.run([optimizer, cost], feed_dict={X: x, Y: y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of',num_epochs,'loss:',epoch_loss)
        correct = tf.equal(tf.argmax(Y_hat), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Train Accuracy:', accuracy.eval({X:X_train, Y:Y_train}))
        print('Test Accuracy:', accuracy.eval({X:X_test, Y:Y_test}))

train_neural_network(30,35,128)
