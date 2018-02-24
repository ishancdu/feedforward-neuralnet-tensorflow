import tensorflow as tf


'''
*The Neural network Archicture starts from here
*The archicture contains 3 layers 
*There are total of 9 neurons in first layer and 5 neurons in the second layer final layer has just one neuron" 
'''

#initializing the smaple input data and the output data that we will feed to the network 
input_train_data=[[1,2,4,2,1,3,4,1,3],[1,3,4,1,3,1,3,1,2],[1,3,2,3,1,4,1,4,2],[1,3,5,1,2,3,1,2,2],[44,43,12,44,23,23,64,23,12],[23,34,21,23,54,24,56,23,43],[23,43,32,43,56,24,56,43,24],[24,16,73,26,26,27,48,16,73]]
labels= [[1],[1],[1],[1],[0],[0],[0],[0]]

#creating the input and output data placeholders for the Neural network(we can input the data using the place holders
inputs=tf.placeholder(dtype=tf.float32,shape=[None,9])
outputs=tf.placeholder(dtype=tf.float32,shape=[None,1])

#layer 1(declaring the weights and biases of the first layer)  
weights_1=tf.get_variable('w1',[9,5],initializer=tf.random_normal_initializer())
bias_1=tf.get_variable('b1',[5],initializer=tf.random_normal_initializer())
Out_1 = tf.nn.sigmoid(tf.matmul(inputs, weights_1) + bias_1)

#layer2(declaring the weights and biases of the output_layer)
weights_2=tf.get_variable('w2',[5,1],initializer=tf.random_normal_initializer())
bias_2=tf.get_variable('b2',[1],initializer=tf.random_normal_initializer())
predicted_Out_2=tf.nn.sigmoid(tf.matmul(Out_1,weights_2) + bias_2)

#initializing the cost function(the cost function is minimized to get the accurate model from the training data)  
cost=tf.reduce_sum((outputs - predicted_Out_2)*(outputs - predicted_Out_2))


#Graient descend we will use the GradientDescentOptimizer(the optimizer function that trains the weights for the neural network to get desired model)(we can pass the lerning rate as the parameter to the function)
training_operation=tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)

#creating the session
sess=tf.Session()

#initializing the Tensorflow Variables(weights and biases)
sess.run(tf.global_variables_initializer())

#neural network training loop
for i in range(30000):
    sess.run(fetches=[training_operation], feed_dict={inputs: input_train_data,outputs:labels})
    #we can see the expected weights and expected outputs of the trained modes every 1000th iteration 
    if i%1000==0:
        print("the weights are : ",sess.run(weights_1))
        print("Expected train Scores : ", sess.run(fetches=predicted_Out_2, feed_dict={inputs: input_train_data}))


#closing the session
sess.close()
