import tensorflow as tf  
import numpy as np  
import os       
 
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
    
       
    # define how many units in each layers  
n_input_layer = 28*28  # input payer  
       
n_layer_1 = 500     # hide layer  
n_layer_2 = 1000    # hide layer  
n_layer_3 = 300     # hide layer  
n_output_layer = 10   # output layer  
     
       
 # define the neural network (feedforward)  
def neural_network(data):
          # define the hide_layer1's weights and biases
    with tf.name_scope('hide_layer1_w_b'):      
         layer_1_w_b = {'w_':tf.Variable(tf.random_normal([n_input_layer, n_layer_1])), 'b_':tf.Variable(tf.random_normal([n_layer_1]))}
        # tf.summary.histogram(data,layer_1_w_b)
        # define the hide_layer2's weights and biases
    with tf.name_scope('hide_layer2_w_b'):
         layer_2_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_1, n_layer_2])), 'b_':tf.Variable(tf.random_normal([n_layer_2]))}  
        # define the hide_layer3's weights and biases
    with tf.name_scope('hide_layer3_w_b'):
         layer_3_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_2, n_layer_3])), 'b_':tf.Variable(tf.random_normal([n_layer_3]))}  
        # define the output_layer's weights and biases
    with tf.name_scope('outputlayer_w_b'):
         layer_output_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_3, n_output_layer])), 'b_':tf.Variable(tf.random_normal([n_output_layer]))}  
         #tf.variable_summaries(outputlayer_w_b)
        # w·x+b
    with tf.name_scope('hide_layer1'):
         layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])  
         layer_1 = tf.nn.relu(layer_1)  # we use relu for active function
    with tf.name_scope('hide_layer2'):
         layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])  
         layer_2 = tf.nn.relu(layer_2 ) # relu active function
    with tf.name_scope('hide_layer3'):
         layer_3 = tf.add(tf.matmul(layer_2, layer_3_w_b['w_']), layer_3_w_b['b_'])  
         layer_3 = tf.nn.relu(layer_3 ) # so on
    with tf.name_scope('outputlayer'):
         layer_output = tf.add(tf.matmul(layer_3, layer_output_w_b['w_']), layer_output_w_b['b_'])      
    return layer_output
    



batch_size = 100  # define batch_Size
with tf.name_scope('input_layer'):      
  X = tf.placeholder('float', [None, 28*28])   
    #[None, 28*28]represent data's hight and width at each image  
  Y = tf.placeholder('float')

  
# back forward  
def train_neural_network(X,Y):  
    predict = neural_network(X)
    tf.summary.histogram('predict',predict) #using tensorboard to visulize the data distribution 
   
    with tf.name_scope('loss'):
         cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    #cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(predict, Y))
    tf.summary.scalar('loss',cost_func)#using tensorboard to visulize the loss changing at the training steps 
    with tf.name_scope('train'):
         optimizer = tf.train.AdamOptimizer().minimize(cost_func)  # learning rate default 0.001
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
             correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
        with tf.name_scope('accuracy'):
             accuracy = tf.reduce_mean(tf.cast(correct,'float'))
    tf.summary.scalar('accuracy',accuracy)#using tensorboard to visulize the accuracy changing at the training steps 
    
    epochs = 13
    with tf.Session() as session:
      #session.run(tf.initialize_all_variables())
#############for this part to check if exist model befor training,if exist than recover it else initialize####        
      saver = tf.train.Saver(max_to_keep = 3) 
      ckpt_path = "/home/mastershaw/program_master/2test/checkpoint"#your path to saver model
      if os.path.exists(ckpt_path):
        print("Restoring Variables from Checkpoint...")
        model_file = tf.train.latest_checkpoint('/home/mastershaw/program_master/2test')
        saver.restore(session,model_file) 
      else:
        print("Initiallizing Variables..")
        session.run(tf.initialize_all_variables())
######################################################################################        
      summary_op = tf.summary.merge_all()
      writer = tf.summary.FileWriter("/home/mastershaw/program_master/2test",session.graph) 
      #you can use a new ternimal typing with the commd-----tensorboard --logdir=file:///home/mastershaw/program_master/2test 
      #!!!!!!be carefull your path are matched in the code##
      max_acc = tf.placeholder('float')
      max_acc = 0
      f = open('/home/mastershaw/program_master/2test/acc.txt','w')#write tha accuray in a text
      epoch_loss = 0  
      for epoch in range(epochs):  
           for i in range( int(mnist.train.num_examples/batch_size) ):
              x, y = mnist.train.next_batch(batch_size)
              #session.run(optimizer, feed_dict={X:x, Y:y})
              _,loss, acc = session.run([optimizer,cost_func, accuracy], feed_dict={X:x, Y:y})
              result = session.run(summary_op,feed_dict={X:x,Y:y})
              writer.add_summary(result,epoch*(int(mnist.train.num_examples/batch_size))+i)
           f.write('epoch:'+str(epoch)+',trainacc:'+str("{:.5f}".format(acc))+'\n')
           #session.run(tf.cond(max_acc < acc, fn1(),fn2()))
           
           print("epoch " + str(epoch) + ",Minibatch Loss = " + \
                  "{:.6f}".format(loss) + ", Training Accuracy = " + \
                  "{:.5f}".format(acc))
##################################to contral save the model of highst accuracy##################
           if acc > max_acc:
               max_acc = acc
               saver.save(session,'/home/mastershaw/program_master/2test/model.ckpt', global_step = epoch)     
               print("training accuracy are rise ->->->updata the model")
           else:
               print("training accuracy are not rise ->->->preserve the current model.......")
           
           model_file = tf.train.latest_checkpoint('/home/mastershaw/program_master/2test')
           saver.restore(session,model_file)
##################################################################################################
           print('test accuracy of present model :',accuracy.eval(feed_dict={X:mnist.test.images , Y:mnist.test.labels}))
           print('\n')
      f.close()
        
train_neural_network(X,Y)  
