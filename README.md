
Protein-DNA binding prediction (CTCF)
=============

Goal: implement DeepBing with TensorFlow
------------


***
## DeepBind:

> <img src="http://www.nature.com/nbt/journal/v33/n8/images/nbt.3300-F2.jpg" width="80%">
> Source: http://www.nature.com/nbt/journal/v33/n8/images/nbt.3300-F2.jpg



```python
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
```


```python
K_FOLD = [4,1] ## A 5 fold
TRAIN_PICKLE = '../../../train.pickle'
TEST_PICKLE  = '../../../test.pickle'

with open(TRAIN_PICKLE, 'rb') as f:
    save = pickle.load(f)
    seq  = save['seq']
    label= save['label']
    len1 = seq.shape[0]
    del save
    print('Loaded train.pickle: ',seq.shape,label.shape)

with open(TEST_PICKLE, 'rb') as f:
    save = pickle.load(f)
    test_seq = save['seq']
    test_label = save['label']
    len2 = test_seq.shape[0]
    del save
    print('Loaded test.pickle: ',test_seq.shape)


_ = int((len1-len1%(sum(K_FOLD)))/(sum(K_FOLD))) ## 15506
r_train = range(0,4*_)
r_valid = range(4*_+1,len1)
r_test  = range(0,len2)

train_dataset, train_labels = seq[r_train,], label[r_train,]  
valid_dataset, valid_labels = seq[r_valid,], label[r_valid,]
test_dataset, test_labels   = seq[r_test,] , label[r_test,]  

print('\nTraining set:\t',     r_train,  train_dataset.shape,  train_labels.shape)
print('Validation set:\t', r_valid,  valid_dataset.shape,  valid_labels.shape)
print('Test set:\t',         r_test,   test_dataset.shape,   test_labels.shape)

  
```

    Loaded train.pickle:  (77531, 121, 4) (77531, 2)
    Loaded test.pickle:  (19383, 121, 4)
    
    Training set:	 range(0, 62024) (62024, 121, 4) (62024, 2)
    Validation set:	 range(62025, 77531) (15506, 121, 4) (15506, 2)
    Test set:	 range(0, 19383) (19383, 121, 4) (19383, 2)


Reformat into a TensorFlow-friendly shape:
- convolutions need the image data formatted as a cube (width by height by #channels)
- labels as float 1-hot encodings.


```python
import numpy as np

num_channels = 1
def reformat(dataset, labels):
    dataset = dataset.reshape((-1, 121, 4, num_channels)).astype(np.float32)
#   labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
```

    Training set (62024, 121, 4, 1) (62024, 2)
    Validation set (15506, 121, 4, 1) (15506, 2)
    Test set (19383, 121, 4, 1) (19383, 2)



```python
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
```

### DeepBind CNN Model
> <img src="http://www.nature.com/nbt/journal/v33/n8/images/nbt.3300-SF1.jpg" width="90%">
>... Shown is an example with __batch_size=5, motif_len=6, num_motifs=4, num_models=3__. Sequences are padded with ‘N’s so that the motif scan operation can find detections at both extremities. Yellow cells represent the reverse complement of the input located above; both strands are fed to the model, and the strand with the maximum score is used for the output prediction (the max strand stage). The output dimension of the pool stage, depicted as num_motifs (*), depends on whether “max” or “max and avg” pooling was used.
> 
> Image source: http://www.nature.com/nbt/journal/v33/n8/fig_tab/nbt.3300_SF1.html



```python
image_size   = [121,4]  ## 101bps, plus 10bps frenking on both end
num_labels   = 2        ## bind or not (1 or 0)
batch_size   = 256      ## TODO: try with double strand input!
filter_size  = [11,4]   ## Motif detector length = 11 (about 1.5 times of expected motif length)
depth        = 16       ## Number of motif detector (num_motif) = 16
num_hidden   = 32       ## 32 ReLU units of no hidden layer at all

graph = tf.Graph()

with graph.as_default():

    # Input data.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size[0], image_size[1], num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
  
    # conv filter, shape=(11,3)
    filter_W = tf.Variable(tf.truncated_normal([11, 3, num_channels, depth], stddev=0.1)) ## difference ?
    filter_b = tf.Variable(tf.zeros([depth]))

    # NN layer, shape=()
    hidden_W = tf.Variable(tf.truncated_normal([1920, 32], stddev=0.1))
    hidden_b = tf.Variable(tf.constant(1.0, shape=[32]))
    
    # output layer, shape=()
    output_W = tf.Variable(tf.truncated_normal([32, 2], stddev=0.1))
    output_b = tf.Variable(tf.constant(1.0, shape=[2]))
 

    ## Model.
    def model(data):
        print('           [batch, height, width, channel]') 
        print('data:     ', data.get_shape().as_list())
        
        # Convolution: (121,4,1) ---- 3x6 filter ---> (121,4,4)
        conv = tf.nn.conv2d(data, filter_W, [1, 1, 1, 1], padding='SAME')
        print('conv:     ', conv.get_shape().as_list())
        
        relu = tf.nn.relu(conv + filter_b)
        print('relu:     ', relu.get_shape().as_list())

        pool = tf.nn.max_pool(relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        print('pooling:  ', pool.get_shape().as_list())
        
        shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
        print('reshape:  ', reshape.get_shape().as_list())
        
        hidden = tf.nn.relu(tf.matmul(reshape, hidden_W) + hidden_b)
        print('hidden:   ', hidden.get_shape().as_list())
        
        output = tf.nn.relu(tf.matmul(hidden, output_W) + output_b)
        print('output:   ', output.get_shape().as_list(),'\n\n')
        return output
  
    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))
```

               [batch, height, width, channel]
    data:      [256, 121, 4, 1]
    conv:      [256, 121, 4, 16]
    relu:      [256, 121, 4, 16]
    pooling:   [256, 60, 2, 16]
    reshape:   [256, 1920]
    hidden:    [256, 32]
    output:    [256, 2] 
    
    
               [batch, height, width, channel]
    data:      [15506, 121, 4, 1]
    conv:      [15506, 121, 4, 16]
    relu:      [15506, 121, 4, 16]
    pooling:   [15506, 60, 2, 16]
    reshape:   [15506, 1920]
    hidden:    [15506, 32]
    output:    [15506, 2] 
    
    
               [batch, height, width, channel]
    data:      [19383, 121, 4, 1]
    conv:      [19383, 121, 4, 16]
    relu:      [19383, 121, 4, 16]
    pooling:   [19383, 60, 2, 16]
    reshape:   [19383, 1920]
    hidden:    [19383, 32]
    output:    [19383, 2] 
    
    



```python
def run_session(num_steps = 1000):
    
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('\t',     'Minibatch\t', 'Minibatch\t',  'Validation')
        print('Step\t', 'Loss\t\t',      'Accuracy\t',  'Accuracy')
        for step in range(num_steps+1):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 50 == 0):
                print('%d\t %f\t %.1f%%\t\t %.1f%%\t' % (
                    step,
                    l,
                    accuracy(predictions, batch_labels),
                    accuracy(valid_prediction.eval(), valid_labels)
                ))
        print('*** TEST ACCURACY: %.1f%% ***' % accuracy(test_prediction.eval(), test_labels))
        prediction_ndarray = test_prediction.eval()

run_session(num_steps = 1000)
```

    	 Minibatch	 Minibatch	 Validation
    Step	 Loss		 Accuracy	 Accuracy
    0	 0.725892	 54.3%		 49.4%	
    50	 0.702979	 48.4%		 51.7%	
    100	 0.705922	 49.6%		 53.3%	
    150	 0.687450	 55.1%		 55.2%	
    200	 0.680434	 58.2%		 55.9%	
    250	 0.668010	 60.2%		 51.9%	
    300	 0.665098	 59.4%		 59.0%	
    350	 0.655646	 62.1%		 58.4%	
    400	 0.749678	 46.9%		 51.8%	
    450	 0.621250	 68.8%		 60.7%	
    500	 0.611697	 71.9%		 63.7%	
    550	 0.680555	 56.2%		 60.9%	
    600	 0.641611	 62.1%		 65.4%	
    650	 0.610430	 68.0%		 66.7%	
    700	 0.607095	 67.2%		 62.7%	
    750	 0.652847	 59.8%		 59.3%	
    800	 0.597264	 65.6%		 66.1%	
    850	 0.571627	 73.8%		 67.0%	
    900	 0.610575	 63.3%		 70.1%	
    950	 0.586581	 70.7%		 70.6%	
    1000	 0.560463	 71.5%		 70.6%	
    *** TEST ACCURACY: 71.8% ***



```python
run_session(num_steps = 2000)
```

    	 Minibatch	 Minibatch	 Validation
    Step	 Loss		 Accuracy	 Accuracy
    0	 0.704212	 46.5%		 50.3%	
    50	 0.711170	 50.0%		 51.7%	
    100	 0.702906	 50.0%		 53.5%	
    150	 0.681891	 53.9%		 55.6%	
    200	 0.694169	 52.7%		 57.0%	
    250	 0.661604	 60.5%		 52.5%	
    300	 0.652424	 63.7%		 60.2%	
    350	 0.648504	 60.9%		 58.5%	
    400	 0.758930	 47.7%		 51.2%	
    450	 0.606714	 69.9%		 61.2%	
    500	 0.602666	 68.0%		 64.5%	
    550	 0.666663	 55.1%		 63.6%	
    600	 0.608256	 68.4%		 66.3%	
    650	 0.617038	 64.8%		 67.7%	
    700	 0.613045	 63.3%		 63.1%	
    750	 0.659635	 58.6%		 59.8%	
    800	 0.584620	 68.0%		 67.6%	
    850	 0.565616	 70.7%		 67.7%	
    900	 0.610997	 65.2%		 70.2%	
    950	 0.587207	 72.3%		 70.8%	
    1000	 0.595937	 65.6%		 68.7%	
    1050	 0.650272	 62.9%		 63.7%	
    1100	 0.549253	 73.0%		 71.5%	
    1150	 0.647494	 62.5%		 65.4%	
    1200	 0.531154	 73.0%		 71.1%	
    1250	 0.545054	 73.0%		 70.8%	
    1300	 0.604557	 70.7%		 67.9%	
    1350	 0.572152	 69.1%		 70.9%	
    1400	 0.535430	 70.7%		 68.5%	
    1450	 0.517623	 73.8%		 71.8%	
    1500	 0.623158	 64.8%		 70.5%	
    1550	 0.529861	 73.0%		 73.4%	
    1600	 0.554052	 71.1%		 72.7%	
    1650	 0.602054	 64.8%		 66.2%	
    1700	 0.559003	 70.3%		 72.6%	
    1750	 0.520869	 76.2%		 72.5%	
    1800	 0.537416	 74.2%		 72.5%	
    1850	 0.587606	 64.8%		 68.4%	
    1900	 0.552073	 69.5%		 72.7%	
    1950	 0.624915	 61.7%		 65.2%	
    2000	 0.531805	 71.1%		 71.7%	
    *** TEST ACCURACY: 72.9% ***



```python
## Export file
DIST = './Prediction_19383.data'


print(prediction_ndarray)
results = np.argmax(prediction_ndarray, axis=1)
print('results: ', results)

with open(DIST, 'w') as f:
    for r in results:
        f.write(str(r)+'\n')        
        f.close
    print('Prediction result saved to',DIST)
```

    [[ 0.36532587  0.63467413]
     [ 0.36353308  0.63646686]
     [ 0.76541942  0.23458053]
     ..., 
     [ 0.29438186  0.70561814]
     [ 0.79588997  0.20411004]
     [ 0.50122339  0.49877667]]
    results:  [1 1 0 ..., 1 0 0]
    Prediction result saved to ./Prediction_19383.data



```python

```
