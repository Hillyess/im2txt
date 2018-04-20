#!/usr/bin/python
import tensorflow as tf

from config import Config
from model import CaptionGenerator
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data

FLAGS = tf.app.flags.FLAGS


tf.flags.DEFINE_string('input_file_pattern', '../data/flickr8k/train-?????-of-00064',
                       'Image feature extracted using faster rcnn and corresponding captions')
tf.flags.DEFINE_string("optimizer", "Adam",
                        "Adam, RMSProp, Momentum or SGD")
tf.flags.DEFINE_string("attention", "fc1",
                        "fc1, fc2, rnn or bias")
tf.flags.DEFINE_integer("number_of_steps", 10, 
                        "Number of training steps.")
# tf.flags.DEFINE_boolean('train_cnn', False,
#                         'Turn on to train both CNN and RNN. \
#                          Otherwise, only RNN is trained')



tf.logging.set_verbosity(tf.logging.INFO)

def main(argv):
    config = Config()
    config.input_file_pattern = FLAGS.input_file_pattern
    config.optimizer = FLAGS.optimizer
    config.attention = FLAGS.attention

    # Create training directory.
    train_dir = config.save_dir
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info("Creating training directory: %s", train_dir)
        tf.gfile.MakeDirs(train_dir)

    # Build the TensorFlow graph.
    g = tf.Graph()
    with g.as_default():
        # Build the model.
        model = CaptionGenerator(config, mode="train")
        model.build()
    
        # Set up the Saver for saving and restoring model checkpoints.
        saver = tf.train.Saver(max_to_keep=config.max_checkpoints_to_keep)
    
    # Run training.
    tf.contrib.slim.learning.train(
        model.opt_op,
        train_dir,
        log_every_n_steps=config.log_every_n_steps,
        graph=g,
        global_step=model.global_step,
        number_of_steps=FLAGS.number_of_steps,
        summary_op=model.summary,
        save_summaries_secs=600,
        init_fn=None,
        saver=saver)

    # model = CaptionGenerator(config, mode="train")
    # model.build()
    # with tf.Session() as sess:
    # #     sess.run(tf.global_variables_initializer())
    #     sess.run(tf.local_variables_initializer())
    # #     if model.init_fn:
    # #         model.init_fn(sess)
        
    

    #     # Start populating the filename queue.
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)

    #     for i in range(1):
    #         # Retrieve a single instance:
    #         img,cap,mask = sess.run([model.image,model.caption, model.mask])
    #         print(img,type(img),img.shape)
    #         print(cap,type(cap),cap.shape)
    #         print(mask,type(mask),mask.shape)

    #     coord.request_stop()
    #     coord.join(threads)


if __name__ == '__main__':
    tf.app.run()