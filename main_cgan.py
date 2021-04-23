import os
## GAN Variants
from WGAN_GP import WGAN_GP
from WGAN_GP2 import WGAN_GP2
from WGAN_GP3 import WGAN_GP3
#from WGAN_GP4 import WGAN_GP4
#from WGAN_GP5 import WGAN_GP5
from CGAN1 import CGAN1
from CGAN2_minus_11 import CGAN2_minus_11
from CGAN2_11 import CGAN2_11
from CGAN2_minus_13 import CGAN2_minus_13
from CGAN2_13 import CGAN2_13
from CGAN2_minus_211 import CGAN2_minus_211
from CGAN2_211 import CGAN2_211
from CGAN2_2212 import CGAN2_2212
from CGAN2_2212_onefifth import CGAN2_2212_onefifth
from CGAN2_2212_onethird import CGAN2_2212_onethird
from SAGAN import SAGAN

from utils_cgan import show_all_variables
from utils_cgan import check_folder

import h5py
import tensorflow as tf
import time
import argparse
import yaml

"""parsing and configuration"""
def parse_args():
    desc = "Select different configuration of GAN network"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--parameter', type=str, default='hyperparam1.yml',
                        #choices=['hyperparam1.yml', 'hyperparam2.yml',  'hyperparam3.yml',
                        #         'hyperparam4.yml', 'hyperparam5.yml', 'hyperparam6.yml',
                        #         'hyperparam7.yml', 'hyperparam8.yml', 'hyperparam9.yml',
                        #         'hyperparam10.yml', 'hyperparam11.yml', 'hyperparam12.yml',
                        #        'hyperparam13.yml'],
                        help='GAN network hyperparameters', required=True)
    parser.add_argument('--structure', type=str, default='structure1.yml',
                        #choices=['structure1.yml', 'structure1.yml', 'structure1.yml'
                        #'structure1.yml', 'structure1.yml', 'structure2.yml'],
                        help='GAN network structures', required=True)
    return parser.parse_args()

def read_yml(yml_type, yml_file):
    yml_path=os.path.join("./yml/"+yml_type,yml_file)
    with open(yml_path) as yml:
        yml_data = yaml.load(yml)
    return yml_data,yml_path

"""checking arguments"""
"""main"""
def main():
    # read config file
    args = parse_args()
    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # declare instance for GAN
        parameter ,parameter_path= read_yml('hyper_parameter', args.parameter)
        structure ,structure_path= read_yml('structure', args.structure)
        gan_type = parameter['gan_type']
        if gan_type == 'GAN':
            gan = WGAN_GP3(sess, epoch=parameter['epoch'], batch_size=parameter['batch_size'], z_dim=parameter['z_dim'], dataset_name=parameter['dataset'],
                      checkpoint_dir=parameter['checkpoint_dir'], result_dir=parameter['result_dir'], log_dir=parameter['log_dir'],test_dir = parameter['test_dir'],
                      input_height=parameter['input_height'], input_width=parameter['input_width'], output_height=parameter['output_height'],
                      output_width=parameter['output_width'], lambd=parameter['lambd'], disc_iters=parameter['disc_iters'],
                      learning_rate=parameter['learning_rate'], beta1=parameter['beta1'], epoch_num=parameter['epoch_num'],input_num=parameter['input_num'],
                      output_batch=parameter['output_batch'],structure=structure, parameter_path=parameter_path, structure_path=structure_path)
        elif gan_type == 'WGAN_GP3':
            gan = WGAN_GP3(sess, epoch=parameter['epoch'], batch_size=parameter['batch_size'], z_dim=parameter['z_dim'], dataset_name=parameter['dataset'],
                      checkpoint_dir=parameter['checkpoint_dir'], result_dir=parameter['result_dir'], log_dir=parameter['log_dir'],test_dir = parameter['test_dir'],
                      input_height=parameter['input_height'], input_width=parameter['input_width'], output_height=parameter['output_height'],
                      output_width=parameter['output_width'], lambd=parameter['lambd'], disc_iters=parameter['disc_iters'],
                      learning_rate=parameter['learning_rate'], beta1=parameter['beta1'],epoch_num=parameter['epoch_num'],input_num=parameter['input_num'],
                      structure=structure, parameter_path=parameter_path, structure_path=structure_path)

        #elif gan_type == 'WGAN_GP4':
        #    gan = WGAN_GP4(sess, epoch=parameter['epoch'], batch_size=parameter['batch_size'], z_dim=parameter['z_dim'], dataset_name=parameter['dataset'],
        #              checkpoint_dir=parameter['checkpoint_dir'], result_dir=parameter['result_dir'], log_dir=parameter['log_dir'],test_dir = parameter['test_dir'],
        #              input_height=parameter['input_height'], input_width=parameter['input_width'], output_height=parameter['output_height'],
        #              output_width=parameter['output_width'], lambd=parameter['lambd'], disc_iters=parameter['disc_iters'],
        #              learning_rate=parameter['learning_rate'], beta1=parameter['beta1'],epoch_num=parameter['epoch_num'],input_num=parameter['input_num'],
        #              structure=structure, parameter_path=parameter_path, structure_path=structure_path)

        #elif gan_type == 'WGAN_GP5':
        #    gan = WGAN_GP5(sess, epoch=parameter['epoch'], batch_size=parameter['batch_size'], z_dim=parameter['z_dim'], dataset_name=parameter['dataset'],
        #              checkpoint_dir=parameter['checkpoint_dir'], result_dir=parameter['result_dir'], log_dir=parameter['log_dir'],test_dir = parameter['test_dir'],
        #              input_height=parameter['input_height'], input_width=parameter['input_width'], output_height=parameter['output_height'],
        #             output_width=parameter['output_width'], lambd=parameter['lambd'], disc_iters=parameter['disc_iters'],
        #              learning_rate=parameter['learning_rate'], beta1=parameter['beta1'],epoch_num=parameter['epoch_num'],input_num=parameter['input_num'],
        #              structure=structure, parameter_path=parameter_path, structure_path=structure_path)
        elif gan_type == 'CGAN1':
            gan = CGAN1(sess, epoch=parameter['epoch'], batch_size=parameter['batch_size'], z_dim=parameter['z_dim'],y_dim=parameter['y_dim'],dataset_name=parameter['dataset'],
                      checkpoint_dir=parameter['checkpoint_dir'], result_dir=parameter['result_dir'], log_dir=parameter['log_dir'],test_dir = parameter['test_dir'],
                      input_height=parameter['input_height'], input_width=parameter['input_width'], output_height=parameter['output_height'],
                      output_width=parameter['output_width'], lambd=parameter['lambd'], disc_iters=parameter['disc_iters'],
                      learning_rate=parameter['learning_rate'], beta1=parameter['beta1'],epoch_num=parameter['epoch_num'],input_num=parameter['input_num'],
                      structure=structure, parameter_path=parameter_path, structure_path=structure_path)

        elif gan_type == 'CGAN2_minus_11':
            gan = CGAN2_minus_11(sess, epoch=parameter['epoch'], batch_size=parameter['batch_size'], z_dim=parameter['z_dim'],y_dim=parameter['y_dim'],dataset_name=parameter['dataset'],
                      checkpoint_dir=parameter['checkpoint_dir'], result_dir=parameter['result_dir'], log_dir=parameter['log_dir'],test_dir = parameter['test_dir'],
                      input_height=parameter['input_height'], input_width=parameter['input_width'], output_height=parameter['output_height'],
                      output_width=parameter['output_width'], lambd=parameter['lambd'], disc_iters=parameter['disc_iters'],
                      learning_rate=parameter['learning_rate'], beta1=parameter['beta1'],epoch_num=parameter['epoch_num'],input_num=parameter['input_num'],
                      structure=structure, parameter_path=parameter_path, structure_path=structure_path)

        elif gan_type == 'CGAN2_11':
            gan = CGAN2_11(sess, epoch=parameter['epoch'], batch_size=parameter['batch_size'], z_dim=parameter['z_dim'],y_dim=parameter['y_dim'],dataset_name=parameter['dataset'],
                      checkpoint_dir=parameter['checkpoint_dir'], result_dir=parameter['result_dir'], log_dir=parameter['log_dir'],test_dir = parameter['test_dir'],
                      input_height=parameter['input_height'], input_width=parameter['input_width'], output_height=parameter['output_height'],
                      output_width=parameter['output_width'], lambd=parameter['lambd'], disc_iters=parameter['disc_iters'],
                      learning_rate=parameter['learning_rate'], beta1=parameter['beta1'],epoch_num=parameter['epoch_num'],input_num=parameter['input_num'],
                      structure=structure, parameter_path=parameter_path, structure_path=structure_path)

        elif gan_type == 'CGAN2_minus_13':
            gan = CGAN2_minus_13(sess, epoch=parameter['epoch'], batch_size=parameter['batch_size'], z_dim=parameter['z_dim'],y_dim=parameter['y_dim'],dataset_name=parameter['dataset'],
                      checkpoint_dir=parameter['checkpoint_dir'], result_dir=parameter['result_dir'], log_dir=parameter['log_dir'],test_dir = parameter['test_dir'],
                      input_height=parameter['input_height'], input_width=parameter['input_width'], output_height=parameter['output_height'],
                      output_width=parameter['output_width'], lambd=parameter['lambd'], disc_iters=parameter['disc_iters'],
                      learning_rate=parameter['learning_rate'], beta1=parameter['beta1'],epoch_num=parameter['epoch_num'],input_num=parameter['input_num'],
                      structure=structure, parameter_path=parameter_path, structure_path=structure_path)

        elif gan_type == 'CGAN2_13':
            gan = CGAN2_13(sess, epoch=parameter['epoch'], batch_size=parameter['batch_size'], z_dim=parameter['z_dim'],y_dim=parameter['y_dim'],dataset_name=parameter['dataset'],
                      checkpoint_dir=parameter['checkpoint_dir'], result_dir=parameter['result_dir'], log_dir=parameter['log_dir'],test_dir = parameter['test_dir'],
                      input_height=parameter['input_height'], input_width=parameter['input_width'], output_height=parameter['output_height'],
                      output_width=parameter['output_width'], lambd=parameter['lambd'], disc_iters=parameter['disc_iters'],
                      learning_rate=parameter['learning_rate'], beta1=parameter['beta1'],epoch_num=parameter['epoch_num'],input_num=parameter['input_num'],
                      structure=structure, parameter_path=parameter_path, structure_path=structure_path)

        elif gan_type == 'CGAN2_minus_211':
            gan = CGAN2_minus_211(sess, epoch=parameter['epoch'], batch_size=parameter['batch_size'], z_dim=parameter['z_dim'],y_dim=parameter['y_dim'],dataset_name=parameter['dataset'],
                      checkpoint_dir=parameter['checkpoint_dir'], result_dir=parameter['result_dir'], log_dir=parameter['log_dir'],test_dir = parameter['test_dir'],
                      input_height=parameter['input_height'], input_width=parameter['input_width'], output_height=parameter['output_height'],
                      output_width=parameter['output_width'], lambd=parameter['lambd'], disc_iters=parameter['disc_iters'],
                      learning_rate=parameter['learning_rate'], beta1=parameter['beta1'],epoch_num=parameter['epoch_num'],input_num=parameter['input_num'],
                      structure=structure, parameter_path=parameter_path, structure_path=structure_path)

        elif gan_type == 'CGAN2_211':
            gan = CGAN2_211(sess, epoch=parameter['epoch'], batch_size=parameter['batch_size'], z_dim=parameter['z_dim'],y_dim=parameter['y_dim'],dataset_name=parameter['dataset'],
                      checkpoint_dir=parameter['checkpoint_dir'], result_dir=parameter['result_dir'], log_dir=parameter['log_dir'],test_dir = parameter['test_dir'],
                      input_height=parameter['input_height'], input_width=parameter['input_width'], output_height=parameter['output_height'],
                      output_width=parameter['output_width'], lambd=parameter['lambd'], disc_iters=parameter['disc_iters'],
                      learning_rate=parameter['learning_rate'], beta1=parameter['beta1'],epoch_num=parameter['epoch_num'],input_num=parameter['input_num'],
                      structure=structure, parameter_path=parameter_path, structure_path=structure_path)

        elif gan_type == 'CGAN2_2212':
            gan = CGAN2_2212(sess, epoch=parameter['epoch'], batch_size=parameter['batch_size'], z_dim=parameter['z_dim'],y_dim=parameter['y_dim'],dataset_name=parameter['dataset'],
                      checkpoint_dir=parameter['checkpoint_dir'], result_dir=parameter['result_dir'], log_dir=parameter['log_dir'],test_dir = parameter['test_dir'],
                      input_height=parameter['input_height'], input_width=parameter['input_width'], output_height=parameter['output_height'],
                      output_width=parameter['output_width'], lambd=parameter['lambd'], disc_iters=parameter['disc_iters'],
                      learning_rate=parameter['learning_rate'], beta1=parameter['beta1'],epoch_num=parameter['epoch_num'],input_num=parameter['input_num'],
                      structure=structure, parameter_path=parameter_path, structure_path=structure_path)

        elif gan_type == 'CGAN2_2212_onefifth':
            gan = CGAN2_2212_onefifth(sess, epoch=parameter['epoch'], batch_size=parameter['batch_size'], z_dim=parameter['z_dim'],y_dim=parameter['y_dim'],dataset_name=parameter['dataset'],
                      checkpoint_dir=parameter['checkpoint_dir'], result_dir=parameter['result_dir'], log_dir=parameter['log_dir'],test_dir = parameter['test_dir'],
                      input_height=parameter['input_height'], input_width=parameter['input_width'], output_height=parameter['output_height'],
                      output_width=parameter['output_width'], lambd=parameter['lambd'], disc_iters=parameter['disc_iters'],
                      learning_rate=parameter['learning_rate'], beta1=parameter['beta1'],epoch_num=parameter['epoch_num'],input_num=parameter['input_num'],
                      structure=structure, parameter_path=parameter_path, structure_path=structure_path)

        elif gan_type == 'CGAN2_2212_onethird':
            gan = CGAN2_2212_onethird(sess, epoch=parameter['epoch'], batch_size=parameter['batch_size'], z_dim=parameter['z_dim'],y_dim=parameter['y_dim'],dataset_name=parameter['dataset'],
                      checkpoint_dir=parameter['checkpoint_dir'], result_dir=parameter['result_dir'], log_dir=parameter['log_dir'],test_dir = parameter['test_dir'],
                      input_height=parameter['input_height'], input_width=parameter['input_width'], output_height=parameter['output_height'],
                      output_width=parameter['output_width'], lambd=parameter['lambd'], disc_iters=parameter['disc_iters'],
                      learning_rate=parameter['learning_rate'], beta1=parameter['beta1'],epoch_num=parameter['epoch_num'],input_num=parameter['input_num'],
                      structure=structure, parameter_path=parameter_path, structure_path=structure_path)



        gan.build_model()

        # show network architecture
        show_all_variables()

        # launch the graph in a session
        time_train = time.time()
        gan.train()
        print(" [*] Training finished!")
        time_generate = time.time()
        gan.output_csv(parameter['output_batch'],parameter['label_num'],parameter['data_name'])
        print(" [*] Testing finished!")
        print("training time:%.2f, generating time:%.2f"%(time.time()-time_train, time.time()-time_generate))


if __name__ == '__main__':
    main()
