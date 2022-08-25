#Recurrent neural network model that trains by reinforcement learning

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import scipy.signal
import threading
import time
import os
import multiprocessing
import sys
#sys.path.append('***')
from python_modules import humanVmonkey_env as hVm

###Misc functions
#Function to specify which gpu to use
def set_gpu(gpu, frac):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=frac)
    return gpu_options

#create fxn that allows worker to make a working copy of the central_network
def get_network_vars(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope) #get the values of the collection from from_scope
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope) #get values from to_scope

    op_holder = [] #list to hold the from_scope values
    for from_var,to_var in zip(from_vars,to_vars): #for each corresponding pair of values in from_scope and to_scope
        op_holder.append(to_var.assign(from_var)) #assign the from_scope value to the to_scope value and append it to op_holder 
    return op_holder #returns the from_scope values in a list

def worker_choose_action(policy_rec):
	action_chosen_index = np.argmax(policy_rec) #choose action with highest prob
	action_chosen = np.zeros(policy_rec.shape[1])
	action_chosen[action_chosen_index] = 1
	return(action_chosen) #1-hot action vector

def worker_act_f(env, state, action):
	r_cur, s_new, trial_term = hVm.do_action_in_environment_f(env,state,action)
	return r_cur, s_new, trial_term

def worker_act(env, state, action):
	r_cur, s_new, trial_term = hVm.do_action_in_environment(env,state,action)
	return r_cur, s_new, trial_term

#reward discounting fxn
def worker_discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

###

#network model
class the_network():
	def __init__(self,state_space, action_space, name, trainer):
		global NETSZ
		with tf.variable_scope(name):
			self.name = name #defines if its the central_network or a worker's working_copy_network

			#tensorflow definition of network - if you give it a state, it'll output policy rec and value pred

			#placeholder for inputs
			self.inputs = tf.placeholder(shape=[None,state_space],dtype=tf.float32) #environmental inputs (first just inputs from state status)

			#make the LSTM - receives from input and outputs to two fully connected layers, 1 for policy and 1 for value
			sizeoflstmcell = NETSZ
			#print("LSTM has " + str(sizeoflstmcell) + " neurons")
			lstm = tf.contrib.rnn.BasicLSTMCell(sizeoflstmcell,state_is_tuple=True) #inputs feed to lstm cell
			#reformats inputs so can go into LSTM
			rnn_in = tf.expand_dims(self.inputs,[0])
			#self.checksize = self.inputs
			#define the lstm states ct and ht
			c_start = np.zeros((1,lstm.state_size.c), np.float32)
			h_start = np.zeros((1,lstm.state_size.h), np.float32)
			self.lstm_state_init = [c_start, h_start] #this is an attribute of self because it will be called when a network is made
			c_in = tf.placeholder(tf.float32, [1,lstm.state_size.c])
			h_in = tf.placeholder(tf.float32, [1,lstm.state_size.h])
			self.state_in = (c_in, h_in) # attribute of self because it will be called when using the network to predict
			state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in) #form of c and h that can be passed back into the LSTM
			batch_size = tf.shape(self.inputs)[:1]
			#connect inputs to lstm and parse lstm outputs
			lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm,rnn_in,initial_state=state_in, sequence_length=batch_size)
			lstm_c, lstm_h = lstm_state
			self.state_out = (lstm_c[:1, :], lstm_h[:1, :]) #will call this to keep track of c and h states
			rnn_out = tf.reshape(lstm_outputs, [-1,sizeoflstmcell]) #output of each of the 256 units in the LSTM; this is the same as lstm_h
			# self.check_rnn_out = rnn_out #this is the same as lstm_h

			#fully connected layers at end to give policy and value
			self.policy_layer_output = slim.fully_connected(rnn_out,action_space,
				activation_fn=tf.nn.softmax,
				weights_initializer=tf.contrib.layers.xavier_initializer(),#normalized_columns_initializer(0.01),
				biases_initializer=None)
			
			self.value_layer_output = slim.fully_connected(rnn_out,1,
				activation_fn=None,
				weights_initializer=tf.contrib.layers.xavier_initializer(),#normalized_columns_initializer(1.0),
				biases_initializer=None)

			#central network exists separate from the worker class (see below) - worker calculates gradients and updates to update central network

			# this is a section of the network that only worker class has - it takes the data they've collected from an episode and
			# uses it to calculate losses and from these, gradients for each parameter (weight)
			# the worker will then send these gradients to the central network and apply them to tune it
			if name != 'central_network':
				#to calc gradients need
					#state, action, policy, value, discounted_reward
				#to get gradients:
					#will give network s, ch_state_in -> these will generate
					#self.policy_layer_output and self.value_layer_output
					#then also give action, discounted_R

				self.A = tf.placeholder(shape=[None,action_space],dtype=tf.float32) #1-hot action taken from this state
				self.R = tf.placeholder(shape=[None,1],dtype=tf.float32) #reward estimate of this state based on rest of episode experience: rt + gamma**1 * rt+1 +...+gamma**k * V(s_end)

				selection_from_policy = tf.reduce_sum(self.policy_layer_output * self.A, [1]) #this is pi(A,S)
				sfp = tf.reshape(selection_from_policy,[-1,1]) #makes it (batch_size, 1)
				advantage = self.R - self.value_layer_output
				#define loss function: Total_loss = Policy_loss + Value_loss + Entropy_loss
				Policy_loss = - tf.log(sfp + 1e-10) * tf.stop_gradient(advantage)
				Value_loss = tf.square(advantage)
				Entropy_loss = - tf.reduce_sum(self.policy_layer_output * tf.log(self.policy_layer_output + 1e-10))

				c_V = 0.05
				c_E = 0.05

				#Total_loss is a vector [#_step_in_experience,1]
				Total_loss = Policy_loss + c_V*Value_loss - c_E*Entropy_loss

				#calculate the gradient of the loss function - use this to update the network
				local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
				self.gradient_loss = tf.gradients(Total_loss,local_vars) #worker will send these gradients to central_network's gradient_list	

				grads_to_apply = self.gradient_loss

				#worker applies the gradients to the central_network
				global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'central_network')
				self.apply_gradients = trainer.apply_gradients(zip(grads_to_apply,global_vars))


	def get_policy_and_value(self, state, ch_state_in, sess):
		feed_dict = {self.inputs:state, self.state_in[0]:ch_state_in[0], self.state_in[1]:ch_state_in[1]}
		p, v, ch_state_out = sess.run([self.policy_layer_output,self.value_layer_output,self.state_out],feed_dict=feed_dict)
		return p, v, ch_state_out	


#worker can only be created if there is a global the_network object defined called central_network and a global variables
#defined: GLOBAL_EPISODE_COUNTER, STATE_SPACE, ACTION_SPACE, GAMMA
class worker():
	def __init__(self,name,trainer):
		self.name = 'worker' + str(name)
		self.env = hVm.make_new_env() #each worker creates own instance of environment to interact with
		self.trainer = trainer
		self.working_copy_network = the_network(state_space=STATE_SPACE, action_space=ACTION_SPACE, name=self.name, trainer=trainer)
		self.working_copy_network_params = get_network_vars('central_network',self.name)
		self.default_graph = tf.get_default_graph()
		self.writestatus = 'w'

	#when episode is done, worker gathers data and processes it, calculates gradients, applies those gradients to the central_network
	def train(self,training_data,bootstrap_value,gamma,sess):
		#first replace the rewards with the discounted-rewards because this is what the network needs to calc losses
		array_training_data = np.array(training_data)
		step_rewards = [ritem for sublist in array_training_data[:,3] for ritem in sublist] #list of the step by step rewards
		step_rewards = step_rewards + [bootstrap_value]

		discR = worker_discount(step_rewards,gamma)[:-1] #cut of the last value because it was just used to give discounted reward estimate

		discR_listed = [[item] for item in discR]
		array_training_data[:,3] = discR_listed

		stacked_states = np.vstack(array_training_data[:,0])
		stacked_action = np.vstack(array_training_data[:,2])
		stacked_reward = np.vstack(array_training_data[:,3])

		feed_dict = {self.working_copy_network.inputs:stacked_states,
			self.working_copy_network.state_in[0]:self.train_rnn_state[0],
			self.working_copy_network.state_in[1]:self.train_rnn_state[1],
			self.working_copy_network.A:stacked_action,
			self.working_copy_network.R:stacked_reward}

		self.train_rnn_state, _ = sess.run([self.working_copy_network.state_out,
			self.working_copy_network.apply_gradients],
			feed_dict=feed_dict)


	def get_experience(self,traindatapath,trialtype,cuedecksize,trialsize,sess,coord):
		print ("Starting " + self.name)
		global GLOBAL_EPISODE_COUNTER, PERFTHRESH
		with sess.as_default(), sess.graph.as_default():   #with this session and session graph set to default
			
			perf_assess = []

			while not coord.should_stop():
				#get copy of the central_network parameters
				sess.run(self.working_copy_network_params)

				training_data = []
				#begin trial
				self.env = hVm.make_new_env(trialtype,cuedecksize,trialsize) #generate new env
				start_state = hVm.get_start_state_from_env(self.env)
				s_cur = start_state
				ch_state_in = self.working_copy_network.lstm_state_init #defines ch_state_in as zeros
				self.train_rnn_state = ch_state_in
				trial_number = 0
				trials_to_train_on = 3
				while trial_number < trials_to_train_on: #every x trials, do training
					#keep track of [s_cur,a_cur,r_cur,s_new,trial_term]
					#feed st to network to get policy and value output
					policy_rec, value_pred, ch_state_out = self.working_copy_network.get_policy_and_value(s_cur,ch_state_in,sess)

					#choose action based on policy_rec
					a_cur = worker_choose_action(policy_rec) #a_cur is the 1-hot action vector
					r_cur, s_new, trial_term = worker_act_f(self.env,s_cur,a_cur)

					new_step_in_environment = [s_cur, ch_state_in, a_cur, r_cur, s_new]
					training_data.append(new_step_in_environment) #this is the data to calculate gradients with

					s_cur = s_new
					ch_state_in = ch_state_out

					if trial_term==True:
						trial_number += 1

					#check if max episode length has been reached
						if trial_number == trials_to_train_on:
							#use s_cur and ch_state_in to get v(s_cur)
							_, value_pred, _ = self.working_copy_network.get_policy_and_value(s_cur,ch_state_in,sess)
							bootstrap_value = value_pred #this is scalar passed to train() below

				if self.name=='worker0': #for writing data to output files
					array_training_data = np.array(training_data)
					stacked_states = np.vstack(array_training_data[:,0])
					stacked_action = np.vstack(array_training_data[:,2])
					stacked_reward = np.vstack(array_training_data[:,3])

					sp=[item[0] for item in stacked_states if item[0]!=0] #non-zero cues
					rp=[item[0] for item in stacked_reward if item[0]!=0] #non-zero rewards
					ap=[np.argmax(item) for item in stacked_action[np.where(stacked_reward!=0)[0]]]

					with open(traindatapath+'_rewards.csv', self.writestatus) as file:
						file.write('\n'.join([str(x) for x in rp]) + '\n')
					with open(traindatapath + '_actions.csv',self.writestatus) as file:
						file.write('\n'.join([str(x) for x in ap]) + '\n')
					with open(traindatapath + '_cues.csv',self.writestatus) as file:
						file.write('\n'.join([str(x) for x in sp]) + '\n')
					if self.writestatus=='w':
						self.writestatus='a'

					if stop_on_perf_ass==True: #if using performance criteria as stop criteria
						perf_assess = perf_assess + [1 if item==5 else 0 for item in rp] #non-zero rewards
						if len(perf_assess)>=100:
							perf_assess = perf_assess[-100:]
							if np.nanmean(perf_assess) >= PERFTHRESH:
								coord.request_stop()

				#when episode==done
				#train processes the training data then runs it through the worker's network to calculate gradients,
				#calculates gradients, then takes these to the central_network and uses them to update the central_network
				self.train(training_data,bootstrap_value,GAMMA,sess)

				GLOBAL_EPISODE_COUNTER += trial_number
				if GLOBAL_EPISODE_COUNTER % 1000 == 0:
					print('GEC: ' + str(GLOBAL_EPISODE_COUNTER))
				if GLOBAL_EPISODE_COUNTER >= EPS_TO_TRAIN_ON:
					coord.request_stop()

	def test(self,sess,testdatapath,getnumep,trialtype,cuedecksize,trialsize):
		print ("Starting " + self.name + " for testing")
		with sess.as_default(), sess.graph.as_default():
			sess.run(self.working_copy_network_params)
			ep_num=0
			trials_to_reset_on = getnumep #1 means reset every trial; getnumep means never reset
			while ep_num < getnumep:
				training_data = []
				#begin episode
				self.env = hVm.make_new_env(trialtype,cuedecksize,trialsize) #for each ep make a new env obj to get new baseline probs of A and B
				start_state = hVm.get_start_state_from_env(self.env)
				s_cur = start_state
				ch_state_in = self.working_copy_network.lstm_state_init #defines ch_state_in as zeros
				self.train_rnn_state = ch_state_in #to do training need to create this self variable to pass the LSTM state in and out
				trial_num = 0
				trial_term = False
				while trial_num < trials_to_reset_on:
					#keep track of [s_cur,a_cur,r_cur,s_new,trial_term]
					#feed st to network to get policy and value output
					policy_rec, value_pred, ch_state_out = self.working_copy_network.get_policy_and_value(s_cur,ch_state_in,sess)

					#choose action based on policy_rec
					a_cur = worker_choose_action(policy_rec) #a_cur is the 1-hot action vector
					r_cur, s_new, trial_term = worker_act(self.env,s_cur,a_cur)

					new_step_in_environment = [s_cur, ch_state_out, a_cur, r_cur, s_new, trial_term]
					training_data.append(new_step_in_environment)

					s_cur = s_new
					ch_state_in = ch_state_out

					#check if max episode length has been reached
					if trial_term == True: #when the episode is over
						ep_num += 1
						trial_num += 1

				#for output files
				td = np.array(training_data)
				stacked_states = np.vstack(td[:,0])
				stacked_action = np.vstack(td[:,2])
				stacked_reward = np.vstack(td[:,3])

				sp=[item[0] for item in stacked_states if item[0]!=0] #non-zero cues
				rp=[item[0] for item in stacked_reward if item[0]!=0] #non-zero rewards
				ap=[np.argmax(item) for item in stacked_action[np.where(stacked_reward!=0)[0]]]

				with open(testdatapath+'_rewards.csv', self.writestatus) as file:
					file.write('\n'.join([str(x) for x in rp]) + '\n')
				with open(testdatapath + '_actions.csv',self.writestatus) as file:
					file.write('\n'.join([str(x) for x in ap]) + '\n')
				with open(testdatapath + '_cues.csv',self.writestatus) as file:
					file.write('\n'.join([str(x) for x in sp]) + '\n')
				with open(testdatapath + '_rnn_c_activity.csv',self.writestatus) as file:
				if self.writestatus=='w':
					self.writestatus='a'


			print ("Done testing")
			if self.writestatus=='a':
				self.writestatus='w'

######################################################################################################
######################################################MAIN############################################


#main
tf.reset_default_graph()

#first arg is the network size: sys.argv[1]
#second arg is the number of episodes to use for training: sys.argv[2]
#third arg is the performance threshold: sys.argv[3]
#fourth arg is run number: sys.argv[4]
#fifth arg is the gpu: sys.argv[5]
#so: python3 a3c_btsuda_humanVmonkey_task.py [NETSZ] [PERFTHRESH] [RUNNO] [GPU]
#e.g. python3 a3c_btsuda_humanVmonkey_task.py 25 5000 1.0 0 0

NETSZ = int(sys.argv[1]) #number of neurons in the RNN
EPS_TO_TRAIN_ON = int(sys.argv[2])
PERFTHRESH = float(sys.argv[3])
runno = int(sys.argv[4])

TRAINER = tf.train.AdamOptimizer(learning_rate=1e-3)
GAMMA = 0
NUMBER_OF_WORKERS = 1
#query the environment to see what the state-space is --need to know the input size to create the network
STATE_SPACE = hVm.get_state_space() #size of state space
ACTION_SPACE = hVm.get_action_space() #size of action space

stop_on_perf_ass = False #False means train for the number of episodes regardless of performance; true means train until a performance level is reached

#create central_network
central_network = the_network(state_space=STATE_SPACE, action_space=ACTION_SPACE, name='central_network', trainer=None)

#create worker that runs the network in the environment
workers = []
for i in range(NUMBER_OF_WORKERS):
	workers.append(worker(name=i,trainer=TRAINER))
the_test_worker = worker(name='_testman',trainer=TRAINER)

trialtype = 0 #0 is Same-Different; 1 is Match-First; 2 is Match-Any; 3 is a shaping task, same as match-any but with random zero cues mixed in
trialsize = 2 #2 for Same-Different; for Match type trials either 4 or 8
cuedecksize = 0 #0 for small, 1 for large

#Tell what to do with network - save, restore, train, test------------------------------------------------------------------
#saver
filerootpath = '/destinationdir/'
saver = tf.train.Saver()
dosave = False #set to True to save; False to not save
savepath = filerootpath + 'models/working_model.ckpt'
dorestore = False
restorepath = filerootpath + 'models/working_model.ckpt'
trainnetwork = True #set to True if want to train network; False if just restoring and network and testing it
testnetwork = True #set to True if want to do test run of network
testdatapath = filerootpath + 'data/test' #give path and prefix for data to save
traindatapath = filerootpath + 'data/train'
getnumep = 5000

#---------------------------------------------------------------------------------------------------------------------------

#create the tf.Session
UGPU = str(sys.argv[5])
with tf.Session(config=tf.ConfigProto(gpu_options=set_gpu(UGPU, .08))) as sess:
# with tf.Session() as sess:
	GLOBAL_EPISODE_COUNTER = 0
	if dorestore == True:
		saver.restore(sess,restorepath)
		print("Model restored: " + restorepath)
	elif dorestore == False:
		sess.run(tf.global_variables_initializer()) #initialize variables

	#if training network
	if trainnetwork == True:
		coord = tf.train.Coordinator()
		worker_threads = [] #list of worker threads; can parallelize the running of this with multiple threads if wanted
		t0=time.time()
		for worker in workers:
			worker_experience = lambda: worker.get_experience(traindatapath=traindatapath,trialtype=trialtype,cuedecksize=cuedecksize,trialsize=trialsize,sess=sess,coord=coord)
			newthread = threading.Thread(target=(worker_experience))
			newthread.start()
			time.sleep(0.5)
			worker_threads.append(newthread)
		coord.join(worker_threads)
	if dosave == True:
		save_path = saver.save(sess, savepath)
		print("saved to " + savepath)
	if testnetwork == True:
		the_test_worker.test(sess=sess,testdatapath=testdatapath,getnumep=getnumep,trialtype=trialtype,cuedecksize=cuedecksize,trialsize=trialsize)












