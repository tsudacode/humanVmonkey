#Wittig et al. task environment
import numpy as np

class humanVmonkey_env():
	def __init__(self,trialtype=0,cuedecksize=0,trialsize=4):
		self.trialtype = trialtype #0 is Same-Different, 1 is Match-First, 2 is Match-Any
		self.cuedecksize = cuedecksize #size of cue deck; 0 means small (2 for Same-Different, 5 for Match)
									#1 means large (100 possible cues)
		self.trialsize = trialsize #length of trial seq for Match trials; short is 4, long is 8
		self.action_space_size = 2 #action space is {a_NO, a_YES} i.e. no_match or match
		self.possible_actions = ["NO", "YES"] #action_command=0 is NO; action_command=1 is YES
		self.state_space_size = 3 #state is represented by cue and previous action (a_t-1) and reward (r_t-1)

		#these will change as the trial is created and run
		self.trial_term = False #trial is terminated

		#create cue-deck depending on trialtype
		if cuedecksize==0: #small deck
			if trialtype==0: #Same-Different
				self.decksize = 2 # self.h_cue[0] is first cue, self.h_cue[1] is second cue
			else: #Match-First or Match-Any small deck
				self.decksize = 5
		else: #large deck
			self.decksize = 100

		self.cuedeck = [i for i in range(1,(self.decksize+1))] #for decksize 100; cues are 1-100; 0 is no_cue

		#draw first cue from cue deck to make start_state
		randomprioraction = np.random.choice([0,1],p=[0.5,0.5])
		randompriorreward = np.random.choice([-5,5],p=[0.5,0.5])
		self.start_state = np.array([0,randomprioraction,randompriorreward],ndmin=2) #initial state is no cue (0), random previous action, random previous reward
		self.seencues = [0] #start the list of cues seen in this trial

	def resetcuedeck(self):
		self.trial_term = False
		self.cuedeck = [i for i in range(1,(self.decksize+1))] #set cuedeck to be the fulldeck to begin with
		self.seencues = [0] #start the list of cues seen in this trial; previous cue seen before reset is always no_cue (0)

	def step(self, action_command):
		reward = 0 #default reward is 0; update if appropriate time
		#Same-Different - all trials are 2 cues long, 4 cues including flanking no_cues
		if self.trialtype==0:
			if len(self.seencues)==1: #get next cue; reward is 0 no matter action; len(self.seencues)=1 only for the very first trial
				newcue = self.cuedeck.pop(np.where(self.cuedeck==np.random.choice(self.cuedeck))[0][0])
				self.seencues.append(newcue)
			elif len(self.seencues)==2:	
				newposcue = np.random.choice(self.cuedeck)
				newcue = np.random.choice([newposcue,self.seencues[1]])
				self.seencues.append(newcue)
			elif len(self.seencues)==3: #check if correct or incorrect action and give appropriate reward; give no cue
				if ((action_command==0) & (self.seencues[1]!=self.seencues[2])) | ((action_command==1) & (self.seencues[1]==self.seencues[2])):
					reward = 5
				else:
					reward = -5
				newcue = 0
				self.seencues.append(newcue)
				self.trial_term = True #action on second card
			elif len(self.seencues)==4: #prompt new trial; give new firstcue, reward is 0
				self.resetcuedeck() #this makes len(self.seencues)=1
				newcue = self.cuedeck.pop(np.where(self.cuedeck==np.random.choice(self.cuedeck))[0][0])
				self.seencues.append(newcue) #this makes len(self.seencues)=2
		#Match-First
		elif self.trialtype==1:
			if len(self.seencues) < self.trialsize: #len=4 means [blank cue1 cue2 cue3] so just get more cues
				newcue = self.cuedeck.pop(np.where(self.cuedeck==np.random.choice(self.cuedeck))[0][0])
				self.seencues.append(newcue)
			elif len(self.seencues) == self.trialsize: #next cue is 50-50 cue that matches first or not
				newposcue = np.random.choice(self.cuedeck+self.seencues[2:]) #50-50 first cue or not first cue
				newcue = np.random.choice([newposcue,self.seencues[1]])
				self.seencues.append(newcue)
			elif len(self.seencues) == (self.trialsize+1):
				if ((action_command==0) & (self.seencues[1]!=self.seencues[(self.trialsize)])) | ((action_command==1) & (self.seencues[1]==self.seencues[(self.trialsize)])):
					reward = 5
				else:
					reward = -5
				newcue = 0
				self.seencues.append(newcue)
				self.trial_term = True
			elif len(self.seencues) == (self.trialsize+2):
				self.resetcuedeck() #this makes len(self.seencues)=1
				newcue = self.cuedeck.pop(np.where(self.cuedeck==np.random.choice(self.cuedeck))[0][0])
				self.seencues.append(newcue) #this makes len(self.seencues)=2
		#Match-Any
		elif self.trialtype==2:
			if len(self.seencues) < self.trialsize: #len=4 means [blank cue1 cue2 cue3] so just get more cues
				newcue = self.cuedeck.pop(np.where(self.cuedeck==np.random.choice(self.cuedeck))[0][0])
				self.seencues.append(newcue)
			elif len(self.seencues) == self.trialsize: #next cue is 50-50 cue that matches one of previous or not
				newposcue1 = np.random.choice(self.cuedeck)
				newposcue2 = np.random.choice(self.seencues[1:])
				newcue = np.random.choice([newposcue1,newposcue2])
				self.seencues.append(newcue)
			elif len(self.seencues) == (self.trialsize+1):
				if ((action_command==0) & (self.seencues[self.trialsize] not in self.seencues[1:self.trialsize])) | ((action_command==1) & (self.seencues[self.trialsize] in self.seencues[1:self.trialsize])):
					reward = 5
				else:
					reward = -5
				newcue = 0
				self.seencues.append(newcue)
				self.trial_term = True
			elif len(self.seencues) == (self.trialsize+2):
				self.resetcuedeck() #this makes len(self.seencues)=1
				newcue = self.cuedeck.pop(np.where(self.cuedeck==np.random.choice(self.cuedeck))[0][0])
				self.seencues.append(newcue) #this makes len(self.seencues)=2

		new_state = np.array([newcue, action_command, reward],ndmin=2) #gives a (1,3) array

		return new_state, reward, self.trial_term


def get_state_space():
	env = humanVmonkey_env()
	return(env.state_space_size)

def get_action_space():
	env = humanVmonkey_env()
	return(env.action_space_size)

def make_new_env(trialtype=0,cuedecksize=0,trialsize=4):
	env = humanVmonkey_env(trialtype=trialtype,cuedecksize=cuedecksize,trialsize=trialsize)
	return(env)

#returns array that defines the start state
def get_start_state_from_env(env):
	obs = env.start_state
	return(obs)

def do_action_in_environment(env,state,action):
	action_command = np.argmax(action) #in humanVmonkey_env NO is 0 and YES is 1; action is 1-hot so NO is [1,0], YES is [0,1]
	new_state_raw, reward_raw, trial_term = env.step(action_command)
	new_state = np.resize(new_state_raw,(1,3))
	reward = np.resize(reward_raw,(1,1))
	return reward, new_state, trial_term


