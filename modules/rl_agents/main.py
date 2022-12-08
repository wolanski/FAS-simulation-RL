################################################################################
#                           1 Import packages                                  #
################################################################################

import numpy as np
import pandas as pd
import torch.optim as optim
import modules.rl_agents.common as cm
from modules.rl_agents import ddql
from modules.process.production_line import ProductionLine
# Supress all warnings (e.g. deprecation warnings) for regular use
import warnings
warnings.filterwarnings("ignore")

################################################################################
#                                 7 Main program                               #
################################################################################

def ddqn_main():
    """Main program loop"""
    
    ############################################################################
    #                          8 Set up environment                            #
    ############################################################################
    
    # Set up game environemnt
    sim = ProductionLine(sim_duration=cm.SIM_DURATION, time_step=cm.TIME_STEP, debug=False)
    sim.reset()

    # Get number of observations returned for state
    observation_space = sim.observation_size
    
    # Get number of actions possible
    action_space = sim.action_size
    
    ############################################################################
    #                    9 Set up policy and target nets                       #
    ############################################################################
    
    # Set up policy and target neural nets
    policy_net = ddql.DQN(observation_space, action_space)
    target_net = ddql.DQN(observation_space, action_space)
    
    # Set loss function and optimizer
    policy_net.optimizer = optim.Adam(
            params=policy_net.parameters(), lr=cm.LEARNING_RATE)
    
    # Copy weights from policy_net to target
    target_net.load_state_dict(policy_net.state_dict())
    
    # Set target net to eval rather than training mode
    # We do not train target net - ot is copied from policy net at intervals
    target_net.eval()
    
    ############################################################################
    #                            10 Set up memory                              #
    ############################################################################
        
    # Set up memomry
    memory = cm.Memory()
    
    ############################################################################
    #                     11 Set up + start training loop                      #
    ############################################################################
    
    # Set up run counter and learning loop    
    run = 0
    all_steps = 0
    continue_learning = True
    
    # Set up list for results
    results_run = []
    results_exploration = []
    results_score = []
    
    # Continue repeating games (episodes) until target complete
    while continue_learning:
        
        ########################################################################
        #                           12 Play episode                            #
        ########################################################################
        
        # Increment run (episode) counter
        run += 1
        
        ########################################################################
        #                             13 Reset game                            #
        ########################################################################
        
        # Reset game environment and get first state observations
        state = sim.reset()
        
        # Trackers for state
        time = []
        line_state = []
        next_maint = []
        conv_state = []
        conv_wear = []
        #last_fail = []
        rewards = []
        
        # Reset total reward
        total_reward = 0
        
        # Reshape state into 2D array with state obsverations as first 'row'
        state = np.reshape(state, [1, observation_space])
              
        # Continue loop until episode complete
        while True:
            
        ########################################################################
        #                       14 Game episode loop                           #
        ########################################################################
            
            ####################################################################
            #                       15 Get action                              #
            ####################################################################
            
            # Get action to take (se eval mode to avoid dropout layers)
            policy_net.eval()
            action = policy_net.act(state)
            
            ####################################################################
            #                 16 Play action (get S', R, T)                    #
            ####################################################################
            
            # Act 
            state_next, reward, terminal = sim.step(action)
            total_reward += reward

            # Update trackers
            time.append(state_next[1])
            line_state.append(state_next[0])
            next_maint.append(state_next[4])
            conv_state.append(state_next[5])
            conv_wear.append([v for k,v in sim.get_resource_states()['hidden states'].items()][0])
            #last_fail.append(p_d.last_fail)
            rewards.append(reward)
                                                          
            # Reshape state into 2D array with state obsverations as first 'row'
            state_next = np.reshape(state_next, [1, observation_space])
            
            # Update display if needed
            if cm.DISPLAY_ON_SCREEN:
                sim.render()
            
            ####################################################################
            #                  17 Add S/A/R/S/T to memory                      #
            ####################################################################
            
            # Record state, action, reward, new state & terminal
            memory.remember(state, action, reward, state_next, terminal)
            
            # Update state
            state = state_next
            
            ####################################################################
            #                  18 Check for end of episode                     #
            ####################################################################
            
            # Actions to take if end of game episode
            if terminal:
                # Get exploration rate
                exploration = policy_net.exploration_rate
                # Clear print row content
                clear_row = '\r' + ' '*79 + '\r'
                print (clear_row, end ='')
                print (f'Episode: {run}, ', end='')
                print (f'Exploration: {exploration: .3f}, ', end='')
                average_reward = total_reward/cm.SIM_DURATION
                print(f'Average reward: {average_reward:4.1f}, ', end='')
                print(f'Total reward: {total_reward:4.1f}, ', end='')
                print(f'Steps: {sim.step_no}, ', end='')
                print(f'Processes: {sim.process_no}, ', end='')
                print(f'Env time: {sim.env.now}', end='')
                print('')
                
                # Add to results lists
                results_run.append(run)
                results_exploration.append(exploration)
                results_score.append(total_reward)
                
                ################################################################
                #             18b Check for end of learning                    #
                ################################################################
                
                if run == cm.TRAINING_EPISODES:
                    continue_learning = False
                
                # End episode loop
                break
            
            
            ####################################################################
            #                        19 Update policy net                      #
            ####################################################################
            
            # Avoid training model if memory is not of sufficient length
            if len(memory.memory) > cm.REPLAY_START_SIZE:
        
                # Update policy net
                cm.optimize(policy_net, target_net, memory.memory)

                ################################################################
                #             20 Update target net periodically                #
                ################################################################
                
                # Use load_state_dict method to copy weights from policy net
                if all_steps % cm.SYNC_TARGET_STEPS == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                
    ############################################################################
    #                      21 Learning complete - plot results                 #
    ############################################################################
    
    # Add last run to DataFrame. summarise, and return
    run_details = pd.DataFrame()
    run_details['time'] = time 
    run_details['line_state'] = line_state
    run_details['next_maint'] = next_maint
    run_details['conv_state'] = conv_state
    run_details['conv_wear'] = conv_wear
    #run_details['last_fail'] = last_fail
    run_details['reward'] = rewards  
        
    # Target reached. Plot results
    cm.plot_results(results_run, results_exploration, results_score, run_details)
    
    return run_details