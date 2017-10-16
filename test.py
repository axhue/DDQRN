import MalmoPython
import os
import sys
import time
import random


agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print 'ERROR:',e
    print agent_host.getUsage()
    exit(1)
if agent_host.receivedArgument("help"):
    print agent_host.getUsage()
    exit(0)



# load world
with open('pig.xml','r') as f:
    my_mission = MalmoPython.MissionSpec(f.read(), True)
my_mission_record = MalmoPython.MissionRecordSpec()

# Attempt to start a mission:
max_retries = 3

for retry in range(max_retries):
    try:
        agent_host.startMission( my_mission, my_mission_record )
        break
    except RuntimeError as e:
        if retry == max_retries - 1:
            print "Error starting mission:",e
            exit(1)
        else:
            time.sleep(2)

# Loop until mission starts:
print "Waiting for the mission to start ",
world_state = agent_host.getWorldState()

while not world_state.has_mission_begun:
    sys.stdout.write(".")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print "Error:",error.text



try:
	while world_state.is_mission_running:
	    time.sleep(2)
	    world_state = agent_host.getWorldState()
	    #chose_act = random.choice(ractions)
	    # print(chose_act)
	    print(world_state.observations[-1].text)
	    # agent_host.sendCommand(chose_act)
	    for error in world_state.errors:
	        print "Error:",error.text
except KeyboardInterrupt:
	agent_host.sendCommand("quit")
	print("clean exit")

print