# Import some necessary libs
import turtle
import time
import numpy as np
import random
import yaml

# Set up the screen
screen = turtle.Screen()
screen.setup(1200, 1200)
screen.title("CXK CHASING BALL")
screen.addshape("ball.GIF")
screen.addshape("cxk.GIF")

# Create a turtle for the score board
score_turtle = turtle.Turtle()
score_turtle.penup()
score_turtle.goto(-250, 400)
score_turtle.hideturtle()
score= 0
prey_score= 0
score_turtle.write(f"CXK's Score: {score}  BALL's Score: {prey_score}", font=("Arial", 25, "bold"))

# Create a turtle for the target
target_turtle = turtle.Turtle()
target_turtle.speed(1000)
target_turtle.shape("ball.GIF")
target_turtle.penup()
target_turtle.goto(-325, -325)
target_turtle.pendown()  # Draw a black square bound
target_turtle.color("black")
target_turtle.goto(-325, 325)
target_turtle.goto(325, 325)
target_turtle.goto(325, -325)
target_turtle.goto(-325, -325)
target_turtle.penup()  # Let the target go to an initial start point
target_turtle.goto(300, 300)

# Create a turtle for the agent
agent_turtle = turtle.Turtle()
agent_turtle.shape("cxk.GIF")
agent_turtle.speed(1000)
agent_turtle.penup()
agent_turtle.goto(-300, -300)

# Create obstacle turtle
obstacles = [(0, 0), (50, 0), (50, -100), (50, -150), (50, -200), (50, 200), (100, 0), (0, 50), (0, 100), (-100, 0), (-50, -50), (-100, 250),
             (250, -200), (200, -200), (250, -150), (250, -50), (250, 0), (250, 100),
             (250, 250), (200, 250), (150, 250), (-200, -250), (-150, -250), (-200, -150),
             (-200, -100), (-250, -50), (-250, 0), (-250, 50), (-250, 150), (-250, 250)]
for obstacle in obstacles:
    obstacle_turtle = turtle.Turtle()
    obstacle_turtle.speed(1000)
    obstacle_turtle.shape("square")
    obstacle_turtle.shapesize(2.5)  # The default width and height of an obstacle are both 20, here we set to 50.
    obstacle_turtle.color("red")
    obstacle_turtle.penup()
    obstacle_turtle.goto(obstacle)  # Set the obstacle turtle's position

# Define the state space
states = []
for x in range(-300, 301, 50):
    for y in range(-300, 301, 50):
        states.append((x, y))

goal_states = []
for state in states:
    if state not in obstacles:
        goal_states.append(state)

# Define the action space
actions = ['up', 'down', 'left', 'right']

"""
In the yaml file, the first two position values are for Tom's position. And the second two position values are for 
Jerry's position.And during Q-learning, we are supposed to let the programme update the Q table itself. This part is 
for opening the existing yaml file after you have training
"""
with open('trained_predator.yaml', 'r') as f:
    predator_table = yaml.load(f, Loader=yaml.FullLoader)
with open('trained_prey.yaml', 'r') as f:
    prey_table = yaml.load(f, Loader=yaml.FullLoader)


# Define two empty to construct a Q table
# """
# In these new constructed dictionary, keys will be the position values of Tom and Jerry. And values will be the action
# probabilities. After you have finished training, you can note these codes below.
# """
# predator_table = {}
# prey_table = {}

# Define the parameters for Q-Learning
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount rate
epsilon = 0.1  # Exploration rate

# Q-Learning algorithm
predator_state = (-300, -300)  # Start from the initial state
prey_state = (300, 300)


for s in goal_states:  # s is to control the predator positions
    print(f"Training for CXK state {s}.")
    # with open('trained_predator.yaml', 'w') as f:
    #     yaml.dump(predator_table, f)
    # with open('trained_prey.yaml', 'w') as f:
    #     yaml.dump(prey_table, f)

    for g_state in goal_states:
        min_duration = 0  # This variety is to save one single time that Tom catch Jerry.
        min_duration_time = 0  # This variety is to save the times that Tom catch Jerry within a time limit.

        while min_duration_time != 1000:

            randomm = True  # If True, then predator and prey will appear in random position. Otherwise they will follow the loop.
            show = True  # If true, the score table will update and you will see detailed movements of predator and prey.

            if not randomm:
                predator_state = s  # Start from the initial state
                prey_state = g_state
            if show:
                agent_turtle.goto(predator_state)
                score_turtle.clear()  # To clear the previous result to avoid overlap display
                score_turtle.write(f"CXK's Score: {score}  BALL's Score: {prey_score}", font=("Arial", 25, "bold"))
                agent_turtle.speed(3.5)
                target_turtle.speed(3)
                target_turtle.goto(prey_state)

            done = False  # One way to make 'done' True is that predator catch prey, another is that prey move 70 steps.
            start = time.time()  # Give us the current time in sec

            step = 0
            while not done:
                step +=1

                """Train Prey"""
                x, y = predator_state
                goal_x, goal_y = prey_state
                condition = (x, y, goal_x, goal_y)
                if condition not in prey_table.keys():
                    prey_table[condition] = {action: 0 for action in actions}
                # Choose an action using epsilon-greedy policy
                if np.random.uniform() < epsilon:
                    action = np.random.choice(actions)  # Explore
                else:
                    action = max(prey_table[condition], key=prey_table[condition].get)  # Exploit

                # Get the next state
                goal_x, goal_y = prey_state
                # actions = ['up', 'down', 'left', 'right']
                if action == 'up':
                    next_state = (goal_x, goal_y + 50)
                elif action == 'down':
                    next_state = (goal_x, goal_y - 50)
                elif action == 'left':
                    next_state = (goal_x - 50, goal_y)
                else:  # 'right'
                    next_state = (goal_x + 50, goal_y)

                # Prey reward
                if next_state not in states:
                    next_state = prey_state
                    reward = -5  # Penalty for hitting the wall
                elif ((next_state[0]-predator_state[0])**2 + (next_state[1]-predator_state[1])**2) <= 5000:
                    reward = -10  # Reached the target turtle
                elif next_state in obstacles:
                    next_state = prey_state
                    reward = -5  # Hit the obstacle turtle
                else:
                    current_distance= ((prey_state[0]-predator_state[0])**2 + (prey_state[1]-predator_state[1])**2)**(1/2)
                    next_distance = ((next_state[0]-predator_state[0])**2 + (next_state[1]-predator_state[1])**2)**(1/2)
                    reward = (next_distance - current_distance)/10


                x, y = predator_state
                goal_x, goal_y = next_state
                next_condition = (x, y, goal_x, goal_y)
                if next_condition not in prey_table.keys():
                    prey_table[next_condition] = {action: 0 for action in actions}

                # Update the Q-table
                prey_table[condition][action] += alpha * (
                        reward + gamma * max(prey_table[next_condition].values()) -
                        prey_table[condition][action])

                # Update the state and move the agent turtle
                prey_state = next_state
                if show:
                    target_turtle.goto(prey_state)

                """Train Predator"""
                x, y = predator_state
                goal_x, goal_y = prey_state
                condition = (x, y, goal_x, goal_y)
                if condition not in predator_table.keys():
                    predator_table[condition] = {action: 0 for action in actions}
                # Choose an action using epsilon-greedy policy
                if np.random.uniform() < epsilon:
                    action = np.random.choice(actions)  # Explore
                else:
                    action = max(predator_table[condition], key=predator_table[condition].get)  # Exploit

                # Get the next state
                x, y = predator_state
                if action == 'up':
                    next_state = (x, y + 50)
                elif action == 'down':
                    next_state = (x, y - 50)
                elif action == 'left':
                    next_state = (x - 50, y)
                else:  # 'right'
                    next_state = (x + 50, y)

                # Predator reward
                if next_state not in states:
                    next_state = predator_state
                    reward = -5 # Penalty for hitting the wall
                elif next_state == prey_state:
                    reward = 10 # Reached the target turtle
                elif next_state in obstacles:
                    next_state = predator_state
                    reward = -5  # Hit the obstacle turtle
                else:
                    reward = -1  # Time pressure penalty

                x, y = next_state
                goal_x, goal_y = prey_state
                next_condition = (x, y, goal_x, goal_y)
                if next_condition not in predator_table.keys():
                    predator_table[next_condition] = {action: 0 for action in actions}

                # Update the Q-table
                predator_table[condition][action] += alpha * (
                        reward + gamma * max(predator_table[next_condition].values()) -
                        predator_table[condition][action])

                # Update the state and move the agent turtle
                predator_state = next_state
                if show:
                    agent_turtle.goto(predator_state)

                # Check if the target turtle escaped
                if step == 70:
                    done = True

                    if randomm:
                        prey_state = random.choice(goal_states)
                    if show:
                        prey_score +=1
                        target_turtle.hideturtle()
                        target_turtle.goto(prey_state)
                        target_turtle.showturtle()
                # Check if the target turtle is reached
                if ((prey_state[0]-predator_state[0])**2 + (prey_state[1]-predator_state[1])**2) <= 5000:

                    done = True

                    if randomm:
                        prey_state = random.choice(goal_states)
                    if show:
                        score += 1
                        target_turtle.hideturtle()
                        target_turtle.goto(prey_state)
                        target_turtle.showturtle()

            duration = time.time() - start
            if duration > min_duration:
                min_duration = duration
                min_duration_time = 0
            else:
                min_duration_time += 1

    print("Finish current training.")
#
with open('trained_predator.yaml', 'w') as f:
    yaml.dump(predator_table, f)
with open('trained_predator.yaml', 'w') as f:
    yaml.dump(prey_table, f)

# Keep the screen open until it's closed manually
turtle.done()