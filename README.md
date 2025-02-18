# CDS524_Assignment1
This is my assignment1 for CDS524 Machine Learning course.
# 🏀 CXK Chasing Ball: A Q-Learning Simulation

![Demo Preview](cxk.gif) ![Demo Preview](ball.gif)

A predator-prey simulation system based on the Q-Learning algorithm, implemented using the Python turtle library for visualization. CXK (the predator) uses Q-learning strategies to chase BALL (the prey), while BALL also learns escape strategies.

## 🌟 Core Features
- **Dual-Agent Q-Learning**: Both predator and prey have autonomous learning capabilities
- **Visualized Training Process**: Observe the agents' movement strategies in real-time
- **Complex Environment Configuration**: Arena with multiple fixed obstacles
- **Dynamic Scoring System**: Track predation success rate in real-time
- **Pre-trained Model Support**: Provides optimized Q-tables for direct use

## 📦 File Description
| File Name | Description |
|--------|------|
| `Assignment1.py` | Main program file |
| `cxk.gif` | Predator character sprite |
| `ball.gif` | Prey character sprite |
| `trained_predator.yaml` | Pre-trained predator Q-table |
| `trained_prey.yaml` | Pre-trained prey Q-table |

## ⚙️ Running Environment
- Python 3.6+
- PyCharm
- Dependencies: `turtle`, `numpy`, `PyYAML`

## ⚙️ Terminal installation
pip install turtle
pip install numpy
pip install numpy pyyaml

## 🧠 Algorithm Parameters
Parameter	Value	Description
α (Learning Rate)	0.1	Speed of new knowledge absorption
γ (Discount Factor)	0.9	Importance of future rewards
ε (Exploration Rate)	0.1	Probability of random exploration

## ⚙️ Running Effects
Red squares: Obstacles
Black border: Arena boundary
Real-time score display for both sides
Automatic position reset upon successful predation
Prey automatically escapes if not caught within 70 steps

📚 Training Instructions
The code includes training logic (commented by default). To retrain:
Uncomment the Q-table initialization section
Comment out the pre-trained model loading section
Adjust training loop parameters
