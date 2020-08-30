# Unity Mouse Environment

This git repository contains the PPO code for training the mouse agent and pre-trained resnet with cutout. To train the agent, an Unity executable or editor containing the environment is required.


## Required Packages
```python
pip install torch torchvision
pip install tensorboard
pip install mlagents==0.16.0
```
At the time of writing, updates are made to mlagents frequently so later versions may have different function names which would require modifications to the script.

## Usage

```console
# To train through the editor, enter the following command in the terminal and then press the play button in the editor.
python ppo_train.py

# To train using the executable, add the file path at the end and the executable will start automatically.
python ppo_train.py </file/path/executable_name>

# Similarly for the test script.
python ppo_test.py </file/path/executable_name>
```

## Python Script
The scripts is modified off of nikhilbarhate99's (https://github.com/nikhilbarhate99/PPO-PyTorch) PPO script to work with the Unity MLAgent environment. The scripts reads from a pretrained resnet18 cutout model and use it to process the images from both eyes. Original source of the cutout model is found at (https://github.com/uoguelph-mlrg/Cutout). The image vectors are then combine with proprio vector containing the odour gradients to be used as input features.

The current training script supports a single agent and additional agent will require more cameras which generates additional overheads.

Hyperparameters in the script must be tuned manually inside the script and most of the parameters are constant except for the action stdev and clipping which supports decay.

## Unity Executable and Editor

To access the mouse environment, you need to be assigned a member of this project. This can be done by going into the Unity DashBoard as the owner and add your Unity account to the member list. When logging into the Unity Hub with that account, the environment should be downloadable from the cloud.

Once the environment is completely downloaded if you encounter error in the console, try (Help > Reset Package to defaults) and go to (Window > Package Manager) and reinstall following packages: ML Agent (version 1.0.3), Post Processing (version 2.3.0), and High Definition RP (version 7.3.1). Then do a (Asset > Reimport All) and the environment should have no errors. This has only been tested on Linux and MacOS.

The training script can be started in either the editor or using the executable file. You can create an executable file inside the editor (File > Build Settings...) and select the OS on which the executable is expected to run.  

## Environment Editor

In the editor, both an empty training environment and the actual meadow environment are available in (Project > Assets > Scenes).
The models for the agent and targets are in (Project > Assets > Prefabs). The scripts containing code for sending the observations to python script and controlling the environment are in (Project > Assets > Scripts):

- The agent script contains properties can be adjusted for training such as penalties on action values, rewards for the amount/change in amount of odour concentration detected and reward for reaching targets.
- Adjustable movement parameters include side-ways/backwards movement speed reduction and an approximate of speed reduction when mice move through grass based on the number of grass model within range.
- Olfactory can be adjusted to use a particle system approximation of odour diffusion with wind effect or a simplified odour concentration gradient base on distance from object.
- Spawn controller script spawns randomly from a list of prefab objects. Other parameter includes spawn distance from agent, spawn speed, and spawn limits.
