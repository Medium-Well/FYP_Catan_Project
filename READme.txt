# FYP_Catan_Project
This is my public code repository for my FYP submission, it is a project that aimed to create and train a pre-programmed AI agent and a machine learning AI agent to play Catan. Kane is the pre-programmed agent and Abel is the machine learning agent, there are 2 versions of Abel that were trained concurrently with only differences in the model layer architecture.

The files provided are structured as such:
"root_files" - Folder containing files that should be inside the imported 'catanatron' folder.
"Command-Line-Interface User Interface (Responses).xlsx" - User Survey Data.
"import_catan.ipynb" - Simple Commands for importing Catanatron github library.
"statistics.ipynb" - Data Statistics.

Within the 'root_files' folder:
"kane_player.py" - Python file for running the agents and user interface classes
"kane_test.py" - Python file for collecting data on Kane
"kane.csv" - CSV file of the data collected from Kane's 750 simulated games

"testmodel4_com.py" - Python file for running Complex Abel model, training and collecting data
"testmodel4_lite.py" - Python file for running Basic Abel model, training and collecting data  
"rl_test_com.h5" - Exported Complex Abel model file for loading/training
"rl_test_lite.h5" - Exported Basic Abel model file for loading/training
"test1_com.csv" - CSV file of the data collected from Complex Abel's 100 simulated games
"test1_lite.csv" - CSV file of the data collected from Basic Abel's 100 simulated games

#--------------------------------------------------------------------------------------------------------------#

CLI commands for running the models for testing and data collection:

--code: To link the python file for the class to be called from
--players: How many players and which players are to be in the game
--num: How many games to be simulated

examples:
catanatron-play --code=kane_test.py --players=W,W,KANE --num=250 
catanatron-play --code=kane_player.py --players=W,W,HUMAN,W --num=1

catanatron-play --code=testmodel4_com.py --players=W,W,R,AI --num=25
catanatron-play --code=testmodel4_lite.py --players=W,W,R,AI --num=50

