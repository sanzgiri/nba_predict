# nba_predictions
NBA Predictions

This repo contains code for algorithms that attempt to predict the outcome and probabilitiies of outcomes of NBA basketball games. 



## Repo Contents
Dockerfile: to build a Docker container to run code/notebooks### code
### data
### notebooks
### tasks
* scripts to build and run docker container


## repo work 

### model
Currently the following are worth additional investigation:

#### high priority 
- using previous season raptor ratings as a prior instead of smoothing to the replacement level 
- fitting the data on multipl seasons of data instead of the 2008 season 
- allocating the minutes using the current depth charts instead of the current hack
- getting more reasonable gametime decisions

#### exploratory 
- including a range of predictions with different player assumptions to illustrate how much "playing time" risk there is
- generating better probabablistic predictions than we get from using logistic regression 
- explore the benefit of using position-level ratpor ratings for each team instead of aggregating to the team level

## tech / eng

- logging data on a regular cadence to build record of production performance 
- logging data that is aquired to make the script more modular
- seperating different parts of the script that can be run on different cadence
