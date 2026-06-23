Welcome to this page! Let me explain what each of these files do

_**dashboard.py**_:
- This is the dashboard python program, this is what makes the streamlit app possible. This program displays all the data that the model processes and the predictions

_**fixtures.csv**_:
- This csv file looks at all the games and the results, the model refers to this directly in order to look at when each of the FIFA games are. So far, this
  is only working for the group stage games. Once the R32 matchups are finalized, along with subsequent further game stages, this file will be adequately updated.

_**predictions.json**_:
- The middleman between 'predictor.py' and 'dashboard.py'. This file is updated every time predictor.py is rerun, and it feeds information regarding
  the newest probabilities and predictions into 'dashboard.py'

_**predictor.py**_:
- The brains behind this entire project. Whenever "the model" is mentioned, this is THE model. It does all the thinking and brainstorming, and then writes to
  'predictions.json', which then feeds to 'dashboard.py'.
- This file first reads data regarding **every recorded international match** going all the way back over 100-150 years, and is trained on this data
- This file then computes three key indicators:
      - **An Elo Rating system**: Every team starts as a baseline number of points. Then, for every game won, points are added, and for
        every game lost, points are lost. This is a large portion of this model.
      - **The team's recent form**: This is how the team in question has performed over the course of the last 10 games. Historical data going back decades shouldn't
        be the "end all say all," so I chose to include this metric. This catches recent hot streaks for some teams, and recent downturns for others.
      - **Squad strength**: What I'm going to say may sound super funny, but its worked so far both in backtest and applications. This model also uses player data
        from the FIFA video game (yeah I know lmao) in order to compute team quality. Turns out, the game actually rates these players surprisingly well, and it works
        as a pretty good metric. It takes the average quality of the top 23 players on the team of a country in order to determine squad strength.
- After computing these metrics and putting them together for a baseline, the model is the trained on football games between 2018 and 2022. The squad strength data set
  starts at 2015, so I chose 2018 as the starting for this training set. Then, I chose 2022 as the ending for this training set, because I didn't want this model to have
  knowledge on the 2022 World Cup. The reasoning was so that when backtests were run for the 2022WC and all the international matches after, there wouldn't be "leaks" where the
  model can look into the actual match results that its trying to predict. It creates a whole paradox, me no likey.
      - With regards to model training, the model feeds the three key indicators, mentioned above, into an XGBoost classifier, which then outputs win/loss/draw probabilities for
        any matchup. Then, this model is trained only once, and then frozen, so that every refresh of predictor.py doesn't create any noise with the training set constantly. Also,
        a home team/host boost is applied to the US, Mexico, and Canada. The boost is 88 points.
- The model then goes on to project the group stage games. It looks at games already won and points achieved per team, expected points (xPts) from games not played yet,
  while also including the tie-breaking "goal difference" metric. Then, it also runs something called clinch detection, which uses probabilities to calculate if its mathematically
  certain for a team to proceed into the R32.
- The model then simulates the knockouts, it builds the real 2026 bracket (all the way from the R32 to the final), running a Monte Carlo simulation 10,000 times to produce championship
  odds and each team's chance of getting to each round and advancing. It's conditioned on what really happens in the game in real life, so if a team wins an R32 game, its taken into account for these simulations
  going forward.
- This model also validates itself through the use of backtests. Its a leak-free backtest, so the data is trained up till 01-01-2022, so no 2022WC data. The backtests ran to validate include the 2022WC along with a series of
  a few hundred matches between 2025 and pre-FIFA. This is where the model gets the "60% accuracy goal" number from
- ALL of this above is packaged and written to predictions.json everytime I manually run a file called "refresh.py", and once pushed to GitHub, shows the updated odds on dashboard. Typically I run the refresh file either after big matches, end of the day,
  or whenever I just feel like updating the dashboard with the most up-to-date info.

_**refresh.py**_:
- Oh look what I coincidence, 100 years, I was just talking about you. This is the file used to refresh 'predictor.py'.

_**requirements.txt**_:
- This is a file which mentions all the dependencies for this project, which includes streamlit, pandas, requests, and tzdata. It pretty much gives the needed dependencies to run the website on streamlit.

_**squad_strength.csv**_:
- This is the FIFA player set I was talking about, used to determine the squad strength for 'predictor.py'.
