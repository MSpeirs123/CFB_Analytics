import cfbd
import numpy as np
import pandas as pd
from api import *
from fastai.tabular import *
from fastai.tabular.all import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

games = []
lines = []

for year in range(2015, 2022):
    response = gamesApi().get_games(year=year)
    games = [*games, *response]

    response = bettingApi().get_lines(year=year)
    lines = [*lines, *response]

games = [g for g in games if
        g.home_conference is not None and g.away_conference is not None and
         g.home_points is not None and g.away_points is not None] ##Removing none FBS teams as conference comes back as none for FCS teams

games = [
    dict(
        id=g.id,
        year=g.season,
        week=g.week,
        neutral_site=g.neutral_site,
        home_team=g.home_team,
        home_conference=g.home_conference,
        home_points=g.home_points,
        home_elo=g.home_pregame_elo,
        away_team=g.away_team,
        away_conference=g.away_conference,
        away_points=g.away_points,
        away_elo=g.away_pregame_elo
    ) for g in games]

for game in games:
    game_lines = [l for l in lines if l.id == game['id']]

    if len(game_lines) > 0:
        game_line = [l for l in game_lines[0].lines if l['provider'] == 'consensus']

        if len(game_line) > 0 and game_line[0]['spread'] is not None:
            game['spread'] = float(game_line[0]['spread']) ##Add consensus spread

games = [g for g in games if 'spread' in g and g['spread'] is not None] #Remove lines where there is no consensus spread

for game in games:
    game['margin'] = game['away_points'] - game['home_points']

df = pd.DataFrame.from_records(games).dropna()
df.head()

test_df = df.query("year == 2021")
train_df = df.query("year != 2021")

excluded = ['id','year','week','home_team','away_team','margin', 'home_points', 'away_points']
cat_features = ['home_conference','away_conference','neutral_site']
cont_features = [c for c in df.columns.to_list() if c not in cat_features and c not in excluded] #get home_elo, away_elo and spread

splits = RandomSplitter(valid_pct=0.2)(range_of(train_df)) #validation set



to = TabularPandas(train_df, procs=[Categorify, Normalize],
                    y_names="margin",
                    cat_names = cat_features,
                    cont_names = cont_features,
                   splits=splits)

dls = to.dataloaders(bs=64)
dls.show_batch()

learn = tabular_learner(dls, metrics=mae, lr=10e-2)
learn.fit(5)

learn.show_results()

pdf = test_df.copy()
dl = learn.dls.test_dl(pdf)
pdf['predicted'] = learn.get_preds(dl=dl)[0].numpy()

print(pdf[['home_team','away_team','spread','predicted', 'margin']].round(1))
learn.export('talking_tech_neural_net')

learn = load_learner('talking_tech_neural_net')