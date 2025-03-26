import cfbd
import numpy as np
import pandas as pd
from pandas import json_normalize
import math
from api import *
from fastai.tabular import *
from fastai.tabular.all import *

from keras.layers import BatchNormalization, Dense, Input, Dropout
from keras.models import Model
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from collections import Counter

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def getLines(year, gameId):
    bets = bettingApi().get_lines(year=year, game_id=gameId)
    spread = "0"
    overUnder = "0"
    homeMoneyLine = 0
    awayMoneyLine = 0
    lines = bets[0].lines
    for line in lines:
        if line["provider"] != "Bovada":
            continue
        spread = line["spread"]
        overUnder = line["overUnder"]
        homeMoneyLine = line["homeMoneyline"]
        awayMoneyLine = line["awayMoneyline"]
        break
    if spread == "0":
        for line in lines:
            if line["provider"] != "teamrankings":
                continue
            spread = line["spread"]
            overUnder = line["overUnder"]
            homeMoneyLine = line["homeMoneyline"]
            awayMoneyLine = line["awayMoneyline"]
            break
    if spread == "0":
        for line in lines:
            if line["provider"] != "consensus":
                continue
            spread = line["spread"]
            overUnder = line["overUnder"]
            homeMoneyLine = line["homeMoneyline"]
            awayMoneyLine = line["awayMoneyline"]
            break
    homeodds=getDecimalFromAmerican(homeMoneyLine)
    awayodds=getDecimalFromAmerican(awayMoneyLine)
    if not spread:
        spread = 0
    if not overUnder:
        overUnder = 0
    if not homeodds:
        homeodds = 0
    if not awayodds:
        awayodds = 0
    line = [float(spread), float(overUnder), homeodds, awayodds]
    return line


def getDecimalFromAmerican(odds):
    if not odds:
        return 0
    if odds > 0:
        decimal = (odds/100) + 1
    else:
        decimal = 1 - (100/odds)
    return decimal


def getHomeAdvantages():
    df = pd.read_excel(r'/Users/Matt/PycharmProjects/CFB Analytics/Home_Advantages.xlsx')
    return df


def getFBSTeams():
    teams = teamsApi().get_fbs_teams()
    fbsTeams = []
    for team in teams:
        fbsTeams.append(team.school)
    return fbsTeams


def getPredictions(year, startWeek, endWeek):
    #ha = getHomeAdvantages()
    headers = ['Week', 'Home Team', 'Away Team']
    sts = gamesApi().get_team_game_stats(year=year, week=1, team="Alabama")
    for i in sts[0].teams[0]["stats"]:
        headers.append("Away "+i["category"])
    for i in sts[0].teams[0]["stats"]:
        headers.append("Home "+i["category"])
    headers.extend(['Home Moneyline', 'Away Moneyline', 'Spread', 'OverUnder', 'Home ELO', 'Away ELO', 'Home Points',
                    'Away Points', 'Winner', 'Spread Winner'])
    allMatches = pd.DataFrame()
    fbsTeams = getFBSTeams()
    for week in range(startWeek, endWeek):
        #matches = pd.DataFrame([],
        #                       columns=headers)
        games = gamesApi().get_games(year=year, week=week)
        for game in games:
            if game.home_team not in fbsTeams or game.away_team not in fbsTeams:
                continue
            #match = [week, game.home_team, game.away_team]
            line = getLines(year=year, week=week, homeTeam=game.home_team, awayTeam=game.away_team)
            if game.home_points > game.away_points:
                winner = 1
            else:
                winner = 2
            if game.home_points + line[0] > game.away_points:
                spreadWinner = game.home_team+" "+str(line[0])
            else:
                spreadWinner = game.away_team + " " + str(-line[0])
            totalPoints = game.home_points + game.away_points
            if totalPoints > line[1]:
                overUnderWinner = "Over"
            elif totalPoints < line[1]:
                overUnderWinner = "Under"
            else:
                overUnderWinner = "N/a"
            stats = gamesApi().get_team_game_stats(year=year, week=week, team=game.away_team)
            df = json_normalize(stats[0].teams[0]["stats"])
            df = df.transpose()
            df.columns = "Away " + df.iloc[0]
            df = df[1:]

            df.insert(loc=0, column='Week', value=[week])
            df.insert(loc=1, column='Home Team', value=[game.home_team])
            df.insert(loc=2, column='Away Team', value=[game.away_team])

            df2 = json_normalize(stats[0].teams[1]["stats"])
            df2 = df2.transpose()
            df2.columns = "Home " + df2.iloc[0]
            df2 = df2[1:]

            df = df.join(df2)
            df.insert(loc=3, column="Winner", value=[winner])
            df.insert(loc=4, column="Spread Winner", value=[spreadWinner])
            df.insert(loc=5, column="Over/Under Winner", value=[overUnderWinner])
            df.insert(loc=6, column="Home Moneyline", value=[line[2]])
            df.insert(loc=7, column="Away Moneyline", value=[line[3]])
            df.insert(loc=8, column="Spread", value=[line[0]])
            df.insert(loc=9, column="OverUnder", value=[line[1]])
            df.insert(loc=10, column="Home Points", value=[game.home_points])
            df.insert(loc=11, column="Away Points", value=[game.away_points])
            df.insert(loc=12, column="Total", value=[totalPoints])
            df.insert(loc=13, column="Home ELO", value=[game.home_pregame_elo])
            df.insert(loc=14, column="Away ELO", value=[game.away_pregame_elo])

            allMatches = pd.concat([allMatches, df])

    allMatches.fillna(0, inplace=True)
    allMatches[['A', 'B']] = allMatches['Home possessionTime'].str.split(':', expand=True)
    allMatches['Home possessionTime'] = allMatches['A'].astype('float64') * 60 + allMatches['B'].astype('float64')
    allMatches[['A', 'B']] = allMatches['Away possessionTime'].str.split(':', expand=True)
    allMatches['Away possessionTime'] = allMatches['A'].astype('float64') * 60 + allMatches['B'].astype('float64')

    allMatches[['A', 'B']] = allMatches['Home totalPenaltiesYards'].str.split('-', expand=True)
    allMatches.loc[allMatches['A'] == 0, 'Home totalPenaltiesYards'] = 0
    allMatches.loc[allMatches['A'] != 0, 'Home totalPenaltiesYards'] = allMatches['A'].astype('float64') / allMatches['B'].astype('float64')

    allMatches[['A', 'B']] = allMatches['Home completionAttempts'].str.split('-', expand=True)
    allMatches.loc[allMatches['A'] == 0, 'Home completionAttempts'] = 0
    allMatches.loc[allMatches['A'] != 0, 'Home completionAttempts'] = allMatches['A'].astype('float64') / allMatches['B'].astype('float64')

    allMatches[['A', 'B']] = allMatches['Home fourthDownEff'].str.split('-', expand=True)
    allMatches.loc[allMatches['A'] == 0, 'Home fourthDownEff'] = 0
    allMatches.loc[allMatches['A'] != 0, 'Home fourthDownEff'] = allMatches['A'].astype('float64') / allMatches['B'].astype('float64')

    allMatches[['A', 'B']] = allMatches['Home thirdDownEff'].str.split('-', expand=True)
    allMatches.loc[allMatches['A'] == 0, 'Home thirdDownEff'] = 0
    allMatches.loc[allMatches['A'] != 0, 'Home thirdDownEff'] = allMatches['A'].astype('float64') / allMatches['B'].astype('float64')

    allMatches[['A', 'B']] = allMatches['Away totalPenaltiesYards'].str.split('-', expand=True)
    allMatches.loc[allMatches['A'] == 0, 'Away totalPenaltiesYards'] = 0
    allMatches.loc[allMatches['A'] != 0, 'Away totalPenaltiesYards'] = allMatches['A'].astype('float64') / allMatches['B'].astype('float64')

    allMatches[['A', 'B']] = allMatches['Away completionAttempts'].str.split('-', expand=True)
    allMatches.loc[allMatches['A'] == 0, 'Away completionAttempts'] = 0
    allMatches.loc[allMatches['A'] != 0, 'Away completionAttempts'] = allMatches['A'].astype('float64') / allMatches['B'].astype('float64')

    allMatches[['A', 'B']] = allMatches['Away fourthDownEff'].str.split('-', expand=True)
    allMatches.loc[allMatches['A'] == 0, 'Away fourthDownEff'] = 0
    allMatches.loc[allMatches['A'] != 0, 'Away fourthDownEff'] = allMatches['A'].astype('float64') / allMatches['B'].astype('float64')

    allMatches[['A', 'B']] = allMatches['Away thirdDownEff'].str.split('-', expand=True)
    allMatches.loc[allMatches['A'] == 0, 'Away thirdDownEff'] = 0
    allMatches.loc[allMatches['A'] != 0, 'Away thirdDownEff'] = allMatches['A'].astype('float64') / allMatches['B'].astype('float64')

    del allMatches['A']
    del allMatches['B']

    writer = pd.ExcelWriter("Model"+str(year)+".xlsx")
    allMatches.to_excel(writer, 'Model')
    writer.save()
    return allMatches


def winner_loss_fn(y_true, y_pred):
    """
    The function implements the custom loss function

    Inputs
    true : a vector of dimension batch_size, 7. A label encoded version of the output and the backp1_a and backp1_b
    pred : a vector of probabilities of dimension batch_size , 5.

    Returns
    the loss value
    """
    winHome = y_true[:, 0:1]
    winAway = y_true[:, 1:2]
    #spreadHome = y_true[:, 2:3]
    #spreadAway = y_true[:, 3:4]
    no_bet = y_true[:, 2:3]
    odds_a = y_true[:, 3:4]
    odds_b = y_true[:, 4:5]
    gain_loss_vector = K.concatenate([winHome * (odds_a - 1) + (1 - winHome) * -1,
                                      winAway * (odds_b - 1) + (1 - winAway) * -1,
                                      K.zeros_like(odds_a)], axis=1)
    return -1 * K.mean(K.sum(gain_loss_vector * y_pred, axis=1))


def get_data():
    data = getPredictions(2021, 1, 3)
    X = data.values[:, 6:-1]
    y = data.values[:, 3]
    #y_full = np.zeros((X.shape[0], 8))
    #for i, y_i in enumerate(y):
    #    if y_i == 1:
    #        y_full[i, 0] = 1.0
    #        y_full[i, 1] = 1.0
    #    if y_i == 2:
    #        y_full[i, 2] = 1.0
    #        y_full[i, 3] = 1.0
    #    if y_i == 3:
    #        y_full[i, 1] = 1.0
    #        y_full[i, 3] = 1.0
    #        y_full[i, 4] = 1.0
    #    y_full[i, 6] = X[i, 1] # ADD ODDS OF HOME TEAM
    #    y_full[i, 7] = X[i, 2] # ADD ODDS OF AWAY TEAM
    return X, y#y_full, y


def get_model(input_dim, output_dim, base=1000, multiplier=0.25, p=0.2):
    inputs = Input(shape=(input_dim,))
    l = BatchNormalization()(inputs)
    l = Dropout(p)(l)
    n = base
    l = Dense(n, activation='relu')(l)
    l = BatchNormalization()(l)
    l = Dropout(p)(l)
    n = int(n * multiplier)
    l = Dense(n, activation='relu')(l)
    l = BatchNormalization()(l)
    l = Dropout(p)(l)
    n = int(n * multiplier)
    l = Dense(n, activation='relu')(l)
    outputs = Dense(output_dim, activation='softmax')(l)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='Nadam', loss=mse)#odds_loss)
    return model
#X, y = get_data()
# SPLIT THE DATA IN TRAIN AND TEST DATASET.
#train_x, test_x, train_y, test_y, = train_test_split(X,  y)
#model = get_model(len(X), 2, 1000, 0.9, 0.7)
#train_x = np.asarray(train_x).astype('float32')
#train_y = np.asarray(train_y).astype('float32')
#test_x = np.asarray(test_x).astype('float32')
#test_y = np.asarray(test_y).astype('float32')
#history = model.fit(train_x, train_y, validation_data=(test_x, test_y),
#          epochs=200, batch_size=5, callbacks=[EarlyStopping(patience=25),ModelCheckpoint('odds_loss.hdf5',save_best_only=True)])

#print('Training Loss : {}\nValidation Loss : {}'.format(model.evaluate(train_x, train_y), model.evaluate(test_x, test_y)))


def get_analysis(year, startWeek, endWeek):
    headers = ['Week', 'Home Team', 'Away Team']
#    sts = gamesApi().get_team_game_stats(year=year, week=1, team="Alabama")
#    for i in sts[0].teams[0]["stats"]:
#        headers.append("Away "+i["category"])
#    for i in sts[0].teams[0]["stats"]:
#        headers.append("Home "+i["category"])
    headers.extend(['Home Moneyline', 'Away Moneyline', 'Spread', 'OverUnder', 'Home ELO', 'Away ELO', 'Home Points',
                    'Away Points', 'Winner', 'Spread Winner'])
    allMatches = pd.DataFrame()
    fbsTeams = getFBSTeams()
    for week in range(startWeek, endWeek):
        print("Week "+str(week)+" of 13")
        games = gamesApi().get_games(year=year, week=week)
        for game in games:
            if game.home_team not in fbsTeams or game.away_team not in fbsTeams:
                continue
            if game.home_points is None or game.away_points is None:
                continue
            line = getLines(year=year, gameId=game.id)
            if float(game.home_points) > float(game.away_points):
                winner = "Home"
            else:
                winner = "Away"
            if float(game.home_points) + line[0] > float(game.away_points):
                spreadWinner = "Home"
            else:
                spreadWinner = "Away"
            #if -1.5 < float(game.home_points) + line[0] - float(game.away_points) < 1.5:
            #    spreadClose = "Yes"
            #else:
            #    spreadClose = "No"
            #if float(game.home_points) + line[0] - float(game.away_points) >= 1.5 or float(game.home_points) + line[0] - float(game.away_points) <= -1.5:
            #    spreadBet = -0.1
            #elif float(game.home_points) + line[0] - float(game.away_points) == 1 or float(game.home_points) + line[0] - float(game.away_points) == -1:
            #    spreadBet = 0.8
            #else:
            #    spreadBet = 1.6
            totalPoints = float(game.home_points) + float(game.away_points)
            if totalPoints > line[1]:
                overUnderWinner = "Over"
            elif totalPoints < line[1]:
                overUnderWinner = "Under"
            else:
                overUnderWinner = "N/a"
            #if -1.5 < totalPoints - line[1] < 1.5:
            #    overUnderClose = "Yes"
            #else:
            #    overUnderClose = "No"
            #if totalPoints - line[1] >= 1.5 or totalPoints - line[1] <= -1.5:
            #    overUnderBet = -0.1
            #elif totalPoints - line[1] == 1 or totalPoints - line[1] == -1:
            #    overUnderBet = 0.8
            #else:
            #    overUnderBet = 1.6


            df = pd.DataFrame()
            df.insert(loc=0, column='Week', value=[week])
            df.insert(loc=1, column='Home Team', value=[game.home_team])
            df.insert(loc=2, column='Away Team', value=[game.away_team])
            df.insert(loc=3, column="Winner", value=[winner])
            df.insert(loc=4, column="Spread Winner", value=[spreadWinner])
            df.insert(loc=5, column="Over/Under Winner", value=[overUnderWinner])
            df.insert(loc=6, column="Home Moneyline", value=[line[2]])
            df.insert(loc=7, column="Away Moneyline", value=[line[3]])
            df.insert(loc=8, column="Spread", value=[line[0]])
            df.insert(loc=9, column="OverUnder", value=[line[1]])
            df.insert(loc=10, column="Home Points", value=[game.home_points])
            df.insert(loc=11, column="Away Points", value=[game.away_points])
            df.insert(loc=12, column="Total", value=[totalPoints])
            #df.insert(loc=13, column="Spread is close", value=[spreadClose])
            #df.insert(loc=14, column="Spread Bet Value", value=[spreadBet])
            #df.insert(loc=15, column="Over/Under is close", value=[overUnderClose])
            #df.insert(loc=16, column="Over/Under Bet Value", value=[overUnderBet])



            allMatches = pd.concat([allMatches, df])
    return allMatches


def writeToExcel(df, fileName, tabName):
    writer = pd.ExcelWriter(fileName +".xlsx")
    df.to_excel(writer, tabName)
    writer.save()


def get_spread_diff(year):
    games = gamesApi().get_games(year=year)
    spreadDiff = []
    ouDiff = []
    fbsTeams = getFBSTeams()
    for game in games:
        if game.home_team not in fbsTeams or game.away_team not in fbsTeams:
            continue
        print("Week " + str(game.week))
        line = getLines(year=year, gameId=game.id)
        spreadDiff.append(game.home_points + line[0] - game.away_points)
        ouDiff.append(line[1] - game.home_points - game.away_points)
    return [spreadDiff, ouDiff]

#df = get_analysis(2023, 1, 14)
#writer = pd.ExcelWriter("All Matches 2023" + ".xlsx")
#df.to_excel(writer, 'Matches')
#writer.save()

#allOpeners=pd.DataFrame()
allGames=pd.DataFrame()
for year in range(2021, 2024):
    print(year)
#    if year == 2020:
#        continue
    yearAnalysis = get_analysis(year, 1, 14)
    allGames = pd.concat([allGames, yearAnalysis])
writer = pd.ExcelWriter("All Matches 2021-23" + ".xlsx")
allGames.to_excel(writer, 'Matches')
writer.save()
#    print(str(year) + " spread percentage profit = " + str(yearAnalysis['Spread Bet Value'].sum()))
#    print(str(year) + " Over/Under percentage profit = " + str(yearAnalysis['Over/Under Bet Value'].sum()))

#    allOpeners = pd.concat([allOpeners, yearAnalysis])

#print(allGames['Spread is close'].value_counts())
#print(allGames['Spread Winner'].value_counts())
#print(allGames['Winner'].value_counts())
#print(allGames['Spread Bet Value'].value_counts())
#print(allGames['Over/Under is close'].value_counts())
#print(allGames['Over/Under Bet Value'].value_counts())
#print("Spread percentage profit = "+str(allGames['Spread Bet Value'].sum()))
#print("Over/Under percentage profit = "+str(allGames['Over/Under Bet Value'].sum()))
#teams = getFBSTeams()
#df = pd.DataFrame(teams, columns = ["Team", "School"])
#writeToExcel(df, "All Football Teams", "Teams")
#allSpreadDiff = []
#allOUDiff = []
#for year in range(2015, 2022):
#    print(year)
#    if year == 2020:
#        continue
#    diffs = get_spread_diff(year)
#    allSpreadDiff.append(diffs[0])
#    allOUDiff.append(diffs[1])
#print(allSpreadDiff)
#allSpreadDiff = [["+1.5", "+1.5", "-1.5"]]
#allOUDiff = [["+1.5", "-1.5", "-1.5"]]
#spreadCounter = Counter(allSpreadDiff[0])
#ouCounter = Counter(allOUDiff[0])
#df1 = pd.DataFrame.from_dict(spreadCounter, orient='index').reset_index()
#df2 = pd.DataFrame.from_dict(ouCounter, orient='index').reset_index()
#dfSorted1 = df1.sort_values(by='index', inplace=False, ascending=True)
#dfSorted2 = df2.sort_values(by='index', inplace=False, ascending=True)
#dfSorted1['Spread Percent'] = 100*dfSorted1[dfSorted1.columns[1]]/(dfSorted1[dfSorted1.columns[1]].sum())
#dfSorted2['Total Percent'] = 100*dfSorted2[dfSorted2.columns[1]]/(dfSorted2[dfSorted2.columns[1]].sum())

#writeToExcel(dfSorted1, "Spread Difference", "Spread")
#writeToExcel(dfSorted2, "Over or Under Difference", "Over or Under")
