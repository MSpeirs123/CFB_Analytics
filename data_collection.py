import cfbd
import numpy as np
import pandas as pd
import math
from api import *


def getWinner(row):
    if row['Home Points'] > row['Away Points']:
        return 'Home'
    else:
        return 'Away'


def getSpreadWinner(row):
    if row['Home Points'] + row['Spread'] > row['Away Points']:
        return 'Home'
    if row['Home Points'] + row['Spread'] < row['Away Points']:
        return 'Away'
    if row['Home Points'] + row['Spread'] == row['Away Points']:
        return 'Push'


def getTotalWinner(row):
    if row['Home Points'] + row['Away Points'] < row['OverUnder']:
        return 'Under'
    if row['Home Points'] + row['Away Points'] > row['OverUnder']:
        return 'Over'
    if row['Home Points'] + row['Away Points'] == row['OverUnder']:
        return 'Push'


def getFBSTeams():
    teams = teamsApi().get_fbs_teams()
    fbsTeams = []
    for team in teams:
        fbsTeams.append(team.school)
    return fbsTeams


def getDecimalFromAmerican(odds):
    if not odds:
        return 0
    if odds > 0:
        decimal = (odds/100) + 1
    else:
        decimal = 1 - (100/odds)
    return decimal


def writeToExcel(df, fileName, tabName):
    writer = pd.ExcelWriter(fileName + ".xlsx")
    df.to_excel(writer, tabName)
    writer.save()


def getLines(year, gameId):
    bets = bettingApi().get_lines(year=year, game_id=gameId)
    spread = "0"
    overUnder = "0"
    homeMoneyLine = 0
    awayMoneyLine = 0
    lines = bets[0].lines
    for line in lines:
        spread = line["spread"]
        overUnder = line["overUnder"]
        homeMoneyLine = line["homeMoneyline"]
        awayMoneyLine = line["awayMoneyline"]
        if homeMoneyLine:
            break
    homeOdds = getDecimalFromAmerican(homeMoneyLine)
    awayOdds = getDecimalFromAmerican(awayMoneyLine)
    if not spread:
        spread = 0
    if not overUnder:
        overUnder = 0
    if not homeOdds:
        homeOdds = 0
    if not awayOdds:
        awayOdds = 0
    line = [float(spread), float(overUnder), homeOdds, awayOdds]
    return line


def calculateHomeAdvantage():
    fbsTeams = teamsApi().get_fbs_teams()
    homeAdvantages = pd.DataFrame([], columns=['Team', 'Average Points Scored at Home',
                                               'Average Points Conceded at Home',
                                               'Average Points Scored Away (and neutral sites)',
                                               'Average Points Conceded Away (and neutral sites)'])

    for team1 in fbsTeams:
        print(team1.school)
        totalHomeGames = 0
        totalAwayGames = 0
        totalHomePointsScored = 0
        totalHomePointsConceded = 0
        totalAwayPointsScored = 0
        totalAwayPointsConceded = 0

        for year in range(2021, 2024):
            if year == 2020:
                continue
            games = gamesApi().get_games(year=year, team=team1.school)
            for game in games:
                if game.neutral_site is False and game.home_team == team1.school:
                    totalHomeGames = totalHomeGames + 1
                    totalHomePointsScored = totalHomePointsScored + game.home_points
                    totalHomePointsConceded = totalHomePointsConceded + game.away_points
                else:
                    totalAwayGames = totalAwayGames + 1
                    totalAwayPointsScored = totalAwayPointsScored + game.away_points
                    totalAwayPointsConceded = totalAwayPointsConceded + game.home_points
            if totalHomeGames == 0:
                continue
            if totalAwayGames == 0:
                continue
        averageHomePointsScored = totalHomePointsScored / totalHomeGames
        averageHomePointsConceded = totalHomePointsConceded / totalHomeGames
        averageAwayPointsScored = totalAwayPointsScored / totalAwayGames
        averageAwayPointsConceded = totalAwayPointsConceded / totalAwayGames
        homeAdvantage = [team1.school, averageHomePointsScored, averageHomePointsConceded, averageAwayPointsScored, averageAwayPointsConceded]
        row = len(homeAdvantages)
        homeAdvantages.loc[row] = homeAdvantage

    homeAdvantages['Points Scored Difference'] = homeAdvantages['Average Points Scored at Home'] - \
                                                 homeAdvantages['Average Points Scored Away (and neutral sites)']
    homeAdvantages['Points Conceded Difference'] = homeAdvantages['Average Points Conceded Away (and neutral sites)'] - \
                                                 homeAdvantages['Average Points Conceded at Home']
    writer = pd.ExcelWriter("Home_Advantages_2021-23.xlsx")
    homeAdvantages.to_excel(writer, 'Home Advantages')
    writer.save()


def get_analysis(year, startWeek, endWeek):
    allMatches = pd.DataFrame()
    fbsTeams = getFBSTeams()
    for week in range(startWeek, endWeek):
        print("Week " + str(week) + " of 13")
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

            totalPoints = float(game.home_points) + float(game.away_points)
            if totalPoints > line[1]:
                overUnderWinner = "Over"
            elif totalPoints < line[1]:
                overUnderWinner = "Under"
            else:
                overUnderWinner = "Push"

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

            allMatches = pd.concat([allMatches, df])
    return allMatches


def get_btb_analysis(year):
    allMatches = pd.DataFrame()
    fbsTeams = getFBSTeams()
    for team in fbsTeams:
        #print(team)
        if team == "UMass" or team == "Connecticut" or team == "Liberty" or team == "Charlotte" or team == "Coastal Carolina" \
                or team == "James Madison" or team == "Sam Houston State" or team == "Jacksonville State" or team == "Army":
            continue
        print(team)
        btbGames = []
        games = gamesApi().get_games(year=year, away=team)
        weeks = []
        for game in games:
            if game.neutral_site:
                continue
            weeks.append(game.week)
        for i in range(1, len(weeks)-1):
            if weeks[(i+1)] - weeks[i] == 1:
                btbGames.append(games[(i+1)])
        if not btbGames:
            continue
        for game in btbGames:
            if game.away_conference == "Mountain West" or game.away_conference == "American Athletic" or game.away_conference == "Mid-American"\
                    or game.away_conference == "Sun Belt" or game.away_conference == "Conference USA":
                continue
            line = getLines(year=year, gameId=game.id)
            if line[0] < -21:
                continue
#            if float(game.home_points) > float(game.away_points):
#                winner = "Home"
#            else:
#                winner = "Away"
#            if float(game.home_points) + line[0] > float(game.away_points):
#                spreadWinner = "Home"
#            else:
#                spreadWinner = "Away"

#            totalPoints = float(game.home_points) + float(game.away_points)
#            if totalPoints > line[1]:
#                overUnderWinner = "Over"
#            elif totalPoints < line[1]:
#                overUnderWinner = "Under"
#            else:
#                overUnderWinner = "Push"

            df = pd.DataFrame()
            df.insert(loc=0, column='Year', value=[year])
            df.insert(loc=1, column='Week', value=[game.week])
            df.insert(loc=2, column='Home Team', value=[game.home_team])
            df.insert(loc=3, column='Away Team', value=[game.away_team])
#            df.insert(loc=4, column="Winner", value=[winner])
#            df.insert(loc=4, column="Spread Winner", value=[spreadWinner])
#            df.insert(loc=5, column="Over/Under Winner", value=[overUnderWinner])
            df.insert(loc=4, column="Home Moneyline", value=[line[2]])
            df.insert(loc=5, column="Away Moneyline", value=[line[3]])
            df.insert(loc=6, column="Spread", value=[line[0]])
            df.insert(loc=7, column="OverUnder", value=[line[1]])
            df.insert(loc=8, column="Home Points", value=[game.home_points])
            df.insert(loc=9, column="Away Points", value=[game.away_points])
#            df.insert(loc=12, column="Total", value=[totalPoints])

            allMatches = pd.concat([allMatches, df])
    allMatches['Winner'] = allMatches.apply(getWinner, axis=1)
    allMatches['Spread Winner'] = allMatches.apply(getSpreadWinner, axis=1)
    allMatches['Total Winner'] = allMatches.apply(getTotalWinner, axis=1)
    allMatches['Total'] = allMatches['Home Points'] + allMatches['Away Points']

    return allMatches


allGames = pd.DataFrame()
for year in range(2014, 2024):
    print(year)
    if year == 2020:
        continue
    yearAnalysis = get_btb_analysis(year)
    allGames = pd.concat([allGames, yearAnalysis])

writeToExcel(allGames, "Back to Back P5 away 2014-23", "Matches")

#allGames = pd.DataFrame()
#for year in range(2014, 2023):
#    print(year)
#    if year == 2020:
#        continue
#    yearAnalysis = get_analysis(year, 1, 14)
#    allGames = pd.concat([allGames, yearAnalysis])
#writeToExcel(allGames, "All Matches 2014-22", "Matches")

