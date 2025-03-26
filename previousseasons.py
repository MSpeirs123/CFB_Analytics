import cfbd
import numpy as np
import pandas as pd
import math
from api import *
from fastai.tabular import *
from fastai.tabular.all import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def getPlayersDraftLost(college, year):
    response = draftApi().get_draft_picks(year=year, college=college)
    return response


def getAllPlayersDraftLost(year):
    fbsTeams = teamsApi().get_fbs_teams()
    teamAttrition = pd.DataFrame([], columns=['College', 'Number of Players Lost', 'ELO Loss Effect'])
    allPlayersLost = draftApi().get_draft_picks(year=year)

    for team in fbsTeams:
        playersLost = []
        eloEffect = 0
        for player in allPlayersLost:
            if player.college_team == team.school:
                playersLost.append(player)
                allPlayersLost.remove(player)
                eloEffect = eloEffect + 1/(int(player.overall)+1)
        eloEffect = 1-eloEffect-0.05
        collegeAttrition = [team.school, len(playersLost), eloEffect]
        row = len(teamAttrition)
        teamAttrition.loc[row] = collegeAttrition
    return teamAttrition


def getPlayersTransferPortal(year):
    return playersApi().get_transfer_portal(year = year)


def getPlayersTransPortalLost(college, year):
    players = getPlayersTransferPortal(year)
    response = [p for p in players if p.origin == college]
    return response


def getPlayersTransPortalGain(college, year):
    players = getPlayersTransferPortal(year)
    response = [p for p in players if p.destination == college]
    return response


def getPlayersRecruiting(college, year):
    response = recruitingApi().get_recruiting_players(team=college, year=year)
    return response


def getAllTeamRecruitingImpact(year):
    fbsTeams = teamsApi().get_fbs_teams()
    recuitingImpact = pd.DataFrame([], columns=['College', 'ELO Gain Effect'])
    freshmanClass = recruitingApi().get_recruiting_teams(year=year)
    sophmoreClass = recruitingApi().get_recruiting_teams(year=year-1)
    juniorClass = recruitingApi().get_recruiting_teams(year=year-2)
    seniorClass = recruitingApi().get_recruiting_teams(year=year-3)

    for team in fbsTeams:
        freshmanEloEffect = 0
        sophmoreEloEffect = 0
        juniorEloEffect = 0
        seniorEloEffect = 0
        for yearClass in freshmanClass:
            if yearClass.team == team.school:
                freshmanClass.remove(yearClass)
                freshmanEloEffect = int(yearClass.points)
                continue
        for yearClass in sophmoreClass:
            if yearClass.team == team.school:
                sophmoreClass.remove(yearClass)
                sophmoreEloEffect = int(yearClass.points)*int(yearClass.points)
                continue
        for yearClass in juniorClass:
            if yearClass.team == team.school:
                juniorClass.remove(yearClass)
                juniorEloEffect = int(yearClass.points)*int(yearClass.points)*int(yearClass.points)
                continue
        for yearClass in seniorClass:
            if yearClass.team == team.school:
                seniorClass.remove(yearClass)
                seniorEloEffect = int(yearClass.points)*int(yearClass.points)*int(yearClass.points)*2
                continue
        eloEffect = 1+(freshmanEloEffect+sophmoreEloEffect+juniorEloEffect+seniorEloEffect)/170000000
        teamRecuitingImpact = [team.school, eloEffect]
        row = len(recuitingImpact)
        recuitingImpact.loc[row] = teamRecuitingImpact
    return recuitingImpact


def getAnnualRatingChange(year):
    draftLoss = getAllPlayersDraftLost(year=year)
    recruitmentGain = getAllTeamRecruitingImpact(year=year)
    ratingChange = pd.DataFrame(draftLoss, columns=['College', 'ELO Loss Effect'])
    ratingChange['ELO Gain Effect']=recruitmentGain['ELO Gain Effect']
    ratingChange['Overall ELO Effect']=ratingChange['ELO Gain Effect']+ratingChange['ELO Loss Effect']-1
    writer = pd.ExcelWriter("Player Turnover.xlsx")
    ratingChange.to_excel(writer, 'Turnover')
    writer.save()
    return ratingChange


def calculateHomeAdvantage():
    fbsTeams = teamsApi().get_fbs_teams()
    totalGames = 0
    totalHomeGames = 0
    totalPointsScored = 0
    totalPointsConceded = 0
    totalHomePointsScored = 0
    totalHomePointsConceded = 0
    homeAdvantages = pd.DataFrame([], columns = ['Team', 'Points Scored Advantage', 'Points Conceded Advantage'])

    for team1 in fbsTeams:
        print(team1.school)
        totalGames = 0
        totalHomeGames = 0
        totalPointsScored = 0
        totalPointsConceded = 0
        totalHomePointsScored = 0
        totalHomePointsConceded = 0
        for team2 in fbsTeams:
            if team1.school == team2.school:
                continue
            history = teamsApi().get_team_matchup(team1=team1.school, team2=team2.school, min_year=2010)
            games=history.games
            for game in games:
                totalGames = totalGames + 1
                if game["homeTeam"] == team1.school:
                    totalPointsScored = totalPointsScored + game["homeScore"]
                    totalPointsConceded = totalPointsConceded + game["awayScore"]
                else:
                    totalPointsScored = totalPointsScored + game["awayScore"]
                    totalPointsConceded = totalPointsConceded + game["homeScore"]
                if game["neutralSite"] == False and game["homeTeam"] == team1.school:
                    totalHomeGames = totalHomeGames + 1
                    totalHomePointsScored = totalHomePointsScored + game["homeScore"]
                    totalHomePointsConceded = totalHomePointsConceded + game["awayScore"]
        if totalGames == 0:
            continue
        if totalHomeGames == 0:
            continue
        averagePointsScored = totalPointsScored/totalGames
        averagePointsConceded = totalPointsConceded/totalGames
        averageHomePointsScored = totalHomePointsScored/totalHomeGames
        averageHomePointsConceded = totalHomePointsConceded/totalHomeGames
        pointsScoredHomeAdvantage = averageHomePointsScored/averagePointsScored - 1
        pointsConcededHomeAdvantage = -(averageHomePointsConceded/averagePointsConceded - 1)
        homeAdvantage = [team1.school, pointsScoredHomeAdvantage, pointsConcededHomeAdvantage]
        row = len(homeAdvantages)
        homeAdvantages.loc[row] = homeAdvantage

    writer = pd.ExcelWriter("Home_Advantages.xlsx")
    homeAdvantages.to_excel(writer, 'Home Advantages')
    writer.save()


def getHomeAdvantages():
    df = pd.read_excel(r'/Users/Matt/PycharmProjects/CFB Analytics/Home_Advantages.xlsx')
    return df


def get_expected_score(rating, opp_rating):
    exp = (opp_rating - rating) / 400
    return 1 / (1 + 10**exp)


def getLines(year, week, homeTeam, awayTeam):
    bets = bettingApi().get_lines(year=year, week=week, home=homeTeam, away=awayTeam)
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
        homeMoneyline = line["homeMoneyline"]
        awayMoneyline = line["awayMoneyline"]
        break
    line = [float(spread), float(overUnder), getDecimalFromAmerican(homeMoneyline), getDecimalFromAmerican(awayMoneyline)]
    return line


def getDecimalFromAmerican(odds):
    if not odds:
        return 0
    if odds > 0:
        decimal = (odds/100) + 1
    else:
        decimal = 1 - (100/odds)
    return decimal


def getPredictions(year):
    rc = getAnnualRatingChange(year=year)
    ha = getHomeAdvantages()
    allMatches = pd.DataFrame([], columns=['Week', 'Home Team', 'Away Team', 'Home Predicted Win', 'Home Win Probability', 'Away Predicted Win', 'Away Win Probability',
                                            'Predicted Winner', 'Favourite', 'Bet', 'Winner', 'Bet Won', 'Spread', 'Points Difference'])
    for week in range(1, 15):
        matches = pd.DataFrame([], columns=['Week', 'Home Team', 'Away Team', 'Home Predicted Win', 'Home Win Probability', 'Away Predicted Win', 'Away Win Probability',
                                            'Predicted Winner', 'Favourite', 'Bet', 'Winner', 'Bet Won', 'Spread', 'Points Difference'])
        games=gamesApi().get_games(year=year, week=week)
        preGameWinProbs=metricsApi().get_pregame_win_probabilities(year=year, week=week)
        for game in games:
            line = getLines(year=year, week=week, homeTeam=game.home_team, awayTeam=game.away_team)

            homePlayerTurnover = rc.loc[rc['College'] == game.home_team]['Overall ELO Effect'].values
            awayPlayerTurnover = rc.loc[rc['College'] == game.away_team]['Overall ELO Effect'].values
            homeTeamAdvantage = ha.loc[ha['Team'] == game.home_team]
            if homeTeamAdvantage['Points Scored Advantage'].empty:
                continue
            try:
                homeRating = int(game.home_pregame_elo)
            except TypeError:
                continue
            try:
                awayRating = int(game.away_pregame_elo)
            except TypeError:
                continue
            try:
                homeWinProb = float(preGameWinProbs.home_win_prob)
            except TypeError:
                continue
            try:
                spread = float(preGameWinProbs.spread)
            except TypeError:
                continue
            awayWinProb = 1-homeWinProb
            newHomeRating = homeRating * (1+homeTeamAdvantage['Points Scored Advantage'].values[0]) * homePlayerTurnover
            newAwayRating = awayRating * (1-homeTeamAdvantage['Points Conceded Advantage'].values[0]) * awayPlayerTurnover
            #totalRatingProb = newHomeRating + newAwayRating

            winner = ""
            predictedWinner = ""
            winnerCorrect = ""
            favourite = ""
            favouriteWin = ""
            if game.home_points > game.away_points:
                winner = "Home"
            else:
                winner = "Away"
            if get_expected_score(newHomeRating, newAwayRating) > get_expected_score(newAwayRating, newHomeRating):
                predictedWinner = "Home"
            else:
                predictedWinner = "Away"
            if winner == predictedWinner:
                winnerCorrect = "Yes"
            else:
                winnerCorrect = "No"
            if homeWinProb > awayWinProb:
                favourite = "Home"
            else:
                favourite = "Away"
            if favourite == winner:
                favouriteWin = "Yes"
            else:
                favouriteWin = "No"
            if favourite == predictedWinner:
                bet = favourite
            else:
                bet = "N/A"
            if bet == winner:
                betWon = "Yes"
            elif bet == "N/A":
                betWon = "N/A"
            else:
                betWon = "No"
            pointsDifference = game.away_points - game.home_points

            match = [week, game.home_team, game.away_team, get_expected_score(newHomeRating, newAwayRating),
                   homeWinProb, get_expected_score(newAwayRating, newHomeRating), awayWinProb, predictedWinner,
                     favourite, bet, winner, betWon, spread, pointsDifference]
            row = len(matches)
            matches.loc[row] = match
        allMatches = pd.concat([allMatches, matches])
        rc['Overall ELO Effect'] = rc['Overall ELO Effect']**(1/2)
    print(allMatches['Bet Won'].value_counts())

    writer = pd.ExcelWriter("All Matches"+str(year)+".xlsx")
    allMatches.to_excel(writer, 'Matches')
    writer.save()

getPredictions(2022)
#getAnnualRatingChange(2022)

