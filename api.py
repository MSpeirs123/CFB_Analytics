import cfbd

configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = 'TBDORyowV/QVqgKzBeHuESCh2hDjsSeYNFMMuZmz0WK7D1+Swc/N5FCxLhl189Of'
configuration.api_key_prefix['Authorization'] = 'Bearer'
api_config = cfbd.ApiClient(configuration)

#4372016
def gamesApi():
    return cfbd.GamesApi(api_config)


def drivesApi():
    return cfbd.DrivesApi(api_config)


def playsApi():
    return cfbd.PlaysApi(api_config)


def teamsApi():
    return cfbd.TeamsApi(api_config)


def conferencesApi():
    return cfbd.ConferencesApi(api_config)


def venuesApi():
    return cfbd.VenuesApi(api_config)


def coachesApi():
    return cfbd.CoachesApi(api_config)


def playersApi():
    return cfbd.PlayersApi(api_config)


def rankingsApi():
    return cfbd.RankingsApi(api_config)


def bettingApi():
    return cfbd.BettingApi(api_config)


def recruitingApi():
    return cfbd.RecruitingApi(api_config)


def ratingsApi():
    return cfbd.RatingsApi(api_config)


def metricsApi():
    return cfbd.MetricsApi(api_config)


def statsApi():
    return cfbd.StatsApi(api_config)


def draftApi():
    return cfbd.DraftApi(api_config)
