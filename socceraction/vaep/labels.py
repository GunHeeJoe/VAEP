"""Implements the label tranformers of the VAEP framework."""
"""Implements the label tranformers of the VAEP framework."""
import pandas as pd  # type: ignore
from pandera.typing import DataFrame

import socceraction.spadl.config as spadl
from socceraction.spadl.schema import SPADLSchema

from tqdm import tqdm
#레이블 : #k(1~10)개 미래의 결과가 득점이면(같은팀의 득점/상대팀의 자책골), 레이블=1 부여
#그런데, 실점 후에 바로 득점을 하거나 후반전 바로 득점을 하는 경우 이 코드가 타당할까?
#실점 후에 바로 득점을 할 때, 실점했을 때 액션도 score=1이 부여되는 오류가 발생한다. 이는 shift만 사용하다보니까 이러한 현상이 발생함
#후반전 시작 후에 바로 득점을 할 때, 전반전의 마지막 액션도 score=1이 부여되는 오류가 발생함
#즉, 득점/실점 & 전/후반 이것을 기준으로 레이블을 적용해야함
def scores(actions: DataFrame[SPADLSchema], nr_actions: int = 10) -> pd.DataFrame:
    """Determine whether the team possessing the ball scored a goal within the next x actions.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.
    nr_actions : int, default=10  # noqa: DAR103
        Number of actions after the current action to consider.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'scores' and a row for each action set to
        True if a goal was scored by the team possessing the ball within the
        next x actions; otherwise False.
    """

    # merging goals, owngoals and team_ids
    actions['goal'] = actions['type_name'].str.contains('shot') & (
        actions['result_id'] == spadl.results.index('success')
    )
    actions['owngoal'] = actions['type_name'].str.contains('shot') & (
        actions['result_id'] == spadl.results.index('owngoal')
    )

    actions['scores'] = False

    for i in range(len(actions)):
        team_id = actions.loc[i,'team_id']
        period_id = actions.loc[i,'period_id']

        future_actions = actions.loc[i:i+nr_actions-1]
        future_goal_indices = future_actions.loc[future_actions['goal'] | future_actions['owngoal']].index.to_list()

        #미래 10개의 action내 득점/실점이 있는 경우
        if future_goal_indices:
            #가장 가까운 시점의 득점/실점 정보를 활용해야함
            #기존 SPADL의 label.py의 shitf함수를 활용하면, 득실점이 붙어있을 때, score=1 & concede=1인 에러가 발생함
            #득점 후에 바로 실점을 하면 득점에 대한 action의 concede=1이 부여되는 현상이 발생함
            index = future_goal_indices[0]
            
            #미래의 득점/실점이 전반/후반이 바뀐 시점이면 사용해서는 안됨
            #발례)후반전 초기에 득점을 할 경우, 전반전 마지막에 score=1이 부여해서는 안됨
            period_condition = (actions.loc[index,'period_id'] == period_id)

            #우리팀이 득점 | 상대팀이 자책골
            score_condition = ((actions.loc[index,'goal']) & (actions.loc[index,'team_id'] == team_id)) | ((actions.loc[index,'owngoal']) & (actions.loc[index,'team_id'] != team_id))
            actions.loc[i,'scores'] = score_condition & period_condition

    return actions['scores']


def concedes(actions: DataFrame[SPADLSchema], nr_actions: int = 10) -> pd.DataFrame:
    """Determine whether the team possessing the ball conceded a goal within the next x actions.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.
    nr_actions : int, default=10  # noqa: DAR103
        Number of actions after the current action to consider.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'concedes' and a row for each action set to
        True if a goal was conceded by the team possessing the ball within the
        next x actions; otherwise False.
    """
    # merging goals, owngoals and team_ids
    actions['goal'] = actions['type_name'].str.contains('shot') & (
        actions['result_id'] == spadl.results.index('success')
    )
    actions['owngoal'] = actions['type_name'].str.contains('shot') & (
        actions['result_id'] == spadl.results.index('owngoal')
    )

    actions['concedes'] = False

    for i in range(len(actions)):
        team_id = actions.loc[i,'team_id']
        period_id = actions.loc[i,'period_id']

        future_actions = actions.loc[i:i+nr_actions-1]
        future_goal_indices = future_actions[future_actions['goal'] | future_actions['owngoal']].index.to_list()

        #미래 10개의 action내 득점/실점이 있는 경우
        if future_goal_indices:
            #가장 가까운 시점의 득점/실점 정보를 활용해야함
            #기존 SPADL의 label.py의 shitf함수를 활용하면, 득실점이 붙어있을 때, score=1 & concede=1인 에러가 발생함
            #득점 후에 바로 실점을 하면 득점에 대한 action의 concede=1이 부여되는 현상이 발생함
            index = future_goal_indices[0]
            
            #미래의 득점/실점이 전반/후반이 바뀐 시점이면 사용해서는 안됨
            #발례)후반전 초기에 득점을 할 경우, 전반전 마지막에 score=1이 부여해서는 안됨
            period_condition = (actions.loc[index,'period_id'] == period_id)
            
            #우리팀이 득점 | 상대팀이 자책골
            concede_condition = ((actions.loc[index,'goal']) & (actions.loc[index,'team_id'] != team_id)) | ((actions.loc[index,'owngoal']) & (actions.loc[index,'team_id'] == team_id)) 
            actions.loc[i,'concedes'] = concede_condition & period_condition


    return actions['concedes']



def goal_from_shot(actions: DataFrame[SPADLSchema]) -> pd.DataFrame:
    """Determine whether a goal was scored from the current action.

    This label can be use to train an xG model.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'goal' and a row for each action set to
        True if a goal was scored from the current action; otherwise False.
    """
    goals = actions['type_name'].str.contains('shot') & (
        actions['result_id'] == spadl.results.index('success')
    )

    return pd.DataFrame(goals, columns=['goal_from_shot'])
