"""Baselines consume exactly the normalized observation and action mask used by DQN."""
from .fcfs import fcfs
from .timetable_policy import timetable_policy
from .priority_policy import priority_policy
from .lookahead import lookahead

POLICIES = {"fcfs": fcfs, "timetable": timetable_policy, "priority": priority_policy, "greedy": lookahead}
