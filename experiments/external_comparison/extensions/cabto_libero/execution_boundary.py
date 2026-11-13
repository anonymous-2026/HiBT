"""Preserve CABTO's complete execute_step boundary over an async motor carrier.

Only the callback execution boundary changes. The shared step implementation
still enforces every action precondition, stage/attempt/global limit, policy
chunk and seed. Generated-program guards and model-effect checks are not edited.
"""


class OfficialGoalReached(Exception):
    """The unchanged official evaluator became true at a physical step."""


def complete_native_skill(advance, official_goal):
    """Finish one explicitly generated API call before returning to its BT.

    ``advance`` is the original NativeEpisode.execute, not a replacement motor
    policy or symbolic state transition. Its RUNNING means exactly one real
    physical step happened. It returns SUCCESS/FAILURE under the original
    effect/limit rules. Reactivity resumes after this complete API invocation;
    conditions explicitly inside the generated program are still interpreted.
    """
    while True:
        status = advance()
        if status == 'running' and official_goal():
            # Original NativeEpisode.run checks the official evaluator after
            # every physical-step tick. Preserve that immediate success exit,
            # including when later generated code would change the state.
            raise OfficialGoalReached()
        if status != 'running':
            return status
