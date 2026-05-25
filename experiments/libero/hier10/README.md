# LIBERO-Hier10 Scaffold

Status: scaffold only. BDDL generation and validation are the next implementation step.

| ID | Type | Task | Key Test | Source Hint |
|---:|---|---|---|---|
| 0 | multi_object | put alphabet soup and tomato sauce in basket | subgoal decomposition | libero_10:0 |
| 1 | distractor_aware | put soup and tomato sauce in basket while ignoring ketchup | wrong-object / distractor | libero_90:LIVING_ROOM_SCENE1/2 basket tasks |
| 2 | articulated | put black bowl in bottom drawer and close it | open/close stable state | libero_10:3 |
| 3 | multi_stage | put bowl in drawer and close it, then place wine bottle on rack | cross-receptacle planning | libero_90:KITCHEN_SCENE4 drawer/rack tasks |
| 4 | stove_state | turn on stove and put moka pot on it | state + placement | libero_10:2 |
| 5 | microwave_state | put mug in microwave and close it | articulated closure | libero_10:9 |
| 6 | two_plates | put white mug on left plate and yellow-white mug on right plate | spatial binding/order | libero_10:4 |
| 7 | stack_place | stack bowls then place in tray | compositional manipulation | libero_90:LIVING_ROOM_SCENE4 stack bowl tasks |
| 8 | caddy_compartment | place book in back compartment then mug right of caddy | target-region specificity | libero_10/libero_90:STUDY_SCENE1/3 caddy tasks |
| 9 | constraint | put target object in tray without moving distractor | strict success vs final success | libero_90:tray tasks with distractors |
