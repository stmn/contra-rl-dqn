# Plan: Lives-Based Scroll Reward Multiplier

## Idea
Agent gets more scroll reward when it has more lives. Dying reduces future scroll rewards, making lives more valuable.

## How it works
- 3 lives (full): scroll_reward * 1.0
- 2 lives: scroll_reward * 0.7
- 1 life: scroll_reward * 0.4

## Why it should help
- Agent currently doesn't value lives enough — death penalty -100 is a one-time cost
- With multiplier, dying permanently reduces all future scroll rewards in that episode
- Agent learns: staying alive = more reward per step = higher total reward
- Still moves forward after dying (scroll > 0 > standing) but knows it would be better alive

## Implementation
- Read lives from RAM ($32, BCD format)
- Calculate multiplier based on remaining lives
- Apply to scroll_reward before adding to total_reward
- Lives count: $02 = 2 extra (3 total), $01 = 1 extra (2 total), $00 = last life (1 total)

## No counter-arguments found
- Agent won't stop moving after losing life — scroll still gives positive reward, standing gives penalty
- Incentive is clear: keep lives = keep full reward multiplier
