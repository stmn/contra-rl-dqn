# Plan: Extra Lives Exploration

## Pomysł

Dać agentowi 30 żyć (`$32 = 29`) zamiast 3 żeby mógł eksplorować dalsze sekcje mapy i zbierać doświadczenia. Potem przywrócić 3 życia do właściwego treningu.

## RAM

- `$32` = Player 1 lives (0 = ostatnie życie, 2 = 3 życia standardowo)
- `$38` = P1 Game Over Status (0 = gra, 1 = game over)
- Źródło: https://datacrystal.tcrf.net/wiki/Contra_(NES)/RAM_map

## Warianty

### A. Stałe 30 żyć
- `nes[0x32] = 29` po każdym reset
- Agent zawsze ma 30 żyć, dociera dalej
- Problem: death penalty -500 × 30 = -15000, total reward ujemny
- Rozwiązanie: zmniejszyć death penalty lub ją wyłączyć

### B. Exploration + training
- Epizod z 30 życiami BEZ treningu (exploration only, zbieraj doświadczenia do buffera)
- W losowym momencie: przywróć 3 życia (`nes[0x32] = 2`), WŁĄCZ trening
- Bufer ma doświadczenia z całej mapy

### C. Konami Code
- Alternatywa: wpisać Up,Up,Down,Down,Left,Right,Left,Right,B,A na title screen
- Daje 30 żyć bez modyfikacji RAM
- Wymaga przejścia przez ekran tytułowy (wolniejszy boot)

## Implementacja (RAM)
```python
# W ContraEnv._boot() lub reset(), po załadowaniu initial_state:
if settings.extra_lives:
    self._nes[0x32] = 29  # 30 lives
```

## Status
Pomysł na przyszłość. Do wdrożenia gdy agent stagnuje i potrzebuje doświadczeń z dalszych sekcji mapy.
