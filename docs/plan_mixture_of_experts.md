# Plan: Mixture of Experts — osobne modele per sekcja mapy

## Pomysł

Podzielić level na sekcje (np. co 1000-2000 scrollu). Każda sekcja ma osobny model trenowany do perfekcji. Router przełącza między modelami na podstawie scroll position.

## Flow treningu

1. Trenuj Model A na sekcji 0-2000 (normalny start) → aż opanuje
2. Save State na granicy sekcji 2000 → trenuj Model B na sekcji 2000-4000 → aż opanuje
3. Save State na 4000 → trenuj Model C → aż opanuje
4. Połącz: router (if scroll < 2000 → A, elif < 4000 → B, else → C)

## Zalety

- Każdy model specjalizuje się w swojej sekcji
- Zero catastrophic forgetting — modele nie nadpisują się wzajemnie
- Trenowanie sekcji 3 nie wymaga przechodzenia sekcji 1+2 za każdym razem (Save State)
- Można trenować sekcje równolegle na różnych maszynach

## Wyzwania do rozwiązania

- Transfer wiedzy: "unikaj pocisków" trzeba nauczyć w każdym modelu osobno
  - Możliwe rozwiązanie: shared CNN backbone, osobne heady per sekcja
  - Lub: pre-train bazowy model na sekcji 1, fine-tune kopie na kolejnych sekcjach
- Granice sekcji: smooth handoff między modelami
  - Możliwe: overlap zone (np. 200px) gdzie oba modele dają Q-values, interpolacja
- Router: prosty (scroll threshold) lub learned (meta-controller)

## Status

Pomysł na przyszłość. Wymaga implementacji:
- Multi-model manager (ładowanie/przełączanie modeli)
- Save State per sekcja (auto-save na granicach)
- Router logic w DQN trainer
- Dashboard: widoczność który model jest aktywny
