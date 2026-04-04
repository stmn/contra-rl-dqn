# Plan: Mixture of Experts — Segments

## Pomysł

Podzielić level na segmenty. Każdy segment ma osobny model (sieć + replay buffer). Router przełącza na podstawie scroll position. Agent trenuje wszystkie segmenty jednocześnie.

## Kluczowy problem: rzadkie segmenty

Jeśli agent dochodzi do bossa w 5% epizodów, model bossa dostaje 20x mniej doświadczeń. Rozwiązanie: **auto-zbieranie Save State'ów na granicach segmentów**.

### Auto Save States
- Gdy agent przekracza granicę segmentu → NES state (`nes.save()`, ~28KB) zapisywany do puli
- Przy respawnie w segmencie → losowy state z puli (różne sytuacje startowe)
- 100 stanów = ~2.7MB, 1000 = ~27MB — bez limitu
- Agent widzi różne ścieżki/bronie/pozycje zamiast jednego zamrożonego momentu

## Architektura

### Segments (znaczniki na mapie)
```
segments = [
    {"scroll": 30000},  # granica: start → 30K = Segment 0
    {"scroll": 60000},  # granica: 30K → 60K = Segment 1
    {"scroll": 90000},  # granica: 60K → boss = Segment 2
]
# Segment 0: od startu do 30K
# Segment 3: od 90K do końca levelu (boss)
```
- Ustawiane przez użytkownika w dashboardzie (klik na progress bar lub wpisz wartość)
- Widoczne na Level Progress bar jako pionowe linie
- Brak segmentów = jeden model (jak teraz)

### SegmentManager (`contra/training/segments.py`)
- N instancji QNetwork/HybridQNetwork + N target networks + N replay bufferów
- Pula Save State'ów per segment (auto-zbierane na granicach)
- `get_segment(scroll)` → index segmentu
- `select_action(obs, segment_idx, epsilon)` → forward przez właściwy model
- `store(segment_idx, transition)` → do właściwego buffera
- `train_step(segment_idx)` → trenuj właściwy model
- `save(dir)` / `load(dir)` → wszystkie modele + states

### DQN Trainer zmiany
- Delegacja do SegmentManager zamiast bezpośredniego q_network/buffer
- Epsilon, global_step — wspólne
- Auto-rollback per segment

### Dashboard
- **Segments tab** (zastępuje Practice):
  - Lista segmentów z scroll position, buffer size, status
  - Add/Remove segment
  - Pula Save State'ów per segment (ilość zebranych)
- **Level Progress bar** — znaczniki segmentów jako pionowe linie
- Aktywny segment widoczny (podświetlony)

### Respawn flow
1. Agent ginie → episode ends
2. Nowy episode: wybierz segment do treningu (round-robin lub priorytet)
3. Załaduj losowy Save State z puli wybranego segmentu
4. Lub: graj od początku (segment 0) — automatyczne przełączanie modeli

### Tryb "wszystkie naraz"
Agent gra od początku. Każdy segment używa swojego modelu. Doświadczenia trafiają do właściwego buffera. Na granicach auto-save state do puli. Jeśli agent ginie w segmencie 2, model segmentu 2 się uczy. Model segmentu 0 też się uczył (z początku tego samego runa).

## Wyzwania

1. **Transfer wiedzy** — "unikaj pocisków" trzeba nauczyć w każdym modelu
   - Rozwiązanie: pre-train na segmencie 0, skopiuj wagi jako start dla nowych segmentów
2. **Granice** — smooth handoff między modelami
   - Overlap zone (200px) z interpolacją Q-values
3. **RAM** — N modeli × ~77MB = dużo
   - Modele są małe (~2-5MB wag), 77MB to optimizer state
   - 5 segmentów × 5MB = 25MB wag, akceptowalne
4. **Save State timing** — state zapisany na granicy może nie być idealny
   - Pula 100+ stanów mityguje to — różnorodność

## Status

Do wdrożenia. Wymaga:
- `contra/training/segments.py` — SegmentManager
- Refactor DQN trainer
- Dashboard Segments tab
- API endpoints
- Usunięcie Practice mode
