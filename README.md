# nnlab – Neuronale Netze trainieren und beim Lernen zusehen

Eine Lern-App für Leute, die wissen wollen, was beim Training eines neuronalen
Netzes wirklich passiert. Alles läuft echt: eigene Implementierung von
Vorwärtsrechnung, Backpropagation und Optimierern in Rust, ohne ML-Bibliothek.
Trainiert wird auf MNIST (Ziffern) oder Fashion-MNIST.

## Starten

```
cargo run --release
```

Der erste Build dauert ein paar Minuten. Beim ersten Start lädt die App den
Datensatz (ca. 11 MB) nach `~/Library/Application Support/nnlab`.

Ohne Oberfläche, z. B. für einen Leistungstest:

```
cargo run --release --example bench -- cnn 3 8     # Architektur, Epochen, Threads
cargo run --release --example bench -- mlp 5 8 60000 fashion
```

Die Korrektheit der Backpropagation prüfen (Vergleich mit numerischer Ableitung):

```
cargo test --release
```

## Was man einstellen kann

- **Daten:** MNIST oder Fashion-MNIST, Größe des Trainingssets (klein = Overfitting),
  Normierung, Augmentierung durch zufälliges Verschieben
- **Architektur:** beliebige Folge aus Dense, Conv2D (k×k, same/valid), MaxPool,
  BatchNorm, Dropout und Aktivierungen (ReLU, LeakyReLU, Sigmoid, Tanh, GELU);
  am Ende immer Dense(10) + Softmax. Vorlagen für typische Experimente.
- **Initialisierung:** He, Xavier, LeCun, kleine Normalverteilung, Null
- **Verlust:** Kreuzentropie oder MSE (jeweils nach Softmax), Label Smoothing
- **Optimierer:** SGD, Momentum, Nesterov, RMSProp, Adam, AdamW, L2/Weight Decay,
  Gradient Clipping
- **Lernraten-Plan:** konstant, Stufen, exponentiell, Cosine, Warmup;
  zusätzlich ein Faktor, den man während des Trainings live verstellen kann
- **Rechenleistung:** Batchgröße und Anzahl der CPU-Threads; oben werden
  Durchsatz (Bilder/s) und GFLOP/s angezeigt. Die Zeitlupe bremst das Training,
  damit man zusehen kann.

## Was man sieht

- **Verlauf:** Verlust und Genauigkeit (Train/Test), Lernrate, Gradientennorm,
  Generalisierungslücke pro Epoche
- **Netz & Aktivierungen:** ein Testbild (oder die eigene Zeichnung) läuft durch
  alle Schichten; jede Feature-Map und jedes Neuron live
- **Gewichte:** Filter bzw. Gewichtsbilder jeder Schicht, Histogramme von
  Gewichten und Gradienten
- **Gradienten-Dynamik:** ‖∂L/∂W‖ und Update-Verhältnis ‖ΔW‖/‖W‖ pro Schicht,
  Aktivierungsstatistik, Anteil toter bzw. gesättigter Einheiten
- **Auswertung:** Konfusionsmatrix und die überzeugtesten Fehlklassifikationen
- **Zeichnen:** eigene Ziffern malen und klassifizieren lassen

## Experimente zum Ausprobieren

1. **Vanishing Gradients:** Vorlage „Tiefes Sigmoid-MLP“. Im Tab
   Gradienten-Dynamik liegen die vorderen Schichten Größenordnungen unter den
   hinteren. Dann die Aktivierungen auf ReLU stellen oder BatchNorm einfügen.
2. **Overfitting:** 1000 Trainingsbilder, MLP 2×256 ohne Dropout, 50 Epochen.
   Dann Dropout, L2 oder Augmentierung dazunehmen.
3. **Lernrate:** SGD mit η = 1.0 (divergiert), 0.1, 0.01, 0.001 vergleichen. Oder
   während des Trainings den Live-Faktor verändern.
4. **Symmetrie:** Initialisierung „Null“ – alle Neuronen einer Schicht bleiben gleich.
5. **Batchgröße und Threads:** Batch 8 gegen 256, 1 Thread gegen alle Kerne –
   und dabei die Aktivitätsanzeige beobachten.

## Aufbau

| Datei | Inhalt |
|---|---|
| `src/nn.rs` | Schichten, Backpropagation, Verlust, Optimierer, Gradiententests |
| `src/data.rs` | Download, IDX-Format, Normierung, Augmentierung |
| `src/trainer.rs` | Trainingsschleife im Hintergrund-Thread, Metriken |
| `src/main.rs` | Oberfläche (egui) |
| `src/viz.rs`, `src/draw.rs` | Farbskalen, Kachelbilder, Zeichenfläche |

Matrixprodukte laufen über `matrixmultiply` (SIMD, auf Apple Silicon NEON),
parallelisiert mit rayon. Faltungen werden per im2col auf Matrixprodukte
zurückgeführt.
